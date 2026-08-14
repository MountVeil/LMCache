# SPDX-License-Identifier: Apache-2.0
"""ConfKV B3 GPU persistent-crypto bridge.

Keeps persistent KV ciphertext opaque outside the GPU:
store performs GPU seal before D2H; retrieve performs H2D before GPU authenticated open.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import threading
from types import ModuleType
from typing import Any, Literal, Sequence

import torch

from lmcache.v1.confkv.key_provider import (
    KeyProvider,
    create_key_provider,
)
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_management import MemoryObj

FRAME_OVERHEAD = 29
IV_LEN = 12
_BINDING: ModuleType | None = None
_BINDING_LOCK = threading.Lock()


class ConfKVAuthenticationError(RuntimeError):
    """Persistent KV frame failed GPU AES-GCM authentication."""


def confkv_gpu_crypto_enabled() -> bool:
    return os.getenv("CONFKV_GPU_SEAL_ENABLE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }



def confkv_gpu_execution_mode() -> Literal["naive", "optimized"]:
    """Return the selected data-plane implementation.

    ``naive`` serializes every chunk on the LMCache stream and introduces a
    CPU-visible completion barrier. ``optimized`` uses batched staging and
    independent crypto streams/slots.
    """
    selected = os.getenv("CONFKV_GPU_MODE", "optimized").strip().lower()
    if selected not in {"naive", "optimized"}:
        raise ValueError(
            "CONFKV_GPU_MODE must be 'naive' or 'optimized'; "
            f"got {selected!r}"
        )
    return selected


def validate_confkv_l2_adapter_config(
    config: Any,
) -> None:
    """Reject L2 transforms that would reinterpret a B3 sealed frame.

    In B3, the object handed to the L2 stack is already the final
    application-level persistent representation:

        [version][IV][ciphertext][tag]

    Therefore an LMCache L2 serde layer must not serialize,
    quantize, compress, or encrypt that object again.

    This validator intentionally does not blacklist storage backends
    merely because they use a different transport mechanism. Backends
    are allowed as long as they accept the MemoryObj as an opaque
    P+29-byte payload.
    """
    if not confkv_gpu_crypto_enabled():
        return

    serde_config = getattr(
        config,
        "serde_config",
        None,
    )

    if serde_config is None:
        return

    serde_type = getattr(
        serde_config,
        "type",
        type(serde_config).__name__,
    )

    raise RuntimeError(
        "ConfKV B3 is enabled, but L2 serde is also configured "
        f"({serde_type!r}). B3 already persists GPU-sealed P+29 "
        "frames, so every L2 adapter must use serde=None."
    )


def layout_nbytes(layout: MemoryLayoutDesc) -> int:
    total = 0
    for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True):
        numel = 1
        for dim in shape:
            numel *= int(dim)
        total += numel * torch.empty((), dtype=dtype).element_size()
    return total


def sealed_layout_desc(plaintext_layout: MemoryLayoutDesc) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([layout_nbytes(plaintext_layout) + FRAME_OVERHEAD])],
        dtypes=[torch.uint8],
    )


def persistent_layout_descs(
    group_layout_descs: dict[int, MemoryLayoutDesc],
) -> dict[int, MemoryLayoutDesc]:
    """Return L1/L2 object layouts for the active ConfKV mode.

    B0/B1/B2 keep LMCache's plaintext KV layouts. B3 allocates P+29
    opaque byte objects so L2 prefetch restores the exact persistent
    ciphertext frame rather than allocating a P-byte plaintext object.
    """
    if not confkv_gpu_crypto_enabled():
        return group_layout_descs
    return {
        gid: sealed_layout_desc(desc)
        for gid, desc in group_layout_descs.items()
    }


def _confkv_root() -> Path:
    root = os.getenv("CONFKV_ROOT")
    return Path(root).expanduser().resolve() if root else Path(__file__).resolve().parents[4]


def _load_binding() -> ModuleType:
    global _BINDING
    with _BINDING_LOCK:
        if _BINDING is not None:
            return _BINDING
        path = Path(
            os.getenv(
                "CONFKV_GPU_AESGCM_BINDING",
                str(_confkv_root() / "experiments/gpu_kv_seal/gpu/gpu_aesgcm.py"),
            )
        ).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ConfKV GPU binding not found: {path}")
        spec = importlib.util.spec_from_file_location("_confkv_gpu_aesgcm", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load ConfKV GPU binding: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "GpuAesGcmLibrary") or not hasattr(module, "GpuAes128GcmKey"):
            raise ImportError(f"Unexpected ConfKV GPU binding API: {path}")
        _BINDING = module
        return module


def _normalize_cuda_device(device: Any) -> torch.device:
    dev = torch.device("cuda", int(device)) if isinstance(device, int) else torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(f"ConfKV GPU seal requires CUDA, got {device!r}")
    index = dev.index if dev.index is not None else int(torch.cuda.current_device())
    return torch.device("cuda", index)


def _stream_ptr(stream: Any) -> int:
    ptr = getattr(stream, "cuda_stream", None)
    if ptr is None:
        raise RuntimeError("ConfKV requires a CUDA stream exposing cuda_stream")
    return int(ptr)


class _CryptoSlot:
    """One independently schedulable GPU AES-GCM pipeline slot."""

    def __init__(
        self,
        *,
        key: Any,
        sealed: torch.Tensor,
        auth: torch.Tensor,
        stream: Any,
    ) -> None:
        self.key = key
        self.sealed = sealed
        self.auth = auth
        self.stream = stream


class ConfKVGpuCrypto:
    """Own the GPU-side ConfKV persistent crypto context.

The context owns one provisioned K_store handle plus reusable
GPU staging for persistent P+29 frames and authentication state.
"""

    def __init__(
        self,
        device: Any,
        *,
        max_plaintext_len: int,
        num_slots: int = 1,
        key_provider: KeyProvider | None = None,
    ) -> None:
        if max_plaintext_len <= 0:
            raise ValueError(
                "max_plaintext_len must be positive"
            )
        if num_slots <= 0:
            raise ValueError(
                "num_slots must be positive"
            )

        self._device = _normalize_cuda_device(device)
        self._lock = threading.Lock()
        self._closed = False
        self._capacity = int(max_plaintext_len)
        self._execution_mode = confkv_gpu_execution_mode()
        self._key_provider = key_provider or create_key_provider()

        configured = os.getenv(
            "CONFKV_GPU_CRYPTO_SLOTS",
            "",
        ).strip()

        if configured:
            try:
                requested_slots = int(configured)
            except ValueError as exc:
                raise ValueError(
                    "CONFKV_GPU_CRYPTO_SLOTS must be "
                    "a positive integer"
                ) from exc

            if requested_slots <= 0:
                raise ValueError(
                    "CONFKV_GPU_CRYPTO_SLOTS must be "
                    "a positive integer"
                )
        else:
            # 8 is the prototype default; never exceed the number
            # of independent LMCache batch staging slots.
            requested_slots = min(
                int(num_slots),
                8,
            )

        self._num_slots = (
            1
            if self._execution_mode == "naive"
            else min(int(num_slots), requested_slots)
        )

        binding = _load_binding()

        lib_path = Path(
            os.getenv(
                "CONFKV_GPU_AESGCM_LIB",
                str(
                    _confkv_root()
                    / "experiments/gpu_kv_seal/gpu/libgpu_aesgcm.so"
                ),
            )
        ).expanduser().resolve()

        self._library = binding.GpuAesGcmLibrary(
            lib_path
        )

        keys = self._key_provider.provision_keys(
            binding=binding,
            library=self._library,
            device=int(self._device.index),
            count=self._num_slots,
            max_plaintext_len=self._capacity,
        )
        self._slots: list[_CryptoSlot] = []

        try:
            with torch.cuda.device(self._device):
                for key in keys:
                    native_frame = int(
                        self._library.frame_size(
                            self._capacity
                        )
                    )

                    expected_frame = (
                        self._capacity
                        + FRAME_OVERHEAD
                    )

                    if native_frame != expected_frame:
                        raise RuntimeError(
                            "ConfKV frame contract mismatch: "
                            f"native={native_frame}, "
                            f"expected={expected_frame}"
                        )

                    sealed = torch.empty(
                        native_frame,
                        dtype=torch.uint8,
                        device=self._device,
                    )

                    auth = torch.empty(
                        1,
                        dtype=torch.int32,
                        device=self._device,
                    )

                    slot_stream = torch.cuda.Stream(
                        device=self._device
                    )

                    self._slots.append(
                        _CryptoSlot(
                            key=key,
                            sealed=sealed,
                            auth=auth,
                            stream=slot_stream,
                        )
                    )

        except Exception:
            for key in keys:
                try:
                    key.close()
                except Exception:
                    pass

            self._slots.clear()
            raise

        if not self._slots:
            raise RuntimeError(
                "ConfKV failed to create GPU crypto slots"
            )

        # Preserve compatibility with the original single-item
        # helpers and their existing unit tests.
        self._key = self._slots[0].key
        self._sealed = self._slots[0].sealed
        self._auth = self._slots[0].auth

    @property
    def num_slots(self) -> int:
        """Number of independent GPU crypto pipelines."""
        return self._num_slots

    @property
    def execution_mode(self) -> Literal["naive", "optimized"]:
        return self._execution_mode

    @property
    def key_provider_name(self) -> str:
        return self._key_provider.name

    @property
    def security_mode(self) -> str:
        return self._key_provider.security_mode

    def quiesce_after_error(self, *, stream: Any | None = None) -> None:
        """Synchronize all streams before error-path buffers are released.

        This is intentionally absent from the steady-state path. It protects
        pinned L1 and GPU staging lifetimes when an enqueue operation raises
        after earlier work has already reached one or more CUDA streams.
        """
        if stream is not None:
            synchronize = getattr(stream, "synchronize", None)
            if synchronize is not None:
                synchronize()

        for slot in getattr(self, "_slots", ()):
            synchronize = getattr(slot.stream, "synchronize", None)
            if synchronize is not None:
                synchronize()

    def _validate_plaintext(
        self,
        plaintext_gpu: torch.Tensor,
    ) -> tuple[int, int]:
        if self._closed:
            raise RuntimeError(
                "ConfKV GPU crypto is closed"
            )

        if plaintext_gpu.device != self._device:
            raise RuntimeError(
                "ConfKV staging device mismatch: "
                f"{plaintext_gpu.device} != {self._device}"
            )

        if (
            plaintext_gpu.dtype != torch.uint8
            or plaintext_gpu.ndim != 1
        ):
            raise ValueError(
                "ConfKV expects flat uint8 "
                "object-group staging"
            )

        if not plaintext_gpu.is_contiguous():
            raise ValueError(
                "ConfKV staging must be contiguous"
            )

        plaintext_len = int(
            plaintext_gpu.nbytes
        )

        if plaintext_len > self._capacity:
            raise ValueError(
                "ConfKV staging exceeds registered capacity: "
                f"{plaintext_len} > {self._capacity}"
            )

        return (
            plaintext_len,
            plaintext_len + FRAME_OVERHEAD,
        )

    def seal_batch_and_copy_d2h(
        self,
        plaintext_gpus: Sequence[torch.Tensor],
        dsts: Sequence[MemoryObj],
        *,
        stream: Any,
    ) -> None:
        """Seal one LMCache chunk batch in parallel, then D2H ciphertext.

        The LMCache stream has already gathered each chunk into a different
        batch-slot staging buffer.

        A gather-ready event fans the batch out to independent crypto streams.
        Each slot performs:

            plaintext staging
              -> AES-GCM seal
              -> sealed P+29 staging
              -> D2H

        Different slots may execute concurrently. The LMCache stream then
        waits on every used slot before it is allowed to reuse staging buffers
        or publish finish_write().
        """
        if len(plaintext_gpus) != len(dsts):
            raise ValueError(
                "ConfKV store batch length mismatch"
            )

        if not plaintext_gpus:
            return

        validated: list[tuple[int, int]] = []

        for plaintext_gpu, dst in zip(
            plaintext_gpus,
            dsts,
            strict=True,
        ):
            plaintext_len, frame_len = (
                self._validate_plaintext(
                    plaintext_gpu
                )
            )

            if dst.get_size() != frame_len:
                raise ValueError(
                    "ConfKV sealed destination mismatch: "
                    f"{dst.get_size()} != {frame_len}"
                )

            validated.append(
                (plaintext_len, frame_len)
            )

        with (
            self._lock,
            torch.cuda.device(self._device),
        ):
            # All gather kernels preceding this record on the LMCache stream
            # must finish before any crypto slot consumes plaintext staging.
            gather_ready = torch.cuda.Event(
                enable_timing=False
            )
            gather_ready.record(stream)

            used_slots: list[int] = []
            used_set: set[int] = set()

            for idx, (
                plaintext_gpu,
                dst,
            ) in enumerate(
                zip(
                    plaintext_gpus,
                    dsts,
                    strict=True,
                )
            ):
                slot_idx = (
                    idx % self._num_slots
                )
                slot = self._slots[slot_idx]

                if slot_idx not in used_set:
                    slot.stream.wait_event(
                        gather_ready
                    )
                    used_set.add(slot_idx)
                    used_slots.append(slot_idx)

                plaintext_len, frame_len = (
                    validated[idx]
                )

                # Existing frame semantics: fresh random 96-bit IV per frame.
                iv = os.urandom(IV_LEN)

                with torch.cuda.stream(
                    slot.stream
                ):
                    sealed = slot.sealed[
                        :frame_len
                    ]

                    slot.key.seal_async(
                        src_ptr=int(
                            plaintext_gpu.data_ptr()
                        ),
                        plaintext_len=plaintext_len,
                        dst_ptr=int(
                            sealed.data_ptr()
                        ),
                        dst_capacity=frame_len,
                        iv=iv,
                        stream_ptr=_stream_ptr(
                            slot.stream
                        ),
                    )

                    # The copy helper is non-blocking. Because the slot stream
                    # is current here, seal -> D2H remains FIFO within this
                    # slot while other slots can progress independently.
                    lmcache_memcpy_async_d2h(
                        sealed,
                        dst,
                    )

            # Join crypto/copy streams back into the LMCache stream.
            #
            # This is GPU-side synchronization only; Store does not block the
            # CPU here.
            for slot_idx in used_slots:
                done = torch.cuda.Event(
                    enable_timing=False
                )
                done.record(
                    self._slots[
                        slot_idx
                    ].stream
                )
                stream.wait_event(done)

    def seal_and_copy_d2h(
        self, plaintext_gpu: torch.Tensor, dst: MemoryObj, *, stream: Any
    ) -> None:
        # Optimized production instances use the parallel slot scheduler even
        # for one item. Naive mode intentionally falls through to the single
        # LMCache-stream implementation and synchronizes once per chunk.
        if (
            hasattr(self, "_slots")
            and getattr(self, "_execution_mode", "naive") == "optimized"
        ):
            self.seal_batch_and_copy_d2h(
                [plaintext_gpu],
                [dst],
                stream=stream,
            )
            return

        if self._closed:
            raise RuntimeError("ConfKV GPU crypto context is closed")
        if plaintext_gpu.device != self._device:
            raise RuntimeError(
                f"ConfKV staging device mismatch: {plaintext_gpu.device} != {self._device}"
            )
        if plaintext_gpu.dtype != torch.uint8 or plaintext_gpu.ndim != 1:
            raise ValueError("ConfKV expects flat uint8 object-group staging")
        if not plaintext_gpu.is_contiguous():
            raise ValueError("ConfKV staging must be contiguous")

        plaintext_len = int(plaintext_gpu.nbytes)
        if plaintext_len > self._capacity:
            raise ValueError(
                f"ConfKV staging exceeds registered capacity: {plaintext_len} > {self._capacity}"
            )
        frame_len = plaintext_len + FRAME_OVERHEAD
        if dst.get_size() != frame_len:
            raise ValueError(
                f"ConfKV sealed destination mismatch: {dst.get_size()} != {frame_len}"
            )

        # Host enqueue lock + one fixed LMCache stream preserve the key handle's
        # non-overlap invariant and the required gather -> seal -> D2H order.
        with self._lock:
            sealed = self._sealed[:frame_len]
            self._key.seal_async(
                src_ptr=int(plaintext_gpu.data_ptr()),
                plaintext_len=plaintext_len,
                dst_ptr=int(sealed.data_ptr()),
                dst_capacity=frame_len,
                iv=os.urandom(IV_LEN),
                stream_ptr=_stream_ptr(stream),
            )
            lmcache_memcpy_async_d2h(sealed, dst)

            if getattr(self, "_execution_mode", "naive") == "naive":
                synchronize = getattr(stream, "synchronize", None)
                if synchronize is None:
                    raise RuntimeError(
                        "ConfKV CUDA stream does not support synchronize()"
                    )
                synchronize()

    def copy_h2d_and_open_batch(
        self,
        srcs: Sequence[MemoryObj],
        plaintext_gpus: Sequence[torch.Tensor],
        *,
        stream: Any,
    ) -> None:
        """Parallel H2D + authenticated GPU open for one LMCache batch.

        No paged-KV scatter is allowed until every chunk in the current wave
        authenticates successfully.

        A buffers-free event also prevents the crypto streams from overwriting
        LMCache staging buffers while the previous batch's scatter is still
        consuming them.
        """
        if len(srcs) != len(plaintext_gpus):
            raise ValueError(
                "ConfKV load batch length mismatch"
            )

        if not srcs:
            return

        validated: list[tuple[int, int]] = []

        for src, plaintext_gpu in zip(
            srcs,
            plaintext_gpus,
            strict=True,
        ):
            plaintext_len, frame_len = (
                self._validate_plaintext(
                    plaintext_gpu
                )
            )

            if src.get_size() != frame_len:
                raise ValueError(
                    "ConfKV sealed source mismatch: "
                    f"{src.get_size()} != {frame_len}"
                )

            validated.append(
                (plaintext_len, frame_len)
            )

        synchronize = getattr(
            stream,
            "synchronize",
            None,
        )

        if synchronize is None:
            raise RuntimeError(
                "ConfKV CUDA stream does not "
                "support synchronize()"
            )

        with (
            self._lock,
            torch.cuda.device(self._device),
        ):
            # Everything already queued on the LMCache stream -- including
            # the preceding batch's scatter -- must stop using these staging
            # buffers before slot streams may overwrite them.
            buffers_free = torch.cuda.Event(
                enable_timing=False
            )
            buffers_free.record(stream)

            for wave_start in range(
                0,
                len(srcs),
                self._num_slots,
            ):
                wave_end = min(
                    wave_start
                    + self._num_slots,
                    len(srcs),
                )

                used: list[
                    tuple[int, int]
                ] = []

                for absolute_idx in range(
                    wave_start,
                    wave_end,
                ):
                    slot_idx = (
                        absolute_idx
                        - wave_start
                    )

                    slot = self._slots[
                        slot_idx
                    ]

                    src = srcs[
                        absolute_idx
                    ]

                    plaintext_gpu = (
                        plaintext_gpus[
                            absolute_idx
                        ]
                    )

                    plaintext_len, frame_len = (
                        validated[
                            absolute_idx
                        ]
                    )

                    slot.stream.wait_event(
                        buffers_free
                    )

                    with torch.cuda.stream(
                        slot.stream
                    ):
                        sealed = slot.sealed[
                            :frame_len
                        ]

                        lmcache_memcpy_async_h2d(
                            src,
                            sealed,
                        )

                        slot.auth.zero_()

                        slot.key.open_async(
                            src_ptr=int(
                                sealed.data_ptr()
                            ),
                            frame_len=frame_len,
                            dst_ptr=int(
                                plaintext_gpu.data_ptr()
                            ),
                            plaintext_len=plaintext_len,
                            auth_ok_ptr=int(
                                slot.auth.data_ptr()
                            ),
                            stream_ptr=_stream_ptr(
                                slot.stream
                            ),
                        )

                    done = torch.cuda.Event(
                        enable_timing=False
                    )
                    done.record(
                        slot.stream
                    )

                    # Main LMCache stream cannot advance to scatter until
                    # every open in this wave is complete.
                    stream.wait_event(done)

                    used.append(
                        (
                            absolute_idx,
                            slot_idx,
                        )
                    )

                # One CPU-visible authentication barrier per wave, not one
                # barrier per chunk.
                synchronize()

                failures = [
                    absolute_idx
                    for (
                        absolute_idx,
                        slot_idx,
                    ) in used
                    if int(
                        self._slots[
                            slot_idx
                        ].auth.item()
                    )
                    != 1
                ]

                if failures:
                    # Native open already zeroizes plaintext for the failing
                    # frame. More importantly, the caller has not enqueued any
                    # scatter for this batch.
                    raise ConfKVAuthenticationError(
                        "ConfKV persistent KV authentication "
                        f"failed for batch indices {failures}; "
                        "refusing scatter"
                    )

    def copy_h2d_and_open(
        self, src: MemoryObj, plaintext_gpu: torch.Tensor, *, stream: Any
    ) -> None:
        if (
            hasattr(self, "_slots")
            and getattr(self, "_execution_mode", "naive") == "optimized"
        ):
            self.copy_h2d_and_open_batch(
                [src],
                [plaintext_gpu],
                stream=stream,
            )
            return

        """Copy one persistent P+29 frame to GPU and authenticate/open it.

        Ordering is deliberately strict for the first correctness-oriented B3
        implementation:

            CPU sealed MemoryObj
                -> H2D sealed staging
                -> GPU AES-GCM verify/open
                -> synchronize this LMCache CUDA stream
                -> inspect auth_ok
                -> return only on authentication success

        The caller performs paged-KV scatter only after this method returns.
        Therefore a failed tag/version check raises before scatter can be
        enqueued. The native GPU primitive also zeroizes ``plaintext_gpu`` on
        authentication failure.
        """
        if self._closed:
            raise RuntimeError("ConfKV GPU crypto context is closed")
        if plaintext_gpu.device != self._device:
            raise RuntimeError(
                f"ConfKV staging device mismatch: {plaintext_gpu.device} != {self._device}"
            )
        if plaintext_gpu.dtype != torch.uint8 or plaintext_gpu.ndim != 1:
            raise ValueError("ConfKV expects flat uint8 object-group staging")
        if not plaintext_gpu.is_contiguous():
            raise ValueError("ConfKV staging must be contiguous")

        plaintext_len = int(plaintext_gpu.nbytes)
        if plaintext_len > self._capacity:
            raise ValueError(
                f"ConfKV staging exceeds registered capacity: {plaintext_len} > {self._capacity}"
            )
        frame_len = plaintext_len + FRAME_OVERHEAD
        if src.get_size() != frame_len:
            raise ValueError(
                f"ConfKV sealed source mismatch: {src.get_size()} != {frame_len}"
            )

        with self._lock:
            sealed = self._sealed[:frame_len]

            # The surrounding LMCache retrieve code has cache_context.stream as
            # the current CUDA stream. Thus H2D -> open executes FIFO on the
            # same stream used later by the scatter kernel.
            lmcache_memcpy_async_h2d(src, sealed)
            self._auth.zero_()
            self._key.open_async(
                src_ptr=int(sealed.data_ptr()),
                frame_len=frame_len,
                dst_ptr=int(plaintext_gpu.data_ptr()),
                plaintext_len=plaintext_len,
                auth_ok_ptr=int(self._auth.data_ptr()),
                stream_ptr=_stream_ptr(stream),
            )

            # First B3 load implementation favors a hard auth-before-scatter
            # boundary over overlap. We synchronize only this stream, not the
            # whole device, then expose the 32-bit auth result to the CPU.
            synchronize = getattr(stream, "synchronize", None)
            if synchronize is None:
                raise RuntimeError("ConfKV CUDA stream does not support synchronize()")
            synchronize()
            auth_ok = int(self._auth.item())

            if auth_ok != 1:
                raise ConfKVAuthenticationError(
                    "ConfKV persistent KV authentication failed; refusing scatter"
                )

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        slots = getattr(
            self,
            "_slots",
            None,
        )

        # Backward compatibility for CPU fake objects used by the original
        # unit tests.
        if slots is None:
            key = getattr(
                self,
                "_key",
                None,
            )
            if key is not None:
                key.close()
            return

        # Teardown may block; steady-state Store does not.
        # Never destroy a handle/workspace while its stream still uses it.
        for slot in slots:
            slot.stream.synchronize()

        for slot in slots:
            slot.key.close()

        slots.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def create_confkv_gpu_crypto(
    device: Any,
    *,
    max_plaintext_len: int,
    num_slots: int = 1,
) -> ConfKVGpuCrypto | None:
    if not confkv_gpu_crypto_enabled():
        return None

    return ConfKVGpuCrypto(
        device,
        max_plaintext_len=max_plaintext_len,
        num_slots=num_slots,
    )
