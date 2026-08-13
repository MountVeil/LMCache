# SPDX-License-Identifier: Apache-2.0
"""ConfKV B3 GPU persistent-crypto bridge.

Keeps persistent KV ciphertext opaque outside the GPU:
store performs GPU seal before D2H; retrieve performs H2D before GPU authenticated open.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat
import threading
from types import ModuleType
from typing import Any

import torch

from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_management import MemoryObj

FRAME_OVERHEAD = 29
KEY_LEN = 16
IV_LEN = 12
_DEFAULT_KEY_PATH = "/run/secrets/confkv/k_store"
_BINDING: ModuleType | None = None
_BINDING_LOCK = threading.Lock()


class ConfKVAuthenticationError(RuntimeError):
    """Persistent KV frame failed GPU AES-GCM authentication."""


def confkv_gpu_crypto_enabled() -> bool:
    return os.getenv("CONFKV_GPU_SEAL_ENABLE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }



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


def _wipe(buf: bytearray) -> None:
    for i in range(len(buf)):
        buf[i] = 0


def _read_key() -> bytearray:
    path = Path(os.getenv("CONFKV_STORE_KEY_PATH", _DEFAULT_KEY_PATH)).expanduser().resolve()
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    buf = bytearray(KEY_LEN + 1)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"ConfKV K_store is not a regular file: {path}")
        if st.st_mode & 0o077:
            raise PermissionError(f"ConfKV K_store must be mode 0600/0400: {path}")
        nread = os.readv(fd, [buf])
    finally:
        os.close(fd)
    if nread != KEY_LEN:
        _wipe(buf)
        raise ValueError(f"ConfKV K_store must contain exactly {KEY_LEN} raw bytes: {path}")
    del buf[KEY_LEN:]
    return buf


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


class ConfKVGpuCrypto:
    """Own the GPU-side ConfKV persistent crypto context.

The context owns one provisioned K_store handle plus reusable
GPU staging for persistent P+29 frames and authentication state.
"""

    def __init__(self, device: Any, *, max_plaintext_len: int) -> None:
        if max_plaintext_len <= 0:
            raise ValueError("max_plaintext_len must be positive")
        self._device = _normalize_cuda_device(device)
        self._lock = threading.Lock()
        self._closed = False
        self._capacity = int(max_plaintext_len)

        binding = _load_binding()
        lib_path = Path(
            os.getenv(
                "CONFKV_GPU_AESGCM_LIB",
                str(_confkv_root() / "experiments/gpu_kv_seal/gpu/libgpu_aesgcm.so"),
            )
        ).expanduser().resolve()
        self._library = binding.GpuAesGcmLibrary(lib_path)

        key_material = _read_key()
        try:
            self._key = binding.GpuAes128GcmKey(
                self._library, key_material, device=int(self._device.index)
            )
        finally:
            _wipe(key_material)

        try:
            self._key.reserve(self._capacity)
            native_frame = int(self._library.frame_size(self._capacity))
            expected_frame = self._capacity + FRAME_OVERHEAD
            if native_frame != expected_frame:
                raise RuntimeError(
                    f"ConfKV frame contract mismatch: native={native_frame}, expected={expected_frame}"
                )
            self._sealed = torch.empty(
                native_frame, dtype=torch.uint8, device=self._device
            )
            self._auth = torch.empty(
                1, dtype=torch.int32, device=self._device
            )
        except Exception:
            self._key.close()
            raise

    def seal_and_copy_d2h(
        self, plaintext_gpu: torch.Tensor, dst: MemoryObj, *, stream: Any
    ) -> None:
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

    def copy_h2d_and_open(
        self, src: MemoryObj, plaintext_gpu: torch.Tensor, *, stream: Any
    ) -> None:
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
        self._key.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def create_confkv_gpu_crypto(
    device: Any, *, max_plaintext_len: int
) -> ConfKVGpuCrypto | None:
    if not confkv_gpu_crypto_enabled():
        return None
    return ConfKVGpuCrypto(device, max_plaintext_len=max_plaintext_len)
