# SPDX-License-Identifier: Apache-2.0
# Modified for the ConfKV research prototype.
# Based on LMCache; see LICENSE, THIRD_PARTY.md, and MODIFICATIONS.md.

"""Native direct-buffer AES-GCM backend for LMCache."""

import ctypes
import os
from pathlib import Path

import torch
from cryptography.exceptions import InvalidTag

from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import (
    Deserializer,
    SerdeProcessor,
    Serializer,
)
from lmcache.v1.distributed.serde.key_provider import KeyProvider
from lmcache.v1.memory_management import MemoryObj


logger = init_logger(__name__)

_VERSION = 1
_IV_LEN = 12
_TAG_LEN = 16
_HDR_LEN = 1 + _IV_LEN
_FRAME_OVERHEAD = _HDR_LEN + _TAG_LEN


def _default_lib_path() -> Path:
    # .../LMCache/lmcache/v1/distributed/serde/native_aesgcm.py
    # parents[4] == repository root
    repo_root = Path(__file__).resolve().parents[4]

    return (
        repo_root
        / "experiments"
        / "gpu_kv_seal"
        / "native"
        / "libcpu_seal.so"
    )


def _load_native_lib():
    configured = os.getenv(
        "LMCACHE_AESGCM_NATIVE_LIB",
        "",
    ).strip()

    lib_path = (
        Path(configured)
        if configured
        else _default_lib_path()
    )

    if not lib_path.exists():
        raise RuntimeError(
            "Native AES-GCM library not found: "
            f"{lib_path}. "
            "Build experiments/gpu_kv_seal/native/libcpu_seal.so "
            "or set LMCACHE_AESGCM_NATIVE_LIB."
        )

    lib = ctypes.CDLL(
        str(lib_path)
    )

    lib.lmcache_aes128gcm_seal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.lmcache_aes128gcm_seal.restype = ctypes.c_int

    lib.lmcache_aes128gcm_open.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.lmcache_aes128gcm_open.restype = ctypes.c_int

    logger.info(
        "Loaded native AES-GCM backend: %s",
        lib_path,
    )

    return lib


_LIB = None


def _lib():
    global _LIB

    if _LIB is None:
        _LIB = _load_native_lib()

    return _LIB


def _plaintext_bytes(
    layout_desc: MemoryLayoutDesc,
) -> int:

    total = 0

    for shape, dtype in zip(
        layout_desc.shapes,
        layout_desc.dtypes,
        strict=True,
    ):
        n = 1

        for dim in shape:
            n *= int(dim)

        total += (
            n
            * torch.empty(
                (),
                dtype=dtype,
            ).element_size()
        )

    return total


def _byte_view(
    obj: MemoryObj,
):
    return memoryview(
        obj.byte_array
    ).cast("B")


def _address(view) -> int:
    if len(view) == 0:
        raise ValueError(
            "native aesgcm does not support zero-length buffers"
        )

    return ctypes.addressof(
        ctypes.c_ubyte.from_buffer(view)
    )


def _key_holder(
    raw_key: bytes,
):
    if len(raw_key) != 16:
        raise ValueError(
            "native AES-GCM backend currently supports "
            "AES-128 only"
        )

    KeyArray = (
        ctypes.c_ubyte
        * len(raw_key)
    )

    holder = KeyArray.from_buffer_copy(
        raw_key
    )

    return (
        holder,
        ctypes.addressof(holder),
    )


class NativeAesGcmSerializer(Serializer):

    def __init__(
        self,
        key_provider: KeyProvider,
    ) -> None:
        self._keys = key_provider

    def serialize(
        self,
        src: MemoryObj,
        dst: MemoryObj,
        key: ObjectKey,
    ) -> int:

        src_view = _byte_view(src)
        dst_view = _byte_view(dst)

        plaintext_len = len(src_view)

        frame_len = (
            plaintext_len
            + _FRAME_OVERHEAD
        )

        if len(dst_view) < frame_len:
            raise ValueError(
                "native aesgcm: destination buffer too small"
            )

        # Same LMCache per-cache_salt key lookup.
        dek = self._keys.get_key(
            key.cache_salt
        )

        holder, key_ptr = _key_holder(
            dek
        )

        rc = _lib().lmcache_aes128gcm_seal(
            key_ptr,
            _address(src_view),
            plaintext_len,
            _address(dst_view),
            len(dst_view),
        )

        # Keep key memory alive until native call has returned.
        _ = holder

        if rc != 0:
            raise RuntimeError(
                "native aesgcm seal failed: "
                f"rc={rc}"
            )

        return frame_len

    def estimate_serialized_size(
        self,
        layout_desc: MemoryLayoutDesc,
    ) -> int:

        return (
            _plaintext_bytes(layout_desc)
            + _FRAME_OVERHEAD
        )


class NativeAesGcmDeserializer(Deserializer):

    def __init__(
        self,
        key_provider: KeyProvider,
    ) -> None:
        self._keys = key_provider

    def deserialize(
        self,
        src: MemoryObj,
        dst: MemoryObj,
        key: ObjectKey,
    ) -> None:

        src_view = _byte_view(src)
        dst_view = _byte_view(dst)

        plaintext_len = len(dst_view)

        frame_len = (
            plaintext_len
            + _FRAME_OVERHEAD
        )

        if len(src_view) < frame_len:
            raise ValueError(
                "native aesgcm: malformed frame"
            )

        dek = self._keys.get_key(
            key.cache_salt
        )

        holder, key_ptr = _key_holder(
            dek
        )

        rc = _lib().lmcache_aes128gcm_open(
            key_ptr,
            _address(src_view),
            frame_len,
            _address(dst_view),
            plaintext_len,
        )

        _ = holder

        if rc == -11:
            raise InvalidTag(
                "native aesgcm authentication failed"
            )

        if rc != 0:
            raise ValueError(
                "native aesgcm open failed: "
                f"rc={rc}"
            )


def create_native_aesgcm_processor(
    provider: KeyProvider,
    max_workers: int,
) -> SerdeProcessor:

    return AsyncSerdeProcessor(
        NativeAesGcmSerializer(
            provider
        ),
        NativeAesGcmDeserializer(
            provider
        ),
        max_workers=max_workers,
    )
