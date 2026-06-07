# SPDX-License-Identifier: Apache-2.0
"""Local disk encryption helpers for KV cache experiments."""

# Standard
from typing import Any, Callable, Optional
import json
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat

logger = init_logger(__name__)

_MAGIC = b"LMCKVAE1"
_NONCE_SIZE = 12


def _canonicalize(value: Any) -> Any:
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, MemoryFormat):
        return value.value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return value


def build_local_disk_kv_identity_aad(
    key: CacheEngineKey,
    shape: torch.Size,
    dtype: torch.dtype,
    fmt: MemoryFormat,
    cached_positions: Optional[torch.Tensor] = None,
) -> bytes:
    """Build authenticated identity metadata for a local-disk KV block.

    The returned bytes are used as AES-GCM AAD. They are not encrypted, but
    they are authenticated, so ciphertext copied from another block with a
    different key, shape, dtype, format, or cached position set will fail to
    decrypt.
    """
    aad = {
        "version": 1,
        "cache_key": key.to_string(),
        "shape": _canonicalize(shape),
        "dtype": _canonicalize(dtype),
        "fmt": _canonicalize(fmt),
        "cached_positions": _canonicalize(cached_positions),
    }
    return json.dumps(aad, sort_keys=True, separators=(",", ":")).encode("utf-8")


class LocalDiskAesGcmCodec:
    """AES-GCM codec for LocalDiskBackend.

    This experimental codec encrypts the serialized KV bytes before they are
    written to disk. Callers provide AAD derived from the KV block identity so
    valid ciphertexts cannot be silently swapped across incompatible blocks.
    """

    def __init__(
        self,
        key: bytes,
        aad_provider: Optional[Callable[[CacheEngineKey], bytes]] = None,
    ) -> None:
        try:
            # Third Party
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise RuntimeError(
                "Local disk AES-GCM encryption requires the 'cryptography' "
                "package. Install dependencies from requirements/common.txt."
            ) from exc

        if len(key) not in (16, 24, 32):
            raise ValueError("AES-GCM key must be 16, 24, or 32 bytes")

        self._aesgcm = AESGCM(key)
        self._aad_provider = aad_provider or (lambda _key: b"")

    def encrypt(
        self,
        key: CacheEngineKey,
        plaintext: bytes,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Encrypt plaintext KV bytes for storage."""
        nonce = os.urandom(_NONCE_SIZE)
        ciphertext = self._aesgcm.encrypt(
            nonce,
            plaintext,
            self._aad_provider(key) if aad is None else aad,
        )
        return _MAGIC + nonce + ciphertext

    def decrypt(
        self,
        key: CacheEngineKey,
        stored_bytes: bytes,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Decrypt bytes read from storage."""
        if not stored_bytes.startswith(_MAGIC):
            raise ValueError("encrypted KV file has an unknown magic header")
        offset = len(_MAGIC)
        nonce = stored_bytes[offset : offset + _NONCE_SIZE]
        ciphertext = stored_bytes[offset + _NONCE_SIZE :]
        if len(nonce) != _NONCE_SIZE:
            raise ValueError("encrypted KV file is truncated before nonce")
        return self._aesgcm.decrypt(
            nonce,
            ciphertext,
            self._aad_provider(key) if aad is None else aad,
        )


def load_local_disk_aes_gcm_key(extra_config: Optional[dict]) -> bytes:
    """Load the AES-GCM key for local disk KV encryption."""
    key_hex = None
    if extra_config is not None:
        configured_key = extra_config.get("local_disk_aes_gcm_key_hex")
        if configured_key is not None:
            key_hex = str(configured_key)
    if key_hex is None:
        key_hex = os.getenv("LMCACHE_LOCAL_DISK_AES_GCM_KEY_HEX")
    if key_hex is not None:
        return bytes.fromhex(key_hex)

    logger.warning(
        "No local disk AES-GCM key configured; using a process-local random "
        "key. Cached KV files from this run cannot be decrypted after restart."
    )
    return os.urandom(32)
