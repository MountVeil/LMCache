# SPDX-License-Identifier: Apache-2.0
"""Secure KV capsule serialization and authenticated metadata helpers."""

# Standard
from dataclasses import dataclass
from typing import Any, Optional
import base64
import hashlib
import json

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat

CAPSULE_MAGIC = b"LMCKVSC1"
NONCE_SIZE = 12


def _canonicalize(value: Any) -> Any:
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, MemoryFormat):
        return value.value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return value


@dataclass(frozen=True)
class KvObjectMetadata:
    """Trusted-side expected identity for a KV object."""

    cache_key: str
    shape: tuple[int, ...]
    dtype: str
    fmt: str
    version: int
    session_id: str = "default-session"
    model_id: str = ""
    chunk_id: str = ""
    direction: str = "store"
    cached_positions: Optional[Any] = None

    @staticmethod
    def from_lmcache(
        key: CacheEngineKey,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        version: int,
        cached_positions: Optional[torch.Tensor] = None,
        session_id: str = "default-session",
        direction: str = "store",
    ) -> "KvObjectMetadata":
        """Build trusted expected metadata from LMCache runtime context."""
        return KvObjectMetadata(
            cache_key=key.to_string(),
            shape=tuple(shape),
            dtype=str(dtype),
            fmt=fmt.value,
            version=version,
            session_id=session_id,
            model_id=key.model_name,
            chunk_id=str(key.chunk_hash),
            direction=direction,
            cached_positions=_canonicalize(cached_positions),
        )


@dataclass(frozen=True)
class SecureKvCapsule:
    """Stored encrypted KV object."""

    metadata: KvObjectMetadata
    nonce: bytes
    ciphertext: bytes


def serialize_metadata(metadata: KvObjectMetadata) -> bytes:
    """Serialize metadata canonically for AAD, manifests, and nonce derivation."""
    raw = {
        "cache_key": metadata.cache_key,
        "cached_positions": _canonicalize(metadata.cached_positions),
        "chunk_id": metadata.chunk_id,
        "direction": metadata.direction,
        "dtype": metadata.dtype,
        "fmt": metadata.fmt,
        "model_id": metadata.model_id,
        "session_id": metadata.session_id,
        "shape": list(metadata.shape),
        "version": metadata.version,
    }
    return json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_aad(metadata: KvObjectMetadata) -> bytes:
    """Build AES-GCM AAD from trusted expected metadata."""
    return b"lmcache-secure-kv-v1:" + serialize_metadata(metadata)


def digest_plaintext(plaintext: bytes) -> str:
    """Digest plaintext for trusted manifests."""
    return hashlib.sha256(plaintext).hexdigest()


def manifest_entry_digest(metadata: KvObjectMetadata, plaintext_digest: str) -> str:
    """Digest a manifest entry binding identity, version, and content digest."""
    h = hashlib.sha256()
    h.update(serialize_metadata(metadata))
    h.update(b"\0")
    h.update(plaintext_digest.encode("ascii"))
    return h.hexdigest()


def manifest_root(entry_digests: list[str]) -> str:
    """Compute an order-sensitive manifest root."""
    h = hashlib.sha256()
    for digest in entry_digests:
        h.update(digest.encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def serialize_capsule(capsule: SecureKvCapsule) -> bytes:
    """Serialize an encrypted capsule for untrusted storage."""
    doc = {
        "ciphertext": base64.b64encode(capsule.ciphertext).decode("ascii"),
        "metadata": json.loads(serialize_metadata(capsule.metadata).decode("utf-8")),
        "nonce": base64.b64encode(capsule.nonce).decode("ascii"),
    }
    body = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return CAPSULE_MAGIC + body


def deserialize_capsule(stored_bytes: bytes) -> SecureKvCapsule:
    """Deserialize an encrypted capsule from untrusted storage bytes."""
    if not stored_bytes.startswith(CAPSULE_MAGIC):
        raise ValueError("secure KV capsule has an unknown magic header")
    doc = json.loads(stored_bytes[len(CAPSULE_MAGIC) :].decode("utf-8"))
    raw_meta = doc["metadata"]
    metadata = KvObjectMetadata(
        cache_key=raw_meta["cache_key"],
        cached_positions=raw_meta.get("cached_positions"),
        chunk_id=raw_meta["chunk_id"],
        direction=raw_meta["direction"],
        dtype=raw_meta["dtype"],
        fmt=raw_meta["fmt"],
        model_id=raw_meta["model_id"],
        session_id=raw_meta["session_id"],
        shape=tuple(raw_meta["shape"]),
        version=int(raw_meta["version"]),
    )
    return SecureKvCapsule(
        metadata=metadata,
        nonce=base64.b64decode(doc["nonce"]),
        ciphertext=base64.b64decode(doc["ciphertext"]),
    )
