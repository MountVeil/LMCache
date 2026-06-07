# SPDX-License-Identifier: Apache-2.0
"""Mock trusted state manager for TEE-aware KV cache experiments."""

# Standard
from dataclasses import dataclass
from typing import Optional
import hashlib
import hmac
import os
import threading

try:
    # Third Party
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError as exc:  # pragma: no cover - import-time dependency guard
    AESGCM = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# First Party
from lmcache.logging import init_logger
from lmcache_secure.capsule import (
    NONCE_SIZE,
    KvObjectMetadata,
    SecureKvCapsule,
    build_aad,
    deserialize_capsule,
    digest_plaintext,
    manifest_entry_digest,
    manifest_root,
    serialize_capsule,
    serialize_metadata,
)

logger = init_logger(__name__)


class KvSecurityFailure(Exception):
    """Raised when trusted KV state validation fails."""


@dataclass
class ManifestRecord:
    """Trusted manifest state for a sequence of KV objects."""

    session_id: str
    request_id: str
    entry_digests: list[str]
    root_hash: str


class TrustedStateManager:
    """Interface for trusted KV object protection.

    A real TEE implementation would keep root keys, epochs, and manifests in
    enclave/CVM-protected memory. This mock implementation keeps them in-process
    so the LMCache integration points and failure semantics are explicit.
    """

    def derive_session_key(self, session_id: str, model_id: str) -> bytes:
        raise NotImplementedError

    def seal_kv_object(self, metadata: KvObjectMetadata, plaintext: bytes) -> bytes:
        raise NotImplementedError

    def open_kv_object(
        self,
        expected_metadata: KvObjectMetadata,
        stored_bytes: bytes,
    ) -> bytes:
        raise NotImplementedError

    def register_manifest(
        self,
        session_id: str,
        request_id: str,
        entries: list[tuple[KvObjectMetadata, str]],
    ) -> str:
        raise NotImplementedError

    def check_epoch(self, metadata: KvObjectMetadata) -> None:
        raise NotImplementedError

    def verify_before_restore(
        self,
        session_id: str,
        request_id: str,
        entries: list[tuple[KvObjectMetadata, str]],
    ) -> None:
        raise NotImplementedError


class MockTrustedStateManager(TrustedStateManager):
    """In-process TEE mock for secure KV object experiments."""

    def __init__(self, root_key: bytes) -> None:
        if AESGCM is None:
            raise RuntimeError(
                "MockTrustedStateManager requires the 'cryptography' package"
            ) from _IMPORT_ERROR
        if len(root_key) < 32:
            raise ValueError("root key must be at least 32 bytes")
        self._root_key = root_key
        self._latest_versions: dict[str, int] = {}
        self._manifests: dict[tuple[str, str], ManifestRecord] = {}
        self._sealed_identities: set[bytes] = set()
        self._lock = threading.RLock()

    @staticmethod
    def from_config(extra_config: Optional[dict]) -> "MockTrustedStateManager":
        """Create a mock TEE manager from config or environment."""
        key_hex = None
        if extra_config is not None:
            configured_key = extra_config.get("secure_kv_root_key_hex")
            if configured_key is not None:
                key_hex = str(configured_key)
        if key_hex is None:
            key_hex = os.getenv("LMCACHE_SECURE_KV_ROOT_KEY_HEX")
        if key_hex is not None:
            root_key = bytes.fromhex(key_hex)
        else:
            logger.warning(
                "No secure KV root key configured; using a process-local "
                "mock TEE key. Secure KV objects will not survive restart."
            )
            root_key = os.urandom(32)
        return MockTrustedStateManager(root_key)

    def derive_session_key(self, session_id: str, model_id: str) -> bytes:
        """Derive a session/model key from the mock TEE root key."""
        info = f"lmcache-secure-session:{session_id}:{model_id}".encode("utf-8")
        return hmac.new(self._root_key, info, hashlib.sha256).digest()

    def _derive_chunk_key(self, metadata: KvObjectMetadata) -> bytes:
        session_key = self.derive_session_key(metadata.session_id, metadata.model_id)
        info = b"lmcache-secure-chunk:" + serialize_metadata(metadata)
        return hmac.new(session_key, info, hashlib.sha256).digest()

    def _derive_nonce(self, metadata: KvObjectMetadata) -> bytes:
        """Derive a deterministic AES-GCM nonce from trusted identity state.

        Nonce uniqueness comes from the tuple encoded in ``metadata``:
        session_id, cache_key, chunk_id, version, direction, model_id, shape,
        dtype, fmt, and cached positions. The same root key must never be used
        with the same metadata tuple for different plaintexts. Version/epoch
        is therefore part of the trusted state contract.
        """
        nonce_key = hmac.new(
            self._root_key,
            b"lmcache-secure-nonce",
            hashlib.sha256,
        ).digest()
        return hmac.new(nonce_key, serialize_metadata(metadata), hashlib.sha256).digest()[
            :NONCE_SIZE
        ]

    def next_version(self, metadata: KvObjectMetadata) -> int:
        """Advance and return the trusted version for this KV identity."""
        with self._lock:
            current = self._latest_versions.get(metadata.cache_key, 0)
            next_version = current + 1
            self._latest_versions[metadata.cache_key] = next_version
            return next_version

    def current_version(self, cache_key: str) -> int:
        """Return the trusted current version for a KV identity."""
        with self._lock:
            return self._latest_versions.get(cache_key, 0)

    def seal_kv_object(self, metadata: KvObjectMetadata, plaintext: bytes) -> bytes:
        """Encrypt and authenticate a KV object under trusted metadata."""
        with self._lock:
            latest = self._latest_versions.get(metadata.cache_key, 0)
            if metadata.version < latest:
                raise KvSecurityFailure(
                    f"stale version during seal: {metadata.version} < {latest}"
                )
            metadata_bytes = serialize_metadata(metadata)
            if metadata_bytes in self._sealed_identities:
                raise KvSecurityFailure("refusing to reuse AES-GCM nonce identity")
            self._sealed_identities.add(metadata_bytes)
            self._latest_versions[metadata.cache_key] = metadata.version

        aesgcm = AESGCM(self._derive_chunk_key(metadata))
        nonce = self._derive_nonce(metadata)
        ciphertext = aesgcm.encrypt(nonce, plaintext, build_aad(metadata))
        return serialize_capsule(
            SecureKvCapsule(
                metadata=metadata,
                nonce=nonce,
                ciphertext=ciphertext,
            )
        )

    def open_kv_object(
        self,
        expected_metadata: KvObjectMetadata,
        stored_bytes: bytes,
    ) -> bytes:
        """Validate metadata, freshness, and tag before returning plaintext."""
        capsule = deserialize_capsule(stored_bytes)
        if capsule.metadata != expected_metadata:
            raise KvSecurityFailure("metadata mismatch")
        self.check_epoch(expected_metadata)

        expected_nonce = self._derive_nonce(expected_metadata)
        if capsule.nonce != expected_nonce:
            raise KvSecurityFailure("nonce mismatch")

        aesgcm = AESGCM(self._derive_chunk_key(expected_metadata))
        try:
            return aesgcm.decrypt(
                capsule.nonce,
                capsule.ciphertext,
                build_aad(expected_metadata),
            )
        except Exception as exc:
            raise KvSecurityFailure("tag mismatch") from exc

    def register_manifest(
        self,
        session_id: str,
        request_id: str,
        entries: list[tuple[KvObjectMetadata, str]],
    ) -> str:
        """Register a trusted manifest over an ordered KV object sequence."""
        entry_digests = [
            manifest_entry_digest(metadata, plaintext_digest)
            for metadata, plaintext_digest in entries
        ]
        root_hash = manifest_root(entry_digests)
        with self._lock:
            self._manifests[(session_id, request_id)] = ManifestRecord(
                session_id=session_id,
                request_id=request_id,
                entry_digests=entry_digests,
                root_hash=root_hash,
            )
        return root_hash

    def check_epoch(self, metadata: KvObjectMetadata) -> None:
        """Reject stale object versions according to trusted state."""
        with self._lock:
            latest = self._latest_versions.get(metadata.cache_key)
        if latest is not None and metadata.version < latest:
            raise KvSecurityFailure(
                f"rollback detected: {metadata.version} < latest {latest}"
            )

    def verify_before_restore(
        self,
        session_id: str,
        request_id: str,
        entries: list[tuple[KvObjectMetadata, str]],
    ) -> None:
        """Verify an ordered manifest before restoring a KV prefix."""
        with self._lock:
            expected = self._manifests.get((session_id, request_id))
        if expected is None:
            raise KvSecurityFailure("missing manifest")

        entry_digests = [
            manifest_entry_digest(metadata, plaintext_digest)
            for metadata, plaintext_digest in entries
        ]
        root_hash = manifest_root(entry_digests)
        if root_hash != expected.root_hash:
            raise KvSecurityFailure("manifest mismatch")

    def digest_plaintext(self, plaintext: bytes) -> str:
        """Return the trusted digest used by manifests."""
        return digest_plaintext(plaintext)
