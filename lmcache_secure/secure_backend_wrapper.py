# SPDX-License-Identifier: Apache-2.0
"""Generic secure backend wrapper sketch for untrusted LMCache backends."""

# Standard
from typing import Protocol

# First Party
from lmcache_secure.capsule import KvObjectMetadata
from lmcache_secure.trusted_state_manager import (
    KvSecurityFailure,
    TrustedStateManager,
)


class ByteBackend(Protocol):
    """Minimal untrusted byte backend protocol."""

    def put_bytes(self, key: str, data: bytes) -> None:
        """Store opaque bytes."""

    def get_bytes(self, key: str) -> bytes | None:
        """Load opaque bytes or return None."""


class SecureBackendWrapper:
    """Protect a byte backend with a trusted state manager.

    This wrapper captures the intended architecture:
    LMCache backends remain untrusted storage/movement agents, while the trusted
    state manager performs key derivation, nonce derivation, AAD construction,
    version checks, and tag verification.
    """

    def __init__(self, backend: ByteBackend, tsm: TrustedStateManager) -> None:
        self._backend = backend
        self._tsm = tsm

    def put(self, key: str, plaintext: bytes, metadata: KvObjectMetadata) -> None:
        """Seal and store an object."""
        capsule = self._tsm.seal_kv_object(metadata, plaintext)
        self._backend.put_bytes(key, capsule)

    def get(self, key: str, expected_metadata: KvObjectMetadata) -> bytes | None:
        """Load, verify, and open an object, returning None on validation fail."""
        stored = self._backend.get_bytes(key)
        if stored is None:
            return None
        try:
            return self._tsm.open_kv_object(expected_metadata, stored)
        except KvSecurityFailure:
            return None
