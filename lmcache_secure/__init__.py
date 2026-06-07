# SPDX-License-Identifier: Apache-2.0
"""TEE-style KV cache protection prototype for LMCache experiments."""

from lmcache_secure.capsule import (
    KvObjectMetadata,
    SecureKvCapsule,
    build_aad,
    deserialize_capsule,
    serialize_capsule,
    serialize_metadata,
)
from lmcache_secure.trusted_state_manager import (
    KvSecurityFailure,
    MockTrustedStateManager,
    TrustedStateManager,
)

__all__ = [
    "KvObjectMetadata",
    "KvSecurityFailure",
    "MockTrustedStateManager",
    "SecureKvCapsule",
    "TrustedStateManager",
    "build_aad",
    "deserialize_capsule",
    "serialize_capsule",
    "serialize_metadata",
]
