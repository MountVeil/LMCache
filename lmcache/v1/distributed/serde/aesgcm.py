# SPDX-License-Identifier: Apache-2.0
"""AES-GCM encryption serde for L2-at-rest confidentiality.

Encrypts KV cache bytes on store (L1->L2) and decrypts on load (L2->L1),
keyed per ``cache_salt`` so each tenant's ciphertext is distinct. Content is
treated as an opaque byte blob (not tensor-typed), an exact round-trip.

Scope is the L2 tier only: L1 and L0 hold plaintext. See
``docs/design/v1/distributed/serde/aesgcm.md`` for the threat model, key
model, and residual-metadata notes.
"""

# Standard
from dataclasses import dataclass
import os
from typing import Sequence

# Third Party
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider, KeyProvider
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

# Wire frame per chunk: [1B version][IV_LEN IV][ciphertext || TAG_LEN tag].
_VERSION = 1
_IV_LEN = 12  # 96-bit nonce (GCM native size)
_TAG_LEN = 16  # GCM authentication tag
_HDR_LEN = 1 + _IV_LEN
_FRAME_OVERHEAD = _HDR_LEN + _TAG_LEN

_SUPPORTED_AES_BITS = (128, 256)

# HKDF ``info`` domain-separation prefix for keys derived for this serde.
_HKDF_INFO_PREFIX = b"lmcache-l2-aesgcm-v1"


def _byte_view(obj: MemoryObj) -> memoryview:
    """Return a zero-copy byte view over a memory object."""
    return memoryview(obj.byte_array).cast("B")


def _plaintext_bytes(layout_desc: MemoryLayoutDesc) -> int:
    """Return the total byte size of the KV data described by ``layout_desc``."""
    total = 0
    for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
        n = 1
        for dim in shape:
            n *= int(dim)
        total += n * torch.empty((), dtype=dtype).element_size()
    return total


@dataclass(frozen=True)
class PackedObjectRange:
    """Location of one independently authenticated object in a packed buffer."""

    object_index: int
    offset: int
    length: int


@dataclass(frozen=True)
class ObjectPackGroup:
    """One per-layer group of objects sharing a contiguous host buffer.

    ``first_object`` is an index into that layer's ordered hit-object list,
    rather than the object's logical KV-cache index. Keeping the plan in
    terms of the hit list lets the restore scheduler retain the real object
    keys and GPU destinations without making the pack planner depend on them.
    """

    layer_index: int
    first_object: int
    object_count: int
    packed_bytes: int

    @property
    def coalesced(self) -> bool:
        """Whether this group combines more than one object."""
        return self.object_count > 1


def plan_uniform_object_packs(
    layer_hit_counts: Sequence[int],
    object_size: int,
    *,
    min_pack_bytes: int = 4 << 20,
    max_pack_bytes: int = 8 << 20,
) -> list[ObjectPackGroup]:
    """Plan balanced 4--8 MiB packs for uniform-size hit objects.

    Planning is performed independently for each layer, and objects retain
    their order in that layer's hit list. Objects of at least
    ``min_pack_bytes`` are returned as singleton groups: they are already
    large enough to decrypt/H2D pipeline directly.

    For smaller objects, the planner first minimizes the number of groups
    subject to the maximum pack size, then distributes objects as evenly as
    possible among those groups. This avoids a small final tail such as
    ``[8, 1] MiB`` when ``[5, 4] MiB`` is possible. If the layer contains too
    little data, or the object-size granularity makes the interval
    impossible, the result contains the smallest unavoidable underfilled
    group; it never exceeds ``max_pack_bytes``.

    Args:
        layer_hit_counts: Number of hit objects in each layer.
        object_size: Plaintext bytes per object. All objects must be equal.
        min_pack_bytes: Preferred inclusive lower bound (default 4 MiB).
        max_pack_bytes: Hard inclusive upper bound (default 8 MiB).

    Returns:
        Per-layer groups expressed as slices of each layer's hit-object list.
    """
    if object_size <= 0:
        raise ValueError("object pack: object_size must be positive")
    if min_pack_bytes <= 0 or max_pack_bytes < min_pack_bytes:
        raise ValueError("object pack: invalid preferred size interval")
    if any(count < 0 for count in layer_hit_counts):
        raise ValueError("object pack: layer hit counts must be non-negative")

    groups: list[ObjectPackGroup] = []

    # Objects that already reach the preferred lower bound should bypass
    # coalescing, even when they exceed the nominal upper bound.
    if object_size >= min_pack_bytes:
        for layer_index, hit_count in enumerate(layer_hit_counts):
            groups.extend(
                ObjectPackGroup(layer_index, object_index, 1, object_size)
                for object_index in range(hit_count)
            )
        return groups

    max_objects = max_pack_bytes // object_size
    if max_objects == 0:
        # Unreachable under the validation above, but keep an explicit guard
        # if the interval policy changes later.
        raise ValueError("object pack: one object exceeds the maximum pack size")

    for layer_index, hit_count in enumerate(layer_hit_counts):
        if hit_count == 0:
            continue

        # The fewest transfers that can respect the hard upper bound.
        group_count = (hit_count + max_objects - 1) // max_objects

        # A balanced quotient/remainder partition maximizes the smallest group
        # and prevents a pathological undersized tail.
        base_count, larger_group_count = divmod(hit_count, group_count)
        first_object = 0
        for group_index in range(group_count):
            count = base_count + (group_index < larger_group_count)
            groups.append(
                ObjectPackGroup(
                    layer_index=layer_index,
                    first_object=first_object,
                    object_count=count,
                    packed_bytes=count * object_size,
                )
            )
            first_object += count

    return groups


class AesGcmSerializer(Serializer):
    """Encrypt KV bytes with AES-GCM, keyed per ``cache_salt``.

    Args:
        key_provider: Supplies the AES key for each key's ``cache_salt``.
    """

    def __init__(self, key_provider: KeyProvider) -> None:
        self._keys = key_provider

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Encrypt ``src`` into ``dst`` as ``[version][iv][ciphertext||tag]``.

        Args:
            src: Source KV bytes (read-locked).
            dst: Destination byte buffer, sized by ``estimate_serialized_size``.
            key: Object key; ``key.cache_salt`` selects the encryption key.

        Returns:
            Number of bytes written to ``dst``.
        """
        dek = self._keys.get_key(key.cache_salt)
        iv = os.urandom(_IV_LEN)
        ciphertext = AESGCM(dek).encrypt(iv, bytes(src.byte_array), None)
        # Cast to native "B": MemoryObj.byte_array is a ctypes-backed memoryview
        # with format "<B", which does not support slice assignment.
        out = memoryview(dst.byte_array).cast("B")
        out[0:1] = bytes((_VERSION,))
        out[1:_HDR_LEN] = iv
        out[_HDR_LEN : _HDR_LEN + len(ciphertext)] = ciphertext
        return _HDR_LEN + len(ciphertext)

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return exact framed size: plaintext + version + IV + tag.

        AES-GCM ciphertext is the plaintext length plus a 16-byte tag (no
        padding), so this is exact rather than an upper bound.
        """
        return _plaintext_bytes(layout_desc) + _FRAME_OVERHEAD


class AesGcmDeserializer(Deserializer):
    """Decrypt AES-GCM ciphertext produced by :class:`AesGcmSerializer`.

    Args:
        key_provider: Supplies the AES key for each key's ``cache_salt``.
    """

    def __init__(self, key_provider: KeyProvider) -> None:
        self._keys = key_provider

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Decrypt ``src``'s frame into ``dst``'s KV buffer.

        The ciphertext length is taken from ``dst``'s size (the plaintext
        length), not ``src``, which may be padded larger than the stored frame.

        Args:
            src: Source byte buffer holding the stored frame (may be padded).
            dst: Destination KV buffer (write-locked); its size is the plaintext
                length.
            key: Object key; ``key.cache_salt`` selects the decryption key.

        Raises:
            ValueError: If the frame header/version is malformed.
            cryptography.exceptions.InvalidTag: If the ciphertext was tampered
                with or the wrong key is used (surfaces as a load miss).
        """
        out = memoryview(dst.byte_array).cast("B")
        plaintext_len = len(out)
        blob = bytes(src.byte_array)
        frame_end = _HDR_LEN + plaintext_len + _TAG_LEN
        if len(blob) < frame_end or blob[0] != _VERSION:
            raise ValueError("aesgcm serde: malformed frame or unknown version")
        dek = self._keys.get_key(key.cache_salt)
        plaintext = AESGCM(dek).decrypt(
            blob[1:_HDR_LEN], blob[_HDR_LEN:frame_end], None
        )
        out[: len(plaintext)] = plaintext


class DirectAesGcmSerializer(AesGcmSerializer):
    """AES-GCM serializer that writes ciphertext directly into ``dst``."""

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Encrypt ``src`` directly into the caller-provided frame buffer."""
        plaintext = _byte_view(src)
        out = _byte_view(dst)
        frame_len = len(plaintext) + _FRAME_OVERHEAD
        if len(out) < frame_len:
            raise ValueError(
                "aesgcm serde: destination buffer is smaller than the frame"
            )

        dek = self._keys.get_key(key.cache_salt)
        iv = os.urandom(_IV_LEN)
        out[0] = _VERSION
        out[1:_HDR_LEN] = iv
        written = AESGCM(dek).encrypt_into(
            iv,
            plaintext,
            None,
            out[_HDR_LEN:frame_len],
        )
        if written != len(plaintext) + _TAG_LEN:
            raise RuntimeError(
                "aesgcm serde: encrypt_into returned an unexpected length"
            )
        return _HDR_LEN + written


class DirectAesGcmDeserializer(AesGcmDeserializer):
    """AES-GCM deserializer supporting direct and packed destinations."""

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Decrypt ``src`` directly into ``dst`` without a plaintext temporary."""
        self.decrypt_into(src, _byte_view(dst), key)

    def decrypt_into(
        self,
        src: MemoryObj,
        dst: memoryview,
        key: ObjectKey,
    ) -> int:
        """Authenticate and decrypt an object into a caller-provided region.

        Args:
            src: Source byte buffer holding one stored AES-GCM frame.
            dst: Exact-size writable plaintext destination. It may be a slice
                of a larger packed host buffer and must not overlap ``src``.
            key: Object key whose ``cache_salt`` selects the decryption key.

        Returns:
            Number of plaintext bytes written to ``dst``.

        Raises:
            ValueError: If the frame is malformed or ``dst`` has an invalid
                size or overlaps the ciphertext input.
            cryptography.exceptions.InvalidTag: If authentication fails.

        Notes:
            Callers must not consume ``dst`` unless this method returns
            successfully. On authentication failure its contents are
            unspecified and should be discarded or zeroed before reuse.
        """
        out = memoryview(dst).cast("B")
        plaintext_len = len(out)
        frame = _byte_view(src)
        frame_end = _HDR_LEN + plaintext_len + _TAG_LEN
        if len(frame) < frame_end or frame[0] != _VERSION:
            raise ValueError("aesgcm serde: malformed frame or unknown version")
        dek = self._keys.get_key(key.cache_salt)
        written = AESGCM(dek).decrypt_into(
            frame[1:_HDR_LEN],
            frame[_HDR_LEN:frame_end],
            None,
            out,
        )
        if written != plaintext_len:
            raise RuntimeError(
                "aesgcm serde: decrypt_into returned an unexpected length"
            )
        return written

    def decrypt_pack_into(
        self,
        src_objs: Sequence[MemoryObj],
        dst: memoryview,
        keys: Sequence[ObjectKey],
        plaintext_sizes: Sequence[int],
    ) -> list[PackedObjectRange]:
        """Decrypt independently authenticated objects into one packed buffer.

        Args:
            src_objs: Encrypted object frames in restore order.
            dst: Writable contiguous host buffer. It may be larger than the
                resulting pack; only the returned ranges contain plaintext.
            keys: Object keys aligned with ``src_objs``.
            plaintext_sizes: Plaintext byte lengths aligned with ``src_objs``.

        Returns:
            Offset and length of every object in ``dst``.

        Raises:
            ValueError: If argument lengths differ, a size is negative, the
                destination is too small, or a frame is malformed.
            cryptography.exceptions.InvalidTag: If any object fails its own
                AES-GCM authentication. The caller must discard the whole pack.
        """
        count = len(src_objs)
        if len(keys) != count or len(plaintext_sizes) != count:
            raise ValueError("aesgcm pack: sources, keys, and sizes must align")
        if any(size < 0 for size in plaintext_sizes):
            raise ValueError("aesgcm pack: plaintext sizes must be non-negative")

        out = memoryview(dst).cast("B")
        packed_size = sum(plaintext_sizes)
        if len(out) < packed_size:
            raise ValueError(
                f"aesgcm pack: destination has {len(out)} bytes, "
                f"requires {packed_size}"
            )

        ranges: list[PackedObjectRange] = []
        offset = 0
        for object_index, (src, key, size) in enumerate(
            zip(src_objs, keys, plaintext_sizes, strict=True)
        ):
            end = offset + size
            self.decrypt_into(src, out[offset:end], key)
            ranges.append(PackedObjectRange(object_index, offset, size))
            offset = end
        return ranges


def _create_aesgcm_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    """Build the aesgcm serde from ``SerdeConfig.kwargs``.

    Kwargs:
        key_provider: ``"hkdf"`` (default). ``"keyring"`` is reserved for
            per-tenant provisioned keys and is not yet implemented.
        aes_bits: ``128`` (default) or ``256``.
        master_key_path: Path to the master-key file (required for ``hkdf``).
        max_workers: Serde thread-pool size (default ``1``).
        backend: ``"stock"`` (default) preserves LMCache's allocating API;
            ``"direct"`` uses ``encrypt_into`` / ``decrypt_into`` and exposes
            packed direct-buffer decryption. ``LMCACHE_AESGCM_BACKEND`` is used
            only when this option is omitted.

    Raises:
        ValueError: On an unsupported ``key_provider`` / ``aes_bits``, or an
            empty ``master_key_path``.
        OSError: If ``master_key_path`` is set but cannot be read.
    """
    provider_name = str(kwargs.get("key_provider", "hkdf"))
    aes_bits = int(kwargs.get("aes_bits", 128))  # type: ignore[call-overload]
    if aes_bits not in _SUPPORTED_AES_BITS:
        raise ValueError(
            f"aes_bits must be one of {_SUPPORTED_AES_BITS}, got {aes_bits}"
        )

    if provider_name == "hkdf":
        master_key_path = str(kwargs.get("master_key_path", ""))
        if not master_key_path:
            raise ValueError("aesgcm serde 'hkdf' requires 'master_key_path'")
        with open(master_key_path, "rb") as f:
            master_key = f.read()
        provider: KeyProvider = HkdfKeyProvider(
            master_key, key_len=aes_bits // 8, info_prefix=_HKDF_INFO_PREFIX
        )
    else:
        raise ValueError(
            f"unsupported key_provider {provider_name!r} (only 'hkdf' is implemented)"
        )

    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[call-overload]
    backend = str(
        kwargs.get(
            "backend",
            os.getenv("LMCACHE_AESGCM_BACKEND", "stock"),
        )
    ).strip().lower()

    if backend in {"stock", "original"}:
        serializer: Serializer = AesGcmSerializer(provider)
        deserializer: Deserializer = AesGcmDeserializer(provider)
    elif backend in {"direct", "copy_free", "copy-free"}:
        serializer = DirectAesGcmSerializer(provider)
        deserializer = DirectAesGcmDeserializer(provider)
    else:
        raise ValueError(
            f"unsupported aesgcm backend {backend!r}; expected 'stock' or 'direct'"
        )

    logger.info("AES-GCM serde backend: %s", backend)
    return AsyncSerdeProcessor(
        serializer,
        deserializer,
        max_workers=max_workers,
    )


register_serde_factory("aesgcm", _create_aesgcm_serde)
