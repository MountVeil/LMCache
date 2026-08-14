# SPDX-License-Identifier: Apache-2.0
"""ConfKV GPU key-provider boundary.

The data plane only consumes provisioned GPU key handles. How the key is
established is a control-plane decision:

* DevKeyProvider reads one raw AES-128 test key from a protected file.
  It is only for functional and performance experiments.
* TdxGpuCcKeyProvider reserves the future attested TDX-to-GPU-CC path and
  deliberately fails until that implementation is installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
import stat
from typing import Any

KEY_LEN = 16
DEFAULT_DEV_KEY_PATH = "/run/secrets/confkv/k_store"


def _wipe(buf: bytearray) -> None:
    """Best-effort overwrite of an application-owned mutable key buffer."""
    for idx in range(len(buf)):
        buf[idx] = 0


def _read_secret_file(path: Path, expected_len: int = KEY_LEN) -> bytearray:
    """Read an exact-length regular file without following its final symlink."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd = os.open(path, flags)
    buf = bytearray(expected_len + 1)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"ConfKV key is not a regular file: {path}")
        if info.st_mode & 0o077:
            raise PermissionError(
                f"ConfKV key must be mode 0600/0400: {path}"
            )
        nread = os.readv(fd, [buf])
    finally:
        os.close(fd)

    if nread != expected_len:
        _wipe(buf)
        raise ValueError(
            f"ConfKV key must contain exactly {expected_len} raw bytes: {path}"
        )

    del buf[expected_len:]
    return buf


def _close_keys(keys: list[Any]) -> None:
    for key in keys:
        try:
            key.close()
        except Exception:
            pass
    keys.clear()


class KeyProvider(ABC):
    """Provision one or more equivalent GPU-resident key handles."""

    name: str
    security_mode: str

    @abstractmethod
    def provision_keys(
        self,
        *,
        binding: Any,
        library: Any,
        device: int,
        count: int,
        max_plaintext_len: int,
    ) -> list[Any]:
        """Return count independent handles for the same K_store."""


class DevKeyProvider(KeyProvider):
    """File-backed provider for non-CC data-plane experiments."""

    name = "dev"
    security_mode = "dev-no-cc"

    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.getenv(
            "CONFKV_DEV_KEY_PATH",
            os.getenv("CONFKV_STORE_KEY_PATH", DEFAULT_DEV_KEY_PATH),
        )

        # Do not call Path.resolve() here. The final path component must remain
        # unresolved so O_NOFOLLOW can reject symbolic links.
        self.path = Path(configured).expanduser().absolute()

    def provision_keys(
        self,
        *,
        binding: Any,
        library: Any,
        device: int,
        count: int,
        max_plaintext_len: int,
    ) -> list[Any]:
        if count <= 0:
            raise ValueError("key handle count must be positive")
        if max_plaintext_len <= 0:
            raise ValueError("max_plaintext_len must be positive")

        key_material = _read_secret_file(self.path)
        keys: list[Any] = []

        try:
            for _ in range(count):
                key = binding.GpuAes128GcmKey(
                    library,
                    key_material,
                    device=device,
                )
                try:
                    key.reserve(max_plaintext_len)
                except Exception:
                    key.close()
                    raise
                keys.append(key)

            return keys
        except Exception:
            _close_keys(keys)
            raise
        finally:
            # The host copy is no longer needed after the independent GPU
            # handles have expanded the key and reserved their workspaces.
            _wipe(key_material)


class TdxGpuCcKeyProvider(KeyProvider):
    """Reserved provider for attested TDX-to-GPU-CC provisioning."""

    name = "tdx_gpu_cc"
    security_mode = "tdx-gpu-cc"

    def provision_keys(
        self,
        *,
        binding: Any,
        library: Any,
        device: int,
        count: int,
        max_plaintext_len: int,
    ) -> list[Any]:
        del binding, library, device, count, max_plaintext_len
        raise RuntimeError(
            "TdxGpuCcKeyProvider is not installed in this pre-CC build. "
            "Use CONFKV_KEY_PROVIDER=dev only for non-security experiments."
        )


def create_key_provider(name: str | None = None) -> KeyProvider:
    """Create the configured provider without silently falling back."""
    selected = (
        name or os.getenv("CONFKV_KEY_PROVIDER", "dev")
    ).strip().lower()

    if selected == "dev":
        return DevKeyProvider()

    if selected in {"tdx_gpu_cc", "tdx-gpu-cc"}:
        return TdxGpuCcKeyProvider()

    raise ValueError(
        "CONFKV_KEY_PROVIDER must be one of: dev, tdx_gpu_cc; "
        f"got {selected!r}"
    )


__all__ = [
    "DEFAULT_DEV_KEY_PATH",
    "KEY_LEN",
    "DevKeyProvider",
    "KeyProvider",
    "TdxGpuCcKeyProvider",
    "create_key_provider",
]
