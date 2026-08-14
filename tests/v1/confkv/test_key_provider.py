# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from lmcache.v1.confkv.key_provider import (
    DevKeyProvider,
    TdxGpuCcKeyProvider,
    create_key_provider,
)
from lmcache.v1.distributed.serde.native_aesgcm import _default_lib_path


class _FakeKey:
    instances: list["_FakeKey"] = []
    fail_reserve_at: int | None = None

    def __init__(self, _library, material, *, device: int):
        self.material = bytes(material)
        self.device = device
        self.capacity: int | None = None
        self.closed = False
        self.index = len(self.instances)
        self.instances.append(self)

    def reserve(self, capacity: int) -> None:
        if self.index == self.fail_reserve_at:
            raise RuntimeError("reserve failed")
        self.capacity = capacity

    def close(self) -> None:
        self.closed = True


class _FakeBinding:
    GpuAes128GcmKey = _FakeKey


@pytest.fixture(autouse=True)
def _reset_fake_key():
    _FakeKey.instances.clear()
    _FakeKey.fail_reserve_at = None


def _write_key(path: Path, data: bytes = bytes(range(16))) -> None:
    path.write_bytes(data)
    path.chmod(0o600)


def test_dev_provider_provisions_independent_handles(tmp_path):
    path = tmp_path / "k_store"
    _write_key(path)

    keys = DevKeyProvider(path).provision_keys(
        binding=_FakeBinding,
        library=object(),
        device=3,
        count=4,
        max_plaintext_len=4096,
    )

    assert len(keys) == 4
    assert all(key.material == bytes(range(16)) for key in keys)
    assert all(key.device == 3 for key in keys)
    assert all(key.capacity == 4096 for key in keys)
    assert all(not key.closed for key in keys)


def test_dev_provider_rejects_world_readable_key(tmp_path):
    path = tmp_path / "k_store"
    _write_key(path)
    path.chmod(0o644)

    with pytest.raises(PermissionError, match="0600/0400"):
        DevKeyProvider(path).provision_keys(
            binding=_FakeBinding,
            library=object(),
            device=0,
            count=1,
            max_plaintext_len=1,
        )


def test_dev_provider_rejects_wrong_key_length(tmp_path):
    path = tmp_path / "k_store"
    _write_key(path, b"short")

    with pytest.raises(ValueError, match="exactly 16 raw bytes"):
        DevKeyProvider(path).provision_keys(
            binding=_FakeBinding,
            library=object(),
            device=0,
            count=1,
            max_plaintext_len=1,
        )


def test_dev_provider_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    link = tmp_path / "k_store"
    _write_key(target)
    link.symlink_to(target)

    with pytest.raises(OSError):
        DevKeyProvider(link).provision_keys(
            binding=_FakeBinding,
            library=object(),
            device=0,
            count=1,
            max_plaintext_len=1,
        )


def test_dev_provider_closes_partial_provision_on_failure(tmp_path):
    path = tmp_path / "k_store"
    _write_key(path)
    _FakeKey.fail_reserve_at = 2

    with pytest.raises(RuntimeError, match="reserve failed"):
        DevKeyProvider(path).provision_keys(
            binding=_FakeBinding,
            library=object(),
            device=0,
            count=4,
            max_plaintext_len=4096,
        )

    assert len(_FakeKey.instances) == 3
    assert all(key.closed for key in _FakeKey.instances)


def test_tdx_provider_fails_closed_until_installed():
    with pytest.raises(RuntimeError, match="not installed"):
        TdxGpuCcKeyProvider().provision_keys(
            binding=_FakeBinding,
            library=object(),
            device=0,
            count=1,
            max_plaintext_len=1,
        )


def test_create_key_provider(monkeypatch):
    monkeypatch.setenv("CONFKV_KEY_PROVIDER", "dev")
    assert isinstance(create_key_provider(), DevKeyProvider)

    monkeypatch.setenv("CONFKV_KEY_PROVIDER", "tdx_gpu_cc")
    assert isinstance(create_key_provider(), TdxGpuCcKeyProvider)

    monkeypatch.setenv("CONFKV_KEY_PROVIDER", "typo")
    with pytest.raises(ValueError, match="CONFKV_KEY_PROVIDER"):
        create_key_provider()


def test_native_cpu_library_defaults_to_confkv_superproject():
    expected = (
        Path(__file__).resolve().parents[4]
        / "experiments/gpu_kv_seal/native/libcpu_seal.so"
    )
    assert _default_lib_path() == expected
