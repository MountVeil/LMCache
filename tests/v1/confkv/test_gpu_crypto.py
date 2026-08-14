# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
import torch

import lmcache.v1.confkv.gpu_crypto as gpu_crypto
from lmcache.v1.confkv.gpu_crypto import (
    FRAME_OVERHEAD,
    ConfKVAuthenticationError,
    ConfKVGpuCrypto,
    confkv_gpu_execution_mode,
    create_confkv_gpu_crypto,
    layout_nbytes,
    persistent_layout_descs,
    sealed_layout_desc,
)
from lmcache.v1.confkv.gpu_crypto import validate_confkv_l2_adapter_config
from lmcache.v1.distributed.api import MemoryLayoutDesc


def test_layout_nbytes_and_sealed_layout():
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([4, 3]), torch.Size([5])],
        dtypes=[torch.float16, torch.float32],
    )
    plaintext = 4 * 3 * 2 + 5 * 4
    assert layout_nbytes(layout) == plaintext
    sealed = sealed_layout_desc(layout)
    assert sealed.shapes == [torch.Size([plaintext + FRAME_OVERHEAD])]
    assert sealed.dtypes == [torch.uint8]


def test_disabled_factory_does_not_touch_cuda(monkeypatch):
    monkeypatch.setenv("CONFKV_GPU_SEAL_ENABLE", "0")
    assert create_confkv_gpu_crypto(
        "cuda:0", max_plaintext_len=4096
    ) is None



@pytest.mark.parametrize("mode", ["naive", "optimized"])
def test_execution_mode(monkeypatch, mode):
    monkeypatch.setenv("CONFKV_GPU_MODE", mode)
    assert confkv_gpu_execution_mode() == mode


def test_execution_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("CONFKV_GPU_MODE", "fastest")
    with pytest.raises(ValueError, match="CONFKV_GPU_MODE"):
        confkv_gpu_execution_mode()


def test_quiesce_after_error_synchronizes_main_and_slot_streams():
    events: list[str] = []

    class _Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def synchronize(self) -> None:
            events.append(self.name)

    class _Slot:
        def __init__(self, name: str) -> None:
            self.stream = _Stream(name)

    crypto = object.__new__(ConfKVGpuCrypto)
    crypto._slots = [_Slot("slot0"), _Slot("slot1")]
    crypto.quiesce_after_error(stream=_Stream("main"))

    assert events == ["main", "slot0", "slot1"]


def test_persistent_layout_descs_b3(monkeypatch):
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([7])],
        dtypes=[torch.float16],
    )
    monkeypatch.setenv("CONFKV_GPU_SEAL_ENABLE", "1")
    out = persistent_layout_descs({0: layout})
    assert out[0].dtypes == [torch.uint8]
    assert out[0].shapes == [torch.Size([7 * 2 + FRAME_OVERHEAD])]


def test_persistent_layout_descs_disabled(monkeypatch):
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([7])],
        dtypes=[torch.float16],
    )
    original = {0: layout}
    monkeypatch.setenv("CONFKV_GPU_SEAL_ENABLE", "0")
    assert persistent_layout_descs(original) is original


class _FakeMemoryObj:
    def __init__(self, size: int):
        self._size = size

    def get_size(self) -> int:
        return self._size


class _FakeStream:
    cuda_stream = 123

    def __init__(self, events: list[str]):
        self._events = events

    def synchronize(self) -> None:
        self._events.append("sync")


class _FakeKey:
    def __init__(self, owner, auth_ok: int, events: list[str]):
        self._owner = owner
        self._auth_ok = auth_ok
        self._events = events

    def open_async(self, **kwargs) -> None:
        self._events.append("open")
        self._owner._auth.fill_(self._auth_ok)


def _make_fake_opener(auth_ok: int, events: list[str]) -> ConfKVGpuCrypto:
    opener = object.__new__(ConfKVGpuCrypto)
    opener._device = torch.device("cpu")
    opener._lock = threading.Lock()
    opener._closed = False
    opener._capacity = 32
    opener._sealed = torch.empty(32 + FRAME_OVERHEAD, dtype=torch.uint8)
    opener._auth = torch.empty(1, dtype=torch.int32)
    opener._key = _FakeKey(opener, auth_ok, events)
    return opener


def test_copy_h2d_and_open_auth_success(monkeypatch):
    events: list[str] = []
    opener = _make_fake_opener(1, events)
    plaintext = torch.empty(32, dtype=torch.uint8)
    src = _FakeMemoryObj(32 + FRAME_OVERHEAD)
    monkeypatch.setattr(
        gpu_crypto,
        "lmcache_memcpy_async_h2d",
        lambda _src, _dst: events.append("h2d"),
    )

    opener.copy_h2d_and_open(src, plaintext, stream=_FakeStream(events))
    assert events == ["h2d", "open", "sync"]


def test_copy_h2d_and_open_auth_failure(monkeypatch):
    events: list[str] = []
    opener = _make_fake_opener(0, events)
    plaintext = torch.empty(32, dtype=torch.uint8)
    src = _FakeMemoryObj(32 + FRAME_OVERHEAD)
    monkeypatch.setattr(
        gpu_crypto,
        "lmcache_memcpy_async_h2d",
        lambda _src, _dst: events.append("h2d"),
    )

    with pytest.raises(ConfKVAuthenticationError):
        opener.copy_h2d_and_open(src, plaintext, stream=_FakeStream(events))
    assert events == ["h2d", "open", "sync"]



class _FakeSerdeConfig:
    type = "aesgcm"


class _FakeAdapterConfig:
    def __init__(self, serde_config):
        self.serde_config = serde_config


def test_b3_rejects_l2_serde(monkeypatch):
    monkeypatch.setenv(
        "CONFKV_GPU_SEAL_ENABLE",
        "1",
    )

    config = _FakeAdapterConfig(
        _FakeSerdeConfig()
    )

    with pytest.raises(
        RuntimeError,
        match="L2 serde",
    ):
        validate_confkv_l2_adapter_config(
            config
        )


def test_b3_accepts_opaque_l2_without_serde(monkeypatch):
    monkeypatch.setenv(
        "CONFKV_GPU_SEAL_ENABLE",
        "1",
    )

    config = _FakeAdapterConfig(
        None
    )

    validate_confkv_l2_adapter_config(
        config
    )


def test_disabled_b3_does_not_reject_serde(monkeypatch):
    monkeypatch.setenv(
        "CONFKV_GPU_SEAL_ENABLE",
        "0",
    )

    config = _FakeAdapterConfig(
        _FakeSerdeConfig()
    )

    validate_confkv_l2_adapter_config(
        config
    )
