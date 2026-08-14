# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_manager import StorageManager


class _FakeL1Manager:
    def __init__(self) -> None:
        self.calls = []

    def delete(self, keys, *, force=False):
        self.calls.append((list(keys), force))
        return {key: L1Error.SUCCESS for key in keys}


def test_abort_write_force_deletes_reserved_objects():
    manager = object.__new__(StorageManager)
    manager._l1_manager = _FakeL1Manager()
    keys = ["a", "b"]

    manager.abort_write(keys)

    assert manager._l1_manager.calls == [(keys, True)]
