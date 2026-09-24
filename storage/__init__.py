"""存储层工厂函数与对外导出。"""
from __future__ import annotations

from storage.object_store import LocalObjectStore
from storage.repository import BrainRepository, DecisionLedgerEntry
from storage.sqlite_repository import SqliteRepository


def get_repository(backend: str = "sqlite", **kwargs) -> BrainRepository:
    """按 backend 名称构造 BrainRepository。目前仅支持 ``"sqlite"``。"""
    if backend == "sqlite":
        return SqliteRepository(**kwargs)
    raise ValueError(f"不支持的存储后端: {backend!r}")


def get_object_store(path: str) -> LocalObjectStore:
    """构造指向 ``path`` 的本地对象存储。"""
    return LocalObjectStore(base_path=path)


__all__ = [
    "BrainRepository",
    "DecisionLedgerEntry",
    "LocalObjectStore",
    "SqliteRepository",
    "get_object_store",
    "get_repository",
]
