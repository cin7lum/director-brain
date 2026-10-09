"""BrainRepository 抽象基类与决策账本条目数据类。

定义 Director Brain M0 存储层的统一契约：8 类顶层实体均可通过通用
``save / get / list / update / delete`` 接口完成 CRUD。实体类与其主键字段的
映射由具体后端（如 :class:`~storage.sqlite_repository.SqliteRepository`）维护。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DecisionLedgerEntry:
    """决策账本中的一条变更记录（非 Pydantic 顶层记录）。

    每次决策（EDL / 计划 / 提案）发生状态或内容变更时追加一条，用于审计回溯。
    """

    ledger_id: str
    decision_id: str
    action: str
    timestamp: int
    detail: dict[str, Any] | None = None
    project_id: str | None = None


class BrainRepository(ABC):
    """所有持久化后端必须实现的存储接口。

    采用泛型统一接口：调用方传入实体类（或已构造的实体实例），由后端解析
    对应的表与主键字段。支持的实体类见模块文档与各后端实现。
    """

    @abstractmethod
    def save(self, entity: Any) -> None:
        """写入或覆盖一条实体（UPSERT，按主键去重）。"""

    @abstractmethod
    def get(self, entity_cls: type, entity_id: str) -> Any | None:
        """按主键读取一条实体；不存在返回 ``None``。"""

    @abstractmethod
    def list(self, entity_cls: type, project_id: str | None = None) -> list[Any]:
        """列出某类实体；传入 ``project_id`` 时只返回该项目的记录。"""

    @abstractmethod
    def update(self, entity_cls: type, entity_id: str, **fields: Any) -> Any:
        """局部更新指定字段，返回更新后的完整实体。"""

    @abstractmethod
    def delete(self, entity_cls: type, entity_id: str) -> bool:
        """按主键删除；命中删除返回 ``True``，本不存在返回 ``False``。"""

    @abstractmethod
    def invalidate_context(self, context_id: str) -> bool:
        """将指定 :class:`FilmContextSnapshot` 标记为已失效。

        返回是否命中并更新了记录。
        """
