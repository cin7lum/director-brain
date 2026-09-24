"""基于标准库 sqlite3 的 :class:`BrainRepository` 实现。

每张实体表结构统一为::

    <主键字段>  TEXT PRIMARY KEY   -- 如 brief_id / observation_id ...
    project_id  TEXT               -- 建索引，用于按项目过滤
    created_at  INTEGER
    data        TEXT               -- 整条实体的 JSON 快照（含嵌套模型与枚举）

复杂字段（list / dict / 嵌套模型）无需拆列，随整条 ``data`` 一起
``json.dumps`` 存储；读取时用对应 Pydantic 模型（或 dataclass）重建。
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from typing import Any

from director_brain.models import (
    DirectorBrief,
    DirectorDecisionPlan,
    EditorialDecisionList,
    FilmContextSnapshot,
    FilmObservation,
    RevisionProposal,
    StoryGraph,
)
from storage.repository import BrainRepository, DecisionLedgerEntry

# entity_cls -> (表名, 主键字段名)
_TABLES: dict[type, tuple[str, str]] = {
    DirectorBrief: ("director_briefs", "brief_id"),
    FilmContextSnapshot: ("film_context_snapshots", "context_id"),
    FilmObservation: ("film_observations", "observation_id"),
    StoryGraph: ("story_graphs", "graph_id"),
    EditorialDecisionList: ("editorial_decision_lists", "edl_id"),
    DirectorDecisionPlan: ("director_decision_plans", "plan_id"),
    RevisionProposal: ("revision_proposals", "proposal_id"),
    DecisionLedgerEntry: ("decision_ledger", "ledger_id"),
}


class SqliteRepository(BrainRepository):
    """sqlite3 后端的 BrainRepository。"""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    # ------------------------------------------------------------------ schema
    def _create_tables(self) -> None:
        ddl = []
        for _, (table, id_col) in _TABLES.items():
            ddl.append(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                f"  {id_col} TEXT PRIMARY KEY,"
                f"  project_id TEXT,"
                f"  created_at INTEGER,"
                f"  data TEXT NOT NULL"
                f")"
            )
            ddl.append(f"CREATE INDEX IF NOT EXISTS idx_{table}_project "
                       f"ON {table}(project_id)")
        self._conn.executescript(";\n".join(ddl))
        self._conn.commit()

    # ------------------------------------------------------------- (de)serialize
    @staticmethod
    def _dump(entity: Any) -> dict[str, Any]:
        """把实体序列化为可 json.dumps 的纯 dict。"""
        if is_dataclass(entity):
            return asdict(entity)
        # Pydantic v2：mode="json" 会把枚举转成其字面值
        return entity.model_dump(mode="json")

    def _load(self, entity_cls: type, row: sqlite3.Row) -> Any:
        data = json.loads(row["data"])
        if is_dataclass(entity_cls):
            return entity_cls(**data)
        return entity_cls(**data)

    @staticmethod
    def _meta(entity: Any) -> tuple[str | None, int | None]:
        project_id = getattr(entity, "project_id", None)
        created_at = getattr(entity, "created_at", None)
        return project_id, created_at

    # -------------------------------------------------------------------- CRUD
    def save(self, entity: Any) -> None:
        table, id_col = _TABLES[type(entity)]
        entity_id = getattr(entity, id_col)
        project_id, created_at = self._meta(entity)
        payload = json.dumps(self._dump(entity), ensure_ascii=False)
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} "
            f"({id_col}, project_id, created_at, data) VALUES (?, ?, ?, ?)",
            (entity_id, project_id, created_at, payload),
        )
        self._conn.commit()

    def get(self, entity_cls: type, entity_id: str) -> Any | None:
        table, id_col = _TABLES[entity_cls]
        cur = self._conn.execute(
            f"SELECT data FROM {table} WHERE {id_col} = ?", (entity_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._load(entity_cls, row)

    def list(self, entity_cls: type, project_id: str | None = None) -> list[Any]:
        table, _ = _TABLES[entity_cls]
        if project_id is not None:
            cur = self._conn.execute(
                f"SELECT data FROM {table} WHERE project_id = ?", (project_id,)
            )
        else:
            cur = self._conn.execute(f"SELECT data FROM {table}")
        return [self._load(entity_cls, row) for row in cur.fetchall()]

    def update(self, entity_cls: type, entity_id: str, **fields: Any) -> Any:
        table, id_col = _TABLES[entity_cls]
        cur = self._conn.execute(
            f"SELECT data FROM {table} WHERE {id_col} = ?", (entity_id,)
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"{entity_cls.__name__}#{entity_id} 不存在")

        data = json.loads(row["data"])
        data.update(fields)
        payload = json.dumps(data, ensure_ascii=False)
        self._conn.execute(
            f"UPDATE {table} SET data = ? WHERE {id_col} = ?",
            (payload, entity_id),
        )
        self._conn.commit()
        return entity_cls(**data)

    def delete(self, entity_cls: type, entity_id: str) -> bool:
        table, id_col = _TABLES[entity_cls]
        cur = self._conn.execute(
            f"DELETE FROM {table} WHERE {id_col} = ?", (entity_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------- context
    def invalidate_context(self, context_id: str) -> bool:
        table, id_col = _TABLES[FilmContextSnapshot]
        cur = self._conn.execute(
            f"SELECT data FROM {table} WHERE {id_col} = ?", (context_id,)
        )
        row = cur.fetchone()
        if row is None:
            return False
        data = json.loads(row["data"])
        data["invalidated_at"] = int(time.time())
        self._conn.execute(
            f"UPDATE {table} SET data = ? WHERE {id_col} = ?",
            (json.dumps(data, ensure_ascii=False), context_id),
        )
        self._conn.commit()
        return True
