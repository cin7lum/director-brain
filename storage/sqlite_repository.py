"""基于标准库 sqlite3 的 :class:`BrainRepository` 实现。

每张实体表结构统一为::

    <主键字段>  TEXT PRIMARY KEY   -- 如 brief_id / observation_id ...
    project_id  TEXT               -- 建索引，用于按项目过滤
    revision    INTEGER            -- FilmProjectManifest only
    created_at  INTEGER
    data        TEXT               -- 整条实体的 JSON 快照（含嵌套模型与枚举）

复杂字段（list / dict / 嵌套模型）无需拆列，随整条 ``data`` 一起
``json.dumps`` 存储；读取时用对应 Pydantic 模型（或 dataclass）重建。
"""
from __future__ import annotations

import hashlib
import json
import math
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
    FilmProjectManifest,
    ProjectStoryGraph,
    ProjectDirectorShadowComparisonRecord,
    ProjectStoryLinkComparison,
    ProjectStoryLinkReview,
    ProjectStoryMentionReview,
    RevisionProposal,
    StoryGraph,
)
from storage.repository import BrainRepository, DecisionLedgerEntry

_PROJECT_CONFIRMATION_DISPATCH_BLOCK_REASON = (
    "P3 ExecutionPort is not admitted"
)

# entity_cls -> (表名, 主键字段名)
_TABLES: dict[type, tuple[str, str]] = {
    DirectorBrief: ("director_briefs", "brief_id"),
    FilmContextSnapshot: ("film_context_snapshots", "context_id"),
    FilmObservation: ("film_observations", "observation_id"),
    FilmProjectManifest: ("film_project_manifests", "manifest_id"),
    StoryGraph: ("story_graphs", "graph_id"),
    ProjectStoryGraph: ("project_story_graphs", "graph_id"),
    ProjectStoryLinkComparison: (
        "project_story_link_comparisons", "comparison_id"),
    ProjectDirectorShadowComparisonRecord: (
        "project_director_shadow_comparisons", "comparison_id"),
    EditorialDecisionList: ("editorial_decision_lists", "edl_id"),
    DirectorDecisionPlan: ("director_decision_plans", "plan_id"),
    RevisionProposal: ("revision_proposals", "proposal_id"),
    DecisionLedgerEntry: ("decision_ledger", "ledger_id"),
}


class SqliteRepository(BrainRepository):
    """sqlite3 后端的 BrainRepository。"""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        # Make the durability contract explicit instead of relying on the
        # SQLite build's default. EXTRA also syncs the containing directory
        # after deleting a rollback journal before reporting the commit.
        self._conn.execute("PRAGMA synchronous = EXTRA")
        if self._conn.execute("PRAGMA synchronous").fetchone()[0] != 3:
            self._conn.close()
            raise RuntimeError("SQLite EXTRA synchronous durability is unavailable")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def close(self) -> None:
        """Release the request-scoped SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------ schema
    def _create_tables(self) -> None:
        ddl = []
        for entity_cls, (table, id_col) in _TABLES.items():
            candidate_column = (
                "  candidate_id TEXT,"
                if entity_cls is ProjectStoryLinkComparison
                else ""
            )
            revision_column = (
                "  revision INTEGER,"
                if entity_cls is FilmProjectManifest
                else ""
            )
            ddl.append(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                f"  {id_col} TEXT PRIMARY KEY,"
                f"  project_id TEXT,"
                f"{candidate_column}"
                f"{revision_column}"
                f"  created_at INTEGER,"
                f"  data TEXT NOT NULL"
                f")"
            )
            ddl.append(f"CREATE INDEX IF NOT EXISTS idx_{table}_project "
                       f"ON {table}(project_id)")
        self._conn.executescript(";\n".join(ddl))
        self._conn.executescript(
            "CREATE TABLE IF NOT EXISTS analysis_result_cache ("
            "fingerprint TEXT PRIMARY KEY,"
            "source_content_hash TEXT NOT NULL,"
            "analysis_profile TEXT NOT NULL,"
            "created_at INTEGER NOT NULL,"
            "data TEXT NOT NULL"
            ");"
        )
        self._conn.executescript(
            "CREATE TABLE IF NOT EXISTS project_story_link_ranking_snapshots ("
            "fingerprint TEXT PRIMARY KEY,"
            "project_id TEXT NOT NULL,"
            "candidate_set_id TEXT NOT NULL,"
            "ranking_profile_id TEXT NOT NULL,"
            "ranking_model TEXT NOT NULL,"
            "model_digest TEXT NOT NULL,"
            "ranking_metric TEXT NOT NULL,"
            "created_at INTEGER NOT NULL,"
            "data TEXT NOT NULL"
            ");"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_story_link_ranking_scope "
            "ON project_story_link_ranking_snapshots(project_id, candidate_set_id)"
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "idx_film_project_manifests_revision "
            "ON film_project_manifests(project_id, revision)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS "
            "idx_project_story_link_comparisons_project_created "
            "ON project_story_link_comparisons(project_id, created_at DESC, comparison_id DESC)"
        )
        comparison_columns = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(project_story_link_comparisons)")
        }
        if "candidate_id" not in comparison_columns:
            self._conn.execute(
                "ALTER TABLE project_story_link_comparisons ADD COLUMN candidate_id TEXT"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS "
            "idx_project_story_link_comparisons_candidate "
            "ON project_story_link_comparisons(project_id, candidate_id, created_at, comparison_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS "
            "idx_project_director_shadow_comparisons_project_created "
            "ON project_director_shadow_comparisons(project_id, created_at DESC, comparison_id DESC)"
        )
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
        if isinstance(entity, ProjectStoryLinkComparison):
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} "
                f"({id_col}, project_id, candidate_id, created_at, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    entity_id,
                    project_id,
                    entity.candidate_id,
                    created_at,
                    payload,
                ),
            )
        else:
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

        # 校验 fields 的 key 都是模型合法字段，避免脏数据写入数据库后
        # 才在重建模型时报错。Pydantic 模型用 model_fields；dataclass
        # （如 DecisionLedgerEntry）用 __dataclass_fields__。
        if hasattr(entity_cls, "model_fields"):
            valid_keys = entity_cls.model_fields.keys()
        else:
            valid_keys = entity_cls.__dataclass_fields__.keys()
        for key in fields:
            if key not in valid_keys:
                raise ValueError(
                    f"field '{key}' is not a valid field of {entity_cls.__name__}"
                )

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

    def save_project_story_link_comparison(
        self, comparison: ProjectStoryLinkComparison
    ) -> tuple[ProjectStoryLinkComparison, bool]:
        """Atomically append or replay an idempotent shadow comparison.

        Comparison IDs bind project ID and the request's hashed idempotency key.
        A key reused for a different request is rejected; an exact replay returns
        the original immutable model result without a second inference call.
        """
        table, id_col = _TABLES[ProjectStoryLinkComparison]
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                f"SELECT data FROM {table} WHERE {id_col} = ?",
                (comparison.comparison_id,),
            ).fetchone()
            if row is not None:
                existing = self._load(ProjectStoryLinkComparison, row)
                if (
                    existing.project_id != comparison.project_id
                    or existing.idempotency_key_sha256
                    != comparison.idempotency_key_sha256
                    or existing.request_fingerprint != comparison.request_fingerprint
                    or existing.candidate_id != comparison.candidate_id
                    or existing.attempt_number != comparison.attempt_number
                ):
                    raise ValueError("idempotency key is bound to another request")
                self._conn.commit()
                return existing, True

            project_id, created_at = self._meta(comparison)
            payload = json.dumps(self._dump(comparison), ensure_ascii=False)
            self._conn.execute(
                f"INSERT INTO {table} "
                f"({id_col}, project_id, candidate_id, created_at, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    comparison.comparison_id,
                    project_id,
                    comparison.candidate_id,
                    created_at,
                    payload,
                ),
            )
            self._conn.commit()
            return comparison, False
        except Exception:
            self._conn.rollback()
            raise

    def save_project_director_shadow_comparison(
        self, comparison: ProjectDirectorShadowComparisonRecord
    ) -> tuple[ProjectDirectorShadowComparisonRecord, bool]:
        """Append or replay an immutable, idempotent local SHADOW comparison."""
        table, id_col = _TABLES[ProjectDirectorShadowComparisonRecord]
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                f"SELECT data FROM {table} WHERE {id_col} = ?",
                (comparison.comparison_id,),
            ).fetchone()
            if row is not None:
                existing = self._load(ProjectDirectorShadowComparisonRecord, row)
                if (
                    existing.project_id != comparison.project_id
                    or existing.idempotency_key_sha256
                    != comparison.idempotency_key_sha256
                    or existing.request_fingerprint != comparison.request_fingerprint
                ):
                    raise ValueError("idempotency key is bound to another comparison")
                self._conn.commit()
                return existing, True

            project_id, created_at = self._meta(comparison)
            payload = json.dumps(self._dump(comparison), ensure_ascii=False)
            self._conn.execute(
                f"INSERT INTO {table} "
                f"({id_col}, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                (comparison.comparison_id, project_id, created_at, payload),
            )
            self._conn.commit()
            return comparison, False
        except Exception:
            self._conn.rollback()
            raise

    def list_project_director_shadow_comparisons(
        self, project_id: str, *, limit: int = 100, offset: int = 0
    ) -> tuple[list[ProjectDirectorShadowComparisonRecord], int]:
        """Page newest immutable local SHADOW comparisons for one project."""
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("comparison page bounds are invalid")
        table, _id_col = _TABLES[ProjectDirectorShadowComparisonRecord]
        total = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE project_id = ?",
            (project_id,),
        ).fetchone()["n"]
        rows = self._conn.execute(
            f"SELECT data FROM {table} WHERE project_id = ? "
            "ORDER BY created_at DESC, comparison_id DESC LIMIT ? OFFSET ?",
            (project_id, limit, offset),
        ).fetchall()
        return [
            self._load(ProjectDirectorShadowComparisonRecord, row)
            for row in rows
        ], total

    def list_project_story_link_comparisons(
        self, project_id: str, *, limit: int = 100, offset: int = 0
    ) -> tuple[list[ProjectStoryLinkComparison], int]:
        """Page newest immutable comparison records for one project."""
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("comparison page bounds are invalid")
        table, _id_col = _TABLES[ProjectStoryLinkComparison]
        total = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE project_id = ?",
            (project_id,),
        ).fetchone()["n"]
        rows = self._conn.execute(
            f"SELECT data FROM {table} WHERE project_id = ? "
            "ORDER BY created_at DESC, comparison_id DESC LIMIT ? OFFSET ?",
            (project_id, limit, offset),
        ).fetchall()
        return [self._load(ProjectStoryLinkComparison, row) for row in rows], total

    def get_project_story_link_comparisons_by_ids(
        self, project_id: str, comparison_ids: list[str]
    ) -> dict[str, ProjectStoryLinkComparison]:
        """Read an exact, project-scoped set of pair comparisons in one query."""
        if len(comparison_ids) > 200:
            raise ValueError("comparison ID lookup exceeds one candidate page")
        unique_ids = list(dict.fromkeys(comparison_ids))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self._conn.execute(
            "SELECT data FROM project_story_link_comparisons "
            f"WHERE project_id = ? AND comparison_id IN ({placeholders})",
            [project_id, *unique_ids],
        ).fetchall()
        comparisons = [
            self._load(ProjectStoryLinkComparison, row) for row in rows
        ]
        return {item.comparison_id: item for item in comparisons}

    def get_project_story_link_comparisons_by_candidate_ids(
        self, project_id: str, candidate_ids: list[str]
    ) -> dict[str, list[ProjectStoryLinkComparison]]:
        """Read all persisted attempts for a bounded candidate page."""
        if len(candidate_ids) > 200:
            raise ValueError("candidate ID lookup exceeds one candidate page")
        unique_ids = list(dict.fromkeys(candidate_ids))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self._conn.execute(
            "SELECT data FROM project_story_link_comparisons "
            "WHERE project_id = ? AND candidate_id IN ("
            f"{placeholders}) ORDER BY created_at, comparison_id",
            [project_id, *unique_ids],
        ).fetchall()
        result: dict[str, list[ProjectStoryLinkComparison]] = {
            candidate_id: [] for candidate_id in unique_ids
        }
        for row in rows:
            comparison = self._load(ProjectStoryLinkComparison, row)
            if comparison.candidate_id is None:
                raise ValueError("candidate comparison index is inconsistent with its record")
            result[comparison.candidate_id].append(comparison)
        return result

    def latest_project_manifest(self, project_id: str) -> FilmProjectManifest | None:
        """Return the newest immutable manifest revision for one project."""
        row = self._conn.execute(
            "SELECT data FROM film_project_manifests "
            "WHERE project_id = ? ORDER BY revision DESC LIMIT 1",
            (project_id,),
        ).fetchone()
        return self._load(FilmProjectManifest, row) if row else None

    def get_project_manifest(
        self, project_id: str, revision: int
    ) -> FilmProjectManifest | None:
        """Read an exact project manifest revision; revisions are immutable."""
        row = self._conn.execute(
            "SELECT data FROM film_project_manifests "
            "WHERE project_id = ? AND revision = ?",
            (project_id, revision),
        ).fetchone()
        return self._load(FilmProjectManifest, row) if row else None

    def save_project_manifest(
        self,
        manifest: FilmProjectManifest,
        expected_revision: int,
    ) -> None:
        """Append a manifest revision with compare-and-swap concurrency control."""
        if manifest.revision != expected_revision + 1:
            raise ValueError("manifest revision must increment expected_revision by one")
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT MAX(revision) AS revision FROM film_project_manifests "
                "WHERE project_id = ?",
                (manifest.project_id,),
            ).fetchone()
            current_revision = int(row["revision"] or 0)
            if current_revision != expected_revision:
                raise ValueError(
                    f"stale project manifest: expected revision {expected_revision}, "
                    f"current revision is {current_revision}"
                )
            payload = json.dumps(self._dump(manifest), ensure_ascii=False)
            self._conn.execute(
                "INSERT INTO film_project_manifests "
                "(manifest_id, project_id, revision, created_at, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    manifest.manifest_id,
                    manifest.project_id,
                    manifest.revision,
                    manifest.created_at,
                    payload,
                ),
            )
            # A new project revision changes asset membership/order/rights, so
            # prior project snapshots are no longer current for that project.
            rows = self._conn.execute(
                "SELECT context_id, data FROM film_context_snapshots "
                "WHERE project_id = ?",
                (manifest.project_id,),
            ).fetchall()
            invalidated_at = int(time.time())
            for context_row in rows:
                context_data = json.loads(context_row["data"])
                if context_data.get("invalidated_at") is None:
                    context_data["invalidated_at"] = invalidated_at
                    self._conn.execute(
                        "UPDATE film_context_snapshots SET data = ? "
                        "WHERE context_id = ?",
                        (json.dumps(context_data, ensure_ascii=False),
                         context_row["context_id"]),
                    )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get_analysis_result(
        self,
        fingerprint: str,
        source_content_hash: str,
        analysis_profile: str,
    ) -> list[FilmObservation] | None:
        """Read deterministic analysis only when its full identity still matches."""
        row = self._conn.execute(
            "SELECT source_content_hash, analysis_profile, data "
            "FROM analysis_result_cache WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        if row is None:
            return None
        if (row["source_content_hash"].lower() != source_content_hash.lower()
                or row["analysis_profile"] != analysis_profile):
            return None
        return [FilmObservation(**item) for item in json.loads(row["data"])]

    def save_analysis_result(
        self,
        fingerprint: str,
        source_content_hash: str,
        analysis_profile: str,
        observations: list[FilmObservation],
    ) -> None:
        """Persist immutable, project-neutral local analysis by exact input key."""
        payload = json.dumps(
            [item.model_dump(mode="json") for item in observations],
            ensure_ascii=False,
        )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT source_content_hash, analysis_profile, data "
                "FROM analysis_result_cache WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if row is not None:
                if (row["source_content_hash"].lower() != source_content_hash.lower()
                        or row["analysis_profile"] != analysis_profile
                        or row["data"] != payload):
                    raise ValueError("analysis cache identity collision or result drift")
            else:
                self._conn.execute(
                    "INSERT INTO analysis_result_cache "
                    "(fingerprint, source_content_hash, analysis_profile, created_at, data) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (fingerprint, source_content_hash.lower(), analysis_profile,
                     int(time.time()), payload),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @staticmethod
    def _project_story_link_ranking_snapshot_identity(
        project_id: str,
        candidate_set_id: str,
        ranking_profile_id: str,
        ranking_model: str,
        model_digest: str,
        ranking_metric: str,
    ) -> tuple[str, str]:
        identity = (
            project_id,
            candidate_set_id,
            ranking_profile_id,
            ranking_model,
            model_digest.lower(),
            ranking_metric,
        )
        if any(not isinstance(item, str) or not item for item in identity):
            raise ValueError("ranking snapshot identity fields must be non-empty")
        if (
            len(identity[4]) != 64
            or any(char not in "0123456789abcdef" for char in identity[4])
        ):
            raise ValueError("ranking snapshot model digest must be 64 lowercase hex characters")
        encoded_identity = json.dumps(
            identity, ensure_ascii=False, separators=(",", ":"))
        fingerprint = hashlib.sha256(encoded_identity.encode("utf-8")).hexdigest()
        return fingerprint, encoded_identity

    @staticmethod
    def _validate_project_story_link_ranking_entries(
        entries: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        validated: list[tuple[str, float]] = []
        seen: set[str] = set()
        for entry in entries:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not entry[0]
                or entry[0] in seen
                or isinstance(entry[1], bool)
                or not isinstance(entry[1], (int, float))
                or not math.isfinite(entry[1])
                or not -1.0 <= entry[1] <= 1.0
            ):
                raise ValueError("ranking snapshot entries are invalid")
            seen.add(entry[0])
            validated.append((entry[0], float(entry[1])))
        return validated

    def get_project_story_link_ranking_snapshot(
        self,
        project_id: str,
        candidate_set_id: str,
        ranking_profile_id: str,
        ranking_model: str,
        model_digest: str,
        ranking_metric: str,
    ) -> list[tuple[str, float]] | None:
        """Read an immutable ranking only for its complete source/model identity."""
        fingerprint, _ = self._project_story_link_ranking_snapshot_identity(
            project_id, candidate_set_id, ranking_profile_id, ranking_model,
            model_digest, ranking_metric)
        row = self._conn.execute(
            "SELECT project_id, candidate_set_id, ranking_profile_id, "
            "ranking_model, model_digest, ranking_metric, data "
            "FROM project_story_link_ranking_snapshots WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        if row is None:
            return None
        expected_identity = (
            project_id, candidate_set_id, ranking_profile_id, ranking_model,
            model_digest.lower(), ranking_metric)
        actual_identity = (
            row["project_id"], row["candidate_set_id"],
            row["ranking_profile_id"], row["ranking_model"],
            row["model_digest"].lower(), row["ranking_metric"])
        if actual_identity != expected_identity:
            raise ValueError("ranking snapshot identity collision")
        try:
            payload = json.loads(row["data"])
            if not isinstance(payload, list):
                raise ValueError("ranking snapshot payload must be a list")
            entries = [
                (item["candidate_id"], item["ranking_score"])
                for item in payload
                if isinstance(item, dict)
                and set(item) == {"candidate_id", "ranking_score"}
            ]
            if len(entries) != len(payload):
                raise ValueError("ranking snapshot payload entry is invalid")
            return self._validate_project_story_link_ranking_entries(entries)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("ranking snapshot payload is corrupt") from exc

    def save_project_story_link_ranking_snapshot(
        self,
        project_id: str,
        candidate_set_id: str,
        ranking_profile_id: str,
        ranking_model: str,
        model_digest: str,
        ranking_metric: str,
        entries: list[tuple[str, float]],
    ) -> None:
        """Persist an immutable all-candidate ranking for one exact identity."""
        fingerprint, _ = self._project_story_link_ranking_snapshot_identity(
            project_id, candidate_set_id, ranking_profile_id, ranking_model,
            model_digest, ranking_metric)
        normalized_entries = self._validate_project_story_link_ranking_entries(entries)
        payload = json.dumps(
            [
                {"candidate_id": candidate_id, "ranking_score": score}
                for candidate_id, score in normalized_entries
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT project_id, candidate_set_id, ranking_profile_id, "
                "ranking_model, model_digest, ranking_metric, data "
                "FROM project_story_link_ranking_snapshots WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            identity = (
                project_id, candidate_set_id, ranking_profile_id, ranking_model,
                model_digest.lower(), ranking_metric)
            if row is not None:
                stored_identity = (
                    row["project_id"], row["candidate_set_id"],
                    row["ranking_profile_id"], row["ranking_model"],
                    row["model_digest"].lower(), row["ranking_metric"])
                if stored_identity != identity or row["data"] != payload:
                    raise ValueError("ranking snapshot identity collision or result drift")
            else:
                self._conn.execute(
                    "INSERT INTO project_story_link_ranking_snapshots "
                    "(fingerprint, project_id, candidate_set_id, ranking_profile_id, "
                    "ranking_model, model_digest, ranking_metric, created_at, data) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        fingerprint, project_id, candidate_set_id,
                        ranking_profile_id, ranking_model, model_digest.lower(),
                        ranking_metric, int(time.time()), payload,
                    ),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def save_project_context(
        self,
        snapshot: FilmContextSnapshot,
        observations: list[FilmObservation],
        expected_manifest_revision: int,
    ) -> None:
        """Atomically commit evidence and snapshot only for the current revision."""
        if snapshot.project_revision != expected_manifest_revision:
            raise ValueError("snapshot revision does not match expected manifest revision")
        if any(item.project_id != snapshot.project_id for item in observations):
            raise ValueError("all context observations must belong to snapshot.project_id")

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT MAX(revision) AS revision FROM film_project_manifests "
                "WHERE project_id = ?",
                (snapshot.project_id,),
            ).fetchone()
            current_revision = int(row["revision"] or 0)
            if current_revision != expected_manifest_revision:
                raise ValueError(
                    f"stale project context: expected manifest revision "
                    f"{expected_manifest_revision}, current revision is {current_revision}"
                )

            for observation in observations:
                existing = self._conn.execute(
                    "SELECT data FROM film_observations WHERE observation_id = ?",
                    (observation.observation_id,),
                ).fetchone()
                payload = json.dumps(self._dump(observation), ensure_ascii=False)
                if existing is not None:
                    if existing["data"] != payload:
                        raise ValueError(
                            f"observation evidence identity conflict: "
                            f"{observation.observation_id}"
                        )
                    continue
                self._conn.execute(
                    "INSERT INTO film_observations "
                    "(observation_id, project_id, created_at, data) "
                    "VALUES (?, ?, ?, ?)",
                    (observation.observation_id, observation.project_id,
                     observation.created_at, payload),
                )

            snapshot_payload = json.dumps(self._dump(snapshot), ensure_ascii=False)
            self._conn.execute(
                "INSERT OR REPLACE INTO film_context_snapshots "
                "(context_id, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                (snapshot.context_id, snapshot.project_id, snapshot.created_at,
                 snapshot_payload),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def save_project_story_graph(self, graph: ProjectStoryGraph) -> None:
        """Persist a derived graph only while its exact context is current.

        Graph identities are immutable and evidence-fingerprint scoped. The
        transaction rechecks manifest/context bindings so a concurrent
        revision cannot leave a stale graph appearing current.
        """
        payload = json.dumps(self._dump(graph), ensure_ascii=False)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            manifest_row = self._conn.execute(
                "SELECT data FROM film_project_manifests "
                "WHERE project_id = ? ORDER BY revision DESC LIMIT 1",
                (graph.project_id,),
            ).fetchone()
            if manifest_row is None:
                raise ValueError("project manifest does not exist")
            manifest = self._load(FilmProjectManifest, manifest_row)
            if (
                manifest.revision != graph.project_revision
                or manifest.manifest_id != graph.project_manifest_id
            ):
                raise ValueError("project StoryGraph manifest revision is stale")

            context_row = self._conn.execute(
                "SELECT data FROM film_context_snapshots WHERE context_id = ?",
                (graph.context_id,),
            ).fetchone()
            if context_row is None:
                raise ValueError("project StoryGraph context does not exist")
            context = self._load(FilmContextSnapshot, context_row)
            if (
                context.invalidated_at is not None
                or context.project_id != graph.project_id
                or context.project_manifest_id != graph.project_manifest_id
                or context.project_revision != graph.project_revision
                or context.analysis_fingerprint != graph.analysis_fingerprint
                or context.evidence_refs != graph.evidence_refs
            ):
                raise ValueError("project StoryGraph context binding is stale")

            existing = self._conn.execute(
                "SELECT data FROM project_story_graphs WHERE graph_id = ?",
                (graph.graph_id,),
            ).fetchone()
            if existing is not None:
                if existing["data"] != payload:
                    raise ValueError("project StoryGraph identity collision or result drift")
            else:
                self._conn.execute(
                    "INSERT INTO project_story_graphs "
                    "(graph_id, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                    (graph.graph_id, graph.project_id, graph.created_at, payload),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _project_story_link_reviews_unlocked(
        self, project_id: str, graph_id: str | None = None
    ) -> list[ProjectStoryLinkReview]:
        rows = self._conn.execute(
            "SELECT data FROM decision_ledger WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        reviews: list[ProjectStoryLinkReview] = []
        for row in rows:
            entry = self._load(DecisionLedgerEntry, row)
            if entry.action != "project_story_links_revised" or not entry.detail:
                continue
            payload = entry.detail.get("review")
            if not isinstance(payload, dict):
                raise ValueError("project story link ledger entry is malformed")
            review = ProjectStoryLinkReview(**payload)
            if review.project_id != project_id:
                raise ValueError("project story link ledger entry crossed project scope")
            if graph_id is None or review.story_graph_id == graph_id:
                reviews.append(review)
        return sorted(reviews, key=lambda item: (item.review_revision, item.created_at))

    def list_project_story_link_reviews(
        self, project_id: str, graph_id: str
    ) -> list[ProjectStoryLinkReview]:
        """Return immutable review snapshots for one graph, oldest first."""
        self._conn.execute("BEGIN")
        try:
            reviews = self._project_story_link_reviews_unlocked(project_id, graph_id)
            self._conn.commit()
            return reviews
        except Exception:
            self._conn.rollback()
            raise

    def list_project_story_review_histories(
        self, project_id: str, graph_id: str
    ) -> tuple[list[ProjectStoryMentionReview], list[ProjectStoryLinkReview]]:
        """Read both review ledgers from one SQLite snapshot."""
        self._conn.execute("BEGIN")
        try:
            mention_reviews = self._project_story_mention_reviews_unlocked(
                project_id, graph_id)
            link_reviews = self._project_story_link_reviews_unlocked(
                project_id, graph_id)
            self._conn.commit()
            return mention_reviews, link_reviews
        except Exception:
            self._conn.rollback()
            raise

    def _project_story_mention_reviews_unlocked(
        self, project_id: str, graph_id: str | None = None
    ) -> list[ProjectStoryMentionReview]:
        rows = self._conn.execute(
            "SELECT data FROM decision_ledger WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        reviews: list[ProjectStoryMentionReview] = []
        for row in rows:
            entry = self._load(DecisionLedgerEntry, row)
            if entry.action != "project_story_mentions_revised" or not entry.detail:
                continue
            payload = entry.detail.get("review")
            if not isinstance(payload, dict):
                raise ValueError("project story mention ledger entry is malformed")
            review = ProjectStoryMentionReview(**payload)
            if review.project_id != project_id:
                raise ValueError("project story mention ledger entry crossed project scope")
            if graph_id is None or review.story_graph_id == graph_id:
                reviews.append(review)
        return sorted(reviews, key=lambda item: (item.review_revision, item.created_at))

    def list_project_story_mention_reviews(
        self, project_id: str, graph_id: str
    ) -> list[ProjectStoryMentionReview]:
        """Return immutable source mention review snapshots, oldest first."""
        self._conn.execute("BEGIN")
        try:
            reviews = self._project_story_mention_reviews_unlocked(project_id, graph_id)
            self._conn.commit()
            return reviews
        except Exception:
            self._conn.rollback()
            raise

    def save_project_story_mention_review(
        self,
        project_id: str,
        review: ProjectStoryMentionReview,
        *,
        expected_review_revision: int,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> tuple[ProjectStoryMentionReview, bool]:
        """Atomically append one complete source mention review snapshot."""
        from director_brain.project_story_mention_review import (
            validate_project_story_mention_review,
        )

        if review.project_id != project_id:
            raise ValueError("mention review project_id does not match route scope")
        if expected_review_revision < 0:
            raise ValueError("expected_review_revision must be non-negative")
        if not idempotency_key.strip() or len(idempotency_key) > 128:
            raise ValueError("idempotency key must contain 1 to 128 non-whitespace characters")
        if (
            len(request_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in request_fingerprint)
        ):
            raise ValueError("request fingerprint must be 64 lowercase hex characters")

        key_digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        ledger_id = "project_story_mentions_" + hashlib.sha256(
            f"{project_id}\0{review.story_graph_id}\0{key_digest}".encode("utf-8")
        ).hexdigest()[:32]
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing_row = self._conn.execute(
                "SELECT project_id, data FROM decision_ledger WHERE ledger_id = ?",
                (ledger_id,),
            ).fetchone()
            if existing_row is not None:
                entry = self._load(DecisionLedgerEntry, existing_row)
                detail = entry.detail or {}
                if (
                    existing_row["project_id"] != project_id
                    or entry.decision_id != review.story_graph_id
                    or entry.action != "project_story_mentions_revised"
                ):
                    raise ValueError("idempotency key is bound to another operation")
                if detail.get("request_fingerprint") != request_fingerprint:
                    raise ValueError("idempotency key was reused with a different request")
                stored = ProjectStoryMentionReview(**detail["review"])
                self._conn.commit()
                return stored, True

            manifest_row = self._conn.execute(
                "SELECT data FROM film_project_manifests "
                "WHERE project_id = ? ORDER BY revision DESC LIMIT 1",
                (project_id,),
            ).fetchone()
            if manifest_row is None:
                raise ValueError("project manifest does not exist")
            manifest = self._load(FilmProjectManifest, manifest_row)
            if (
                manifest.manifest_id != review.project_manifest_id
                or manifest.revision != review.project_revision
            ):
                raise ValueError("mention review manifest revision is stale")
            context_row = self._conn.execute(
                "SELECT data FROM film_context_snapshots WHERE context_id = ?",
                (review.context_id,),
            ).fetchone()
            if context_row is None:
                raise ValueError("mention review context does not exist")
            context = self._load(FilmContextSnapshot, context_row)
            graph_row = self._conn.execute(
                "SELECT data FROM project_story_graphs WHERE graph_id = ?",
                (review.story_graph_id,),
            ).fetchone()
            if graph_row is None:
                raise ValueError("mention review StoryGraph does not exist")
            graph = self._load(ProjectStoryGraph, graph_row)

            history = self._project_story_mention_reviews_unlocked(
                project_id, review.story_graph_id)
            current_revision = history[-1].review_revision if history else 0
            if current_revision != expected_review_revision:
                raise ValueError(
                    f"stale mention review: expected revision {expected_review_revision}, "
                    f"current revision is {current_revision}"
                )
            if review.review_revision != current_revision + 1:
                raise ValueError("mention review revision must increment by one")

            observations: dict[str, FilmObservation] = {}
            for observation_id in {
                item.anchor.observation_id for item in review.decisions
            }:
                row = self._conn.execute(
                    "SELECT data FROM film_observations "
                    "WHERE observation_id = ? AND project_id = ?",
                    (observation_id, project_id),
                ).fetchone()
                if row is None:
                    raise ValueError("mention review observation is missing")
                observations[observation_id] = self._load(FilmObservation, row)
            validate_project_story_mention_review(
                review, manifest, context, graph, observations)

            payload = json.dumps(review.model_dump(mode="json"), ensure_ascii=False)
            entry = DecisionLedgerEntry(
                ledger_id=ledger_id,
                decision_id=review.story_graph_id,
                action="project_story_mentions_revised",
                timestamp=review.created_at,
                project_id=project_id,
                detail={
                    "review": review.model_dump(mode="json"),
                    "request_fingerprint": request_fingerprint,
                    "idempotency_key_sha256": key_digest,
                    "expected_review_revision": expected_review_revision,
                    "source_graph_mutated": False,
                },
            )
            self._conn.execute(
                "INSERT INTO decision_ledger "
                "(ledger_id, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                (ledger_id, project_id, review.created_at,
                 json.dumps(self._dump(entry), ensure_ascii=False)),
            )
            self._conn.commit()
            return ProjectStoryMentionReview(**json.loads(payload)), False
        except Exception:
            self._conn.rollback()
            raise

    def _validate_project_plan_story_links_unlocked(
        self,
        plan: DirectorDecisionPlan,
        edl: EditorialDecisionList,
        manifest: FilmProjectManifest,
        context: FilmContextSnapshot,
        graph: ProjectStoryGraph,
        observations: dict[str, FilmObservation],
    ) -> None:
        """Require Plan/EDL link refs to match the active review in this transaction."""
        from director_brain.project_story_link_review import (
            project_story_link_refs_for_edit,
            validate_project_story_link_review_for_plan,
        )

        history = self._project_story_link_reviews_unlocked(
            plan.project_id, graph.graph_id)
        active_review = history[-1] if history else None
        mention_history = self._project_story_mention_reviews_unlocked(
            plan.project_id, graph.graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        expected_binding = (
            (active_review.review_id, active_review.review_revision)
            if active_review is not None else (None, None)
        )
        actual_binding = (
            plan.project_story_link_review_id,
            plan.project_story_link_review_revision,
        )
        if actual_binding != expected_binding:
            raise ValueError(
                "project Plan story link review is missing or no longer current")
        if active_review is not None:
            validate_project_story_link_review_for_plan(
                active_review,
                manifest,
                context,
                graph,
                list(observations.values()),
                active_mention_review,
            )
        for edit in edl.ordered_edits:
            expected_refs = project_story_link_refs_for_edit(edit, active_review)
            if edit.project_story_link_refs != expected_refs:
                raise ValueError(
                    "project EDL story link refs do not match its reviewed source intervals")

    def get_idempotent_project_story_link_review(
        self,
        project_id: str,
        graph_id: str,
        *,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> ProjectStoryLinkReview | None:
        """Return an exact prior write before current evidence is revalidated."""
        if not idempotency_key.strip() or len(idempotency_key) > 128:
            raise ValueError("idempotency_key must contain 1 to 128 non-whitespace characters")
        if (
            len(request_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in request_fingerprint)
        ):
            raise ValueError("request_fingerprint must be 64 lowercase hex characters")

        key_digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        ledger_id = "project_story_links_" + hashlib.sha256(
            f"{project_id}\0{graph_id}\0{key_digest}".encode("utf-8")
        ).hexdigest()[:32]
        self._conn.execute("BEGIN")
        try:
            row = self._conn.execute(
                "SELECT project_id, data FROM decision_ledger WHERE ledger_id = ?",
                (ledger_id,),
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            entry = self._load(DecisionLedgerEntry, row)
            detail = entry.detail or {}
            if (
                row["project_id"] != project_id
                or entry.decision_id != graph_id
                or entry.action != "project_story_links_revised"
            ):
                raise ValueError("idempotency key is already bound to another operation")
            if detail.get("request_fingerprint") != request_fingerprint:
                raise ValueError("idempotency key was reused with a different request")
            stored = ProjectStoryLinkReview(**detail["review"])
            self._conn.commit()
            return stored
        except Exception:
            self._conn.rollback()
            raise

    def save_project_story_link_review(
        self,
        project_id: str,
        review: ProjectStoryLinkReview,
        *,
        expected_review_revision: int,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> tuple[ProjectStoryLinkReview, bool]:
        """Append a full caller-asserted link/disposition snapshot with CAS semantics.

        The validation and ledger append share one SQLite write transaction.
        This does not change the source-local graph or execute model inference.
        """
        from director_brain.project_story_link_review import (
            validate_project_story_comparison_dispositions,
            validate_project_story_link_review_source_mentions,
        )

        if review.project_id != project_id:
            raise ValueError("link review project_id does not match route scope")
        if expected_review_revision < 0:
            raise ValueError("expected_review_revision must be non-negative")
        if not idempotency_key.strip() or len(idempotency_key) > 128:
            raise ValueError("idempotency_key must contain 1 to 128 non-whitespace characters")
        if (
            len(request_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in request_fingerprint)
        ):
            raise ValueError("request_fingerprint must be 64 lowercase hex characters")

        key_digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        ledger_id = "project_story_links_" + hashlib.sha256(
            f"{project_id}\0{review.story_graph_id}\0{key_digest}".encode("utf-8")
        ).hexdigest()[:32]

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing_row = self._conn.execute(
                "SELECT project_id, data FROM decision_ledger WHERE ledger_id = ?",
                (ledger_id,),
            ).fetchone()
            if existing_row is not None:
                entry = self._load(DecisionLedgerEntry, existing_row)
                detail = entry.detail or {}
                if (
                    existing_row["project_id"] != project_id
                    or entry.decision_id != review.story_graph_id
                    or entry.action != "project_story_links_revised"
                ):
                    raise ValueError("idempotency key is already bound to another operation")
                if detail.get("request_fingerprint") != request_fingerprint:
                    raise ValueError("idempotency key was reused with a different request")
                stored = ProjectStoryLinkReview(**detail["review"])
                self._conn.commit()
                return stored, True

            manifest_row = self._conn.execute(
                "SELECT data FROM film_project_manifests "
                "WHERE project_id = ? ORDER BY revision DESC LIMIT 1",
                (project_id,),
            ).fetchone()
            if manifest_row is None:
                raise ValueError("project manifest does not exist")
            manifest = self._load(FilmProjectManifest, manifest_row)
            if (
                manifest.manifest_id != review.project_manifest_id
                or manifest.revision != review.project_revision
            ):
                raise ValueError("link review manifest revision is stale")

            context_row = self._conn.execute(
                "SELECT data FROM film_context_snapshots WHERE context_id = ?",
                (review.context_id,),
            ).fetchone()
            if context_row is None:
                raise ValueError("link review context does not exist")
            context = self._load(FilmContextSnapshot, context_row)
            if (
                context.invalidated_at is not None
                or context.project_id != project_id
                or context.project_manifest_id != manifest.manifest_id
                or context.project_revision != manifest.revision
                or context.analysis_fingerprint != review.analysis_fingerprint
            ):
                raise ValueError("link review context is stale or inconsistent")

            graph_row = self._conn.execute(
                "SELECT data FROM project_story_graphs WHERE graph_id = ?",
                (review.story_graph_id,),
            ).fetchone()
            if graph_row is None:
                raise ValueError("link review StoryGraph does not exist")
            graph = self._load(ProjectStoryGraph, graph_row)
            if (
                graph.project_id != project_id
                or graph.project_manifest_id != manifest.manifest_id
                or graph.project_revision != manifest.revision
                or graph.context_id != context.context_id
                or graph.analysis_fingerprint != context.analysis_fingerprint
                or graph.evidence_refs != context.evidence_refs
                or graph.timeline_scope != "project_per_asset"
            ):
                raise ValueError("link review StoryGraph is stale or inconsistent")

            history = self._project_story_link_reviews_unlocked(
                project_id, review.story_graph_id)
            current_revision = history[-1].review_revision if history else 0
            if current_revision != expected_review_revision:
                raise ValueError(
                    f"stale link review: expected revision {expected_review_revision}, "
                    f"current revision is {current_revision}"
                )
            if review.review_revision != current_revision + 1:
                raise ValueError("link review revision must increment the current revision by one")

            mention_history = self._project_story_mention_reviews_unlocked(
                project_id, review.story_graph_id)
            active_mention_review = mention_history[-1] if mention_history else None
            validate_project_story_link_review_source_mentions(
                review, active_mention_review)

            assets = {item.asset_id: item for item in manifest.assets}
            graph_evidence = set(graph.evidence_refs)
            for link in review.links:
                for anchor in link.anchors:
                    asset = assets.get(anchor.project_asset_id)
                    observation_row = self._conn.execute(
                        "SELECT data FROM film_observations WHERE observation_id = ? "
                        "AND project_id = ?",
                        (anchor.observation_id, project_id),
                    ).fetchone()
                    if asset is None or observation_row is None:
                        raise ValueError("link anchor asset or observation is missing")
                    if (
                        asset.rights.state.value != "local_processing_allowed"
                        or asset.source_identity_state != "locally_verified"
                        or asset.source_content_hash is None
                        or asset.source_content_hash.lower()
                        != anchor.source_content_hash.lower()
                    ):
                        raise ValueError("link anchor is not bound to a locally authorized source")
                    observation = self._load(FilmObservation, observation_row)
                    if (
                        anchor.observation_id not in graph_evidence
                        or observation.project_id != project_id
                        or observation.project_asset_id != asset.asset_id
                        or observation.media_hash.lower() != anchor.source_content_hash.lower()
                        or observation.timebase != anchor.timebase
                        or observation.timebase_unit != anchor.timebase_unit
                        or anchor.source_start < observation.start_frame
                        or anchor.source_end > observation.end_frame
                    ):
                        raise ValueError("link anchor does not match current project evidence")

            comparisons: dict[str, ProjectStoryLinkComparison] = {}
            comparison_table, comparison_id_col = _TABLES[
                ProjectStoryLinkComparison]
            for disposition in review.comparison_dispositions:
                comparison_row = self._conn.execute(
                    f"SELECT data FROM {comparison_table} "
                    f"WHERE {comparison_id_col} = ? AND project_id = ?",
                    (disposition.comparison_id, project_id),
                ).fetchone()
                if comparison_row is None:
                    raise ValueError("comparison disposition source is missing")
                comparisons[disposition.comparison_id] = self._load(
                    ProjectStoryLinkComparison, comparison_row)
            validate_project_story_comparison_dispositions(
                review, comparisons)

            payload = json.dumps(review.model_dump(mode="json"), ensure_ascii=False)
            entry = DecisionLedgerEntry(
                ledger_id=ledger_id,
                decision_id=review.story_graph_id,
                action="project_story_links_revised",
                timestamp=review.created_at,
                project_id=project_id,
                detail={
                    "review": review.model_dump(mode="json"),
                    "request_fingerprint": request_fingerprint,
                    "idempotency_key_sha256": key_digest,
                    "expected_review_revision": expected_review_revision,
                    "automatic_inference_state": "not_attempted",
                },
            )
            self._conn.execute(
                "INSERT INTO decision_ledger "
                "(ledger_id, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                (ledger_id, project_id, review.created_at, json.dumps(
                    self._dump(entry), ensure_ascii=False)),
            )
            self._conn.commit()
            return ProjectStoryLinkReview(**json.loads(payload)), False
        except Exception:
            self._conn.rollback()
            raise

    def save_project_plan_bundle(
        self,
        brief: DirectorBrief,
        edl: EditorialDecisionList,
        plan: DirectorDecisionPlan,
        *,
        expected_superseded_plan_hash: str | None = None,
        expected_superseded_edl_hash: str | None = None,
        clarified_by: str | None = None,
    ) -> tuple[DirectorBrief, EditorialDecisionList, DirectorDecisionPlan, bool]:
        """Atomically persist one immutable project Plan and its evidence links.

        Plan IDs are deterministic for the exact request. Repeating that
        request returns the stored records; an ID collision with changed
        content is rejected instead of silently replacing a prior draft.
        The current manifest, context, graph, and observations are rechecked
        inside the write transaction to close revision races.
        """
        from director_brain.plan_state import PlanState

        expected_state = (
            PlanState.READY_FOR_STRATEGY_CONFIRMATION.value
            if plan.validation_status == "valid"
            else PlanState.FAILED_VALIDATION.value
            if plan.validation_status == "invalid"
            else PlanState.NEEDS_INPUT.value
            if plan.validation_status == "needs_input"
            else None
        )
        if plan.state != expected_state or plan.approval_state != "draft":
            raise ValueError(
                "project plans must be validated before persistence and remain unconfirmed"
            )
        supersession_requested = plan.supersedes_plan_id is not None
        if supersession_requested and any(value is None for value in (
            expected_superseded_plan_hash,
            expected_superseded_edl_hash,
            clarified_by,
        )):
            raise ValueError(
                "project Plan supersession requires both reviewed hashes and a caller")
        if not supersession_requested and (
            expected_superseded_plan_hash is not None
            or expected_superseded_edl_hash is not None
            or clarified_by is not None
        ):
            raise ValueError("supersession metadata requires a predecessor Plan")
        if supersession_requested and (
            plan.supersedes_plan_id == plan.plan_id
            or not clarified_by.strip()
            or len(clarified_by) > 200
            or any(
                len(value) != 16
                or any(ch not in "0123456789abcdef" for ch in value)
                for value in (
                    expected_superseded_plan_hash,
                    expected_superseded_edl_hash,
                )
            )
        ):
            raise ValueError("project Plan supersession identity or hashes are invalid")
        if not (
            brief.project_id == edl.project_id == plan.project_id
            and plan.brief_id == brief.brief_id
            and plan.edl_id == edl.edl_id
            and plan.brief_version == brief.version == edl.brief_version
            and plan.film_state_version == plan.project_context_id
            and edl.context_id == plan.project_story_graph_id
        ):
            raise ValueError("project Plan, Brief, and EDL bindings are inconsistent")
        if (
            plan.project_manifest_id is None
            or plan.project_revision is None
            or plan.project_context_id is None
            or plan.project_story_graph_id is None
        ):
            raise ValueError("project Plan requires complete evidence bindings")
        if plan.sequence != [item.source_asset_id for item in edl.ordered_edits]:
            raise ValueError("project Plan sequence differs from its EDL")
        expected_project_assets = [
            item.project_asset_id for item in edl.ordered_edits
        ]
        if plan.sequence_project_asset_ids != expected_project_assets:
            raise ValueError("project Plan asset sequence differs from its EDL")
        edl_evidence_refs = {
            evidence_ref
            for edit in edl.ordered_edits
            for evidence_ref in edit.source_observation_refs
        }
        expected_hashes = list(dict.fromkeys(
            edit.source_media_hash.lower() for edit in edl.ordered_edits
        ))
        if [item.lower() for item in edl.source_asset_hashes] != expected_hashes:
            raise ValueError("project EDL source hashes differ from its edits")

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            manifest = self.latest_project_manifest(plan.project_id)
            if (
                manifest is None
                or manifest.manifest_id != plan.project_manifest_id
                or manifest.revision != plan.project_revision
            ):
                raise ValueError("project Plan manifest revision is no longer current")
            context = self.get(FilmContextSnapshot, plan.project_context_id)
            if (
                context is None
                or context.invalidated_at is not None
                or context.project_id != plan.project_id
                or context.project_manifest_id != manifest.manifest_id
                or context.project_revision != manifest.revision
                or not edl_evidence_refs <= set(context.evidence_refs)
            ):
                raise ValueError("project Plan context evidence is stale or incomplete")
            graph = self.get(ProjectStoryGraph, plan.project_story_graph_id)
            if (
                graph is None
                or graph.project_id != plan.project_id
                or graph.project_manifest_id != manifest.manifest_id
                or graph.project_revision != manifest.revision
                or graph.context_id != context.context_id
                or graph.evidence_refs != context.evidence_refs
            ):
                raise ValueError("project Plan StoryGraph is stale or inconsistent")

            assets = {item.asset_id: item for item in manifest.assets}
            ordered_assets = sorted(manifest.assets, key=lambda item: item.order)
            expected_asset_ids = [item.asset_id for item in ordered_assets]
            expected_source_hashes = [
                item.source_content_hash.lower()
                if item.source_content_hash is not None else None
                for item in ordered_assets
            ]
            if (
                context.timeline_scope != "project_per_asset"
                or context.asset_refs != expected_asset_ids
                or [
                    item.lower() if item is not None else None
                    for item in context.source_content_hashes
                ] != expected_source_hashes
                or len(context.asset_coverage) != len(ordered_assets)
                or [item.asset_id for item in graph.assets] != expected_asset_ids
                or [item.order for item in graph.assets]
                != [item.order for item in ordered_assets]
            ):
                raise ValueError("project Plan context or graph asset set is stale")
            for asset, coverage, graph_asset in zip(
                ordered_assets, context.asset_coverage, graph.assets, strict=True
            ):
                expected_hash = (
                    asset.source_content_hash.lower()
                    if asset.source_content_hash is not None else None
                )
                if (
                    coverage.asset_id != asset.asset_id
                    or coverage.order != asset.order
                    or (coverage.source_content_hash or "").lower()
                    != (expected_hash or "")
                    or coverage.rights_state != asset.rights.state.value
                    or coverage.rights_evidence_state != asset.rights.evidence_state
                    or (graph_asset.source_content_hash or "").lower()
                    != (expected_hash or "")
                    or graph_asset.analysis_state != coverage.analysis_state
                    or graph_asset.evidence_refs != coverage.evidence_refs
                ):
                    raise ValueError("project Plan asset coverage is inconsistent")
            asset_hashes = {
                asset_id: item.source_content_hash.lower()
                for asset_id, item in assets.items()
                if item.source_content_hash is not None
            }
            context_evidence = set(context.evidence_refs)
            observations_by_id: dict[str, FilmObservation] = {}
            for evidence_ref in context.evidence_refs:
                observation = self.get(FilmObservation, evidence_ref)
                if observation is None or observation.project_id != plan.project_id:
                    raise ValueError("project Plan references missing or out-of-scope evidence")
                observation_asset = assets.get(observation.project_asset_id)
                if (
                    observation_asset is None
                    or asset_hashes.get(observation.project_asset_id)
                    != observation.media_hash.lower()
                ):
                    raise ValueError("project observation is not bound to the current manifest")
                observations_by_id[evidence_ref] = observation
            self._validate_project_plan_story_links_unlocked(
                plan, edl, manifest, context, graph, observations_by_id)
            for edit in edl.ordered_edits:
                if (
                    edit.project_asset_id not in assets
                    or asset_hashes.get(edit.project_asset_id) != edit.source_media_hash.lower()
                    or not set(edit.source_observation_refs) <= context_evidence
                ):
                    raise ValueError("project EDL edit is not bound to current project evidence")
                for evidence_ref in edit.source_observation_refs:
                    observation = observations_by_id[evidence_ref]
                    if (
                        observation.project_asset_id != edit.project_asset_id
                        or observation.media_hash.lower() != edit.source_media_hash.lower()
                    ):
                        raise ValueError("project EDL observation identity does not match its asset")
            for decision in plan.decisions:
                for evidence_ref in decision.evidence_refs:
                    observation = observations_by_id.get(evidence_ref)
                    if (
                        observation is None
                        or (decision.project_asset_id is not None
                            and observation.project_asset_id != decision.project_asset_id)
                    ):
                        raise ValueError(
                            "project Plan decision cites mismatched project evidence")

            supersession_replay = False
            predecessor_plan = None
            supersession_ledger_id = None
            supersession_fingerprint = None
            if supersession_requested:
                from director_brain.plan_state import compute_edl_hash, compute_plan_hash

                predecessor_bundle = self._get_project_plan_bundle_unlocked(
                    plan.project_id, plan.supersedes_plan_id)
                if predecessor_bundle is None:
                    raise ValueError("superseded project Plan does not exist")
                _predecessor_brief, predecessor_edl, predecessor_plan = predecessor_bundle
                supersession_identity = {
                    "project_id": plan.project_id,
                    "prior_plan_id": predecessor_plan.plan_id,
                    "successor_plan_id": plan.plan_id,
                    "expected_prior_plan_hash": expected_superseded_plan_hash,
                    "expected_prior_edl_hash": expected_superseded_edl_hash,
                    "clarified_by": clarified_by,
                }
                supersession_fingerprint = hashlib.sha256(json.dumps(
                    supersession_identity, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                supersession_ledger_id = "plan_superseded_" + hashlib.sha256(
                    f"{plan.project_id}\\0{predecessor_plan.plan_id}\\0{plan.plan_id}"
                    .encode("utf-8")
                ).hexdigest()[:32]
                if (
                    predecessor_plan.project_manifest_id != manifest.manifest_id
                    or predecessor_plan.project_revision != manifest.revision
                    or predecessor_plan.project_context_id != context.context_id
                    or predecessor_plan.project_story_graph_id != graph.graph_id
                    or predecessor_plan.approval_state != "draft"
                    or predecessor_edl.approval_state != "draft"
                    or predecessor_plan.validation_status != "needs_input"
                    or compute_plan_hash(predecessor_plan)
                    != expected_superseded_plan_hash
                    or compute_edl_hash(predecessor_edl)
                    != expected_superseded_edl_hash
                ):
                    raise ValueError(
                        "superseded Plan is stale, not awaiting input, or its reviewed hashes changed"
                    )
                if predecessor_plan.state == "superseded":
                    supersession_row = self._conn.execute(
                        "SELECT project_id, data FROM decision_ledger WHERE ledger_id = ?",
                        (supersession_ledger_id,),
                    ).fetchone()
                    if supersession_row is None:
                        raise ValueError("superseded Plan has no matching lifecycle receipt")
                    entry = self._load(DecisionLedgerEntry, supersession_row)
                    detail = entry.detail or {}
                    receipt = detail.get("supersession")
                    successor_bundle = self._get_project_plan_bundle_unlocked(
                        plan.project_id, plan.plan_id)
                    if successor_bundle is None:
                        raise ValueError("supersession receipt has no stored successor Plan")
                    _successor_brief, successor_edl, successor_plan = successor_bundle
                    if (
                        supersession_row["project_id"] != plan.project_id
                        or entry.action != "plan_superseded"
                        or entry.decision_id != predecessor_plan.plan_id
                        or detail.get("request_fingerprint") != supersession_fingerprint
                        or not isinstance(receipt, dict)
                        or receipt.get("prior_plan_id") != predecessor_plan.plan_id
                        or receipt.get("successor_plan_id") != plan.plan_id
                        or receipt.get("prior_state") != "needs_input"
                        or receipt.get("prior_plan_hash") != expected_superseded_plan_hash
                        or receipt.get("prior_edl_hash") != expected_superseded_edl_hash
                        or receipt.get("clarified_by") != clarified_by
                        or receipt.get("actor_identity_state") != "caller_asserted"
                        or receipt.get("state") != "superseded"
                        or successor_plan.supersedes_plan_id != predecessor_plan.plan_id
                        or receipt.get("successor_plan_hash")
                        != compute_plan_hash(successor_plan)
                        or receipt.get("successor_edl_hash")
                        != compute_edl_hash(successor_edl)
                        or compute_plan_hash(successor_plan) != compute_plan_hash(plan)
                        or compute_edl_hash(successor_edl) != compute_edl_hash(edl)
                    ):
                        raise ValueError("stored Plan supersession receipt is inconsistent")
                    supersession_replay = True
                elif predecessor_plan.state != "needs_input":
                    raise ValueError("only a Plan awaiting input can be superseded by clarification")

            def insert_or_reuse(entity: Any) -> tuple[Any, bool]:
                table, id_col = _TABLES[type(entity)]
                entity_data = self._dump(entity)
                entity_id = getattr(entity, id_col)
                row = self._conn.execute(
                    f"SELECT data FROM {table} WHERE {id_col} = ?", (entity_id,)
                ).fetchone()
                if row is not None:
                    stored_data = json.loads(row["data"])
                    compare_data = dict(entity_data)
                    compare_stored = dict(stored_data)
                    compare_data.pop("created_at", None)
                    compare_stored.pop("created_at", None)
                    workflow_fields = set()
                    if isinstance(entity, DirectorDecisionPlan):
                        workflow_fields.update({"state", "approval_state"})
                    elif isinstance(entity, EditorialDecisionList):
                        workflow_fields.add("approval_state")
                    for field in workflow_fields:
                        compare_data.pop(field, None)
                        compare_stored.pop(field, None)
                    if compare_stored != compare_data:
                        raise ValueError(
                            f"project draft identity collision or result drift: {entity_id}"
                        )
                    stored_entity = self._load(type(entity), row)
                    if (
                        isinstance(entity, DirectorDecisionPlan)
                        and stored_entity.state == "draft"
                        and entity.state in {
                            "ready_for_strategy_confirmation",
                            "needs_input",
                            "failed_validation",
                        }
                        and stored_entity.approval_state == "draft"
                    ):
                        # Older P1 drafts were saved before the Plan state
                        # machine was connected. A current revalidation of the
                        # identical request can safely advance that workflow
                        # record while preserving its content identity.
                        from director_brain.plan_state import (
                            PlanState,
                            transition_plan,
                        )

                        context_state = (
                            PlanState.CONTEXT_READY
                            if context.coverage.endswith("_observed")
                            else PlanState.CONTEXT_PARTIAL
                        )
                        target_state = (
                            PlanState.READY_FOR_STRATEGY_CONFIRMATION
                            if entity.validation_status == "valid"
                            else PlanState.NEEDS_INPUT
                            if entity.validation_status == "needs_input"
                            else PlanState.FAILED_VALIDATION
                        )
                        transition_plan(stored_entity, context_state)
                        transition_plan(stored_entity, PlanState.VALIDATING)
                        transition_plan(stored_entity, target_state)
                        stored_payload = json.dumps(
                            self._dump(stored_entity), ensure_ascii=False)
                        self._conn.execute(
                            "UPDATE director_decision_plans SET data = ? "
                            "WHERE plan_id = ? AND project_id = ?",
                            (stored_payload, entity_id, plan.project_id),
                        )
                    return stored_entity, True

                payload = json.dumps(entity_data, ensure_ascii=False)
                project_id, created_at = self._meta(entity)
                self._conn.execute(
                    f"INSERT INTO {table} "
                    f"({id_col}, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                    (entity_id, project_id, created_at, payload),
                )
                return entity, False

            stored_brief, brief_reused = insert_or_reuse(brief)
            stored_edl, edl_reused = insert_or_reuse(edl)
            stored_plan, plan_reused = insert_or_reuse(plan)
            if supersession_requested and not supersession_replay:
                from director_brain.plan_state import (
                    PlanState,
                    compute_edl_hash,
                    compute_plan_hash,
                    transition_plan,
                )

                transition_plan(predecessor_plan, PlanState.SUPERSEDED)
                predecessor_payload = json.dumps(
                    self._dump(predecessor_plan), ensure_ascii=False)
                predecessor_update = self._conn.execute(
                    "UPDATE director_decision_plans SET data = ? "
                    "WHERE plan_id = ? AND project_id = ?",
                    (
                        predecessor_payload,
                        predecessor_plan.plan_id,
                        plan.project_id,
                    ),
                )
                if predecessor_update.rowcount != 1:
                    raise ValueError("predecessor Plan changed during supersession")
                timestamp = int(time.time())
                supersession = {
                    "state": PlanState.SUPERSEDED.value,
                    "prior_state": PlanState.NEEDS_INPUT.value,
                    "prior_plan_id": predecessor_plan.plan_id,
                    "successor_plan_id": stored_plan.plan_id,
                    "prior_plan_hash": expected_superseded_plan_hash,
                    "prior_edl_hash": expected_superseded_edl_hash,
                    "successor_plan_hash": compute_plan_hash(stored_plan),
                    "successor_edl_hash": compute_edl_hash(stored_edl),
                    "clarified_by": clarified_by,
                    "actor_identity_state": "caller_asserted",
                    "superseded_at": timestamp,
                }
                entry = DecisionLedgerEntry(
                    ledger_id=supersession_ledger_id,
                    decision_id=predecessor_plan.plan_id,
                    action="plan_superseded",
                    timestamp=timestamp,
                    project_id=plan.project_id,
                    detail={
                        "supersession": supersession,
                        "request_fingerprint": supersession_fingerprint,
                    },
                )
                self._conn.execute(
                    "INSERT INTO decision_ledger "
                    "(ledger_id, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                    (
                        supersession_ledger_id,
                        plan.project_id,
                        timestamp,
                        json.dumps(self._dump(entry), ensure_ascii=False),
                    ),
                )
            self._conn.commit()
            return (
                stored_brief,
                stored_edl,
                stored_plan,
                brief_reused and edl_reused and plan_reused,
            )
        except Exception:
            self._conn.rollback()
            raise

    def _validate_project_plan_supersession_pair_unlocked(
        self,
        project_id: str,
        prior_plan_id: str,
        successor_plan_id: str,
    ) -> None:
        """Verify both Plan versions and their atomic supersession ledger event."""
        from director_brain.plan_state import compute_edl_hash, compute_plan_hash

        if prior_plan_id == successor_plan_id:
            raise ValueError("project Plan cannot supersede itself")
        prior = self.get(DirectorDecisionPlan, prior_plan_id)
        successor = self.get(DirectorDecisionPlan, successor_plan_id)
        if (
            prior is None
            or successor is None
            or prior.project_id != project_id
            or successor.project_id != project_id
            or prior.state != "superseded"
            or prior.validation_status != "needs_input"
            or prior.approval_state != "draft"
            or successor.supersedes_plan_id != prior_plan_id
            or successor.state not in {
                "ready_for_strategy_confirmation", "needs_input", "failed_validation",
            }
            or successor.approval_state != "draft"
            or successor.validation_status != {
                "ready_for_strategy_confirmation": "valid",
                "needs_input": "needs_input",
                "failed_validation": "invalid",
            }[successor.state]
            or prior.project_manifest_id != successor.project_manifest_id
            or prior.project_revision != successor.project_revision
            or prior.project_context_id != successor.project_context_id
            or prior.project_story_graph_id != successor.project_story_graph_id
        ):
            raise ValueError("stored project Plan supersession states or bindings are inconsistent")

        prior_edl = self.get(EditorialDecisionList, prior.edl_id or "")
        successor_edl = self.get(EditorialDecisionList, successor.edl_id or "")
        if (
            prior_edl is None
            or successor_edl is None
            or prior_edl.project_id != project_id
            or successor_edl.project_id != project_id
            or prior_edl.approval_state != "draft"
            or successor_edl.approval_state != "draft"
        ):
            raise ValueError("stored project Plan supersession EDL bindings are inconsistent")

        prior_plan_hash = compute_plan_hash(prior)
        prior_edl_hash = compute_edl_hash(prior_edl)
        successor_plan_hash = compute_plan_hash(successor)
        successor_edl_hash = compute_edl_hash(successor_edl)
        ledger_id = "plan_superseded_" + hashlib.sha256(
            f"{project_id}\\0{prior_plan_id}\\0{successor_plan_id}".encode("utf-8")
        ).hexdigest()[:32]
        row = self._conn.execute(
            "SELECT project_id, data FROM decision_ledger WHERE ledger_id = ?",
            (ledger_id,),
        ).fetchone()
        if row is None:
            raise ValueError("superseded project Plan has no lifecycle ledger event")
        entry = self._load(DecisionLedgerEntry, row)
        detail = entry.detail or {}
        receipt = detail.get("supersession")
        if not isinstance(receipt, dict):
            raise ValueError("project Plan supersession ledger receipt is missing")
        clarified_by = receipt.get("clarified_by")
        if not isinstance(clarified_by, str) or not clarified_by.strip():
            raise ValueError("project Plan supersession caller is missing")
        identity = {
            "project_id": project_id,
            "prior_plan_id": prior_plan_id,
            "successor_plan_id": successor_plan_id,
            "expected_prior_plan_hash": prior_plan_hash,
            "expected_prior_edl_hash": prior_edl_hash,
            "clarified_by": clarified_by,
        }
        fingerprint = hashlib.sha256(json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if (
            row["project_id"] != project_id
            or entry.project_id != project_id
            or entry.action != "plan_superseded"
            or entry.decision_id != prior_plan_id
            or detail.get("request_fingerprint") != fingerprint
            or receipt.get("state") != "superseded"
            or receipt.get("prior_state") != "needs_input"
            or receipt.get("prior_plan_id") != prior_plan_id
            or receipt.get("successor_plan_id") != successor_plan_id
            or receipt.get("prior_plan_hash") != prior_plan_hash
            or receipt.get("prior_edl_hash") != prior_edl_hash
            or receipt.get("successor_plan_hash") != successor_plan_hash
            or receipt.get("successor_edl_hash") != successor_edl_hash
            or receipt.get("actor_identity_state") != "caller_asserted"
            or receipt.get("superseded_at") != entry.timestamp
        ):
            raise ValueError("project Plan supersession receipt does not match stored artifacts")

    def _get_project_plan_bundle_unlocked(
        self, project_id: str, plan_id: str
    ) -> tuple[DirectorBrief, EditorialDecisionList, DirectorDecisionPlan] | None:
        """Read and validate a project Plan bundle inside the caller's transaction."""
        plan = self.get(DirectorDecisionPlan, plan_id)
        if plan is None or plan.project_id != project_id:
            return None
        if not plan.brief_id or not plan.edl_id:
            raise ValueError("stored project Plan has incomplete Brief/EDL links")
        brief = self.get(DirectorBrief, plan.brief_id)
        edl = self.get(EditorialDecisionList, plan.edl_id)
        if (
            brief is None
            or edl is None
            or brief.project_id != project_id
            or edl.project_id != project_id
            or plan.brief_id != brief.brief_id
            or plan.edl_id != edl.edl_id
            or plan.brief_version != brief.version
            or edl.brief_version != brief.version
            or plan.film_state_version != plan.project_context_id
            or edl.context_id != plan.project_story_graph_id
            or plan.sequence != [item.source_asset_id for item in edl.ordered_edits]
            or plan.sequence_project_asset_ids
            != [item.project_asset_id for item in edl.ordered_edits]
        ):
            raise ValueError("stored project Plan bundle is incomplete or inconsistent")
        if plan.supersedes_plan_id is not None:
            self._validate_project_plan_supersession_pair_unlocked(
                project_id, plan.supersedes_plan_id, plan.plan_id)
        if plan.state == "superseded":
            rows = self._conn.execute(
                "SELECT project_id, data FROM decision_ledger WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            entries = [self._load(DecisionLedgerEntry, row) for row in rows]
            successors = [
                entry.detail.get("supersession", {}).get("successor_plan_id")
                for entry in entries
                if entry.action == "plan_superseded"
                and entry.decision_id == plan.plan_id
                and isinstance(entry.detail, dict)
                and isinstance(entry.detail.get("supersession"), dict)
            ]
            if len(successors) != 1 or not isinstance(successors[0], str):
                raise ValueError("superseded project Plan must have one successor receipt")
            self._validate_project_plan_supersession_pair_unlocked(
                project_id, plan.plan_id, successors[0])
        return brief, edl, plan

    def get_project_plan_bundle(
        self, project_id: str, plan_id: str
    ) -> tuple[DirectorBrief, EditorialDecisionList, DirectorDecisionPlan] | None:
        """Read a project Plan bundle consistently and enforce project ownership."""
        from director_brain.plan_state import PlanState

        self._conn.execute("BEGIN")
        try:
            bundle = self._get_project_plan_bundle_unlocked(project_id, plan_id)
            if bundle is not None and bundle[2].state in {
                PlanState.DISPATCH_ELIGIBLE.value,
                PlanState.EXECUTION_VERIFIED.value,
                PlanState.EXECUTION_BLOCKED.value,
            }:
                raise ValueError(
                    "project Plan execution state requires an admitted P3 ExecutionPort receipt"
                )
            self._conn.commit()
            return bundle
        except Exception:
            self._conn.rollback()
            raise

    def confirm_project_strategy(
        self,
        project_id: str,
        plan_id: str,
        *,
        expected_plan_hash: str,
        expected_edl_hash: str,
        idempotency_key: str,
        confirmed_by: str,
        output_target: str = "delivery",
        notes: str = "",
    ) -> tuple[DirectorBrief, EditorialDecisionList, DirectorDecisionPlan, dict, bool]:
        """Atomically confirm an exact, currently valid project Plan/EDL pair.

        The confirmation receipt, Plan/EDL approvals, state transition, and
        project ledger entry commit together. Replays return the original
        receipt only when the idempotency key and full request fingerprint
        match. This does not make the Plan dispatch eligible for Resolve.
        """
        from director_brain.plan_state import (
            PlanState,
            StrategyConfirmation,
            confirm_strategy,
            compute_edl_hash,
            compute_plan_hash,
            is_confirmation_valid,
            transition_plan,
        )
        from director_brain.plan_validator import validate_plan
        from director_brain.intent_constraints import interpret_constraints

        for label, value in (
            ("plan", expected_plan_hash),
            ("EDL", expected_edl_hash),
        ):
            if (
                len(value) != 16
                or value != value.lower()
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"expected {label} hash must be 16 lowercase hex characters")
        if not idempotency_key.strip() or len(idempotency_key) > 128:
            raise ValueError("idempotency_key must contain 1 to 128 non-whitespace characters")
        if not confirmed_by.strip() or len(confirmed_by) > 200:
            raise ValueError("confirmed_by must contain 1 to 200 non-whitespace characters")
        if output_target not in {"preview", "delivery"}:
            raise ValueError("output_target must be preview or delivery")

        idempotency_digest = hashlib.sha256(
            idempotency_key.encode("utf-8")
        ).hexdigest()
        request_identity = {
            "project_id": project_id,
            "plan_id": plan_id,
            "expected_plan_hash": expected_plan_hash,
            "expected_edl_hash": expected_edl_hash,
            "confirmed_by": confirmed_by,
            "output_target": output_target,
            "notes": notes,
            "idempotency_key_sha256": idempotency_digest,
        }
        request_fingerprint = hashlib.sha256(json.dumps(
            request_identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        ledger_id = "strategy_confirmed_" + hashlib.sha256(
            f"{project_id}\0{idempotency_key}".encode("utf-8")
        ).hexdigest()[:32]

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing_row = self._conn.execute(
                "SELECT project_id, data FROM decision_ledger WHERE ledger_id = ?",
                (ledger_id,),
            ).fetchone()
            if existing_row is not None:
                entry = self._load(DecisionLedgerEntry, existing_row)
                detail = entry.detail or {}
                if (
                    existing_row["project_id"] != project_id
                    or entry.decision_id != plan_id
                    or entry.action != "strategy_confirmed"
                ):
                    raise ValueError("idempotency key is already bound to another operation")
                if detail.get("request_fingerprint") != request_fingerprint:
                    raise ValueError("idempotency key was reused with a different request")
                bundle = self._get_project_plan_bundle_unlocked(project_id, plan_id)
                if bundle is None:
                    raise ValueError("confirmed project Plan does not exist")
                brief, edl, plan = bundle
                confirmation = StrategyConfirmation(**detail["confirmation"])
                if (
                    plan.state != PlanState.STRATEGY_CONFIRMED.value
                    or plan.approval_state != "approved"
                    or edl.approval_state != "approved"
                    or not is_confirmation_valid(confirmation, plan, edl)
                ):
                    raise ValueError("stored project confirmation no longer matches its Plan/EDL")
                self._conn.commit()
                return brief, edl, plan, confirmation.to_dict(), True

            bundle = self._get_project_plan_bundle_unlocked(project_id, plan_id)
            if bundle is None:
                raise ValueError("project Plan does not exist")
            brief, edl, plan = bundle
            if interpret_constraints(brief).unverifiable:
                raise ValueError(
                    "project Plan has unresolved semantic constraints and needs input"
                )
            if (
                plan.state != PlanState.READY_FOR_STRATEGY_CONFIRMATION.value
                or plan.validation_status != "valid"
                or plan.approval_state != "draft"
                or edl.approval_state != "draft"
            ):
                raise ValueError("project Plan is not ready for strategy confirmation")
            if (compute_plan_hash(plan) != expected_plan_hash
                    or compute_edl_hash(edl) != expected_edl_hash):
                raise ValueError("project Plan/EDL changed after the caller reviewed the hashes")

            manifest = self.latest_project_manifest(project_id)
            if (
                manifest is None
                or manifest.manifest_id != plan.project_manifest_id
                or manifest.revision != plan.project_revision
            ):
                raise ValueError("project Plan manifest revision is no longer current")
            context = self.get(FilmContextSnapshot, plan.project_context_id)
            if (
                context is None
                or context.invalidated_at is not None
                or context.project_id != project_id
                or context.project_manifest_id != manifest.manifest_id
                or context.project_revision != manifest.revision
                or context.timeline_scope != "project_per_asset"
                or edl.context_id != plan.project_story_graph_id
            ):
                raise ValueError("project Plan context is stale or inconsistent")
            graph = self.get(ProjectStoryGraph, plan.project_story_graph_id)
            if (
                graph is None
                or graph.project_id != project_id
                or graph.project_manifest_id != manifest.manifest_id
                or graph.project_revision != manifest.revision
                or graph.context_id != context.context_id
                or graph.timeline_scope != "project_per_asset"
                or graph.evidence_refs != context.evidence_refs
            ):
                raise ValueError("project Plan StoryGraph is stale or inconsistent")

            ordered_assets = sorted(manifest.assets, key=lambda item: item.order)
            assets = {asset.asset_id: asset for asset in ordered_assets}
            expected_asset_ids = [asset.asset_id for asset in ordered_assets]
            expected_hashes = [
                asset.source_content_hash.lower()
                if asset.source_content_hash is not None else None
                for asset in ordered_assets
            ]
            if (
                context.asset_refs != expected_asset_ids
                or [value.lower() if value is not None else None
                    for value in context.source_content_hashes] != expected_hashes
                or [asset.asset_id for asset in graph.assets] != expected_asset_ids
                or len(context.asset_coverage) != len(ordered_assets)
                or [asset.order for asset in graph.assets]
                != [asset.order for asset in ordered_assets]
            ):
                raise ValueError("project Plan asset coverage is incomplete")
            for asset, coverage, graph_asset in zip(
                ordered_assets, context.asset_coverage, graph.assets, strict=True
            ):
                expected_hash = (
                    asset.source_content_hash.lower()
                    if asset.source_content_hash is not None else None
                )
                if (
                    coverage.asset_id != asset.asset_id
                    or coverage.order != asset.order
                    or (coverage.source_content_hash or "").lower()
                    != (expected_hash or "")
                    or coverage.rights_state != asset.rights.state.value
                    or coverage.rights_evidence_state != asset.rights.evidence_state
                    or (graph_asset.source_content_hash or "").lower()
                    != (expected_hash or "")
                    or graph_asset.analysis_state != coverage.analysis_state
                    or graph_asset.evidence_refs != coverage.evidence_refs
                ):
                    raise ValueError("project Plan asset coverage differs from its manifest")
            observations_by_id: dict[str, FilmObservation] = {}
            for evidence_ref in context.evidence_refs:
                observation = self.get(FilmObservation, evidence_ref)
                if observation is None or observation.project_id != project_id:
                    raise ValueError("project Plan evidence is missing or out of scope")
                asset = assets.get(observation.project_asset_id)
                if (
                    asset is None
                    or asset.source_content_hash is None
                    or observation.media_hash.lower() != asset.source_content_hash.lower()
                ):
                    raise ValueError("project Plan evidence no longer matches its manifest")
                observations_by_id[evidence_ref] = observation

            self._validate_project_plan_story_links_unlocked(
                plan, edl, manifest, context, graph, observations_by_id)

            context_evidence = set(context.evidence_refs)
            expected_edl_hashes = list(dict.fromkeys(
                edit.source_media_hash.lower() for edit in edl.ordered_edits
            ))
            if [value.lower() for value in edl.source_asset_hashes] != expected_edl_hashes:
                raise ValueError("project EDL source hashes differ from its edits")
            for edit in edl.ordered_edits:
                asset = assets.get(edit.project_asset_id or "")
                if (
                    asset is None
                    or asset.source_content_hash is None
                    or asset.source_content_hash.lower() != edit.source_media_hash.lower()
                    or not edit.source_observation_refs
                    or not set(edit.source_observation_refs) <= context_evidence
                ):
                    raise ValueError("project EDL edit is not bound to current project evidence")
                for evidence_ref in edit.source_observation_refs:
                    observation = observations_by_id[evidence_ref]
                    if (
                        observation.project_asset_id != edit.project_asset_id
                        or observation.media_hash.lower() != edit.source_media_hash.lower()
                    ):
                        raise ValueError("project EDL observation identity does not match its asset")
            for decision in plan.decisions:
                for evidence_ref in decision.evidence_refs:
                    observation = observations_by_id.get(evidence_ref)
                    if (
                        observation is None
                        or (decision.project_asset_id is not None
                            and observation.project_asset_id != decision.project_asset_id)
                    ):
                        raise ValueError("project Plan decision cites mismatched evidence")
            valid, errors = validate_plan(edl, plan, list(observations_by_id.values()))
            if not valid:
                raise ValueError("project Plan validation failed: " + "; ".join(errors[:5]))

            confirmation = confirm_strategy(
                plan, edl, confirmed_by=confirmed_by,
                output_target=output_target, notes=notes,
            )
            transition_plan(plan, PlanState.STRATEGY_CONFIRMED)
            plan.approval_state = "approved"
            edl.approval_state = "approved"
            if not is_confirmation_valid(confirmation, plan, edl):
                raise ValueError("project confirmation hash binding failed")

            plan_payload = json.dumps(self._dump(plan), ensure_ascii=False)
            edl_payload = json.dumps(self._dump(edl), ensure_ascii=False)
            plan_update = self._conn.execute(
                "UPDATE director_decision_plans SET data = ? "
                "WHERE plan_id = ? AND project_id = ?",
                (plan_payload, plan_id, project_id),
            )
            edl_update = self._conn.execute(
                "UPDATE editorial_decision_lists SET data = ? "
                "WHERE edl_id = ? AND project_id = ?",
                (edl_payload, edl.edl_id, project_id),
            )
            if plan_update.rowcount != 1 or edl_update.rowcount != 1:
                raise ValueError("project Plan/EDL disappeared during confirmation")

            timestamp = confirmation.confirmed_at
            detail = {
                "confirmation": confirmation.to_dict(),
                "state": PlanState.STRATEGY_CONFIRMED.value,
                "dispatch_eligible": False,
                "dispatch_block_reason": _PROJECT_CONFIRMATION_DISPATCH_BLOCK_REASON,
                "actor_identity_state": "caller_asserted",
                "idempotency_key_sha256": idempotency_digest,
                "request_fingerprint": request_fingerprint,
            }
            entry = DecisionLedgerEntry(
                ledger_id=ledger_id,
                decision_id=plan_id,
                action="strategy_confirmed",
                timestamp=timestamp,
                project_id=project_id,
                detail=detail,
            )
            self._conn.execute(
                "INSERT INTO decision_ledger "
                "(ledger_id, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                (ledger_id, project_id, timestamp, json.dumps(
                    self._dump(entry), ensure_ascii=False,
                )),
            )
            self._conn.commit()
            return brief, edl, plan, confirmation.to_dict(), False
        except Exception:
            self._conn.rollback()
            raise

    def reject_project_strategy(
        self,
        project_id: str,
        plan_id: str,
        *,
        expected_plan_hash: str,
        expected_edl_hash: str,
        idempotency_key: str,
        rejected_by: str,
        reason: str,
    ) -> tuple[DirectorBrief, EditorialDecisionList, DirectorDecisionPlan, dict, bool]:
        """Atomically reject an exact reviewable project Plan/EDL pair.

        NEEDS_INPUT is explicitly rejectable so an unresolved semantic
        constraint cannot leave the user with a blocked but unterminable Plan.
        The rejection and its audit-ledger receipt commit with the Plan state.
        """
        from director_brain.plan_state import (
            PlanState,
            compute_edl_hash,
            compute_plan_hash,
            transition_plan,
        )

        for label, value in (
            ("plan", expected_plan_hash),
            ("EDL", expected_edl_hash),
        ):
            if (
                len(value) != 16
                or value != value.lower()
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"expected {label} hash must be 16 lowercase hex characters")
        if not idempotency_key.strip() or len(idempotency_key) > 128:
            raise ValueError("idempotency_key must contain 1 to 128 non-whitespace characters")
        rejected_by = rejected_by.strip()
        reason = reason.strip()
        if not rejected_by or len(rejected_by) > 200:
            raise ValueError("rejected_by must contain 1 to 200 non-whitespace characters")
        if not reason or len(reason) > 2000:
            raise ValueError("reason must contain 1 to 2000 non-whitespace characters")

        idempotency_digest = hashlib.sha256(
            idempotency_key.encode("utf-8")
        ).hexdigest()
        request_identity = {
            "project_id": project_id,
            "plan_id": plan_id,
            "expected_plan_hash": expected_plan_hash,
            "expected_edl_hash": expected_edl_hash,
            "rejected_by": rejected_by,
            "reason": reason,
            "idempotency_key_sha256": idempotency_digest,
        }
        request_fingerprint = hashlib.sha256(json.dumps(
            request_identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        ledger_id = "strategy_rejected_" + hashlib.sha256(
            f"{project_id}\\0{idempotency_key}".encode("utf-8")
        ).hexdigest()[:32]

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing_row = self._conn.execute(
                "SELECT project_id, data FROM decision_ledger WHERE ledger_id = ?",
                (ledger_id,),
            ).fetchone()
            if existing_row is not None:
                entry = self._load(DecisionLedgerEntry, existing_row)
                detail = entry.detail or {}
                if (
                    existing_row["project_id"] != project_id
                    or entry.decision_id != plan_id
                    or entry.action != "strategy_rejected"
                ):
                    raise ValueError("idempotency key is already bound to another operation")
                if detail.get("request_fingerprint") != request_fingerprint:
                    raise ValueError("idempotency key was reused with a different request")
                bundle = self._get_project_plan_bundle_unlocked(project_id, plan_id)
                if bundle is None:
                    raise ValueError("rejected project Plan does not exist")
                brief, edl, plan = bundle
                rejection_data = detail.get("rejection")
                rejection = rejection_data if isinstance(rejection_data, dict) else {}
                prior_state = rejection.get("prior_state")
                prior_validation_status = {
                    PlanState.READY_FOR_STRATEGY_CONFIRMATION.value: "valid",
                    PlanState.NEEDS_INPUT.value: "needs_input",
                }.get(prior_state)
                if (
                    plan.state != PlanState.REJECTED.value
                    or plan.approval_state != "draft"
                    or edl.approval_state != "draft"
                    or compute_plan_hash(plan) != expected_plan_hash
                    or compute_edl_hash(edl) != expected_edl_hash
                    or rejection.get("state") != PlanState.REJECTED.value
                    or prior_validation_status is None
                    or plan.validation_status != prior_validation_status
                    or detail.get("state") != PlanState.REJECTED.value
                    or detail.get("actor_identity_state") != "caller_asserted"
                ):
                    raise ValueError("stored project rejection no longer matches its Plan/EDL")
                self._conn.commit()
                return brief, edl, plan, rejection, True

            bundle = self._get_project_plan_bundle_unlocked(project_id, plan_id)
            if bundle is None:
                raise ValueError("project Plan does not exist")
            brief, edl, plan = bundle
            reviewable_states = {
                PlanState.READY_FOR_STRATEGY_CONFIRMATION.value,
                PlanState.NEEDS_INPUT.value,
            }
            if (
                plan.state not in reviewable_states
                or plan.approval_state != "draft"
                or edl.approval_state != "draft"
            ):
                raise ValueError("project Plan is not reviewable for rejection")
            expected_validation_status = (
                "valid"
                if plan.state == PlanState.READY_FOR_STRATEGY_CONFIRMATION.value
                else "needs_input"
            )
            if plan.validation_status != expected_validation_status:
                raise ValueError("project Plan state and validation status are inconsistent")
            if (compute_plan_hash(plan) != expected_plan_hash
                    or compute_edl_hash(edl) != expected_edl_hash):
                raise ValueError("project Plan/EDL changed after the caller reviewed the hashes")

            prior_state = plan.state
            transition_plan(plan, PlanState.REJECTED)
            timestamp = int(time.time())
            rejection = {
                "state": PlanState.REJECTED.value,
                "prior_state": prior_state,
                "rejected_by": rejected_by,
                "rejected_at": timestamp,
                "reason": reason,
                "plan_hash": expected_plan_hash,
                "edl_hash": expected_edl_hash,
            }
            detail = {
                "rejection": rejection,
                "state": PlanState.REJECTED.value,
                "actor_identity_state": "caller_asserted",
                "idempotency_key_sha256": idempotency_digest,
                "request_fingerprint": request_fingerprint,
            }
            plan_payload = json.dumps(self._dump(plan), ensure_ascii=False)
            plan_update = self._conn.execute(
                "UPDATE director_decision_plans SET data = ? "
                "WHERE plan_id = ? AND project_id = ?",
                (plan_payload, plan_id, project_id),
            )
            if plan_update.rowcount != 1:
                raise ValueError("project Plan disappeared during rejection")

            entry = DecisionLedgerEntry(
                ledger_id=ledger_id,
                decision_id=plan_id,
                action="strategy_rejected",
                timestamp=timestamp,
                project_id=project_id,
                detail=detail,
            )
            self._conn.execute(
                "INSERT INTO decision_ledger "
                "(ledger_id, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                (ledger_id, project_id, timestamp, json.dumps(
                    self._dump(entry), ensure_ascii=False,
                )),
            )
            self._conn.commit()
            return brief, edl, plan, rejection, False
        except Exception:
            self._conn.rollback()
            raise

    def get_project_strategy_rejection(
        self, project_id: str, plan_id: str
    ) -> dict | None:
        """Read and verify the persisted user rejection receipt for one Plan."""
        from director_brain.plan_state import (
            PlanState,
            compute_edl_hash,
            compute_plan_hash,
        )

        self._conn.execute("BEGIN")
        try:
            rows = self._conn.execute(
                "SELECT data FROM decision_ledger WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            entries = [self._load(DecisionLedgerEntry, row) for row in rows]
            matches = [
                entry for entry in entries
                if entry.decision_id == plan_id
                and entry.action == "strategy_rejected"
                and entry.detail is not None
            ]
            if not matches:
                self._conn.commit()
                return None
            entry = max(matches, key=lambda item: item.timestamp)
            detail = entry.detail or {}
            rejection_data = detail.get("rejection")
            rejection = rejection_data if isinstance(rejection_data, dict) else {}
            bundle = self._get_project_plan_bundle_unlocked(project_id, plan_id)
            hash_binding_valid = bool(
                bundle is not None
                and rejection.get("plan_hash") == compute_plan_hash(bundle[2])
                and rejection.get("edl_hash") == compute_edl_hash(bundle[1])
            )
            workflow_state_consistent = bool(
                bundle is not None
                and bundle[2].state == PlanState.REJECTED.value
                and bundle[2].approval_state == "draft"
                and bundle[1].approval_state == "draft"
                and rejection.get("state") == PlanState.REJECTED.value
                and bundle[2].validation_status == {
                    PlanState.READY_FOR_STRATEGY_CONFIRMATION.value: "valid",
                    PlanState.NEEDS_INPUT.value: "needs_input",
                }.get(rejection.get("prior_state"))
                and detail.get("state") == PlanState.REJECTED.value
                and detail.get("actor_identity_state") == "caller_asserted"
                and isinstance(rejection.get("reason"), str)
                and bool(rejection.get("reason", "").strip())
            )
            receipt_consistent = hash_binding_valid and workflow_state_consistent
            receipt = {
                **rejection,
                "ledger_id": entry.ledger_id,
                "recorded_at": entry.timestamp,
                "plan_edl_hash_binding_valid": hash_binding_valid,
                "receipt_consistency_state": (
                    "consistent" if receipt_consistent else "inconsistent"),
                "state": (
                    PlanState.REJECTED.value
                    if receipt_consistent else "invalid_receipt"),
                "recorded_state": detail.get("state"),
                "actor_identity_state": (
                    "caller_asserted"
                    if detail.get("actor_identity_state") == "caller_asserted"
                    else "unverified"),
            }
            self._conn.commit()
            return receipt
        except Exception:
            self._conn.rollback()
            raise

    def reject_project_constraint_assessment(
        self,
        project_id: str,
        comparison_id: str,
        *,
        hypothesis_id: str,
        candidate_binding_digest: str,
        constraint_kind: str,
        brief_index: int,
        expected_assessment: str,
        idempotency_key: str,
        rejected_by: str,
        reason_code: str,
    ) -> tuple[dict, bool]:
        """Append an idempotent caller rejection for one exact SHADOW suggestion.

        The receipt is content-free and never changes the comparison, Brief,
        Plan, validation state, or quality state.
        """
        if (not idempotency_key.strip() or len(idempotency_key) > 128
                or not rejected_by.strip() or len(rejected_by.strip()) > 200):
            raise ValueError("constraint rejection identity is invalid")
        if (len(candidate_binding_digest) != 64
                or any(char not in "0123456789abcdef"
                       for char in candidate_binding_digest)):
            raise ValueError("candidate binding digest is invalid")
        if constraint_kind not in {"must_include", "must_avoid"}:
            raise ValueError("constraint kind is invalid")
        if (isinstance(brief_index, bool) or not isinstance(brief_index, int)
                or brief_index < 0):
            raise ValueError("Brief index is invalid")
        if expected_assessment not in {
            "candidate_supported", "candidate_conflicted",
        }:
            raise ValueError("only a concrete model suggestion can be rejected")
        if reason_code not in {
            "source_evidence_insufficient", "misread_constraint", "other",
        }:
            raise ValueError("constraint rejection reason code is invalid")
        rejected_by = rejected_by.strip()

        idempotency_digest = hashlib.sha256(
            idempotency_key.encode("utf-8")
        ).hexdigest()
        request_identity = {
            "project_id": project_id,
            "comparison_id": comparison_id,
            "hypothesis_id": hypothesis_id,
            "candidate_binding_digest": candidate_binding_digest,
            "constraint_kind": constraint_kind,
            "brief_index": brief_index,
            "expected_assessment": expected_assessment,
            "rejected_by": rejected_by,
            "reason_code": reason_code,
            "idempotency_key_sha256": idempotency_digest,
        }
        request_fingerprint = hashlib.sha256(json.dumps(
            request_identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        ledger_id = "director_constraint_assessment_rejected_" + hashlib.sha256(
            f"{project_id}\0{comparison_id}\0{idempotency_key}".encode("utf-8")
        ).hexdigest()[:32]

        def require_exact_assessment():
            record = self.get(ProjectDirectorShadowComparisonRecord, comparison_id)
            if record is None or record.project_id != project_id:
                raise ValueError("project SHADOW comparison does not exist")
            baseline_brief_id = record.comparison.baseline_plan.brief_id
            if not baseline_brief_id:
                raise ValueError("project SHADOW comparison Brief binding is missing")
            candidates = [
                item for item in record.comparison.strategy_candidates
                if item.hypothesis.hypothesis_id == hypothesis_id
            ]
            if len(candidates) != 1:
                raise ValueError("strategy hypothesis does not exist uniquely")
            candidate = candidates[0]
            if (candidate.candidate_binding_digest != candidate_binding_digest
                    or candidate.plan.project_id != project_id
                    or candidate.plan.brief_id != baseline_brief_id):
                raise ValueError("strategy candidate binding is stale or inconsistent")
            matches = [
                item for item in candidate.hypothesis.constraint_assessments
                if item.constraint_kind == constraint_kind
                and item.brief_index == brief_index
            ]
            if len(matches) != 1:
                raise ValueError("candidate does not contain this constraint assessment")
            assessment = matches[0]
            if assessment.assessment != expected_assessment:
                raise ValueError("constraint assessment no longer matches the request")
            return assessment

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing_row = self._conn.execute(
                "SELECT project_id, data FROM decision_ledger WHERE ledger_id = ?",
                (ledger_id,),
            ).fetchone()
            if existing_row is not None:
                entry = self._load(DecisionLedgerEntry, existing_row)
                detail = entry.detail or {}
                if (
                    existing_row["project_id"] != project_id
                    or entry.project_id != project_id
                    or entry.decision_id != comparison_id
                    or entry.action != "director_constraint_assessment_rejected"
                    or detail.get("idempotency_key_sha256") != idempotency_digest
                    or detail.get("actor_identity_state") != "caller_asserted"
                ):
                    raise ValueError("idempotency key is already bound to another operation")
                if detail.get("request_fingerprint") != request_fingerprint:
                    raise ValueError("idempotency key was reused with a different request")
                require_exact_assessment()
                receipt = detail.get("assessment_rejection")
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("ledger_id") != ledger_id
                    or receipt.get("comparison_id") != comparison_id
                    or receipt.get("hypothesis_id") != hypothesis_id
                    or receipt.get("candidate_binding_digest") != candidate_binding_digest
                    or receipt.get("constraint_ref")
                    != f"{constraint_kind}:{brief_index}"
                    or receipt.get("constraint_kind") != constraint_kind
                    or receipt.get("brief_index") != brief_index
                    or receipt.get("expected_assessment") != expected_assessment
                    or receipt.get("rejected_by") != rejected_by
                    or receipt.get("reason_code") != reason_code
                    or receipt.get("recorded_at") != entry.timestamp
                    or detail.get("candidate_mutated") is not False
                    or detail.get("plan_state_unchanged") is not True
                    or detail.get("quality_acceptance") != "NOT_PROVEN"
                ):
                    raise ValueError("stored constraint rejection receipt is inconsistent")
                self._conn.commit()
                return receipt, True

            assessment = require_exact_assessment()
            timestamp = int(time.time())
            receipt = {
                "ledger_id": ledger_id,
                "comparison_id": comparison_id,
                "hypothesis_id": hypothesis_id,
                "candidate_binding_digest": candidate_binding_digest,
                "constraint_ref": f"{constraint_kind}:{brief_index}",
                "constraint_kind": constraint_kind,
                "brief_index": brief_index,
                "expected_assessment": assessment.assessment,
                "rejected_by": rejected_by,
                "reason_code": reason_code,
                "recorded_at": timestamp,
            }
            detail = {
                "assessment_rejection": receipt,
                "actor_identity_state": "caller_asserted",
                "candidate_mutated": False,
                "plan_state_unchanged": True,
                "quality_acceptance": "NOT_PROVEN",
                "idempotency_key_sha256": idempotency_digest,
                "request_fingerprint": request_fingerprint,
            }
            entry = DecisionLedgerEntry(
                ledger_id=ledger_id,
                decision_id=comparison_id,
                action="director_constraint_assessment_rejected",
                timestamp=timestamp,
                project_id=project_id,
                detail=detail,
            )
            self._conn.execute(
                "INSERT INTO decision_ledger "
                "(ledger_id, project_id, created_at, data) VALUES (?, ?, ?, ?)",
                (ledger_id, project_id, timestamp, json.dumps(
                    self._dump(entry), ensure_ascii=False,
                )),
            )
            self._conn.commit()
            return receipt, False
        except Exception:
            self._conn.rollback()
            raise

    def get_project_strategy_confirmation(
        self, project_id: str, plan_id: str
    ) -> dict | None:
        """Read the persisted strategy confirmation receipt for one Plan."""
        from director_brain.plan_state import (
            PlanState,
            compute_edl_hash,
            compute_plan_hash,
        )

        self._conn.execute("BEGIN")
        try:
            rows = self._conn.execute(
                "SELECT data FROM decision_ledger WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            entries = [self._load(DecisionLedgerEntry, row) for row in rows]
            matches = [
                entry for entry in entries
                if entry.decision_id == plan_id
                and entry.action == "strategy_confirmed"
                and entry.detail is not None
            ]
            if not matches:
                self._conn.commit()
                return None
            entry = max(matches, key=lambda item: item.timestamp)
            detail = entry.detail or {}
            bundle = self._get_project_plan_bundle_unlocked(project_id, plan_id)
            confirmation = detail.get("confirmation") or {}
            hash_binding_valid = bool(
                bundle is not None
                and isinstance(confirmation, dict)
                and confirmation.get("plan_hash") == compute_plan_hash(bundle[2])
                and confirmation.get("edl_hash") == compute_edl_hash(bundle[1])
            )
            workflow_state_consistent = bool(
                bundle is not None
                and bundle[2].state == PlanState.STRATEGY_CONFIRMED.value
                and bundle[2].approval_state == "approved"
                and bundle[1].approval_state == "approved"
                and detail.get("state") == PlanState.STRATEGY_CONFIRMED.value
                and detail.get("dispatch_eligible") is False
                and detail.get("dispatch_block_reason")
                == _PROJECT_CONFIRMATION_DISPATCH_BLOCK_REASON
                and detail.get("actor_identity_state") == "caller_asserted"
            )
            receipt_consistent = hash_binding_valid and workflow_state_consistent
            receipt = {
                "ledger_id": entry.ledger_id,
                "recorded_at": entry.timestamp,
                "confirmation": confirmation,
                "plan_edl_hash_binding_valid": hash_binding_valid,
                "receipt_consistency_state": (
                    "consistent" if receipt_consistent else "inconsistent"),
                "state": (
                    PlanState.STRATEGY_CONFIRMED.value
                    if receipt_consistent else "invalid_receipt"),
                "recorded_state": detail.get("state"),
                "dispatch_eligible": False,
                "recorded_dispatch_eligible": detail.get("dispatch_eligible"),
                "dispatch_block_reason": (
                    _PROJECT_CONFIRMATION_DISPATCH_BLOCK_REASON
                    if receipt_consistent
                    else "confirmation_receipt_inconsistent"),
                "actor_identity_state": (
                    "caller_asserted"
                    if detail.get("actor_identity_state") == "caller_asserted"
                    else "unverified"),
                "recorded_actor_identity_state": detail.get(
                    "actor_identity_state"),
            }
            self._conn.commit()
            return receipt
        except Exception:
            self._conn.rollback()
            raise
