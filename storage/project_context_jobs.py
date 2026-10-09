"""SQLite lifecycle store for durable project Context analysis jobs."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from director_brain.models.project_context_job import (
    ProjectContextJob,
    ProjectContextJobProgress,
    ProjectContextJobState,
)


class IdempotencyConflictError(ValueError):
    """A key was reused for a different immutable job request."""


class ProjectContextJobConflictError(ValueError):
    """A job cannot make the requested state transition."""


class ProjectContextJobLeaseLostError(RuntimeError):
    """The worker no longer owns the job lease."""


class SqliteProjectContextJobStore:
    """Atomic job state transitions and append-only lifecycle events."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS project_context_jobs ("
                " job_id TEXT PRIMARY KEY,"
                " project_id TEXT NOT NULL,"
                " idempotency_key TEXT NOT NULL,"
                " request_fingerprint TEXT NOT NULL,"
                " state TEXT NOT NULL,"
                " worker_id TEXT,"
                " updated_at INTEGER NOT NULL,"
                " heartbeat_at INTEGER,"
                " data TEXT NOT NULL,"
                " UNIQUE(project_id, idempotency_key)"
                ");"
                "CREATE INDEX IF NOT EXISTS idx_project_context_jobs_project_state "
                " ON project_context_jobs(project_id, state, updated_at DESC);"
                "CREATE TABLE IF NOT EXISTS project_context_job_events ("
                " event_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " job_id TEXT NOT NULL REFERENCES project_context_jobs(job_id),"
                " created_at INTEGER NOT NULL,"
                " event_type TEXT NOT NULL,"
                " data TEXT NOT NULL"
                ");"
                "CREATE INDEX IF NOT EXISTS idx_project_context_job_events_job "
                " ON project_context_job_events(job_id, event_id);"
                "CREATE TABLE IF NOT EXISTS project_context_job_worker_lease ("
                " singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),"
                " owner_id TEXT NOT NULL,"
                " lease_until INTEGER NOT NULL,"
                " updated_at INTEGER NOT NULL"
                ");"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA synchronous = EXTRA")
            if connection.execute("PRAGMA synchronous").fetchone()[0] != 3:
                raise RuntimeError(
                    "SQLite EXTRA synchronous durability is unavailable")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _load(row: sqlite3.Row | None) -> ProjectContextJob | None:
        if row is None:
            return None
        return ProjectContextJob.model_validate_json(row["data"])

    @staticmethod
    def _write(connection: sqlite3.Connection, job: ProjectContextJob) -> None:
        connection.execute(
            "UPDATE project_context_jobs SET state=?, worker_id=?, updated_at=?, "
            "heartbeat_at=?, data=? WHERE job_id=?",
            (
                job.state.value,
                job.worker_id,
                job.updated_at,
                job.heartbeat_at,
                job.model_dump_json(),
                job.job_id,
            ),
        )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        job_id: str,
        event_type: str,
        data: dict[str, Any],
        created_at: int,
    ) -> None:
        connection.execute(
            "INSERT INTO project_context_job_events "
            "(job_id, created_at, event_type, data) VALUES (?, ?, ?, ?)",
            (job_id, created_at, event_type, json.dumps(data, sort_keys=True)),
        )

    def create(self, job: ProjectContextJob) -> tuple[ProjectContextJob, bool]:
        """Insert once by project/idempotency key; return existing on exact replay."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM project_context_jobs "
                "WHERE project_id=? AND idempotency_key=?",
                (job.project_id, job.idempotency_key),
            ).fetchone()
            if row is not None:
                existing = self._load(row)
                assert existing is not None
                if existing.request_fingerprint != job.request_fingerprint:
                    raise IdempotencyConflictError(
                        "idempotency key is already bound to different inputs")
                return existing, False
            connection.execute(
                "INSERT INTO project_context_jobs "
                "(job_id, project_id, idempotency_key, request_fingerprint, state, "
                "worker_id, updated_at, heartbeat_at, data) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
                (
                    job.job_id,
                    job.project_id,
                    job.idempotency_key,
                    job.request_fingerprint,
                    job.state.value,
                    job.updated_at,
                    job.model_dump_json(),
                ),
            )
            self._event(
                connection, job.job_id, "queued",
                {"manifest_revision": job.manifest_revision}, job.created_at,
            )
            return job, True

    def get(
        self, job_id: str, project_id: str | None = None
    ) -> ProjectContextJob | None:
        with self._connect() as connection:
            if project_id is None:
                row = connection.execute(
                    "SELECT * FROM project_context_jobs WHERE job_id=?",
                    (job_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM project_context_jobs "
                    "WHERE job_id=? AND project_id=?",
                    (job_id, project_id),
                ).fetchone()
        return self._load(row)

    def list_events(self, job_id: str, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT e.event_id, e.created_at, e.event_type, e.data "
                "FROM project_context_job_events e "
                "JOIN project_context_jobs j USING(job_id) "
                "WHERE e.job_id=? AND j.project_id=? ORDER BY event_id",
                (job_id, project_id),
            ).fetchall()
        return [
            {
                "event_id": int(row["event_id"]),
                "created_at": int(row["created_at"]),
                "event_type": row["event_type"],
                "data": json.loads(row["data"]),
            }
            for row in rows
        ]

    def queued_job_ids(self) -> list[str]:
        """Return durable queued jobs in creation order for worker recovery."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id FROM project_context_jobs WHERE state=? "
                "ORDER BY updated_at, job_id",
                (ProjectContextJobState.QUEUED.value,),
            ).fetchall()
        return [str(row["job_id"]) for row in rows]

    def claim(self, job_id: str, worker_id: str, now: int | None = None):
        now = int(time.time()) if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM project_context_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            current = self._load(row)
            if current is None or current.state != ProjectContextJobState.QUEUED:
                return None
            updated = current.model_copy(update={
                "state": ProjectContextJobState.RUNNING,
                "attempt": current.attempt + 1,
                "worker_id": worker_id,
                "updated_at": now,
                "started_at": current.started_at or now,
                "heartbeat_at": now,
                "progress": (
                    current.progress.model_copy(update={
                        "phase": "validating_manifest"
                    })
                    if current.attempt == 0 else current.progress
                ),
                "failure_code": None,
            })
            self._write(connection, updated)
            self._event(connection, job_id, "running", {"attempt": updated.attempt}, now)
            return updated

    def update_progress(
        self,
        job_id: str,
        worker_id: str,
        progress: ProjectContextJobProgress,
        now: int | None = None,
    ) -> ProjectContextJob:
        now = int(time.time()) if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM project_context_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            current = self._load(row)
            if current is None or current.worker_id != worker_id:
                raise ProjectContextJobLeaseLostError(job_id)
            if current.state not in (
                ProjectContextJobState.RUNNING,
                ProjectContextJobState.CANCEL_REQUESTED,
            ):
                raise ProjectContextJobLeaseLostError(job_id)
            updated = current.model_copy(update={
                "progress": progress,
                "updated_at": now,
                "heartbeat_at": now,
            })
            self._write(connection, updated)
            self._event(connection, job_id, "progress", progress.model_dump(), now)
            return updated

    def request_cancel(self, job_id: str, project_id: str, now: int | None = None):
        now = int(time.time()) if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM project_context_jobs WHERE job_id=? AND project_id=?",
                (job_id, project_id),
            ).fetchone()
            current = self._load(row)
            if current is None:
                return None
            if current.state == ProjectContextJobState.QUEUED:
                updated = current.model_copy(update={
                    "state": ProjectContextJobState.CANCELLED,
                    "updated_at": now,
                    "finished_at": now,
                    "cancel_requested_at": now,
                    "failure_code": None,
                })
                event_type = "cancelled_before_start"
            elif current.state == ProjectContextJobState.RUNNING:
                updated = current.model_copy(update={
                    "state": ProjectContextJobState.CANCEL_REQUESTED,
                    "updated_at": now,
                    "cancel_requested_at": now,
                })
                event_type = "cancellation_requested"
            else:
                return current
            self._write(connection, updated)
            self._event(connection, job_id, event_type, {}, now)
            return updated

    def cancellation_requested(self, job_id: str, worker_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM project_context_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        current = self._load(row)
        if current is None or current.worker_id != worker_id:
            raise ProjectContextJobLeaseLostError(job_id)
        return current.state == ProjectContextJobState.CANCEL_REQUESTED

    def finish(
        self, job_id: str, worker_id: str, context_id: str, now: int | None = None
    ) -> ProjectContextJob:
        now = int(time.time()) if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM project_context_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            current = self._load(row)
            self._assert_owner(current, worker_id, job_id)
            updated = current.model_copy(update={
                "state": ProjectContextJobState.SUCCEEDED,
                "worker_id": None,
                "updated_at": now,
                "heartbeat_at": now,
                "finished_at": now,
                "result_context_id": context_id,
                "failure_code": None,
                "progress": current.progress.model_copy(update={
                    "phase": "context_persisted"
                }),
            })
            self._write(connection, updated)
            self._event(
                connection,
                job_id,
                (
                    "succeeded_after_cancel_request"
                    if current.state == ProjectContextJobState.CANCEL_REQUESTED
                    else "succeeded"
                ),
                {"context_id": context_id},
                now,
            )
            return updated

    def fail(
        self, job_id: str, worker_id: str, failure_code: str,
        now: int | None = None,
    ) -> ProjectContextJob:
        now = int(time.time()) if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM project_context_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            current = self._load(row)
            self._assert_owner(current, worker_id, job_id)
            cancelled = current.state == ProjectContextJobState.CANCEL_REQUESTED
            updated = current.model_copy(update={
                "state": (
                    ProjectContextJobState.CANCELLED if cancelled
                    else ProjectContextJobState.FAILED
                ),
                "worker_id": None,
                "updated_at": now,
                "heartbeat_at": now,
                "finished_at": now,
                "failure_code": None if cancelled else failure_code,
            })
            self._write(connection, updated)
            self._event(
                connection,
                job_id,
                "cancelled" if cancelled else "failed",
                {} if cancelled else {"failure_code": failure_code},
                now,
            )
            return updated

    @staticmethod
    def _assert_owner(
        current: ProjectContextJob | None, worker_id: str, job_id: str
    ) -> None:
        if (
            current is None
            or current.worker_id != worker_id
            or current.state not in (
                ProjectContextJobState.RUNNING,
                ProjectContextJobState.CANCEL_REQUESTED,
            )
        ):
            raise ProjectContextJobLeaseLostError(job_id)

    def acquire_worker_lease(
        self,
        owner_id: str,
        now: int | None = None,
        lease_seconds: int = 30,
    ) -> bool:
        now = int(time.time()) if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_id, lease_until FROM project_context_job_worker_lease "
                "WHERE singleton_id=1"
            ).fetchone()
            if row is not None and row["owner_id"] != owner_id and row["lease_until"] > now:
                return False
            connection.execute(
                "INSERT INTO project_context_job_worker_lease "
                "(singleton_id, owner_id, lease_until, updated_at) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(singleton_id) DO UPDATE SET owner_id=excluded.owner_id, "
                "lease_until=excluded.lease_until, updated_at=excluded.updated_at",
                (owner_id, now + lease_seconds, now),
            )
            return True

    def renew_worker_lease(
        self, owner_id: str, now: int | None = None, lease_seconds: int = 30
    ) -> bool:
        now = int(time.time()) if now is None else now
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE project_context_job_worker_lease SET lease_until=?, updated_at=? "
                "WHERE singleton_id=1 AND owner_id=?",
                (now + lease_seconds, now, owner_id),
            )
            return cursor.rowcount == 1

    def release_worker_lease(self, owner_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM project_context_job_worker_lease "
                "WHERE singleton_id=1 AND owner_id=?",
                (owner_id,),
            )

    def recover_inflight_jobs(
        self, new_worker_id: str, now: int | None = None
    ) -> list[str]:
        """Requeue jobs left by a prior dead worker; cached shots make this safe."""
        now = int(time.time()) if now is None else now
        requeued: list[str] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM project_context_jobs "
                "WHERE state IN (?, ?) AND (worker_id IS NULL OR worker_id<>?)",
                (
                    ProjectContextJobState.RUNNING.value,
                    ProjectContextJobState.CANCEL_REQUESTED.value,
                    new_worker_id,
                ),
            ).fetchall()
            for row in rows:
                current = self._load(row)
                assert current is not None
                cancelled = current.state == ProjectContextJobState.CANCEL_REQUESTED
                updated = current.model_copy(update={
                    "state": (
                        ProjectContextJobState.CANCELLED if cancelled
                        else ProjectContextJobState.QUEUED
                    ),
                    "worker_id": None,
                    "heartbeat_at": None,
                    "updated_at": now,
                    "finished_at": now if cancelled else None,
                    "cancel_requested_at": (
                        current.cancel_requested_at if cancelled else None
                    ),
                    "failure_code": None,
                })
                self._write(connection, updated)
                event_type = "cancelled_after_restart" if cancelled else "requeued_after_restart"
                self._event(connection, current.job_id, event_type, {}, now)
                if not cancelled:
                    requeued.append(current.job_id)
        return requeued
