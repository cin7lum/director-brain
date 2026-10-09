"""Cross-process idempotency leases for local Director Reasoner comparisons.

The lease stores only hashed request identity and lifecycle state. It never
stores a prompt, provider response, or candidate content.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_LEASE_SECONDS = 90
_HEARTBEAT_SECONDS = 15
_POLL_SECONDS = 0.05


class ProjectDirectorShadowRunError(RuntimeError):
    """Base error for local Reasoner run coordination."""


class ProjectDirectorShadowRunConflictError(ProjectDirectorShadowRunError):
    """An idempotency identity is bound to a different request."""


class ProjectDirectorShadowRunInProgressError(ProjectDirectorShadowRunError):
    """Another process still owns the same comparison run."""


class ProjectDirectorShadowRunOutcomeUnknownError(ProjectDirectorShadowRunError):
    """A previous owner may have called the provider but left no receipt."""


class ProjectDirectorShadowRunLeaseLostError(ProjectDirectorShadowRunError):
    """The current process no longer owns the comparison lease."""


class SqliteProjectDirectorShadowRunStore:
    """Coordinate one same-key SHADOW run across API worker processes."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS project_director_shadow_runs ("
                " comparison_id TEXT PRIMARY KEY,"
                " project_id TEXT NOT NULL,"
                " request_fingerprint TEXT NOT NULL,"
                " state TEXT NOT NULL CHECK(state IN ('running', 'uncertain')),"
                " owner_token TEXT,"
                " lease_until INTEGER,"
                " provider_started INTEGER NOT NULL DEFAULT 0,"
                " updated_at INTEGER NOT NULL"
                ")"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_project_director_shadow_runs_project "
                "ON project_director_shadow_runs(project_id, state, updated_at)"
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

    def acquire(
        self,
        *,
        comparison_id: str,
        project_id: str,
        request_fingerprint: str,
        wait_timeout_seconds: float = 30,
    ) -> "ProjectDirectorShadowRunClaim":
        """Acquire a durable lease or fail closed without duplicating a call."""
        if (not comparison_id or not project_id
                or len(request_fingerprint) != 64
                or any(char not in "0123456789abcdef"
                       for char in request_fingerprint)):
            raise ValueError("project Director shadow run identity is invalid")
        if wait_timeout_seconds < 0:
            raise ValueError("wait timeout must not be negative")

        owner_token = uuid.uuid4().hex
        deadline = time.monotonic() + wait_timeout_seconds
        while True:
            now = int(time.time())
            uncertain = False
            busy = False
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT project_id, request_fingerprint, state, owner_token, "
                    "lease_until, provider_started "
                    "FROM project_director_shadow_runs WHERE comparison_id=?",
                    (comparison_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO project_director_shadow_runs "
                        "(comparison_id, project_id, request_fingerprint, state, "
                        "owner_token, lease_until, provider_started, updated_at) "
                        "VALUES (?, ?, ?, 'running', ?, ?, 0, ?)",
                        (comparison_id, project_id, request_fingerprint,
                         owner_token, now + _LEASE_SECONDS, now),
                    )
                    acquired = True
                else:
                    if (row["project_id"] != project_id
                            or row["request_fingerprint"] != request_fingerprint):
                        raise ProjectDirectorShadowRunConflictError(
                            "idempotency key is bound to another comparison")
                    if row["state"] == "uncertain":
                        raise ProjectDirectorShadowRunOutcomeUnknownError(
                            "prior comparison outcome is uncertain")
                    lease_until = row["lease_until"]
                    if lease_until is not None and int(lease_until) > now:
                        busy = True
                        acquired = False
                    elif bool(row["provider_started"]):
                        connection.execute(
                            "UPDATE project_director_shadow_runs SET state='uncertain', "
                            "owner_token=NULL, lease_until=NULL, updated_at=? "
                            "WHERE comparison_id=? AND state='running'",
                            (now, comparison_id),
                        )
                        uncertain = True
                        acquired = False
                    else:
                        connection.execute(
                            "UPDATE project_director_shadow_runs SET project_id=?, "
                            "request_fingerprint=?, state='running', owner_token=?, "
                            "lease_until=?, provider_started=0, updated_at=? "
                            "WHERE comparison_id=?",
                            (project_id, request_fingerprint, owner_token,
                             now + _LEASE_SECONDS, now, comparison_id),
                        )
                        acquired = True

            if acquired:
                return ProjectDirectorShadowRunClaim(
                    self, comparison_id, owner_token)
            if uncertain:
                raise ProjectDirectorShadowRunOutcomeUnknownError(
                    "prior comparison outcome is uncertain")
            if busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProjectDirectorShadowRunInProgressError(
                        "comparison is still in progress")
                time.sleep(min(_POLL_SECONDS, remaining))

    def _renew(self, comparison_id: str, owner_token: str) -> bool:
        now = int(time.time())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE project_director_shadow_runs SET lease_until=?, updated_at=? "
                "WHERE comparison_id=? AND state='running' AND owner_token=?",
                (now + _LEASE_SECONDS, now, comparison_id, owner_token),
            )
            return cursor.rowcount == 1

    def _mark_provider_started(self, comparison_id: str, owner_token: str) -> None:
        now = int(time.time())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE project_director_shadow_runs SET provider_started=1, "
                "lease_until=?, updated_at=? "
                "WHERE comparison_id=? AND state='running' AND owner_token=? "
                "AND lease_until>?",
                (now + _LEASE_SECONDS, now, comparison_id, owner_token, now),
            )
            if cursor.rowcount != 1:
                raise ProjectDirectorShadowRunLeaseLostError(
                    "project Director shadow run lease was lost")

    def _assert_owner(self, comparison_id: str, owner_token: str) -> None:
        now = int(time.time())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM project_director_shadow_runs "
                "WHERE comparison_id=? AND state='running' AND owner_token=? "
                "AND lease_until>?",
                (comparison_id, owner_token, now),
            ).fetchone()
        if row is None:
            raise ProjectDirectorShadowRunLeaseLostError(
                "project Director shadow run lease was lost")

    def _complete(self, comparison_id: str, owner_token: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "DELETE FROM project_director_shadow_runs "
                "WHERE comparison_id=? AND state='running' AND owner_token=?",
                (comparison_id, owner_token),
            )
            if cursor.rowcount != 1:
                raise ProjectDirectorShadowRunLeaseLostError(
                    "project Director shadow run lease was lost")

    def _abandon(
        self, comparison_id: str, owner_token: str, provider_started: bool,
    ) -> None:
        now = int(time.time())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if provider_started:
                connection.execute(
                    "UPDATE project_director_shadow_runs SET state='uncertain', "
                    "owner_token=NULL, lease_until=NULL, updated_at=? "
                    "WHERE comparison_id=? AND state='running' AND owner_token=?",
                    (now, comparison_id, owner_token),
                )
            else:
                connection.execute(
                    "DELETE FROM project_director_shadow_runs "
                    "WHERE comparison_id=? AND state='running' AND owner_token=?",
                    (comparison_id, owner_token),
                )

    def get_state(
        self, comparison_id: str, project_id: str,
    ) -> str | None:
        now = int(time.time())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state, lease_until, provider_started "
                "FROM project_director_shadow_runs "
                "WHERE comparison_id=? AND project_id=?",
                (comparison_id, project_id),
            ).fetchone()
        if row is None:
            return None
        if row["state"] == "uncertain":
            return "uncertain"
        if row["lease_until"] is not None and int(row["lease_until"]) > now:
            return "running"
        if bool(row["provider_started"]):
            return "uncertain"
        return "recoverable"


class ProjectDirectorShadowRunClaim:
    """Heartbeat-backed owner token for one comparison attempt."""

    def __init__(
        self,
        store: SqliteProjectDirectorShadowRunStore,
        comparison_id: str,
        owner_token: str,
    ) -> None:
        self._store = store
        self.comparison_id = comparison_id
        self._owner_token = owner_token
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._provider_started = False
        self._completed = False
        self._thread = threading.Thread(
            target=self._heartbeat,
            name="director-shadow-run-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def __enter__(self) -> "ProjectDirectorShadowRunClaim":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        if not self._completed:
            self._store._abandon(
                self.comparison_id,
                self._owner_token,
                self._provider_started,
            )
        return False

    def _heartbeat(self) -> None:
        while not self._stop.wait(_HEARTBEAT_SECONDS):
            try:
                if not self._store._renew(self.comparison_id, self._owner_token):
                    self._lost.set()
                    return
            except sqlite3.Error:
                self._lost.set()
                return

    def mark_provider_started(self) -> None:
        if self._lost.is_set():
            raise ProjectDirectorShadowRunLeaseLostError(
                "project Director shadow run lease was lost")
        self._store._mark_provider_started(
            self.comparison_id, self._owner_token)
        self._provider_started = True

    def assert_owner(self) -> None:
        if self._lost.is_set():
            raise ProjectDirectorShadowRunLeaseLostError(
                "project Director shadow run lease was lost")
        self._store._assert_owner(self.comparison_id, self._owner_token)

    def complete(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        self._store._complete(self.comparison_id, self._owner_token)
        self._completed = True
