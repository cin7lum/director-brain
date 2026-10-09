from __future__ import annotations

import multiprocessing
import os
import sqlite3

import pytest

from storage.project_director_shadow_runs import (
    ProjectDirectorShadowRunConflictError,
    ProjectDirectorShadowRunInProgressError,
    ProjectDirectorShadowRunOutcomeUnknownError,
    SqliteProjectDirectorShadowRunStore,
)


_COMPARISON_ID = "pds_0123456789abcdef0123456789abcdef01234567"
_PROJECT_ID = "synthetic-project"
_FINGERPRINT = "a" * 64


def _hold_claim_in_child(db_path, entered, release, result_queue):
    store = SqliteProjectDirectorShadowRunStore(db_path)
    try:
        with store.acquire(
            comparison_id=_COMPARISON_ID,
            project_id=_PROJECT_ID,
            request_fingerprint=_FINGERPRINT,
            wait_timeout_seconds=5,
        ) as claim:
            claim.mark_provider_started()
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("test did not release the child claim")
            claim.complete()
        result_queue.put("completed")
    except Exception as exc:  # pragma: no cover - child failure reporting
        result_queue.put(type(exc).__name__)
        raise


def _crash_after_provider_start_in_child(db_path, entered):
    store = SqliteProjectDirectorShadowRunStore(db_path)
    claim = store.acquire(
        comparison_id=_COMPARISON_ID,
        project_id=_PROJECT_ID,
        request_fingerprint=_FINGERPRINT,
        wait_timeout_seconds=5,
    )
    claim.mark_provider_started()
    entered.set()
    os._exit(0)


def test_same_key_lease_is_exclusive_across_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    result_queue = context.Queue()
    db_path = str(tmp_path / "shadow-run-lease.sqlite3")
    store = SqliteProjectDirectorShadowRunStore(db_path)
    child = context.Process(
        target=_hold_claim_in_child,
        args=(db_path, entered, release, result_queue),
    )
    child.start()
    try:
        assert entered.wait(timeout=10)
        with pytest.raises(ProjectDirectorShadowRunInProgressError):
            store.acquire(
                comparison_id=_COMPARISON_ID,
                project_id=_PROJECT_ID,
                request_fingerprint=_FINGERPRINT,
                wait_timeout_seconds=0.1,
            )
    finally:
        release.set()
        child.join(timeout=15)
    assert child.exitcode == 0
    assert result_queue.get(timeout=2) == "completed"

    # The API rechecks its persisted success/failure receipt after acquisition.
    # This second claim models that post-wait handoff point.
    with store.acquire(
        comparison_id=_COMPARISON_ID,
        project_id=_PROJECT_ID,
        request_fingerprint=_FINGERPRINT,
    ) as claim:
        claim.complete()


def test_started_run_without_terminal_receipt_fails_closed(tmp_path):
    store = SqliteProjectDirectorShadowRunStore(
        tmp_path / "shadow-run-uncertain.sqlite3")
    with store.acquire(
        comparison_id=_COMPARISON_ID,
        project_id=_PROJECT_ID,
        request_fingerprint=_FINGERPRINT,
    ) as claim:
        claim.mark_provider_started()

    with pytest.raises(ProjectDirectorShadowRunOutcomeUnknownError):
        store.acquire(
            comparison_id=_COMPARISON_ID,
            project_id=_PROJECT_ID,
            request_fingerprint=_FINGERPRINT,
            wait_timeout_seconds=0,
        )


def test_expired_provider_started_claim_after_process_crash_is_unknown(
    tmp_path,
):
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    db_path = str(tmp_path / "shadow-run-crash.sqlite3")
    store = SqliteProjectDirectorShadowRunStore(db_path)
    child = context.Process(
        target=_crash_after_provider_start_in_child,
        args=(db_path, entered),
    )
    child.start()
    child.join(timeout=15)
    assert child.exitcode == 0
    assert entered.is_set()

    # Expire the persisted lease without waiting 90 seconds; this models the
    # worker restart boundary after the former owner has exited.
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE project_director_shadow_runs SET lease_until=0 "
            "WHERE comparison_id=?",
            (_COMPARISON_ID,),
        )

    with pytest.raises(ProjectDirectorShadowRunOutcomeUnknownError):
        store.acquire(
            comparison_id=_COMPARISON_ID,
            project_id=_PROJECT_ID,
            request_fingerprint=_FINGERPRINT,
            wait_timeout_seconds=0,
        )


def test_expired_unstarted_run_is_reclaimed_and_fingerprint_is_bound(tmp_path):
    store = SqliteProjectDirectorShadowRunStore(
        tmp_path / "shadow-run-reclaim.sqlite3")
    with store.acquire(
        comparison_id=_COMPARISON_ID,
        project_id=_PROJECT_ID,
        request_fingerprint=_FINGERPRINT,
    ):
        pass

    with store.acquire(
        comparison_id=_COMPARISON_ID,
        project_id=_PROJECT_ID,
        request_fingerprint=_FINGERPRINT,
    ) as active_claim:
        with pytest.raises(ProjectDirectorShadowRunConflictError):
            store.acquire(
                comparison_id=_COMPARISON_ID,
                project_id=_PROJECT_ID,
                request_fingerprint="b" * 64,
                wait_timeout_seconds=0,
            )
        with sqlite3.connect(store.db_path) as connection:
            connection.execute(
                "UPDATE project_director_shadow_runs SET lease_until=0 "
                "WHERE comparison_id=?",
                (_COMPARISON_ID,),
            )
        with store.acquire(
            comparison_id=_COMPARISON_ID,
            project_id=_PROJECT_ID,
            request_fingerprint=_FINGERPRINT,
            wait_timeout_seconds=0,
        ) as reclaimed:
            reclaimed.complete()


def test_persisted_run_schema_contains_no_request_content(tmp_path):
    db_path = tmp_path / "shadow-run-content-free.sqlite3"
    SqliteProjectDirectorShadowRunStore(db_path)
    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute(
                "PRAGMA table_info(project_director_shadow_runs)")
        }
    assert columns == {
        "comparison_id",
        "project_id",
        "request_fingerprint",
        "state",
        "owner_token",
        "lease_until",
        "provider_started",
        "updated_at",
    }
