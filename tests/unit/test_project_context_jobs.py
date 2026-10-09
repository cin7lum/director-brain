from __future__ import annotations

import pytest

from director_brain.models.project_context_job import (
    ProjectContextJob,
    ProjectContextJobProgress,
    ProjectContextJobState,
)
from storage.project_context_jobs import (
    IdempotencyConflictError,
    SqliteProjectContextJobStore,
)


def _job(job_id: str = "context_job_1", key: str = "request-1", now: int = 1):
    return ProjectContextJob(
        job_id=job_id,
        project_id="project-a",
        manifest_id="manifest-a-r1",
        manifest_revision=1,
        context_id="context-a-r1",
        analysis_profile="local_vlm_shadow_v1",
        request_fingerprint="a" * 64,
        idempotency_key=key,
        state=ProjectContextJobState.QUEUED,
        created_at=now,
        updated_at=now,
    )


def test_idempotency_replay_returns_one_project_job(tmp_path):
    store = SqliteProjectContextJobStore(tmp_path / "jobs.sqlite3")

    first, created_first = store.create(_job())
    replay, created_replay = store.create(_job(job_id="context_job_2", now=2))

    assert created_first is True
    assert created_replay is False
    assert replay.job_id == first.job_id
    assert store.list_events(first.job_id, "project-a")[0]["event_type"] == "queued"

    changed = _job(job_id="context_job_3", now=3).model_copy(
        update={"request_fingerprint": "b" * 64})
    with pytest.raises(IdempotencyConflictError):
        store.create(changed)


def test_job_lease_recovery_resumes_and_persists_terminal_result(tmp_path):
    store = SqliteProjectContextJobStore(tmp_path / "jobs.sqlite3")
    created, _ = store.create(_job())

    assert store.acquire_worker_lease("worker-1", now=10, lease_seconds=20)
    claimed = store.claim(created.job_id, "worker-1", now=11)
    assert claimed.state == ProjectContextJobState.RUNNING
    store.update_progress(
        created.job_id,
        "worker-1",
        ProjectContextJobProgress(
            phase="semantic_analysis",
            current_asset_id="asset-1",
            assets_completed=0,
            assets_total=2,
            units_completed=3,
            units_total=8,
        ),
        now=12,
    )

    assert not store.acquire_worker_lease("worker-2", now=15, lease_seconds=20)
    assert store.acquire_worker_lease("worker-2", now=31, lease_seconds=20)
    assert store.recover_inflight_jobs("worker-2", now=31) == [created.job_id]
    resumed = store.claim(created.job_id, "worker-2", now=32)
    assert resumed.attempt == 2
    assert resumed.progress.units_completed == 3
    finished = store.finish(created.job_id, "worker-2", "context-a-r1", now=33)

    assert finished.state == ProjectContextJobState.SUCCEEDED
    assert finished.result_context_id == "context-a-r1"
    assert [event["event_type"] for event in store.list_events(
        created.job_id, "project-a")] == [
            "queued", "running", "progress", "requeued_after_restart",
            "running", "succeeded",
        ]


def test_cancellation_is_cooperative_and_queued_cancel_is_terminal(tmp_path):
    store = SqliteProjectContextJobStore(tmp_path / "jobs.sqlite3")
    queued, _ = store.create(_job())
    cancelled = store.request_cancel(queued.job_id, "project-a", now=2)
    assert cancelled.state == ProjectContextJobState.CANCELLED

    running_job, _ = store.create(_job("context_job_2", "request-2", now=3))
    store.claim(running_job.job_id, "worker-1", now=4)
    requested = store.request_cancel(running_job.job_id, "project-a", now=5)
    assert requested.state == ProjectContextJobState.CANCEL_REQUESTED
    assert store.cancellation_requested(running_job.job_id, "worker-1") is True
    completed = store.fail(running_job.job_id, "worker-1", "cancelled", now=6)
    assert completed.state == ProjectContextJobState.CANCELLED
    assert completed.failure_code is None


def test_jobs_are_hidden_from_other_projects(tmp_path):
    store = SqliteProjectContextJobStore(tmp_path / "jobs.sqlite3")
    job, _ = store.create(_job())

    assert store.get(job.job_id, "project-b") is None
    assert store.list_events(job.job_id, "project-b") == []
