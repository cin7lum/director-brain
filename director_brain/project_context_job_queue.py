"""SQLite-backed local worker for durable project Context analysis jobs."""
from __future__ import annotations

import atexit
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from huey import SqliteHuey
from huey.consumer import Consumer

from director_brain.models.project_context_job import ProjectContextJobProgress
from storage.project_context_jobs import (
    ProjectContextJobLeaseLostError,
    SqliteProjectContextJobStore,
)

_WORKER_LEASE_SECONDS = 30
_WORKER_HEARTBEAT_SECONDS = 8
_WORKERS_LOCK = threading.Lock()


@dataclass
class _WorkerHandle:
    db_path: str
    owner_id: str
    consumer: object
    consumer_thread: threading.Thread
    lease_stop: threading.Event
    lease_thread: threading.Thread | None
    store: SqliteProjectContextJobStore


_WORKERS: dict[str, _WorkerHandle] = {}


class _EmbeddedConsumer(Consumer):
    """Run Huey's durable SQLite queue in an API-owned thread without signals."""

    def _set_signal_handlers(self) -> None:
        # Python permits signal handler registration only on the main thread.
        # This embedded consumer is stopped through its thread-safe stop flag.
        return None


@lru_cache(maxsize=8)
def _queue_bundle(db_path: str):
    """Reuse Huey's SQLite queue and one registered task per local database."""
    huey = SqliteHuey(
        "director_brain_context_jobs",
        filename=db_path,
        results=False,
        immediate=False,
        fsync=True,
        journal_mode="wal",
        timeout=30,
    )

    def run_context_job(job_id: str) -> None:
        _execute_job(db_path, job_id)

    run_context_job.__module__ = __name__
    run_context_job.__name__ = "run_project_context_job"
    task = huey.task(retries=0)(run_context_job)
    return huey, task


def _failure_code(exc: BaseException) -> str:
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        return {
            404: "manifest_missing",
            409: "manifest_stale_or_conflicted",
            422: "request_invalid",
            503: "analysis_profile_unavailable",
        }.get(exc.status_code, f"http_{exc.status_code}")
    return "analysis_failed"


def _execute_job(db_path: str, job_id: str) -> None:
    store = SqliteProjectContextJobStore(db_path)
    lease = _WORKERS.get(db_path)
    if lease is None:
        return
    worker_id = lease.owner_id
    job = store.claim(job_id, worker_id)
    if job is None:
        return

    def update_progress(
        phase: str,
        asset_id: str | None,
        assets_completed: int,
        assets_total: int,
        units_completed: int = 0,
        units_total: int | None = None,
    ) -> None:
        store.update_progress(
            job_id,
            worker_id,
            ProjectContextJobProgress(
                phase=phase,
                current_asset_id=asset_id,
                assets_completed=assets_completed,
                assets_total=assets_total,
                units_completed=units_completed,
                units_total=units_total,
            ),
        )

    def should_cancel() -> bool:
        return store.cancellation_requested(job_id, worker_id)

    try:
        from api.main import (
            ProjectContextSnapshotRequest,
            _build_project_film_context_snapshot,
        )

        response = _build_project_film_context_snapshot(
            job.project_id,
            ProjectContextSnapshotRequest(
                manifest_revision=job.manifest_revision),
            progress_callback=update_progress,
            cancellation_check=should_cancel,
        )
        data = response.get("data", {})
        result_context_id = data.get("context_id")
        if not isinstance(result_context_id, str):
            raise RuntimeError("context response omitted context_id")
        store.finish(job_id, worker_id, result_context_id)
    except ProjectContextJobLeaseLostError:
        # A newer service owner has recovered this job. Its worker will resume
        # from the repository-backed per-shot analysis cache.
        return
    except Exception as exc:  # noqa: BLE001
        try:
            store.fail(job_id, worker_id, _failure_code(exc))
        except ProjectContextJobLeaseLostError:
            return


def _heartbeat_worker(handle: _WorkerHandle) -> None:
    while not handle.lease_stop.wait(_WORKER_HEARTBEAT_SECONDS):
        try:
            if not handle.store.renew_worker_lease(
                handle.owner_id, lease_seconds=_WORKER_LEASE_SECONDS
            ):
                handle.lease_stop.set()
                handle.consumer.stop(graceful=False)
                return
        except Exception:  # noqa: BLE001
            # A later poll will detect the expired lease and recover the queue.
            handle.lease_stop.set()
            handle.consumer.stop(graceful=False)
            return


def _quiesce_worker(handle: _WorkerHandle) -> None:
    handle.lease_stop.set()
    try:
        handle.consumer.stop(graceful=False)
    except Exception:  # noqa: BLE001
        pass
    threads = [
        thread for _, thread in handle.consumer.worker_threads
        if thread.is_alive()
    ]
    for thread in (handle.consumer_thread, handle.consumer.scheduler):
        if thread.is_alive() and thread not in threads:
            threads.append(thread)
    if handle.lease_thread is not None and handle.lease_thread.is_alive():
        threads.append(handle.lease_thread)
    for thread in threads:
        thread.join(timeout=2)
    if not any(thread.is_alive() for thread in threads):
        handle.store.release_worker_lease(handle.owner_id)
    # If a provider call is still running, leave the lease to expire naturally.


def _stop_worker(db_path: str) -> None:
    with _WORKERS_LOCK:
        handle = _WORKERS.pop(db_path, None)
    if handle is None:
        return
    _quiesce_worker(handle)


def ensure_project_context_worker(
    db_path: str | Path, dispatch_job_id: str | None = None
) -> str | None:
    """Start a single local Huey consumer, or use the current DB lease owner."""
    resolved = str(Path(db_path).resolve())
    with _WORKERS_LOCK:
        current = _WORKERS.get(resolved)
        if current is not None and current.consumer_thread.is_alive():
            if dispatch_job_id is not None:
                _, task = _queue_bundle(resolved)
                task(dispatch_job_id)
            return current.owner_id
        if current is not None:
            _WORKERS.pop(resolved, None)
            _quiesce_worker(current)

        store = SqliteProjectContextJobStore(resolved)
        owner_id = f"context_worker_{uuid.uuid4().hex}"
        if not store.acquire_worker_lease(
            owner_id, lease_seconds=_WORKER_LEASE_SECONDS
        ):
            if dispatch_job_id is not None:
                _, task = _queue_bundle(resolved)
                task(dispatch_job_id)
            return None
        recovered = store.recover_inflight_jobs(owner_id)
        huey, task = _queue_bundle(resolved)
        queue_ids = set(recovered) | set(store.queued_job_ids())
        if dispatch_job_id is not None:
            queue_ids.add(dispatch_job_id)
        for job_id in sorted(queue_ids):
            task(job_id)
        consumer = _EmbeddedConsumer(
            huey,
            workers=1,
            worker_type="thread",
            periodic=False,
        )
        lease_stop = threading.Event()
        consumer_thread = threading.Thread(
            target=consumer.run,
            name="director-brain-context-consumer",
            daemon=True,
        )
        handle = _WorkerHandle(
            resolved, owner_id, consumer, consumer_thread,
            lease_stop, None, store,
        )
        lease_thread = threading.Thread(
            target=_heartbeat_worker,
            args=(handle,),
            name="director-brain-context-worker-lease",
            daemon=True,
        )
        handle.lease_thread = lease_thread
        _WORKERS[resolved] = handle
        consumer_thread.start()
        lease_thread.start()
        atexit.register(_stop_worker, resolved)
        return owner_id
