from __future__ import annotations

import json
import subprocess
import time

from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from director_brain.models import ClaimKind, FilmObservation, TimebaseUnit
from director_brain.utils import file_sha256


def _make_media(path, color: str) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        f"color=c={color}:s=320x240:r=25:d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], capture_output=True, text=True, check=True)


def _data(response):
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_durable_context_job_runs_and_is_project_scoped(tmp_path, monkeypatch):
    from director_brain import project_context_job_queue as queue
    from director_brain.project_context_job_queue import _stop_worker
    from observation_service import pipeline

    db_path = tmp_path / "context-jobs.sqlite3"
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "context-job-test-token")

    media_a = tmp_path / "asset-a.mp4"
    media_b = tmp_path / "asset-b.mp4"
    _make_media(media_a, "red")
    _make_media(media_b, "blue")

    calls: list[str] = []

    def analyze(video_path, cache=None, *, progress_callback=None,
                cancellation_check=None):
        if cancellation_check is not None and cancellation_check():
            from observation_service.analysis_control import AnalysisCancelledError

            raise AnalysisCancelledError("test cancellation")
        calls.append(video_path)
        content_hash = file_sha256(video_path)
        if progress_callback is not None:
            progress_callback("deterministic_analysis", 1, 1)
        return [FilmObservation(
            observation_id=f"shot_{len(calls)}",
            media_asset_id=f"shot_{len(calls)}",
            media_hash=content_hash,
            start_frame=0,
            end_frame=1_000_000,
            timebase=1_000_000,
            timebase_unit=TimebaseUnit.MICROSECONDS,
            observation_type="deterministic_technical",
            claim=json.dumps({"blur_score": 1.0, "exposure_ok": True}),
            provider="deterministic_opencv",
            model_version="test-only",
            prompt_version="n/a",
            confidence=1.0,
            review_state="final",
            claim_kind=ClaimKind.MEASURED,
            project_id="unknown",
            created_at=1_700_000_000,
            producer="test_fixture",
            source_ref=video_path,
        )]

    monkeypatch.setattr(pipeline, "analyze_media", analyze)
    client = TestClient(
        app,
        client=("127.0.0.1", 54180),
        headers={"Authorization": "Bearer context-job-test-token"},
    )
    try:
        manifest = _data(client.put(
            "/v1/projects/job-project/film-manifest",
            json={
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": "test-fixture://job-project",
                "boundary_evidence_refs": ["test-fixture://job-project/manifest"],
                "assets": [
                    {
                        "asset_id": "asset-a",
                        "source_ref": str(media_a),
                        "order": 0,
                        "rights": {
                            "state": "local_processing_allowed",
                            "basis": "owner_permission",
                            "evidence_ref": "test-fixture://rights/asset-a",
                        },
                    },
                    {
                        "asset_id": "asset-b",
                        "source_ref": str(media_b),
                        "order": 1,
                        "rights": {
                            "state": "local_processing_allowed",
                            "basis": "owner_permission",
                            "evidence_ref": "test-fixture://rights/asset-b",
                        },
                    },
                ],
            },
        ))
        assert manifest["current_revision"] == 1

        submitted = client.post(
            "/v1/projects/job-project/film-context:jobs",
            headers={"Idempotency-Key": "multi-asset-job-001"},
            json={"manifest_revision": 1},
        )
        assert submitted.status_code == 202, submitted.text
        job = submitted.json()["data"]["job"]
        assert job["state"] in {"queued", "running", "succeeded"}
        assert "request_fingerprint" not in job
        assert "worker_id" not in job

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = _data(client.get(
                f"/v1/projects/job-project/film-context-jobs/{job['job_id']}"
            ))["job"]
            if current["state"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert current["state"] == "succeeded", current
        assert calls == [str(media_a), str(media_b)]
        assert current["result_context_id"] == current["context_id"]

        replay = client.post(
            "/v1/projects/job-project/film-context:jobs",
            headers={"Idempotency-Key": "multi-asset-job-001"},
            json={"manifest_revision": 1},
        )
        assert replay.status_code == 200
        assert replay.json()["data"]["replayed"] is True
        assert replay.json()["data"]["job"]["job_id"] == job["job_id"]

        events = _data(client.get(
            f"/v1/projects/job-project/film-context-jobs/{job['job_id']}/events"
        ))["events"]
        event_types = [item["event_type"] for item in events]
        assert event_types[0] == "queued"
        assert "running" in event_types
        assert "progress" in event_types
        assert event_types[-1] == "succeeded"

        hidden = client.get(
            f"/v1/projects/another-project/film-context-jobs/{job['job_id']}"
        )
        assert hidden.status_code == 404

        from director_brain.models.film_context import FilmContextSnapshot
        from storage.sqlite_repository import SqliteRepository

        repository = SqliteRepository(str(db_path))
        try:
            snapshot = repository.get(FilmContextSnapshot, current["context_id"])
        finally:
            repository.close()
        assert snapshot is not None
        assert snapshot.asset_refs == ["asset-a", "asset-b"]
        assert snapshot.project_manifest_id == manifest["manifest"]["manifest_id"]

        graph = _data(client.post(
            "/v1/projects/job-project/story-graph:build"
        ))["story_graph"]
        assert graph["context_id"] == current["context_id"]
        assert graph["cross_asset_relations_state"] == "not_attempted"
        assert [item["asset_id"] for item in graph["assets"]] == [
            "asset-a", "asset-b"]
        plan = _data(client.post(
            "/v1/projects/job-project/director-plans:generate",
            json={
                "manifest_revision": 1,
                "target_duration_us": 1_000_000,
                "intent_text": "Create a concise two-asset sequence.",
            },
        ))
        assert plan["persisted"] is True
        assert plan["plan"]["project_manifest_id"] == manifest[
            "manifest"]["manifest_id"]

        _stop_worker(str(db_path.resolve()))
        monkeypatch.setattr(
            queue, "ensure_project_context_worker", lambda *_args: None)
        queued = client.post(
            "/v1/projects/job-project/film-context:jobs",
            headers={"Idempotency-Key": "multi-asset-job-cancel"},
            json={"manifest_revision": 1},
        )
        assert queued.status_code == 202
        queued_job_id = queued.json()["data"]["job"]["job_id"]
        cancelled = _data(client.post(
            f"/v1/projects/job-project/film-context-jobs/"
            f"{queued_job_id}:cancel"
        ))["job"]
        assert cancelled["state"] == "cancelled"
    finally:
        _stop_worker(str(db_path.resolve()))
