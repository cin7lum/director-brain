from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


_CHILD = r'''
import json
import os
import sys
import time
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app
from director_brain.models import ClaimKind, FilmObservation, TimebaseUnit
from director_brain.utils import file_sha256
from observation_service import pipeline
import director_brain.project_context_job_queue as queue

queue._WORKER_LEASE_SECONDS = 4
queue._WORKER_HEARTBEAT_SECONDS = 1
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
events_path = Path(config["events_path"])
started_path = Path(config["started_path"])

def log(event):
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(event + "\n")

def analyze(video_path, cache=None, *, progress_callback=None,
            cancellation_check=None):
    if cancellation_check is not None and cancellation_check():
        from observation_service.analysis_control import AnalysisCancelledError
        raise AnalysisCancelledError("test cancellation")
    source_hash = file_sha256(video_path)
    fingerprint = "restart-probe-v1:" + source_hash
    cached = cache.get(fingerprint) if cache is not None else None
    if cached:
        observations = cached
        log("cache_hit:" + Path(video_path).name)
    else:
        observations = [FilmObservation(
            observation_id="shot_" + Path(video_path).stem,
            media_asset_id="shot_" + Path(video_path).stem,
            media_hash=source_hash,
            start_frame=0,
            end_frame=1_000_000,
            timebase=1_000_000,
            timebase_unit=TimebaseUnit.MICROSECONDS,
            observation_type="deterministic_technical",
            claim=json.dumps({"blur_score": 1.0, "exposure_ok": True}),
            provider="deterministic_opencv",
            model_version="restart-test-only",
            prompt_version="n/a",
            confidence=1.0,
            review_state="final",
            claim_kind=ClaimKind.MEASURED,
            project_id="unknown",
            created_at=1_700_000_000,
            producer="restart_fixture",
            source_ref=video_path,
        )]
        if cache is not None:
            cache.put(fingerprint, observations)
        log("computed:" + Path(video_path).name)
        if (os.environ.get("HOLD_PROVIDER") == "1"
                and Path(video_path).name == "asset-a.mp4"):
            started_path.write_text("provider-running", encoding="utf-8")
            while True:
                time.sleep(0.05)
    if progress_callback is not None:
        progress_callback("deterministic_analysis", 1, 1)
    return observations

pipeline.analyze_media = analyze
headers = {"Authorization": "Bearer " + config["token"]}
with TestClient(app, client=("127.0.0.1", 54183), headers=headers) as client:
    if mode == "submit":
        manifest = client.put(
            "/v1/projects/restart-project/film-manifest",
            json={
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": "test-fixture://restart-project",
                "boundary_evidence_refs": ["test-fixture://restart-project/manifest"],
                "assets": [
                    {
                        "asset_id": asset_id,
                        "source_ref": path,
                        "order": order,
                        "rights": {
                            "state": "local_processing_allowed",
                            "basis": "owner_permission",
                            "evidence_ref": "test-fixture://rights/" + asset_id,
                        },
                    }
                    for order, (asset_id, path) in enumerate(config["assets"])
                ],
            },
        )
        assert manifest.status_code == 200, manifest.text
        response = client.post(
            "/v1/projects/restart-project/film-context:jobs",
            headers={"Idempotency-Key": "restart-recovery-001"},
            json={"manifest_revision": 1},
        )
        assert response.status_code == 202, response.text
        Path(config["job_path"]).write_text(
            json.dumps(response.json()["data"]["job"]), encoding="utf-8")
        while True:
            time.sleep(0.1)
    else:
        job = json.loads(Path(config["job_path"]).read_text(encoding="utf-8"))
        job_path = (
            "/v1/projects/restart-project/film-context-jobs/"
            + job["job_id"]
        )
        client.get(job_path)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            response = client.get(job_path)
            assert response.status_code == 200, response.text
            current = response.json()["data"]["job"]
            if current["state"] in {"succeeded", "failed", "cancelled"}:
                Path(config["result_path"]).write_text(
                    json.dumps(current), encoding="utf-8")
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("job did not reach a terminal state")
        queue._stop_worker(str(Path(config["sqlite_path"]).resolve()))
'''


def _wait_for_file(path: Path, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            raise AssertionError(f"child exited early: {process.returncode}")
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path.name}")


def test_running_context_job_recovers_after_process_crash(tmp_path):
    media_a = tmp_path / "asset-a.mp4"
    media_b = tmp_path / "asset-b.mp4"
    for path, color in ((media_a, "red"), (media_b, "blue")):
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"color=c={color}:s=320x240:r=25:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ], capture_output=True, text=True, check=True)

    db_path = tmp_path / "restart.sqlite3"
    config_path = tmp_path / "config.json"
    started_path = tmp_path / "provider-started"
    job_path = tmp_path / "job.json"
    result_path = tmp_path / "result.json"
    events_path = tmp_path / "analysis-events.log"
    recovery_log_path = tmp_path / "recovery.stderr.log"
    config_path.write_text(json.dumps({
        "sqlite_path": str(db_path),
        "media_root": str(tmp_path),
        "token": "restart-test-token",
        "assets": [["asset-a", str(media_a)], ["asset-b", str(media_b)]],
        "job_path": str(job_path),
        "result_path": str(result_path),
        "started_path": str(started_path),
        "events_path": str(events_path),
    }), encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "SQLITE_PATH": str(db_path),
        "DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT": str(tmp_path),
        "DIRECTOR_BRAIN_PROJECT_API_TOKEN": "restart-test-token",
    })
    project_root = Path(__file__).resolve().parents[2]
    first = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(config_path), "submit"],
        cwd=project_root,
        env={**env, "HOLD_PROVIDER": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    second = None
    try:
        _wait_for_file(job_path, first, timeout=20)
        _wait_for_file(started_path, first, timeout=20)
        first.terminate()
        first.wait(timeout=10)
        if first.stderr is not None:
            first.stderr.close()

        job = json.loads(job_path.read_text(encoding="utf-8"))
        from storage.project_context_jobs import SqliteProjectContextJobStore

        interrupted_job = SqliteProjectContextJobStore(db_path).get(
            job["job_id"], "restart-project")
        assert interrupted_job is not None
        assert interrupted_job.state.value == "running"
        recovery_log = recovery_log_path.open("w", encoding="utf-8")
        second = subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(config_path), "recover"],
            cwd=project_root,
            env={**env, "HOLD_PROVIDER": "0"},
            stdout=subprocess.DEVNULL,
            stderr=recovery_log,
        )
        try:
            second.wait(timeout=25)
        except subprocess.TimeoutExpired:
            second.terminate()
            second.wait(timeout=10)
            recovery_log.close()
            raise AssertionError(
                "recovery process did not exit: "
                + recovery_log_path.read_text(encoding="utf-8"))
        recovery_log.close()
        if second.returncode != 0:
            stderr = recovery_log_path.read_text(encoding="utf-8")
            raise AssertionError(f"recovery process failed: {stderr}")
        assert result_path.is_file()

        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["state"] == "succeeded", result
        assert result["attempt"] == 2
        events = events_path.read_text(encoding="utf-8").splitlines()
        assert events.count("computed:asset-a.mp4") == 1
        assert events.count("cache_hit:asset-a.mp4") == 1
        assert events.count("computed:asset-b.mp4") == 1, events

        from director_brain.models.film_context import FilmContextSnapshot
        from storage.sqlite_repository import SqliteRepository

        repository = SqliteRepository(str(db_path))
        try:
            snapshot = repository.get(
                FilmContextSnapshot, result["result_context_id"])
        finally:
            repository.close()
        assert snapshot is not None
        assert snapshot.asset_refs == ["asset-a", "asset-b"]
        assert [item.analysis_state.value for item in snapshot.asset_coverage] == [
            "observed", "observed"]
    finally:
        for child in (first, second):
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
