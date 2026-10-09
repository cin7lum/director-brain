"""Live HTTP persistence check for the project Plan confirmation lifecycle.

This is local software/runtime evidence only. It uses generated synthetic clips
and does not establish director quality or project-level generalization.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _free_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cached_observation_row_types(connection, source_hash: str | None = None):
    if source_hash is None:
        rows = connection.execute(
            "SELECT data FROM analysis_result_cache"
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT data FROM analysis_result_cache WHERE source_content_hash = ?",
            (source_hash,),
        ).fetchall()
    return [
        tuple(item["observation_type"] for item in json.loads(row[0]))
        for row in rows
    ]


def _http_call(
    base_url: str,
    token: str,
    method: str,
    path: str,
    payload: dict | None = None,
) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        base_url + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _start_api(port: int, env: dict[str, str], repo_root: Path):
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "127.0.0.1", "--port", str(port),
            "--log-level", "critical", "--no-access-log",
        ],
        cwd=repo_root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"Uvicorn exited during startup: {process.returncode}")
        try:
            with urlopen(base_url + "/openapi.json", timeout=1) as response:
                if response.status == 200:
                    return process, base_url
        except (OSError, URLError):
            time.sleep(0.1)
    process.terminate()
    process.wait(timeout=10)
    raise AssertionError("Uvicorn did not become ready within 20 seconds")


def _stop_api(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _data(response: tuple[int, dict]) -> dict:
    status, body = response
    assert status == 200, body
    assert "data" in body
    return body["data"]


def _make_video(path: Path, lavfi_source: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi", "-i", lavfi_source,
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path), "-y",
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def _make_offset_multiaudio_video(path: Path) -> None:
    """Create a synthetic clip with two independently clocked audio tracks."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "testsrc=duration=2:size=320x240:rate=25",
            "-itsoffset", "0.25", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-itsoffset", "0.75", "-f", "lavfi", "-i",
            "sine=frequency=660:sample_rate=44100:duration=1",
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-metadata:s:a:0", "language=eng",
            "-metadata:s:a:1", "language=spa", "-disposition:a:0", "default",
            "-disposition:a:1", "0", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def test_project_plan_receipt_survives_live_api_process_restart(tmp_path):
    """Read and replay an acknowledged project receipt after Uvicorn restarts."""
    media_a = tmp_path / "asset_a.mkv"
    media_b = tmp_path / "asset_b.mp4"
    _make_offset_multiaudio_video(media_a)
    _make_video(media_b, "color=c=blue:s=320x240:r=30000/1001:d=2")

    project_id = "project-process-restart"
    token = "p1-process-restart-test-token"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(tmp_path / "director_brain.sqlite3")
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    port = _free_loopback_port()
    project_path = f"/v1/projects/{project_id}"

    process, base_url = _start_api(port, env, Path(__file__).resolve().parents[2])
    try:
        manifest = _data(_http_call(
            base_url, token, "PUT", project_path + "/film-manifest", {
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": "test-fixture://project/process-restart",
                "boundary_evidence_refs": ["test-fixture://project/manifest"],
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
        source_time_map = manifest["manifest"]["assets"][0]["time_map"]
        assert source_time_map["mapping_state"] == "complete"
        assert [
            (item["source_start_offset_numerator"],
             item["source_start_offset_denominator"])
            for item in source_time_map["audio_streams"]
        ] == [(1, 4), (3, 4)]

        context = _data(_http_call(
            base_url, token, "POST", project_path + "/film-context:snapshot",
            {"manifest_revision": 1},
        ))
        assert context["coverage"] == "project_multi_asset_observed"
        assert len(context["snapshot"]["asset_coverage"]) == 2
        assert context["snapshot"]["asset_coverage"][0]["time_map"] == (
            source_time_map)

        graph = _data(_http_call(
            base_url, token, "POST", project_path + "/story-graph:build"))
        assert graph["story_graph"]["cross_asset_relations_state"] == "not_attempted"
        graph_view = graph["story_graph_view"]
        assert graph_view["source_graph"]["graph_id"] == graph["story_graph"]["graph_id"]
        assert graph_view["link_snapshot_state"] == "absent"
        assert graph_view["cross_asset_relation_state"] == "none"
        assert graph_view["cross_asset_link_count"] == 0
        assert graph_view["automatic_inference_state"] == "not_attempted"

        plan = _data(_http_call(
            base_url, token, "POST", project_path + "/director-plans:generate", {
                "manifest_revision": 1,
                "target_duration_us": 1_600_000,
                "intent_text": "制作一段温暖的家庭旅行记录",
            },
        ))
        assert plan["validation"]["valid"] is True
        assert plan["plan"]["state"] == "ready_for_strategy_confirmation"
        assert plan["evidence_scope"]["quality_acceptance"] == "not_proven"

        confirmation_request = {
            "expected_plan_hash": plan["plan_hash"],
            "expected_edl_hash": plan["edl_hash"],
            "idempotency_key": "process-restart-confirmation-001",
            "confirmed_by": "synthetic-test-caller",
            "output_target": "delivery",
            "notes": "live HTTP restart persistence check",
        }
        confirmed = _data(_http_call(
            base_url, token, "POST",
            f"{project_path}/director-plans/{plan['plan']['plan_id']}:confirm-strategy",
            confirmation_request,
        ))
        assert confirmed["state"] == "strategy_confirmed"
        assert confirmed["dispatch_eligible"] is False

        needs_input = _data(_http_call(
            base_url, token, "POST", project_path + "/director-plans:generate", {
                "manifest_revision": 1,
                "target_duration_us": 1_600_000,
                "intent_text": "制作一段家庭旅行记录，必须包含日出镜头",
            },
        ))
        assert needs_input["validation"]["requires_input"] is True
        assert needs_input["plan"]["state"] == "needs_input"
        rejection_request = {
            "expected_plan_hash": needs_input["plan_hash"],
            "expected_edl_hash": needs_input["edl_hash"],
            "idempotency_key": "process-restart-rejection-001",
            "rejected_by": "synthetic-test-caller",
            "reason": "Synthetic lifecycle test of unresolved semantic input",
        }
        rejection = _data(_http_call(
            base_url, token, "POST",
            f"{project_path}/director-plans/{needs_input['plan']['plan_id']}:reject-strategy",
            rejection_request,
        ))
        assert rejection["state"] == "rejected"
        assert rejection["rejection_receipt"]["receipt_consistency_state"] == (
            "consistent")

        clarification_source = _data(_http_call(
            base_url, token, "POST", project_path + "/director-plans:generate", {
                "manifest_revision": 1,
                "target_duration_us": 1_600_000,
                "intent_text": "制作一段旅行记录，必须包含篝火镜头",
            },
        ))
        assert clarification_source["plan"]["state"] == "needs_input"
        supersession_request = {
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段宁静的旅行记录，若有篝火镜头可优先考虑",
            "supersedes_plan_id": clarification_source["plan"]["plan_id"],
            "expected_superseded_plan_hash": clarification_source["plan_hash"],
            "expected_superseded_edl_hash": clarification_source["edl_hash"],
            "clarified_by": "synthetic-test-caller",
        }
        superseding_plan = _data(_http_call(
            base_url, token, "POST",
            project_path + "/director-plans:generate", supersession_request,
        ))
        assert superseding_plan["plan"]["state"] == "ready_for_strategy_confirmation"
        assert superseding_plan["plan"]["supersedes_plan_id"] == (
            clarification_source["plan"]["plan_id"])
    finally:
        _stop_api(process)

    # Start a new Uvicorn process against the same on-disk SQLite database.
    restarted, restarted_url = _start_api(port, env, Path(__file__).resolve().parents[2])
    try:
        plan_id = plan["plan"]["plan_id"]
        readback = _data(_http_call(
            restarted_url, token, "GET",
            f"{project_path}/director-plans/{plan_id}",
        ))
        assert readback["plan"]["state"] == "strategy_confirmed"
        assert readback["confirmation_receipt"]["plan_edl_hash_binding_valid"] is True

        rejected_plan_id = needs_input["plan"]["plan_id"]
        rejected_readback = _data(_http_call(
            restarted_url, token, "GET",
            f"{project_path}/director-plans/{rejected_plan_id}",
        ))
        assert rejected_readback["plan"]["state"] == "rejected"
        assert rejected_readback["rejection_receipt"][
            "receipt_consistency_state"] == "consistent"
        assert rejected_readback["rejection_receipt"]["reason"] == (
            rejection_request["reason"])

        superseded_id = clarification_source["plan"]["plan_id"]
        superseded_readback = _data(_http_call(
            restarted_url, token, "GET",
            f"{project_path}/director-plans/{superseded_id}",
        ))
        assert superseded_readback["plan"]["state"] == "superseded"
        successor_readback = _data(_http_call(
            restarted_url, token, "GET",
            f"{project_path}/director-plans/{superseding_plan['plan']['plan_id']}",
        ))
        assert successor_readback["plan"]["supersedes_plan_id"] == superseded_id

        graph_readback = _data(_http_call(
            restarted_url, token, "GET", project_path + "/story-graph"))
        assert graph_readback["story_graph_view"] == graph_view

        manifest_readback = _data(_http_call(
            restarted_url, token, "GET", project_path + "/film-manifest?revision=1"))
        assert manifest_readback["manifest"]["assets"][0]["time_map"] == (
            source_time_map)
        context_readback = _data(_http_call(
            restarted_url, token, "GET",
            f"/v1/film-context/{context['context_id']}"))
        assert context_readback["snapshot"]["asset_coverage"][0]["time_map"] == (
            source_time_map)
        project_context_readback = _data(_http_call(
            restarted_url, token, "GET",
            f"{project_path}/film-context/{context['context_id']}"))
        assert project_context_readback["project_id"] == project_id
        assert project_context_readback["snapshot"] == (
            context_readback["snapshot"])
        assert project_context_readback["source_hash_validation_state"] == (
            "not_revalidated_by_read")

        replay = _data(_http_call(
            restarted_url, token, "POST",
            f"{project_path}/director-plans/{plan_id}:confirm-strategy",
            confirmation_request,
        ))
        assert replay["reused"] is True
        assert replay["confirmation"] == confirmed["confirmation"]

        rejection_replay = _data(_http_call(
            restarted_url, token, "POST",
            f"{project_path}/director-plans/{rejected_plan_id}:reject-strategy",
            rejection_request,
        ))
        assert rejection_replay["reused"] is True
        assert rejection_replay["rejection_receipt"] == (
            rejection["rejection_receipt"])

        supersession_replay = _data(_http_call(
            restarted_url, token, "POST",
            project_path + "/director-plans:generate", supersession_request,
        ))
        assert supersession_replay["reused"] is True
        assert supersession_replay["plan"] == superseding_plan["plan"]

        ledger = _data(_http_call(
            restarted_url, token, "GET", project_path + "/decision-ledger"))
        entries = [
            item for item in ledger["entries"]
            if item["decision_id"] == plan_id
            and item["action"] == "strategy_confirmed"
        ]
        assert len(entries) == 1
        assert entries[0]["project_id"] == project_id
        rejection_entries = [
            item for item in ledger["entries"]
            if item["decision_id"] == rejected_plan_id
            and item["action"] == "strategy_rejected"
        ]
        assert len(rejection_entries) == 1
        supersession_entries = [
            item for item in ledger["entries"]
            if item["decision_id"] == superseded_id
            and item["action"] == "plan_superseded"
        ]
        assert len(supersession_entries) == 1
    finally:
        _stop_api(restarted)


def test_project_shadow_failure_receipt_survives_live_api_process_restart(
    tmp_path,
):
    """Retain a fail-closed SHADOW receipt across a real API restart."""
    media_a = tmp_path / "shadow-asset-a.mp4"
    media_b = tmp_path / "shadow-asset-b.mp4"
    _make_video(media_a, "testsrc=duration=1:size=320x240:rate=25")
    _make_video(media_b, "color=c=blue:s=320x240:r=25:d=1")

    project_id = "project-shadow-failure-restart"
    project_path = f"/v1/projects/{project_id}"
    token = "p2-shadow-failure-restart-token"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(tmp_path / "shadow_failure.sqlite3")
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    env["PROJECT_LOCAL_VLM_MODEL"] = ""
    env["PROJECT_LOCAL_VLM_DIGEST"] = ""
    env["PROJECT_LOCAL_VLM_RUNTIME_VERSION"] = ""
    env["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{_free_loopback_port()}"
    env["SHOT_DISCOVERY_BACKEND"] = "scenedetect"
    port = _free_loopback_port()
    repo_root = Path(__file__).resolve().parents[2]

    process, base_url = _start_api(port, env, repo_root)
    try:
        manifest = _data(_http_call(
            base_url, token, "PUT", project_path + "/film-manifest", {
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": "test-fixture://project/shadow-failure",
                "boundary_evidence_refs": [
                    "test-fixture://project/shadow-failure/manifest",
                ],
                "assets": [
                    {
                        "asset_id": "asset-a",
                        "source_ref": str(media_a),
                        "order": 0,
                        "rights": {
                            "state": "local_processing_allowed",
                            "basis": "owner_permission",
                            "evidence_ref": "test-fixture://rights/shadow-a",
                        },
                    },
                    {
                        "asset_id": "asset-b",
                        "source_ref": str(media_b),
                        "order": 1,
                        "rights": {
                            "state": "local_processing_allowed",
                            "basis": "owner_permission",
                            "evidence_ref": "test-fixture://rights/shadow-b",
                        },
                    },
                ],
            },
        ))
        assert manifest["current_revision"] == 1
        context = _data(_http_call(
            base_url, token, "POST", project_path + "/film-context:snapshot",
            {"manifest_revision": 1},
        ))
        graph = _data(_http_call(
            base_url, token, "POST", project_path + "/story-graph:build"))

        intent_marker = "synthetic intent omitted from failure receipt"
        failed_request = {
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": intent_marker,
            "idempotency_key": "shadow-evidence-refusal-001",
        }
        failure_status, failure_body = _http_call(
            base_url, token, "POST",
            project_path + "/director-plans:compare-shadow-strategies",
            failed_request,
        )
        assert failure_status == 422, failure_body

        ledger = _data(_http_call(
            base_url, token, "GET", project_path + "/decision-ledger"))
        failures = [
            item for item in ledger["entries"]
            if item["action"] == "director_strategy_comparison_failed"
        ]
        assert len(failures) == 1
        failure_entry = failures[0]
        detail = failure_entry["detail"]
        assert set(detail) == {
            "stage", "failure_code", "request_fingerprint",
            "idempotency_key_sha256", "manifest_id", "manifest_revision",
            "context_id", "story_graph_id",
        }
        assert detail["stage"] == "reasoner_generation"
        assert detail["failure_code"] == "evidence_too_poor"
        assert detail["manifest_id"] == manifest["manifest"]["manifest_id"]
        assert detail["context_id"] == context["context_id"]
        assert detail["story_graph_id"] == graph["story_graph"]["graph_id"]
        assert len(detail["request_fingerprint"]) == 64
        assert len(detail["idempotency_key_sha256"]) == 64
        assert intent_marker not in json.dumps(failure_entry)
        comparisons = _data(_http_call(
            base_url, token, "GET",
            project_path + "/director-plan-shadow-comparisons"))
        assert comparisons["count"] == 0
    finally:
        _stop_api(process)

    restarted, restarted_url = _start_api(port, env, repo_root)
    try:
        restarted_ledger = _data(_http_call(
            restarted_url, token, "GET", project_path + "/decision-ledger"))
        restarted_failures = [
            item for item in restarted_ledger["entries"]
            if item["action"] == "director_strategy_comparison_failed"
        ]
        assert restarted_failures == [failure_entry]
        restarted_comparisons = _data(_http_call(
            restarted_url, token, "GET",
            project_path + "/director-plan-shadow-comparisons"))
        assert restarted_comparisons["count"] == 0

        replay_status, replay_body = _http_call(
            restarted_url, token, "POST",
            project_path + "/director-plans:compare-shadow-strategies",
            failed_request,
        )
        assert replay_status == 422, replay_body
        assert replay_body["detail"] == (
            "prior comparison attempt failed; use a new idempotency key to retry"
        )
        replayed_ledger = _data(_http_call(
            restarted_url, token, "GET", project_path + "/decision-ledger"))
        replayed_failures = [
            item for item in replayed_ledger["entries"]
            if item["action"] == "director_strategy_comparison_failed"
        ]
        assert replayed_failures == [failure_entry]
        assert _data(_http_call(
            restarted_url, token, "GET",
            project_path + "/director-plan-shadow-comparisons"))["count"] == 0
    finally:
        _stop_api(restarted)


def test_caller_reviewed_cross_asset_link_plan_survives_live_api_restart(
    tmp_path, monkeypatch,
):
    """Persist a caller-asserted relation through Plan confirmation and restart."""
    import importlib.util
    from urllib.parse import urlencode

    support_path = Path(__file__).with_name(
        "test_project_story_link_comparison_api.py")
    spec = importlib.util.spec_from_file_location(
        "_director_brain_story_link_seed_support", support_path)
    assert spec is not None and spec.loader is not None
    support = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = support
    spec.loader.exec_module(support)

    db_path = tmp_path / "linked-project-restart.sqlite3"
    project_id, graph_id = support._seed_project(
        db_path,
        tmp_path,
        monkeypatch,
        people_by_asset={
            "asset-a": ["same subject, red coat"],
            "asset-b": ["same subject, red coat"],
        },
    )
    token = "comparison-test-token"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(db_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = env["NO_PROXY"]
    port = _free_loopback_port()
    project_path = f"/v1/projects/{project_id}"
    process, base_url = _start_api(
        port, env, Path(__file__).resolve().parents[2])
    try:
        query = urlencode({
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "relation_kind": "person_identity",
            "limit": 1,
            "offset": 0,
        })
        candidate_page = _data(_http_call(
            base_url, token, "GET",
            f"{project_path}/story-link-candidates?{query}",
        ))["candidate_page"]
        assert candidate_page["total"] == 1
        candidate = candidate_page["candidates"][0]
        anchors = [
            {
                key: side[key]
                for key in (
                    "observation_id", "story_graph_node_id",
                    "source_start", "source_end",
                )
            }
            for side in (candidate["left_anchor"], candidate["right_anchor"])
        ]
        link_body = {
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "expected_review_revision": 0,
            "idempotency_key": "live-restart-caller-link-v1",
            "links": [{
                "link_id": "caller-reviewed-person",
                "entity_kind": "person_identity",
                "display_label": "caller-reviewed person",
                "anchors": anchors,
            }],
            "relations": [{
                "relation_id": "caller-reviewed-cross-asset-relation",
                "relation_type": "appears_with",
                "from_anchor": anchors[0],
                "to_anchor": anchors[1],
                "privacy_class": "private",
            }],
        }
        review = _data(_http_call(
            base_url, token, "PUT", project_path + "/story-links", link_body))
        assert review["review_revision"] == 1
        assert review["actor_identity_state"] == "caller_asserted"
        assert review["automatic_inference_state"] == "not_attempted"
        assert review["quality_acceptance"] == "not_proven"
        assert review["active_relation_count"] == 1

        graph_view = _data(_http_call(
            base_url, token, "GET", project_path + "/story-graph"
        ))["story_graph_view"]
        assert graph_view["cross_asset_relation_state"] == "caller_asserted"
        assert graph_view["cross_asset_link_count"] == 1
        assert graph_view["cross_asset_relation_edge_state"] == "caller_asserted"
        assert graph_view["cross_asset_relation_edge_count"] == 1
        assert graph_view["cross_asset_relation_edges"][0]["relation_id"] == (
            "caller-reviewed-cross-asset-relation")
        assert graph_view["cross_asset_link_review"] == review["review"]

        plan = _data(_http_call(
            base_url, token, "POST", project_path + "/director-plans:generate", {
                "manifest_revision": 1,
                "target_duration_us": 1_600_000,
                "intent_text": "Preserve the caller-reviewed identity evidence.",
            },
        ))
        assert plan["validation"]["valid"] is True
        assert plan["plan"]["state"] == "ready_for_strategy_confirmation"
        assert plan["plan"]["project_story_link_review_id"] == (
            review["review"]["review_id"])
        assert plan["plan"]["project_story_link_review_revision"] == 1
        assert "caller_asserted_project_links_not_independently_verified" in (
            plan["plan"]["open_questions"])
        assert "caller_asserted_project_relations_not_independently_verified" in (
            plan["plan"]["open_questions"])
        for asset_id in ("asset-a", "asset-b"):
            asset_edits = [
                item for item in plan["edl"]["ordered_edits"]
                if item["project_asset_id"] == asset_id
            ]
            assert asset_edits
            assert any(
                "caller-reviewed-person" in item["project_story_link_refs"]
                for item in asset_edits
            )
            assert any(
                "caller-reviewed-cross-asset-relation"
                in item["project_story_relation_refs"]
                for item in asset_edits
            )

        confirmation_request = {
            "expected_plan_hash": plan["plan_hash"],
            "expected_edl_hash": plan["edl_hash"],
            "idempotency_key": "live-restart-linked-confirmation-v1",
            "confirmed_by": "synthetic-test-caller",
            "output_target": "delivery",
            "notes": "linked-plan persistence regression",
        }
        confirmed = _data(_http_call(
            base_url, token, "POST",
            f"{project_path}/director-plans/{plan['plan']['plan_id']}:confirm-strategy",
            confirmation_request,
        ))
        assert confirmed["state"] == "strategy_confirmed"
        assert confirmed["dispatch_eligible"] is False
    finally:
        _stop_api(process)

    restarted, restarted_url = _start_api(
        port, env, Path(__file__).resolve().parents[2])
    try:
        links_readback = _data(_http_call(
            restarted_url, token, "GET", project_path + "/story-links"))
        assert links_readback["review"] == review["review"]

        graph_readback = _data(_http_call(
            restarted_url, token, "GET", project_path + "/story-graph"
        ))["story_graph_view"]
        assert graph_readback == graph_view

        plan_readback = _data(_http_call(
            restarted_url, token, "GET",
            f"{project_path}/director-plans/{plan['plan']['plan_id']}",
        ))
        assert plan_readback["plan"]["state"] == "strategy_confirmed"
        assert plan_readback["plan"]["project_story_link_review_id"] == (
            review["review"]["review_id"])
        assert plan_readback["plan"]["project_story_link_review_revision"] == 1
        assert plan_readback["edl"] == confirmed["edl"]
        assert plan_readback["confirmation_receipt"][
            "plan_edl_hash_binding_valid"] is True
    finally:
        _stop_api(restarted)


def test_changed_source_preserves_current_context_after_live_api_restart(tmp_path):
    """A fresh API process must reject changed media without replacing Context."""
    import sqlite3

    media_a = tmp_path / "asset_a.mp4"
    media_b = tmp_path / "asset_b.mp4"
    _make_video(media_a, "testsrc=duration=2:size=320x240:rate=25")
    _make_video(media_b, "color=c=blue:s=320x240:r=30000/1001:d=2")

    project_id = "project-context-source-restart"
    token = "p1-context-source-restart-token"
    db_path = tmp_path / "director_brain.sqlite3"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(db_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    port = _free_loopback_port()
    project_path = f"/v1/projects/{project_id}"

    process, base_url = _start_api(port, env, Path(__file__).resolve().parents[2])
    try:
        _data(_http_call(
            base_url, token, "PUT", project_path + "/film-manifest", {
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": "test-fixture://project/context-source-restart",
                "boundary_evidence_refs": ["test-fixture://project/manifest"],
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
        context = _data(_http_call(
            base_url, token, "POST", project_path + "/film-context:snapshot",
            {"manifest_revision": 1},
        ))
        assert context["coverage"] == "project_multi_asset_observed"
        assert len(context["snapshot"]["asset_coverage"]) == 2
        with sqlite3.connect(db_path) as connection:
            original_row = connection.execute(
                "SELECT data FROM film_context_snapshots WHERE context_id = ?",
                (context["context_id"],),
            ).fetchone()
        assert original_row is not None
    finally:
        _stop_api(process)

    # The manifest remains pinned to the original hash across the process stop.
    media_b.write_bytes(media_b.read_bytes() + b"changed-after-api-restart")
    restarted, restarted_url = _start_api(
        port, env, Path(__file__).resolve().parents[2])
    try:
        changed_status, changed_body = _http_call(
            restarted_url, token, "POST",
            project_path + "/film-context:snapshot", {"manifest_revision": 1},
        )
        assert changed_status == 409
        assert "asset-b" in changed_body["detail"]
        assert "现有上下文已保留" in changed_body["detail"]

        readback = _data(_http_call(
            restarted_url, token, "GET",
            f"/v1/film-context/{context['context_id']}",
        ))
        assert readback["snapshot"] == context["snapshot"]
        with sqlite3.connect(db_path) as connection:
            preserved_row = connection.execute(
                "SELECT data FROM film_context_snapshots WHERE context_id = ?",
                (context["context_id"],),
            ).fetchone()
            snapshot_count = connection.execute(
                "SELECT COUNT(*) FROM film_context_snapshots WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
        assert preserved_row == original_row
        assert snapshot_count == 1
    finally:
        _stop_api(restarted)


def test_missing_source_preserves_current_context_after_live_api_restart(tmp_path):
    """A missing registered source must not replace the last valid Context."""
    import sqlite3

    media_a = tmp_path / "asset_a.mp4"
    media_b = tmp_path / "asset_b.mp4"
    _make_video(media_a, "testsrc=duration=2:size=320x240:rate=25")
    _make_video(media_b, "color=c=blue:s=320x240:r=30000/1001:d=2")

    project_id = "project-context-missing-source-restart"
    token = "p1-context-missing-source-token"
    db_path = tmp_path / "director_brain.sqlite3"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(db_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    port = _free_loopback_port()
    project_path = f"/v1/projects/{project_id}"

    process, base_url = _start_api(
        port, env, Path(__file__).resolve().parents[2],
    )
    try:
        manifest_status, manifest_body = _http_call(
            base_url, token, "PUT", project_path + "/film-manifest", {
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": (
                    "test-fixture://project/context-missing-source-restart"),
                "boundary_evidence_refs": [
                    "test-fixture://project/manifest",
                ],
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
        )
        assert manifest_status == 200, manifest_body
        assert manifest_body["data"]["manifest"]["revision"] == 1

        context_status, context_body = _http_call(
            base_url, token, "POST",
            project_path + "/film-context:snapshot",
            {"manifest_revision": 1},
        )
        assert context_status == 200, context_body
        context = context_body["data"]
        assert context["coverage"] == "project_multi_asset_observed"
        assert [item["analysis_state"] for item in
                context["snapshot"]["asset_coverage"]] == [
                    "observed", "observed",
                ]
        with sqlite3.connect(db_path) as connection:
            original_context_row = connection.execute(
                "SELECT data FROM film_context_snapshots WHERE context_id = ?",
                (context["context_id"],),
            ).fetchone()
            original_observation_count = connection.execute(
                "SELECT COUNT(*) FROM film_observations WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
        assert original_context_row is not None
        assert original_observation_count > 0
    finally:
        _stop_api(process)

    media_b.unlink()
    restarted, restarted_url = _start_api(
        port, env, Path(__file__).resolve().parents[2],
    )
    try:
        missing_status, missing_body = _http_call(
            restarted_url, token, "POST",
            project_path + "/film-context:snapshot",
            {"manifest_revision": 1},
        )
        assert missing_status == 409
        assert "asset-b" in missing_body["detail"]
        assert "不可用" in missing_body["detail"]
        assert "现有上下文已保留" in missing_body["detail"]

        readback_status, readback_body = _http_call(
            restarted_url, token, "GET",
            f"/v1/film-context/{context['context_id']}",
        )
        assert readback_status == 200, readback_body
        assert readback_body["data"]["snapshot"] == context["snapshot"]

        with sqlite3.connect(db_path) as connection:
            preserved_context_row = connection.execute(
                "SELECT data FROM film_context_snapshots WHERE context_id = ?",
                (context["context_id"],),
            ).fetchone()
            context_count = connection.execute(
                "SELECT COUNT(*) FROM film_context_snapshots "
                "WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            preserved_observation_count = connection.execute(
                "SELECT COUNT(*) FROM film_observations WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
        assert preserved_context_row == original_context_row
        assert context_count == 1
        assert preserved_observation_count == original_observation_count
    finally:
        _stop_api(restarted)


def test_inflight_local_vlm_crash_leaves_no_partial_cache_and_retry_recovers(
    tmp_path,
):
    """A killed VLM request leaves only complete deterministic shot evidence."""
    import sqlite3
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    model_digest = "a" * 64
    provider_state = {
        "calls": 0,
        "lock": threading.Lock(),
        "first_chat_started": threading.Event(),
        "release_first_chat": threading.Event(),
        "first_chat_finished": threading.Event(),
    }
    reply = (
        "A stable indoor view.\n"
        '{"shot_function":"ACTION","sensory_wet_heat":0.2,'
        '"sensory_mood_intensity":0.4,"motion_amount":"subtle",'
        '"proposed_role_v2":"hero"}'
    )

    class OllamaStubHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/version":
                self._send_json(200, {"version": "0.32.14"})
            elif self.path == "/api/tags":
                self._send_json(200, {"models": [{
                    "name": "qwen3-vl:4b", "digest": model_digest,
                }]})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/chat":
                self._send_json(404, {"error": "not found"})
                return
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with provider_state["lock"]:
                provider_state["calls"] += 1
                call_number = provider_state["calls"]
            if call_number == 1:
                provider_state["first_chat_started"].set()
                provider_state["release_first_chat"].wait(timeout=60)
                provider_state["first_chat_finished"].set()
            try:
                self._send_json(200, {"message": {"content": reply}})
            except (BrokenPipeError, ConnectionResetError):
                pass

    provider_port = _free_loopback_port()
    provider_server = ThreadingHTTPServer(
        ("127.0.0.1", provider_port), OllamaStubHandler,
    )
    provider_thread = threading.Thread(
        target=provider_server.serve_forever, daemon=True,
    )
    provider_thread.start()

    media = tmp_path / "inflight-vlm.mp4"
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi", "-i",
            "color=c=blue:s=320x240:r=25:d=2", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-y", str(media),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    project_id = "project-inflight-vlm-recovery"
    project_path = f"/v1/projects/{project_id}"
    token = "p1-inflight-vlm-recovery-token"
    database_path = tmp_path / "inflight-vlm.sqlite3"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(database_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    env["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{provider_port}"
    env["PROJECT_LOCAL_VLM_MODEL"] = "qwen3-vl:4b"
    env["PROJECT_LOCAL_VLM_DIGEST"] = model_digest
    env["PROJECT_LOCAL_VLM_RUNTIME_VERSION"] = "0.32.14"
    env["SHOT_DISCOVERY_BACKEND"] = "scenedetect"
    api_port = _free_loopback_port()
    api_process = None
    requester = None
    first_request: dict[str, object] = {}
    try:
        api_process, api_url = _start_api(
            api_port, env, Path(__file__).resolve().parents[2],
        )
        _data(_http_call(api_url, token, "PUT", project_path + "/film-manifest", {
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/inflight-vlm",
            "boundary_evidence_refs": ["test-fixture://manifest/inflight-vlm"],
            "analysis_profile": "local_vlm_shadow_v1",
            "assets": [{
                "asset_id": "asset-vlm",
                "source_ref": str(media),
                "order": 0,
                "rights": {
                    "state": "local_processing_allowed",
                    "basis": "owner_permission",
                    "evidence_ref": "test-fixture://rights/inflight-vlm",
                },
            }],
        }))

        def issue_first_snapshot() -> None:
            try:
                first_request["response"] = _http_call(
                    api_url, token, "POST",
                    project_path + "/film-context:snapshot",
                    {"manifest_revision": 1},
                )
            except Exception as exc:  # request is expected to lose its server
                first_request["error"] = type(exc).__name__

        requester = threading.Thread(target=issue_first_snapshot, daemon=True)
        requester.start()
        assert provider_state["first_chat_started"].wait(timeout=30), (
            "the API did not reach the local provider request"
        )

        # The provider call is now deliberately held open. Terminate the API
        # process before the provider returns, then let the abandoned local
        # request unwind after its client process is gone.
        _stop_api(api_process)
        api_process = None
        provider_state["release_first_chat"].set()
        assert provider_state["first_chat_finished"].wait(timeout=5)
        requester.join(timeout=10)
        assert not requester.is_alive()
        assert "error" in first_request
        assert provider_state["calls"] == 1

        with sqlite3.connect(database_path) as connection:
            assert _cached_observation_row_types(connection) == [
                ("deterministic_technical",),
            ]

        restarted, restarted_url = _start_api(
            api_port, env, Path(__file__).resolve().parents[2],
        )
        try:
            recovered = _data(_http_call(
                restarted_url, token, "POST",
                project_path + "/film-context:snapshot",
                {"manifest_revision": 1},
            ))
            assert recovered["cache_hit"] is False
            assert recovered["coverage"] == "project_single_asset_observed"
            assert recovered["snapshot"]["asset_coverage"][0][
                "analysis_cache_state"] == "computed"
            calls_after_recovery = provider_state["calls"]
            assert calls_after_recovery == 2

            replay = _data(_http_call(
                restarted_url, token, "POST",
                project_path + "/film-context:snapshot",
                {"manifest_revision": 1},
            ))
            assert replay["cache_hit"] is True
            assert provider_state["calls"] == calls_after_recovery
            with sqlite3.connect(database_path) as connection:
                cached_rows = _cached_observation_row_types(connection)
            assert sorted(cached_rows) == sorted([
                ("deterministic_technical",),
                ("vlm_semantic",),
                ("deterministic_technical", "vlm_semantic"),
            ])
        finally:
            _stop_api(restarted)
    finally:
        if api_process is not None:
            _stop_api(api_process)
        provider_state["release_first_chat"].set()
        provider_server.shutdown()
        provider_server.server_close()
        provider_thread.join(timeout=5)


def test_multi_asset_inflight_vlm_crash_reuses_completed_asset_after_restart(
    tmp_path,
):
    """A second-asset crash reuses the first asset's completed shot cache."""
    import hashlib
    import sqlite3
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    model_digest = "b" * 64
    provider_state = {
        "calls": 0,
        "lock": threading.Lock(),
        "second_chat_started": threading.Event(),
        "release_second_chat": threading.Event(),
        "second_chat_finished": threading.Event(),
    }
    reply = (
        "A stable indoor view.\n"
        '{"shot_function":"ACTION","sensory_wet_heat":0.2,'
        '"sensory_mood_intensity":0.4,"motion_amount":"subtle",'
        '"proposed_role_v2":"hero"}'
    )

    class OllamaStubHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/version":
                self._send_json(200, {"version": "0.32.14"})
            elif self.path == "/api/tags":
                self._send_json(200, {"models": [{
                    "name": "qwen3-vl:4b", "digest": model_digest,
                }]})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/chat":
                self._send_json(404, {"error": "not found"})
                return
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with provider_state["lock"]:
                provider_state["calls"] += 1
                call_number = provider_state["calls"]
            if call_number == 2:
                provider_state["second_chat_started"].set()
                provider_state["release_second_chat"].wait(timeout=60)
                provider_state["second_chat_finished"].set()
            try:
                self._send_json(200, {"message": {"content": reply}})
            except (BrokenPipeError, ConnectionResetError):
                pass

    provider_port = _free_loopback_port()
    provider_server = ThreadingHTTPServer(
        ("127.0.0.1", provider_port), OllamaStubHandler,
    )
    provider_thread = threading.Thread(
        target=provider_server.serve_forever, daemon=True,
    )
    provider_thread.start()

    media_a = tmp_path / "asset-a-green.mp4"
    media_b = tmp_path / "asset-b-blue.mp4"
    _make_video(media_a, "color=c=green:s=320x240:r=25:d=2")
    _make_video(media_b, "color=c=blue:s=320x240:r=25:d=2")
    hash_a = hashlib.sha256(media_a.read_bytes()).hexdigest()
    hash_b = hashlib.sha256(media_b.read_bytes()).hexdigest()

    project_id = "project-multi-asset-crash-recovery"
    project_path = f"/v1/projects/{project_id}"
    token = "p1-multi-asset-crash-recovery-token"
    database_path = tmp_path / "multi-asset-crash.sqlite3"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(database_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    env["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{provider_port}"
    env["PROJECT_LOCAL_VLM_MODEL"] = "qwen3-vl:4b"
    env["PROJECT_LOCAL_VLM_DIGEST"] = model_digest
    env["PROJECT_LOCAL_VLM_RUNTIME_VERSION"] = "0.32.14"
    env["SHOT_DISCOVERY_BACKEND"] = "scenedetect"
    api_port = _free_loopback_port()
    api_process = None
    requester = None
    first_request: dict[str, object] = {}
    try:
        api_process, api_url = _start_api(
            api_port, env, Path(__file__).resolve().parents[2],
        )
        _data(_http_call(api_url, token, "PUT", project_path + "/film-manifest", {
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/multi-asset-crash",
            "boundary_evidence_refs": ["test-fixture://manifest/multi-asset-crash"],
            "analysis_profile": "local_vlm_shadow_v1",
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
        }))

        def issue_snapshot() -> None:
            try:
                first_request["response"] = _http_call(
                    api_url, token, "POST",
                    project_path + "/film-context:snapshot",
                    {"manifest_revision": 1},
                )
            except Exception as exc:  # request is expected to lose its server
                first_request["error"] = type(exc).__name__

        requester = threading.Thread(target=issue_snapshot, daemon=True)
        requester.start()
        assert provider_state["second_chat_started"].wait(timeout=30), (
            "the API did not start the second asset's local provider request"
        )

        # The first static asset has completed and its per-shot cache is durable;
        # the second asset's provider request is held open and not yet cached.
        deadline = time.monotonic() + 10
        cached_by_source: dict[str, int] = {}
        cached_types_a: list[tuple[str, ...]] = []
        while time.monotonic() < deadline:
            with sqlite3.connect(database_path) as connection:
                cached_by_source = dict(connection.execute(
                    "SELECT source_content_hash, COUNT(*) "
                    "FROM analysis_result_cache GROUP BY source_content_hash"
                ).fetchall())
                cached_types_a = _cached_observation_row_types(connection, hash_a)
            if cached_by_source.get(hash_a, 0) > 0:
                break
            time.sleep(0.05)
        assert cached_by_source.get(hash_a, 0) > 0
        assert sorted(cached_types_a) == sorted([
            ("deterministic_technical",),
            ("vlm_semantic",),
        ])
        with sqlite3.connect(database_path) as connection:
            assert _cached_observation_row_types(connection, hash_b) == [
                ("deterministic_technical",),
            ]

        _stop_api(api_process)
        api_process = None
        provider_state["release_second_chat"].set()
        assert provider_state["second_chat_finished"].wait(timeout=5)
        requester.join(timeout=10)
        assert not requester.is_alive()
        assert "error" in first_request
        assert provider_state["calls"] == 2

        with sqlite3.connect(database_path) as connection:
            cached_after_crash_a = _cached_observation_row_types(connection, hash_a)
            cached_after_crash_b = _cached_observation_row_types(connection, hash_b)
        assert sorted(cached_after_crash_a) == sorted([
            ("deterministic_technical",),
            ("vlm_semantic",),
        ])
        assert cached_after_crash_b == [("deterministic_technical",)]

        restarted, restarted_url = _start_api(
            api_port, env, Path(__file__).resolve().parents[2],
        )
        try:
            recovered = _data(_http_call(
                restarted_url, token, "POST",
                project_path + "/film-context:snapshot",
                {"manifest_revision": 1},
            ))
            assert recovered["cache_hit"] is False
            assert recovered["coverage"] == "project_multi_asset_observed"
            assert [item["asset_id"] for item in recovered["snapshot"][
                "asset_coverage"]] == ["asset-a", "asset-b"]
            assert provider_state["calls"] == 3  # asset-a hit shot cache; asset-b retried

            replay = _data(_http_call(
                restarted_url, token, "POST",
                project_path + "/film-context:snapshot",
                {"manifest_revision": 1},
            ))
            assert replay["cache_hit"] is True
            assert provider_state["calls"] == 3
            with sqlite3.connect(database_path) as connection:
                cached_after_recovery_a = _cached_observation_row_types(
                    connection, hash_a)
                cached_after_recovery_b = _cached_observation_row_types(
                    connection, hash_b)
            expected_rows = sorted([
                ("deterministic_technical",),
                ("vlm_semantic",),
                ("deterministic_technical", "vlm_semantic"),
            ])
            assert sorted(cached_after_recovery_a) == expected_rows
            assert sorted(cached_after_recovery_b) == expected_rows
        finally:
            _stop_api(restarted)
    finally:
        if api_process is not None:
            _stop_api(api_process)
        provider_state["release_second_chat"].set()
        provider_server.shutdown()
        provider_server.server_close()
        provider_thread.join(timeout=5)


def test_concurrent_manifest_updates_reject_stale_revision_without_partial_write(
    tmp_path,
):
    """Two live writers on one revision produce one immutable successor."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    project_id = "project-manifest-revision-race"
    project_path = f"/v1/projects/{project_id}"
    token = "p1-manifest-revision-race-token"
    database_path = tmp_path / "manifest-race.sqlite3"
    media_a = tmp_path / "race-asset-a.mkv"
    media_b = tmp_path / "race-asset-b.mp4"
    _make_video(media_a, "color=c=green:s=320x240:r=25:d=1")
    _make_video(media_b, "color=c=blue:s=320x240:r=30000/1001:d=1")
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(database_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    port = _free_loopback_port()
    process, base_url = _start_api(
        port, env, Path(__file__).resolve().parents[2],
    )

    assets = [
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
    ]
    initial_refs = ["test-fixture://manifest/initial"]
    try:
        initial_status, initial_body = _http_call(
            base_url, token, "PUT", project_path + "/film-manifest", {
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": "test-fixture://project/manifest-race",
                "boundary_evidence_refs": initial_refs,
                "assets": assets,
            },
        )
        assert initial_status == 200, initial_body
        assert initial_body["data"]["manifest"]["revision"] == 1

        barrier = Barrier(2)

        def revise(evidence_ref: str) -> tuple[int, dict]:
            barrier.wait(timeout=10)
            return _http_call(
                base_url, token, "PUT", project_path + "/film-manifest", {
                    "expected_revision": 1,
                    "boundary_basis": "user_project_manifest",
                    "boundary_source_ref": (
                        "test-fixture://project/manifest-race"),
                    "boundary_evidence_refs": [evidence_ref],
                    "assets": assets,
                },
            )

        evidence_refs = [
            "test-fixture://manifest/writer-a",
            "test-fixture://manifest/writer-b",
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(revise, ref) for ref in evidence_refs]
            results = [future.result(timeout=30) for future in futures]

        assert sorted(status for status, _ in results) == [200, 409]
        winner_status, winner_body = next(
            result for result in results if result[0] == 200)
        assert winner_status == 200
        winning_refs = winner_body["data"]["manifest"]["boundary_evidence_refs"]
        assert len(winning_refs) == 1
        assert winning_refs[0] in evidence_refs
        assert winner_body["data"]["manifest"]["revision"] == 2

        latest_status, latest_body = _http_call(
            base_url, token, "GET", project_path + "/film-manifest")
        assert latest_status == 200, latest_body
        assert latest_body["data"]["manifest"]["revision"] == 2
        assert latest_body["data"]["manifest"]["boundary_evidence_refs"] == (
            winning_refs)

        historical_status, historical_body = _http_call(
            base_url, token, "GET",
            project_path + "/film-manifest?revision=1")
        assert historical_status == 200, historical_body
        assert historical_body["data"]["manifest"]["boundary_evidence_refs"] == (
            initial_refs)

        import sqlite3
        with sqlite3.connect(database_path) as connection:
            revisions = connection.execute(
                "SELECT revision FROM film_project_manifests "
                "WHERE project_id = ? ORDER BY revision",
                (project_id,),
            ).fetchall()
        assert revisions == [(1,), (2,)]
    finally:
        _stop_api(process)


def test_context_analysis_cannot_commit_after_manifest_revision_advances(
    tmp_path,
):
    """A long-running analysis must not attach evidence to a superseded manifest."""
    import sqlite3
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    model_digest = "c" * 64
    provider_state = {
        "calls": 0,
        "lock": threading.Lock(),
        "chat_started": threading.Event(),
        "release_chat": threading.Event(),
        "chat_finished": threading.Event(),
    }
    reply = (
        "A stable indoor view.\n"
        '{"shot_function":"ACTION","sensory_wet_heat":0.2,'
        '"sensory_mood_intensity":0.4,"motion_amount":"subtle",'
        '"proposed_role_v2":"hero"}'
    )

    class OllamaStubHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/version":
                self._send_json(200, {"version": "0.32.14"})
            elif self.path == "/api/tags":
                self._send_json(200, {"models": [{
                    "name": "qwen3-vl:4b", "digest": model_digest,
                }]})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/chat":
                self._send_json(404, {"error": "not found"})
                return
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with provider_state["lock"]:
                provider_state["calls"] += 1
            provider_state["chat_started"].set()
            provider_state["release_chat"].wait(timeout=60)
            provider_state["chat_finished"].set()
            try:
                self._send_json(200, {"message": {"content": reply}})
            except (BrokenPipeError, ConnectionResetError):
                pass

    provider_port = _free_loopback_port()
    provider_server = ThreadingHTTPServer(
        ("127.0.0.1", provider_port), OllamaStubHandler,
    )
    provider_thread = threading.Thread(
        target=provider_server.serve_forever, daemon=True,
    )
    provider_thread.start()

    media_a = tmp_path / "asset-a-green.mp4"
    media_b = tmp_path / "asset-b-blue.mp4"
    _make_video(media_a, "color=c=green:s=320x240:r=25:d=2")
    _make_video(media_b, "color=c=blue:s=320x240:r=25:d=2")
    project_id = "project-context-manifest-race"
    project_path = f"/v1/projects/{project_id}"
    token = "p1-context-manifest-race-token"
    database_path = tmp_path / "context-manifest-race.sqlite3"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(database_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    env["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{provider_port}"
    env["PROJECT_LOCAL_VLM_MODEL"] = "qwen3-vl:4b"
    env["PROJECT_LOCAL_VLM_DIGEST"] = model_digest
    env["PROJECT_LOCAL_VLM_RUNTIME_VERSION"] = "0.32.14"
    env["SHOT_DISCOVERY_BACKEND"] = "scenedetect"
    api_process = None
    requester = None
    snapshot_result: dict[str, object] = {}
    try:
        api_process, base_url = _start_api(
            _free_loopback_port(), env, Path(__file__).resolve().parents[2],
        )
        asset_inputs = [
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
        ]
        initial_status, initial_body = _http_call(
            base_url, token, "PUT", project_path + "/film-manifest", {
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": "test-fixture://project/context-race",
                "boundary_evidence_refs": ["test-fixture://manifest/rev-1"],
                "analysis_profile": "local_vlm_shadow_v1",
                "assets": asset_inputs,
            },
        )
        assert initial_status == 200, initial_body
        assert initial_body["data"]["manifest"]["revision"] == 1

        def build_snapshot() -> None:
            try:
                snapshot_result["response"] = _http_call(
                    base_url, token, "POST",
                    project_path + "/film-context:snapshot",
                    {"manifest_revision": 1},
                )
            except Exception as exc:
                snapshot_result["error"] = type(exc).__name__

        requester = threading.Thread(target=build_snapshot, daemon=True)
        requester.start()
        assert provider_state["chat_started"].wait(timeout=30), (
            "project analysis did not reach the guarded local VLM"
        )

        revision_two_status, revision_two_body = _http_call(
            base_url, token, "PUT", project_path + "/film-manifest", {
                "expected_revision": 1,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": "test-fixture://project/context-race",
                "boundary_evidence_refs": ["test-fixture://manifest/rev-2"],
                "analysis_profile": "local_vlm_shadow_v1",
                "assets": asset_inputs,
            },
        )
        assert revision_two_status == 200, revision_two_body
        assert revision_two_body["data"]["manifest"]["revision"] == 2

        provider_state["release_chat"].set()
        assert provider_state["chat_finished"].wait(timeout=10)
        requester.join(timeout=30)
        assert not requester.is_alive()
        assert "response" in snapshot_result
        snapshot_status, snapshot_body = snapshot_result["response"]
        assert snapshot_status == 409, snapshot_body
        assert "stale project context" in snapshot_body["detail"]

        with sqlite3.connect(database_path) as connection:
            context_count = connection.execute(
                "SELECT COUNT(*) FROM film_context_snapshots WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            observation_count = connection.execute(
                "SELECT COUNT(*) FROM film_observations WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
        assert context_count == 0
        assert observation_count == 0
        assert provider_state["calls"] >= 1
    finally:
        provider_state["release_chat"].set()
        if requester is not None:
            requester.join(timeout=10)
        if api_process is not None:
            _stop_api(api_process)
        provider_server.shutdown()
        provider_server.server_close()
        provider_thread.join(timeout=5)


def test_changed_vlm_digest_recomputes_and_persists_new_profile_context(
    tmp_path,
):
    """A new provider digest misses old shot cache, then persists reusable output."""
    import sqlite3
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    digest_a = "a" * 64
    digest_b = "d" * 64
    provider_state = {
        "active_digest": digest_a,
        "calls_by_digest": {digest_a: 0, digest_b: 0},
        "lock": threading.Lock(),
    }
    reply = (
        "A stable indoor view.\n"
        '{"shot_function":"ACTION","sensory_wet_heat":0.2,'
        '"sensory_mood_intensity":0.4,"motion_amount":"subtle",'
        '"proposed_role_v2":"hero"}'
    )

    class OllamaStubHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/version":
                self._send_json(200, {"version": "0.32.14"})
            elif self.path == "/api/tags":
                self._send_json(200, {"models": [{
                    "name": "qwen3-vl:4b",
                    "digest": provider_state["active_digest"],
                }]})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/chat":
                self._send_json(404, {"error": "not found"})
                return
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with provider_state["lock"]:
                digest = provider_state["active_digest"]
                provider_state["calls_by_digest"][digest] += 1
            try:
                self._send_json(200, {"message": {"content": reply}})
            except (BrokenPipeError, ConnectionResetError):
                pass

    provider_server = ThreadingHTTPServer(
        ("127.0.0.1", _free_loopback_port()), OllamaStubHandler,
    )
    provider_thread = threading.Thread(
        target=provider_server.serve_forever, daemon=True,
    )
    provider_thread.start()

    media_a = tmp_path / "asset-a-green.mp4"
    media_b = tmp_path / "asset-b-blue.mp4"
    _make_video(media_a, "color=c=green:s=320x240:r=25:d=2")
    _make_video(media_b, "color=c=blue:s=320x240:r=30000/1001:d=2")
    project_id = "project-vlm-digest-recompute"
    project_path = f"/v1/projects/{project_id}"
    token = "p1-vlm-digest-recompute-token"
    database_path = tmp_path / "digest-recompute.sqlite3"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(database_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    env["OLLAMA_BASE_URL"] = (
        f"http://127.0.0.1:{provider_server.server_address[1]}"
    )
    env["PROJECT_LOCAL_VLM_MODEL"] = "qwen3-vl:4b"
    env["PROJECT_LOCAL_VLM_DIGEST"] = digest_a
    env["PROJECT_LOCAL_VLM_RUNTIME_VERSION"] = "0.32.14"
    env["SHOT_DISCOVERY_BACKEND"] = "scenedetect"
    api_process = None
    try:
        api_process, base_url = _start_api(
            _free_loopback_port(), env, Path(__file__).resolve().parents[2],
        )
        manifest_status, manifest_body = _http_call(
            base_url, token, "PUT", project_path + "/film-manifest", {
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": "test-fixture://project/digest-recompute",
                "boundary_evidence_refs": ["test-fixture://manifest/rev-1"],
                "analysis_profile": "local_vlm_shadow_v1",
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
        )
        assert manifest_status == 200, manifest_body
        assert manifest_body["data"]["manifest"]["revision"] == 1

        first_status, first_body = _http_call(
            base_url, token, "POST",
            project_path + "/film-context:snapshot",
            {"manifest_revision": 1},
        )
        assert first_status == 200, first_body
        first = first_body["data"]
        assert first["cache_hit"] is False
        assert first["coverage"] == "project_multi_asset_observed"
        assert [item["analysis_cache_state"] for item in
                first["snapshot"]["asset_coverage"]] == ["computed", "computed"]
        first_context_id = first["context_id"]
        original_hashes = first["snapshot"]["source_content_hashes"]
        digest_a_calls = provider_state["calls_by_digest"][digest_a]
        assert digest_a_calls > 0

        first_replay = _data(_http_call(
            base_url, token, "POST",
            project_path + "/film-context:snapshot",
            {"manifest_revision": 1},
        ))
        assert first_replay["cache_hit"] is True
        assert provider_state["calls_by_digest"][digest_a] == digest_a_calls
        _stop_api(api_process)
        api_process = None

        provider_state["active_digest"] = digest_b
        env["PROJECT_LOCAL_VLM_DIGEST"] = digest_b
        api_process, base_url = _start_api(
            _free_loopback_port(), env, Path(__file__).resolve().parents[2],
        )
        recomputed_status, recomputed_body = _http_call(
            base_url, token, "POST",
            project_path + "/film-context:snapshot",
            {"manifest_revision": 1},
        )
        assert recomputed_status == 200, recomputed_body
        recomputed = recomputed_body["data"]
        assert recomputed["cache_hit"] is False
        assert recomputed["context_id"] != first_context_id
        assert recomputed["snapshot"]["source_content_hashes"] == original_hashes
        assert [item["analysis_cache_state"] for item in
                recomputed["snapshot"]["asset_coverage"]] == [
                    "computed", "computed"]
        digest_b_calls = provider_state["calls_by_digest"][digest_b]
        assert digest_b_calls > 0

        second_replay = _data(_http_call(
            base_url, token, "POST",
            project_path + "/film-context:snapshot",
            {"manifest_revision": 1},
        ))
        assert second_replay["cache_hit"] is True
        assert second_replay["context_id"] == recomputed["context_id"]
        assert provider_state["calls_by_digest"][digest_b] == digest_b_calls

        old_context_status, old_context_body = _http_call(
            base_url, token, "GET", f"/v1/film-context/{first_context_id}")
        assert old_context_status == 200, old_context_body
        assert old_context_body["data"]["snapshot"]["source_content_hashes"] == (
            original_hashes)

        with sqlite3.connect(database_path) as connection:
            profiles = connection.execute(
                "SELECT COUNT(DISTINCT analysis_profile), "
                "COUNT(DISTINCT source_content_hash) "
                "FROM analysis_result_cache"
            ).fetchone()
            context_count = connection.execute(
                "SELECT COUNT(*) FROM film_context_snapshots "
                "WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
        assert profiles == (2, 2)
        assert context_count == 2
    finally:
        if api_process is not None:
            _stop_api(api_process)
        provider_server.shutdown()
        provider_server.server_close()
        provider_thread.join(timeout=5)


def test_two_project_multi_asset_lifecycle_isolated_after_restart(tmp_path):
    """Two complete project traces stay isolated in one persistent database.

    The projects deliberately reuse byte-identical synthetic media so the
    project-neutral analysis cache can be shared. Project evidence, graph,
    and Plan/EDL records must remain scoped to their owning project.
    """
    import shutil
    import sqlite3

    source_a = tmp_path / "source-a.mp4"
    source_b = tmp_path / "source-b.mp4"
    _make_video(source_a, "testsrc=duration=2:size=320x240:rate=25")
    _make_video(source_b, "color=c=blue:s=320x240:r=30000/1001:d=2")

    projects = [
        {
            "project_id": "project-isolation-alpha",
            "asset_ids": ["alpha-main", "alpha-cover"],
        },
        {
            "project_id": "project-isolation-beta",
            "asset_ids": ["beta-main", "beta-cover"],
        },
    ]
    media_by_project: dict[str, list[Path]] = {}
    for project in projects:
        project_dir = tmp_path / project["project_id"]
        project_dir.mkdir()
        paths = [
            project_dir / "main.mp4",
            project_dir / "cover.mp4",
        ]
        shutil.copyfile(source_a, paths[0])
        shutil.copyfile(source_b, paths[1])
        media_by_project[project["project_id"]] = paths

    token = "p1-two-project-isolation-token"
    database_path = tmp_path / "two_project_isolation.sqlite3"
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(database_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    env["PROJECT_LOCAL_VLM_MODEL"] = ""
    env["PROJECT_LOCAL_VLM_DIGEST"] = ""
    env["PROJECT_LOCAL_VLM_RUNTIME_VERSION"] = ""
    env["OLLAMA_BASE_URL"] = ""
    env["SHOT_DISCOVERY_BACKEND"] = "scenedetect"

    port = _free_loopback_port()
    repo_root = Path(__file__).resolve().parents[2]
    process, base_url = _start_api(port, env, repo_root)
    records: dict[str, dict] = {}
    try:
        for project in projects:
            project_id = project["project_id"]
            asset_ids = project["asset_ids"]
            project_path = f"/v1/projects/{project_id}"
            manifest = _data(_http_call(
                base_url, token, "PUT", project_path + "/film-manifest", {
                    "expected_revision": 0,
                    "boundary_basis": "user_project_manifest",
                    "boundary_source_ref": f"test-fixture://project/{project_id}",
                    "boundary_evidence_refs": [
                        f"test-fixture://project/{project_id}/manifest",
                    ],
                    "assets": [
                        {
                            "asset_id": asset_id,
                            "source_ref": str(media_path),
                            "order": order,
                            "rights": {
                                "state": "local_processing_allowed",
                                "basis": "owner_permission",
                                "evidence_ref": (
                                    f"test-fixture://rights/{project_id}/{asset_id}"
                                ),
                            },
                        }
                        for order, (asset_id, media_path) in enumerate(
                            zip(asset_ids, media_by_project[project_id], strict=True)
                        )
                    ],
                },
            ))
            assert manifest["current_revision"] == 1
            assert [item["asset_id"] for item in manifest["manifest"]["assets"]] == (
                asset_ids)

            context = _data(_http_call(
                base_url, token, "POST", project_path + "/film-context:snapshot",
                {"manifest_revision": 1},
            ))
            assert context["coverage"] == "project_multi_asset_observed"
            assert context["snapshot"]["project_id"] == project_id
            assert [item["asset_id"] for item in
                    context["snapshot"]["asset_coverage"]] == asset_ids

            graph = _data(_http_call(
                base_url, token, "POST", project_path + "/story-graph:build"))
            assert graph["persisted"] is True
            assert graph["story_graph"]["project_id"] == project_id
            assert graph["story_graph"]["cross_asset_relations_state"] == (
                "not_attempted")
            assert [item["asset_id"] for item in graph["story_graph"]["assets"]] == (
                asset_ids)
            assert all(
                item["story_graph"] is None
                or item["story_graph"]["project_id"] == project_id
                for item in graph["story_graph"]["assets"]
            )

            plan = _data(_http_call(
                base_url, token, "POST", project_path + "/director-plans:generate", {
                    "manifest_revision": 1,
                    "target_duration_us": 1_600_000,
                    "intent_text": f"合成隔离验证项目 {project_id}",
                },
            ))
            assert plan["persisted"] is True
            assert plan["validation"]["valid"] is True
            assert plan["plan"]["state"] == "ready_for_strategy_confirmation"
            assert plan["plan"]["project_id"] == project_id
            assert plan["plan"]["project_context_id"] == context["context_id"]
            assert plan["plan"]["project_story_graph_id"] == (
                graph["story_graph"]["graph_id"])
            assert set(plan["plan"]["sequence_project_asset_ids"]) <= set(asset_ids)
            assert {
                edit["project_asset_id"]
                for edit in plan["edl"]["ordered_edits"]
            } <= set(asset_ids)
            assert plan["evidence_scope"]["quality_acceptance"] == "not_proven"

            evidence_ids = context["snapshot"]["evidence_refs"]
            observation_rows = []
            for evidence_id in evidence_ids:
                observation = _data(_http_call(
                    base_url, token, "GET",
                    f"{project_path}/observations/{evidence_id}",
                ))["observation"]
                assert observation["project_id"] == project_id
                assert observation["project_asset_id"] in asset_ids
                observation_rows.append(observation)

            records[project_id] = {
                "manifest": manifest,
                "context": context,
                "graph": graph,
                "plan": plan,
                "observations": observation_rows,
            }

        alpha = projects[0]["project_id"]
        beta = projects[1]["project_id"]
        alpha_records = records[alpha]
        beta_records = records[beta]
        assert alpha_records["context"]["context_id"] != beta_records[
            "context"]["context_id"]
        assert alpha_records["graph"]["story_graph"]["graph_id"] != beta_records[
            "graph"]["story_graph"]["graph_id"]
        assert alpha_records["plan"]["plan"]["plan_id"] != beta_records[
            "plan"]["plan"]["plan_id"]
        assert alpha_records["context"]["snapshot"]["source_content_hashes"] == (
            beta_records["context"]["snapshot"]["source_content_hashes"])
        assert all(
            item["analysis_cache_state"] == "reused"
            for item in beta_records["context"]["snapshot"]["asset_coverage"]
        )
        assert {
            item["observation_id"] for item in alpha_records["observations"]
        }.isdisjoint({
            item["observation_id"] for item in beta_records["observations"]
        })

        for owner, foreign in ((alpha, beta), (beta, alpha)):
            own = records[owner]
            foreign_project_path = f"/v1/projects/{owner}"
            foreign_context_id = records[foreign]["context"]["context_id"]
            foreign_plan_id = records[foreign]["plan"]["plan"]["plan_id"]
            foreign_graph_context_id = records[foreign]["context"]["context_id"]
            foreign_observation_id = records[foreign]["observations"][0][
                "observation_id"]
            assert _http_call(
                base_url, token, "GET",
                f"{foreign_project_path}/film-context/{foreign_context_id}",
            )[0] == 404
            assert _http_call(
                base_url, token, "GET",
                f"{foreign_project_path}/story-graph?revision=1&context_id="
                f"{foreign_graph_context_id}",
            )[0] == 404
            assert _http_call(
                base_url, token, "GET",
                f"{foreign_project_path}/director-plans/{foreign_plan_id}",
            )[0] == 404
            assert _http_call(
                base_url, token, "GET",
                f"{foreign_project_path}/observations/{foreign_observation_id}",
            )[0] == 404

            own_context = own["context"]["context_id"]
            own_plan = own["plan"]["plan"]["plan_id"]
            assert _http_call(
                base_url, token, "GET",
                f"{foreign_project_path}/film-context/{own_context}",
            )[0] == 200
            assert _http_call(
                base_url, token, "GET",
                f"{foreign_project_path}/director-plans/{own_plan}",
            )[0] == 200
    finally:
        _stop_api(process)

    restarted, restarted_url = _start_api(port, env, repo_root)
    try:
        for project in projects:
            project_id = project["project_id"]
            asset_ids = project["asset_ids"]
            project_path = f"/v1/projects/{project_id}"
            original = records[project_id]

            manifest_readback = _data(_http_call(
                restarted_url, token, "GET",
                project_path + "/film-manifest?revision=1",
            ))
            context_readback = _data(_http_call(
                restarted_url, token, "GET",
                f"{project_path}/film-context/{original['context']['context_id']}",
            ))
            graph_readback = _data(_http_call(
                restarted_url, token, "GET",
                f"{project_path}/story-graph?revision=1&context_id="
                f"{original['context']['context_id']}",
            ))
            plan_readback = _data(_http_call(
                restarted_url, token, "GET",
                f"{project_path}/director-plans/"
                f"{original['plan']['plan']['plan_id']}",
            ))

            assert manifest_readback["manifest"] == original["manifest"]["manifest"]
            assert context_readback["snapshot"] == original["context"]["snapshot"]
            assert graph_readback["story_graph"] == original["graph"]["story_graph"]
            assert plan_readback["plan"] == original["plan"]["plan"]
            assert plan_readback["edl"] == original["plan"]["edl"]
            assert [item["asset_id"] for item in manifest_readback[
                "manifest"]["assets"]] == asset_ids
            assert [item["asset_id"] for item in context_readback[
                "snapshot"]["asset_coverage"]] == asset_ids
            assert [item["asset_id"] for item in graph_readback[
                "story_graph"]["assets"]] == asset_ids
            assert set(plan_readback["plan"]["sequence_project_asset_ids"]) <= set(
                asset_ids)
            assert {
                edit["project_asset_id"]
                for edit in plan_readback["edl"]["ordered_edits"]
            } <= set(asset_ids)

        for owner, foreign in ((projects[0], projects[1]),
                               (projects[1], projects[0])):
            project_path = f"/v1/projects/{owner['project_id']}"
            foreign_records = records[foreign["project_id"]]
            assert _http_call(
                restarted_url, token, "GET",
                f"{project_path}/film-context/"
                f"{foreign_records['context']['context_id']}",
            )[0] == 404
            assert _http_call(
                restarted_url, token, "GET",
                f"{project_path}/story-graph?revision=1&context_id="
                f"{foreign_records['context']['context_id']}",
            )[0] == 404
            assert _http_call(
                restarted_url, token, "GET",
                f"{project_path}/director-plans/"
                f"{foreign_records['plan']['plan']['plan_id']}",
            )[0] == 404
            assert _http_call(
                restarted_url, token, "GET",
                f"{project_path}/observations/"
                f"{foreign_records['observations'][0]['observation_id']}",
            )[0] == 404

        with sqlite3.connect(database_path) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert connection.execute(
                "SELECT project_id, COUNT(*) FROM film_context_snapshots "
                "GROUP BY project_id ORDER BY project_id"
            ).fetchall() == [
                ("project-isolation-alpha", 1),
                ("project-isolation-beta", 1),
            ]
    finally:
        _stop_api(restarted)
