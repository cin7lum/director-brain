"""Authenticated, local-source API integration for shadow pair comparisons."""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from director_brain.context_gateway import (
    build_multi_asset_project_context,
    project_analysis_profile,
    project_asset_analysis_fingerprint,
    project_context_id,
)
from director_brain.models import (
    AssetAnalysisState,
    ClaimKind,
    FilmContextSnapshot,
    FilmObservation,
    FilmProjectManifest,
    ProcessingRights,
    ProjectAnalysisProfile,
    ProjectAsset,
    ProjectBoundaryBasis,
    ProjectStoryGraph,
    ProjectStoryLinkComparison,
    ProjectStoryLinkReview,
    ProjectStoryMentionReview,
    RightsBasis,
    RightsState,
    TimebaseUnit,
)
from director_brain.project_story_graph import build_project_story_graph
from director_brain.utils import file_sha256, short_hash
import observation_service.keyframe as keyframe_module
from observation_service.ollama_vlm_adapter import OllamaVLMAdapter
from storage.sqlite_repository import SqliteRepository


def _make_video(path, filter_arg):
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", filter_arg,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path), "-y",
    ], capture_output=True, text=True, check=True)


def _run_shadow_failure_request_in_process(
    db_path,
    project_id,
    payload,
    request_started,
    generator_entered,
    release_generator,
    result_queue,
):
    os.environ["SQLITE_PATH"] = db_path
    os.environ["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = "comparison-test-token"
    import api.main as child_api

    def blocked_failure(*args, **kwargs):
        generator_entered.set()
        if not release_generator.wait(timeout=20):
            raise TimeoutError("test did not release the child generator")
        from director_brain.llm_adapter import LLMStructuredOutputError

        raise LLMStructuredOutputError(
            "SYNTHETIC_CROSS_PROCESS_FAILURE_DETAIL",
            failure_code="segment_emotions",
        )

    child_api.generate_project_shadow_strategy_options = blocked_failure
    path = f"/v1/projects/{project_id}/director-plans:compare-shadow-strategies"
    request_started.set()
    with TestClient(
        child_api.app,
        client=("127.0.0.1", 54342),
        headers={"Authorization": "Bearer comparison-test-token"},
    ) as client:
        response = client.post(path, json=payload)
    result_queue.put({
        "status_code": response.status_code,
        "failure_code": response.headers.get("X-Director-Brain-Failure-Code"),
        "leaked_detail": (
            "SYNTHETIC_CROSS_PROCESS_FAILURE_DETAIL" in response.text),
    })


def _seed_project(
    db_path,
    media_root,
    monkeypatch,
    *,
    project_id="comparison-project",
    media_prefix="",
    observation_prefix="",
    include_event_evidence=False,
    people_by_asset=None,
    include_unverified_asset=False,
    include_unverified_asset_order=None,
    include_third_asset=False,
):
    media_a = media_root / f"{media_prefix}comparison-a.mp4"
    media_b = media_root / f"{media_prefix}comparison-b.mp4"
    media_c = media_root / f"{media_prefix}comparison-c.mp4"
    _make_video(media_a, "testsrc=duration=2:size=320x240:rate=25")
    _make_video(media_b, "color=c=blue:s=320x240:r=25:d=2")
    if include_third_asset:
        _make_video(media_c, "color=c=green:s=320x240:r=25:d=2")
    digest = "e" * 64
    monkeypatch.setenv("PROJECT_LOCAL_VLM_MODEL", "qwen3-vl:4b")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_DIGEST", digest)
    monkeypatch.setenv("PROJECT_LOCAL_VLM_RUNTIME_VERSION", "0.32.14")
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(media_root))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "comparison-test-token")

    def asset(asset_id, path, order):
        return ProjectAsset(
            asset_id=asset_id,
            source_ref=str(path),
            source_content_hash=file_sha256(str(path)).lower(),
            size_bytes=path.stat().st_size,
            order=order,
            duration_us=2_000_000,
            fps=25,
            r_frame_rate="25/1",
            avg_frame_rate="25/1",
            stream_time_base="1/12800",
            has_audio=False,
            probe_ok=True,
            rights=ProcessingRights(
                state=RightsState.LOCAL_PROCESSING_ALLOWED,
                basis=RightsBasis.OWNER_PERMISSION,
                evidence_ref=f"test-fixture://rights/{asset_id}",
            ),
        )

    assets = [asset("asset-a", media_a, 0), asset("asset-b", media_b, 1)]
    if include_third_asset:
        assets.append(asset("asset-c", media_c, 2))
    if include_unverified_asset:
        unverified_order = (
            len(assets)
            if include_unverified_asset_order is None
            else include_unverified_asset_order
        )
        if not 0 <= unverified_order <= len(assets):
            raise ValueError("unverified fixture asset order is out of range")
        unverified_asset = ProjectAsset(
            asset_id="asset-unverified",
            source_ref="dataset://unverified/asset-3",
            source_identity_state="declared_unverified",
            order=unverified_order,
            rights=ProcessingRights(),
        )
        assets = [
            item.model_copy(update={
                "order": item.order + (1 if item.order >= unverified_order else 0),
            })
            for item in assets
        ]
        assets.append(unverified_asset)
        assets.sort(key=lambda item: item.order)
    manifest = FilmProjectManifest(
        manifest_id=f"manifest_{short_hash(project_id)}_r00000001",
        revision=1,
        boundary_basis=ProjectBoundaryBasis.USER_PROJECT_MANIFEST,
        boundary_source_ref="test-fixture://project/comparison",
        boundary_evidence_refs=["test-fixture://project/manifest"],
        analysis_profile=ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1,
        assets=assets,
        project_id=project_id,
        created_at=1_800_000_000,
        producer="comparison-api-test",
        source_ref="test-fixture://project/comparison",
    )
    observations_by_asset = {}
    locally_authorized_assets = [
        item for item in manifest.assets
        if item.rights.state == RightsState.LOCAL_PROCESSING_ALLOWED
    ]
    for asset_index, asset_item in enumerate(locally_authorized_assets):
        observation_id = (
            f"{observation_prefix}obs-{chr(ord('a') + asset_index)}")
        claim = {
            "people": (people_by_asset or {}).get(
                asset_item.asset_id, ["red coat, short hair"]),
        }
        if include_event_evidence:
            claim.update({
                "action_type": "action",
                "scene_description": (
                    "A couple enters the ceremony hall."
                    if asset_index == 0
                    else "Guests stand as the couple reaches the stage."),
                "temporal_notes": (
                    "The couple enters from the rear doors."
                    if asset_index == 0
                    else "The guests rise before the couple reaches the stage."),
            })
        technical = FilmObservation(
            observation_id=(
                f"{observation_prefix}tech-{asset_item.asset_id}"),
            media_asset_id=(
                f"{observation_prefix}shot-tech-{asset_item.asset_id}"),
            project_asset_id=asset_item.asset_id,
            media_hash=asset_item.source_content_hash,
            start_frame=0,
            end_frame=2_000_000,
            timebase=1_000_000,
            timebase_unit=TimebaseUnit.MICROSECONDS,
            observation_type="deterministic_technical",
            claim=json.dumps({"brightness_mean": 100, "exposure_ok": True}),
            provider="deterministic_opencv",
            model_version="test-v1",
            prompt_version="n/a",
            confidence=1.0,
            evidence_refs=[f"tech-{asset_item.asset_id}:frame"],
            review_state="final",
            claim_kind=ClaimKind.MEASURED,
            project_id=project_id,
            created_at=1_800_000_000,
            producer="comparison-api-test",
            source_ref="local-source-observation",
        )
        semantic = FilmObservation(
            observation_id=observation_id,
            media_asset_id=(
                f"{observation_prefix}shot-{asset_item.asset_id}"),
            project_asset_id=asset_item.asset_id,
            media_hash=asset_item.source_content_hash,
            start_frame=0,
            end_frame=1_800_000,
            timebase=1_000_000,
            timebase_unit=TimebaseUnit.MICROSECONDS,
            observation_type="vlm_semantic",
            claim=json.dumps(claim),
            provider="ollama_qwen3_vl",
            model_version=f"qwen3-vl:4b@sha256:{digest}|ollama:0.32.14",
            prompt_version="vlm_prompt_v4_people",
            confidence=0.5,
            evidence_refs=[f"{observation_id}:frame"],
            review_state="auto_generated",
            claim_kind=ClaimKind.MODEL_OBSERVATION,
            project_id=project_id,
            created_at=1_800_000_000,
            producer="comparison-api-test",
            source_ref="local-source-observation",
        )
        observations_by_asset[asset_item.asset_id] = [technical, semantic]
    analysis_state_by_asset = {
        asset_id: AssetAnalysisState.OBSERVED
        for asset_id in observations_by_asset
    }
    if include_unverified_asset:
        observations_by_asset["asset-unverified"] = []
        analysis_state_by_asset["asset-unverified"] = AssetAnalysisState.NOT_AUTHORIZED
    context = build_multi_asset_project_context(
        manifest,
        observations_by_asset,
        analysis_state_by_asset=analysis_state_by_asset,
        blocked_reason_by_asset=(
            {"asset-unverified": "rights_unverified"}
            if include_unverified_asset else None
        ),
    )
    observations = [
        observation
        for asset_observations in observations_by_asset.values()
        for observation in asset_observations
    ]
    graph = build_project_story_graph(manifest, context, observations)
    repo = SqliteRepository(str(db_path))
    repo.save(manifest)
    repo.save(context)
    for observation in observations:
        repo.save(observation)
    repo.save(graph)
    repo.close()
    assert context.context_id == project_context_id(manifest)
    return project_id, graph.graph_id


def test_pair_comparison_is_local_provenance_bound_idempotent_and_readable(
    tmp_path, monkeypatch,
):
    project_id, graph_id = _seed_project(
        tmp_path / "db.sqlite", tmp_path, monkeypatch,
        include_event_evidence=True,
        people_by_asset={
            "asset-a": ["red coat, short hair", "red coat, short hair"],
        },
    )
    calls = []
    frame_extraction_calls = []
    original_extract_keyframes = keyframe_module.extract_keyframes

    def extract_local_keyframes(*args, **kwargs):
        frame_extraction_calls.append({
            "source": args[0],
            "local_only": kwargs.get("local_only"),
        })
        return original_extract_keyframes(*args, **kwargs)

    def verify_runtime(self):
        self._runtime_binding_verified = True

    def compare(self, left_frames, right_frames, **kwargs):
        calls.append((len(left_frames), len(right_frames), kwargs))
        return {
            "status": "OBSERVED",
            "assessment": "possible_match",
            "evidence_for": ["synthetic visual cue"],
            "evidence_against": [],
            "limitation": "synthetic footage; no identity-quality claim",
            "confidence_type": "UNCALIBRATED_MODEL_ASSESSMENT",
            "review_state": "unreviewed",
        }

    monkeypatch.setattr(OllamaVLMAdapter, "verify_runtime_binding", verify_runtime)
    monkeypatch.setattr(OllamaVLMAdapter, "compare_cross_asset_frames", compare)
    monkeypatch.setattr(
        keyframe_module, "extract_keyframes", extract_local_keyframes)
    client = TestClient(
        app,
        client=("127.0.0.1", 54321),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    candidates_path = f"/v1/projects/{project_id}/story-link-candidates"
    candidate_response = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 0,
    })
    assert candidate_response.status_code == 200, candidate_response.text
    candidate_data = candidate_response.json()["data"]
    candidate_page = candidate_data["candidate_page"]
    assert candidate_data["provider_invocation_count"] == 0
    assert candidate_data["automatic_link_created"] is False
    assert candidate_page["source_mention_review_state"] == "absent"
    assert candidate_page["total"] == 2
    assert candidate_page["ranking_state"] == "unranked"
    assert candidate_page["automatic_inference_state"] == "not_attempted"
    assert candidate_page["candidates"][0]["comparison_progress"] == {
        "comparison_run_state": "not_started",
        "comparison_id": None,
        "attempts": [],
        "retry_comparison_request": None,
        "latest_caller_disposition": None,
    }
    assert candidate_page["candidates"][0]["candidate_basis"] == (
        "exhaustive_cross_asset_pair")
    assert "match_score" not in candidate_page["candidates"][0]
    body = candidate_page["candidates"][0]["comparison_request"]
    assert body == {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "idempotency_key": candidate_page["candidates"][0]["candidate_id"],
        "candidate_id": candidate_page["candidates"][0]["candidate_id"],
        "attempt_number": 1,
        "relation_kind": "person_identity",
        "left_observation_id": "obs-a",
        "right_observation_id": "obs-b",
        "left_story_graph_node_id": candidate_page["candidates"][0][
            "left_anchor"]["story_graph_node_id"],
        "right_story_graph_node_id": candidate_page["candidates"][0][
            "right_anchor"]["story_graph_node_id"],
        "left_person_description": "red coat, short hair",
        "right_person_description": "red coat, short hair",
        "source_mention_review_id": None,
        "source_mention_review_revision": 0,
    }
    assert calls == []

    second_candidate_page = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 1,
        "expected_candidate_set_id": candidate_page["candidate_set_id"],
    }).json()["data"]["candidate_page"]
    second_candidate = second_candidate_page["candidates"][0]
    assert second_candidate["left_anchor"]["observation_id"] == "obs-a"
    assert second_candidate["left_person_description"] == (
        candidate_page["candidates"][0]["left_person_description"])
    assert second_candidate["left_anchor"]["story_graph_node_id"] != (
        candidate_page["candidates"][0]["left_anchor"]["story_graph_node_id"])
    assert second_candidate["candidate_id"] != (
        candidate_page["candidates"][0]["candidate_id"])

    event_page = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "event_identity",
        "limit": 1,
        "offset": 0,
    })
    assert event_page.status_code == 200, event_page.text
    event_candidate = event_page.json()["data"]["candidate_page"]["candidates"][0]
    assert event_candidate["left_event_evidence"]["scene_description"].startswith(
        "A couple enters")
    assert event_candidate["comparison_request"]["left_person_description"] is None

    empty_page = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 2,
        "expected_candidate_set_id": candidate_page["candidate_set_id"],
    })
    assert empty_page.status_code == 200, empty_page.text
    assert empty_page.json()["data"]["candidate_page"]["candidates"] == []
    assert empty_page.json()["data"]["candidate_page"]["candidate_set_id"] == (
        candidate_page["candidate_set_id"])
    assert empty_page.json()["data"]["candidate_page"]["has_more"] is False

    stale_page = client.get(candidates_path, params={
        "manifest_revision": 2,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
    })
    assert stale_page.status_code == 409

    path = f"/v1/projects/{project_id}/story-link-comparisons"
    first_response = client.post(path, json=body)
    assert first_response.status_code == 200, first_response.text
    first = first_response.json()["data"]
    comparison = first["comparison"]
    assert len(frame_extraction_calls) == 2
    assert all(item["local_only"] is True for item in frame_extraction_calls)
    assert first["reused"] is False
    assert first["automatic_link_created"] is False
    assert first["director_plan_updated"] is False
    assert comparison["run_state"] == "completed_unreviewed"
    assert comparison["review_state"] == "unreviewed"
    assert comparison["confidence_type"] == "uncalibrated_model_assessment"
    assert comparison["left_anchor"]["source_content_hash"]
    assert comparison["right_anchor"]["source_content_hash"]
    assert str(tmp_path / "comparison-a.mp4") not in json.dumps(first)
    after_comparison = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 0,
    }).json()["data"]["candidate_page"]["candidates"][0][
        "comparison_progress"]
    assert after_comparison == {
        "comparison_run_state": "completed_unreviewed",
        "comparison_id": comparison["comparison_id"],
        "attempts": [{
            "attempt_number": 1,
            "comparison_id": comparison["comparison_id"],
            "run_state": "completed_unreviewed",
            "failure_code": None,
        }],
        "retry_comparison_request": None,
        "latest_caller_disposition": None,
    }
    assert calls == [(3, 3, {
        "relation_kind": "person_identity",
        "left_person_description": "red coat, short hair",
        "right_person_description": "red coat, short hair",
        "left_event_evidence": None,
        "right_event_evidence": None,
    })]

    replay = client.post(path, json=body).json()["data"]
    assert replay["reused"] is True
    assert replay["comparison"] == comparison
    assert len(calls) == 1

    readback = client.get(
        path + "/" + comparison["comparison_id"])
    assert readback.status_code == 200
    assert readback.json()["data"]["comparison"] == comparison
    listing = client.get(path, params={"limit": 1, "offset": 0})
    assert listing.status_code == 200
    assert listing.json()["data"]["total"] == 1
    assert listing.json()["data"]["comparisons"] == [comparison]

    child_code = r"""
import json, sys
from fastapi.testclient import TestClient
from api.main import app
args = json.loads(sys.argv[1])
with TestClient(app, client=("127.0.0.1", 54322), headers=args["headers"]) as client:
    response = client.get(args["path"])
    print(json.dumps({"status": response.status_code, "data": response.json().get("data")}))
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, json.dumps({
            "headers": {"Authorization": "Bearer comparison-test-token"},
            "path": path + "/" + comparison["comparison_id"],
        })],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ.copy(),
    )
    child_data = json.loads(child.stdout)
    assert child_data["status"] == 200
    assert child_data["data"]["comparison"] == comparison

    conflict = client.post(path, json={
        **body,
        "relation_kind": "event_identity",
        "left_person_description": None,
        "right_person_description": None,
    })
    assert conflict.status_code == 409
    assert len(calls) == 1

    foreign_project = client.get(
        f"/v1/projects/another-project/story-link-comparisons/"
        f"{comparison['comparison_id']}")
    assert foreign_project.status_code == 404

    bad_subject = client.post(path, json={
        **body,
        "idempotency_key": "pair-comparison-bad-subject",
        "left_person_description": "not in stored evidence",
    })
    assert bad_subject.status_code == 422
    assert len(calls) == 1

    wrong_mention_node = client.post(path, json={
        **body,
        "left_story_graph_node_id": body["right_story_graph_node_id"],
    })
    assert wrong_mention_node.status_code == 409, wrong_mention_node.text
    assert len(calls) == 1

    accepted_link_id = "caller-accepted-pair"
    accepted_body = {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "expected_review_revision": 0,
        "idempotency_key": "review-pair-accepted-001",
        "links": [{
            "link_id": accepted_link_id,
            "entity_kind": "person_identity",
            "display_label": "caller-reviewed subject",
            "source_comparison_ids": [comparison["comparison_id"]],
            "anchors": [
                {
                    "observation_id": comparison["left_anchor"]["observation_id"],
                    "story_graph_node_id": comparison["left_anchor"][
                        "story_graph_node_id"],
                    "source_start": comparison["left_anchor"]["source_start"],
                    "source_end": comparison["left_anchor"]["source_end"],
                },
                {
                    "observation_id": comparison["right_anchor"]["observation_id"],
                    "story_graph_node_id": comparison["right_anchor"][
                        "story_graph_node_id"],
                    "source_start": comparison["right_anchor"]["source_start"],
                    "source_end": comparison["right_anchor"]["source_end"],
                },
            ],
        }],
        "comparison_dispositions": [{
            "comparison_id": comparison["comparison_id"],
            "decision": "accepted_as_caller_asserted",
            "link_id": accepted_link_id,
            "rationale": "The caller reviewed the source intervals.",
        }],
    }
    link_review_path = f"/v1/projects/{project_id}/story-links"
    accepted = client.put(link_review_path, json=accepted_body)
    assert accepted.status_code == 200, accepted.text
    accepted_data = accepted.json()["data"]
    accepted_review = accepted_data["review"]
    assert accepted_review["schema_version"] == "1.5"
    assert accepted_data["comparison_results_remain_unreviewed"] is True
    assert accepted_review["actor_identity_state"] == "caller_asserted"
    assert accepted_review["automatic_inference_state"] == "not_attempted"
    assert accepted_review["comparison_dispositions"] == [{
        "comparison_id": comparison["comparison_id"],
        "decision": "accepted_as_caller_asserted",
        "link_id": accepted_link_id,
        "rationale": "The caller reviewed the source intervals.",
    }]
    after_acceptance = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 0,
    }).json()["data"]["candidate_page"]["candidates"][0][
        "comparison_progress"]
    assert after_acceptance["comparison_run_state"] == "completed_unreviewed"
    assert after_acceptance["comparison_id"] == comparison["comparison_id"]
    assert after_acceptance["latest_caller_disposition"] == {
        "disposition": accepted_review["comparison_dispositions"][0],
        "review_id": accepted_review["review_id"],
        "review_revision": 1,
        "is_current": True,
    }
    assert accepted_review["links"][0]["source_comparison_ids"] == [
        comparison["comparison_id"]]
    graph_path = f"/v1/projects/{project_id}/story-graph"
    accepted_graph_view_response = client.get(graph_path)
    assert accepted_graph_view_response.status_code == 200
    accepted_graph_view = accepted_graph_view_response.json()["data"][
        "story_graph_view"]
    assert accepted_graph_view["source_graph"]["graph_id"] == graph_id
    assert accepted_graph_view["cross_asset_link_review"] == accepted_review
    assert accepted_graph_view["cross_asset_relation_state"] == "caller_asserted"
    assert accepted_graph_view["cross_asset_link_count"] == 1
    assert accepted_graph_view["schema_version"] == "1.4"
    assert accepted_graph_view["cross_asset_relations"] == [{
        "relation_id": accepted_link_id,
        "relation_kind": "person_identity",
        "display_label": "caller-reviewed subject",
        "members": [
            {
                "anchor": comparison["left_anchor"],
                "node_binding_state": "exact_mention_node",
            },
            {
                "anchor": comparison["right_anchor"],
                "node_binding_state": "exact_mention_node",
            },
        ],
        "source_review_id": accepted_review["review_id"],
        "source_review_revision": accepted_review["review_revision"],
        "source_link_id": accepted_link_id,
        "source_comparison_ids": [comparison["comparison_id"]],
        "assertion_state": "caller_asserted",
        "identity_verification_state": "not_independently_verified",
    }]
    assert accepted_graph_view["automatic_inference_state"] == "not_attempted"
    still_unreviewed = client.get(
        path + "/" + comparison["comparison_id"]
    ).json()["data"]["comparison"]
    assert still_unreviewed["review_state"] == "unreviewed"

    accepted_replay = client.put(link_review_path, json=accepted_body)
    assert accepted_replay.status_code == 200
    assert accepted_replay.json()["data"]["reused"] is True
    assert accepted_replay.json()["data"]["review"] == accepted_review

    drifted_anchor_body = json.loads(json.dumps(accepted_body))
    drifted_anchor_body["expected_review_revision"] = 2
    drifted_anchor_body["idempotency_key"] = "review-pair-anchor-drift"
    drifted_anchor_body["links"][0]["anchors"][0]["source_start"] += 1
    drifted_anchor = client.put(link_review_path, json=drifted_anchor_body)
    assert drifted_anchor.status_code == 409

    wrong_kind_body = json.loads(json.dumps(accepted_body))
    wrong_kind_body["expected_review_revision"] = 2
    wrong_kind_body["idempotency_key"] = "review-pair-kind-mismatch"
    wrong_kind_body["links"][0]["entity_kind"] = "event_identity"
    wrong_kind = client.put(link_review_path, json=wrong_kind_body)
    assert wrong_kind.status_code == 409

    second_comparison_request = {
        **body,
        "idempotency_key": "pair-comparison-002",
    }
    second_comparison_request.pop("candidate_id")
    second_comparison_request.pop("attempt_number")
    second_response = client.post(path, json=second_comparison_request)
    assert second_response.status_code == 200, second_response.text
    second_comparison = second_response.json()["data"]["comparison"]
    assert second_comparison["comparison_id"] != comparison["comparison_id"]
    assert len(calls) == 2
    rejected = client.put(
        f"/v1/projects/{project_id}/story-links",
        json={
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "expected_review_revision": 1,
            "idempotency_key": "review-pair-rejected-002",
            "links": [],
            "comparison_dispositions": [{
                "comparison_id": second_comparison["comparison_id"],
                "decision": "rejected",
                "rationale": "The caller judged the supplied views insufficiently alike.",
            }],
        },
    )
    assert rejected.status_code == 200, rejected.text
    rejected_review = rejected.json()["data"]["review"]
    assert rejected_review["review_revision"] == 2
    assert rejected_review["links"] == []
    assert rejected_review["comparison_dispositions"][0]["decision"] == "rejected"
    after_superseding_review = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 0,
    }).json()["data"]["candidate_page"]["candidates"][0][
        "comparison_progress"]
    assert after_superseding_review["latest_caller_disposition"][
        "is_current"] is False
    assert after_superseding_review["latest_caller_disposition"][
        "disposition"]["decision"] == "accepted_as_caller_asserted"
    rejected_graph_view = client.get(graph_path).json()["data"]["story_graph_view"]
    assert rejected_graph_view["source_graph"]["graph_id"] == graph_id
    assert rejected_graph_view["cross_asset_link_review"] == rejected_review
    assert rejected_graph_view["cross_asset_relation_state"] == "none"
    assert rejected_graph_view["cross_asset_link_count"] == 0
    assert rejected_graph_view["view_id"] != accepted_graph_view["view_id"]
    assert client.get(path + "/" + second_comparison["comparison_id"]).json()[
        "data"]["comparison"]["review_state"] == "unreviewed"

    restart_code = r"""
import json, sys
from fastapi.testclient import TestClient
from api.main import app
args = json.loads(sys.argv[1])
with TestClient(app, client=("127.0.0.1", 54323), headers=args["headers"]) as client:
    comparison = client.get(args["comparison_path"]).json()["data"]["comparison"]
    review = client.get(args["review_path"]).json()["data"]["review"]
    graph_view = client.get(args["graph_path"]).json()["data"]["story_graph_view"]
    candidate_page = client.get(args["candidate_path"], params=args["candidate_query"]).json()["data"]["candidate_page"]
    candidate_progress = candidate_page["candidates"][0]["comparison_progress"]
    print(json.dumps({"comparison": comparison, "review": review, "graph_view": graph_view, "candidate_progress": candidate_progress}))
"""
    restarted = subprocess.run(
        [sys.executable, "-c", restart_code, json.dumps({
            "headers": {"Authorization": "Bearer comparison-test-token"},
            "comparison_path": path + "/" + comparison["comparison_id"],
            "review_path": link_review_path,
            "graph_path": graph_path,
            "candidate_path": candidates_path,
            "candidate_query": {
                "manifest_revision": 1,
                "story_graph_id": graph_id,
                "relation_kind": "person_identity",
                "limit": 1,
                "offset": 0,
            },
        })],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ.copy(),
    )
    restarted_data = json.loads(restarted.stdout)
    assert restarted_data["comparison"]["review_state"] == "unreviewed"
    assert restarted_data["review"]["review_revision"] == 2
    assert restarted_data["review"]["schema_version"] == "1.5"
    assert restarted_data["review"]["comparison_dispositions"][0]["decision"] == (
        "rejected")
    assert restarted_data["graph_view"]["view_id"] == rejected_graph_view["view_id"]
    assert restarted_data["graph_view"]["cross_asset_link_count"] == 0
    assert restarted_data["candidate_progress"] == after_superseding_review

    with (tmp_path / "comparison-a.mp4").open("ab") as media:
        media.write(b"source-drift")
    stale_source_request = {
        **body,
        "idempotency_key": "pair-comparison-stale-source",
    }
    stale_source_request.pop("candidate_id")
    stale_source_request.pop("attempt_number")
    stale_source = client.post(path, json=stale_source_request)
    assert stale_source.status_code == 409
    assert len(calls) == 2


def test_candidate_read_rejects_a_previous_context_schema(tmp_path, monkeypatch):
    project_id, graph_id = _seed_project(
        tmp_path / "stale-context-db.sqlite", tmp_path, monkeypatch)
    repo = SqliteRepository(str(tmp_path / "stale-context-db.sqlite"))
    graph = repo.get(ProjectStoryGraph, graph_id)
    context = repo.get(FilmContextSnapshot, graph.context_id)
    stale_context_id = "ctx_project_previous_schema"
    repo.save(context.model_copy(update={
        "context_id": stale_context_id,
        "schema_version": "1.4",
    }))
    repo.save(graph.model_copy(update={"context_id": stale_context_id}))
    repo.close()

    client = TestClient(
        app,
        client=("127.0.0.1", 54323),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    response = client.get(
        f"/v1/projects/{project_id}/story-link-candidates",
        params={
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "relation_kind": "person_identity",
            "limit": 1,
            "offset": 0,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "candidate StoryGraph context schema is stale")


def test_context_schema_refresh_reuses_asset_cache_and_resumes_project_plan(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "schema-refresh-db.sqlite"
    project_id, graph_id = _seed_project(db_path, tmp_path, monkeypatch)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    repo = SqliteRepository(str(db_path))
    manifest = repo.latest_project_manifest(project_id)
    # The shared fixture uses generic persistence; the public snapshot route
    # deliberately reads the append-only manifest revision table.
    repo.delete(FilmProjectManifest, manifest.manifest_id)
    repo.save_project_manifest(manifest, expected_revision=0)
    old_context = repo.get(
        FilmContextSnapshot,
        project_context_id(manifest),
    )
    old_graph = repo.get(ProjectStoryGraph, graph_id)
    observations = repo.list(FilmObservation, project_id)
    profile = project_analysis_profile(manifest.analysis_profile)
    for asset in manifest.assets:
        cached_observations = [
            item for item in observations
            if item.project_asset_id == asset.asset_id
        ]
        repo.save_analysis_result(
            project_asset_analysis_fingerprint(
                asset, manifest.analysis_profile),
            asset.source_content_hash,
            profile,
            cached_observations,
        )

    stale_context_id = "ctx_project_previous_schema"
    stale_context = old_context.model_copy(update={
        "context_id": stale_context_id,
        "schema_version": "1.4",
        "invalidated_at": None,
    })
    repo.save(stale_context)
    repo.delete(FilmContextSnapshot, old_context.context_id)
    repo.delete(ProjectStoryGraph, old_graph.graph_id)
    stale_graph = build_project_story_graph(
        manifest, stale_context, observations)
    repo.save(stale_graph)
    repo.close()

    import observation_service.pipeline as pipeline

    def fail_if_analysis_runs(*_args, **_kwargs):
        raise AssertionError("schema refresh should reuse the asset analysis cache")

    monkeypatch.setattr(pipeline, "analyze_media_full", fail_if_analysis_runs)
    monkeypatch.setattr(pipeline, "analyze_media", fail_if_analysis_runs)

    client = TestClient(
        app,
        client=("127.0.0.1", 54322),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    stale_candidates = client.get(
        f"/v1/projects/{project_id}/story-link-candidates",
        params={
            "manifest_revision": manifest.revision,
            "story_graph_id": stale_graph.graph_id,
            "relation_kind": "person_identity",
            "limit": 1,
            "offset": 0,
        },
    )
    assert stale_candidates.status_code == 409

    refreshed = client.post(
        f"/v1/projects/{project_id}/film-context:snapshot",
        json={"manifest_revision": manifest.revision},
    )
    assert refreshed.status_code == 200, refreshed.text
    refreshed_data = refreshed.json()["data"]
    assert refreshed_data["cache_hit"] is False
    assert refreshed_data["context_id"] == project_context_id(manifest)
    assert refreshed_data["snapshot"]["schema_version"] == "1.5"
    assert [item["analysis_cache_state"] for item in
            refreshed_data["snapshot"]["asset_coverage"]] == ["reused", "reused"]

    graph_response = client.post(
        f"/v1/projects/{project_id}/story-graph:build")
    assert graph_response.status_code == 200, graph_response.text
    refreshed_graph = graph_response.json()["data"]["story_graph"]
    assert refreshed_graph["context_id"] == refreshed_data["context_id"]
    assert refreshed_graph["graph_id"] != stale_graph.graph_id

    plan_response = client.post(
        f"/v1/projects/{project_id}/director-plans:generate",
        json={
            "manifest_revision": manifest.revision,
            "target_duration_us": 1_600_000,
            "intent_text": "Continue planning from the refreshed project context.",
        },
    )
    assert plan_response.status_code == 200, plan_response.text
    plan_data = plan_response.json()["data"]
    assert plan_data["persisted"] is True
    assert plan_data["plan"]["project_context_id"] == refreshed_data["context_id"]
    plan_readback = client.get(
        f"/v1/projects/{project_id}/director-plans/"
        f"{plan_data['plan']['plan_id']}"
    )
    assert plan_readback.status_code == 200, plan_readback.text
    assert plan_readback.json()["data"]["plan"] == plan_data["plan"]
    assert plan_readback.json()["data"]["edl"] == plan_data["edl"]


def test_candidate_pages_are_exhaustive_stable_and_unranked(tmp_path, monkeypatch):
    project_id, graph_id = _seed_project(
        tmp_path / "candidate-db.sqlite",
        tmp_path,
        monkeypatch,
        people_by_asset={
            "asset-a": ["red coat", "blue coat"],
            "asset-b": ["person near doorway"],
        },
        include_unverified_asset=True,
    )
    configured_project_vlm_digest = os.environ.get("PROJECT_LOCAL_VLM_DIGEST")
    monkeypatch.delenv("PROJECT_LOCAL_VLM_DIGEST", raising=False)
    client = TestClient(
        app,
        client=("127.0.0.1", 54324),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    path = f"/v1/projects/{project_id}/story-link-candidates"
    query = {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
    }

    first_page = client.get(path, params={**query, "offset": 0})
    assert first_page.status_code == 200, first_page.text
    first = first_page.json()["data"]["candidate_page"]
    second_page = client.get(path, params={
        **query,
        "offset": 1,
        "expected_candidate_set_id": first["candidate_set_id"],
    })
    missing_set_id = client.get(path, params={**query, "offset": 1})
    stale_set_id = client.get(path, params={
        **query,
        "offset": 1,
        "expected_candidate_set_id": "stale-candidate-set",
    })
    repeated_first_page = client.get(path, params={**query, "offset": 0})
    assert second_page.status_code == 200, second_page.text
    assert missing_set_id.status_code == 422
    assert stale_set_id.status_code == 409
    second = second_page.json()["data"]["candidate_page"]
    repeated = repeated_first_page.json()["data"]["candidate_page"]
    assert first["source_hash_validation_state"] == "not_revalidated_by_read"
    assert first["total"] == 2
    assert first["has_more"] is True
    assert second["has_more"] is False
    assert first["candidate_set_id"] == second["candidate_set_id"]
    assert first["candidate_set_id"] == repeated["candidate_set_id"]
    assert first["candidates"] == repeated["candidates"]
    candidates = first["candidates"] + second["candidates"]
    assert len({item["candidate_id"] for item in candidates}) == 2
    assert [item["left_person_description"] for item in candidates] == [
        "red coat", "blue coat"]
    assert all(item["right_person_description"] == "person near doorway"
               for item in candidates)
    assert all(item["ranking_state"] == "unranked" for item in candidates)
    assert all(item["automatic_inference_state"] == "not_attempted"
               for item in candidates)
    assert first["selection_semantics"] == (
        "all_eligible_cross_asset_pairs_in_manifest_order")
    first_asset_anchor_id = candidates[0]["left_anchor"][
        "story_graph_node_id"]
    second_asset_anchor_id = candidates[0]["right_anchor"][
        "story_graph_node_id"]
    first_anchor_page = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": first_asset_anchor_id,
        "offset": 0,
    })
    second_anchor_page = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": second_asset_anchor_id,
        "offset": 0,
    })
    assert first_anchor_page.status_code == second_anchor_page.status_code == 200
    first_anchor_data = first_anchor_page.json()["data"]["candidate_page"]
    second_anchor_data = second_anchor_page.json()["data"]["candidate_page"]
    assert first_anchor_data["selected_anchor_story_graph_node_id"] == (
        first_asset_anchor_id)
    assert second_anchor_data["selected_anchor_story_graph_node_id"] == (
        second_asset_anchor_id)
    assert first_anchor_data["selection_semantics"] == (
        "all_eligible_cross_asset_pairs_for_selected_mention_in_manifest_order")
    assert first_anchor_data["total"] == 1
    assert second_anchor_data["total"] == 2
    assert first_anchor_data["candidate_set_id"] != first["candidate_set_id"]
    assert first_anchor_data["candidates"][0]["candidate_id"] == (
        candidates[0]["candidate_id"])
    assert all(
        first_anchor_data["selected_anchor_story_graph_node_id"] in {
            item["left_anchor"]["story_graph_node_id"],
            item["right_anchor"]["story_graph_node_id"],
        }
        for item in first_anchor_data["candidates"]
    )
    second_anchor_tail = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": second_asset_anchor_id,
        "offset": 1,
        "expected_candidate_set_id": second_anchor_data["candidate_set_id"],
    })
    assert second_anchor_tail.status_code == 200
    assert second_anchor_tail.json()["data"]["candidate_page"]["total"] == 2
    assert second_anchor_tail.json()["data"]["candidate_page"]["has_more"] is False
    ineligible_anchor = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": "missing-current-person-mention",
    })
    assert ineligible_anchor.status_code == 422
    assert ineligible_anchor.json()["detail"] == (
        "selected anchor must be an eligible current mention of the "
        "requested relation kind")
    assert first["asset_coverage"] == [
        {
            "asset_id": "asset-a",
            "order": 0,
            "state": "eligible_under_manifest_declaration",
            "rights_state": "local_processing_allowed",
            "rights_evidence_state": "declared_unverified",
            "source_identity_state": "locally_verified",
            "eligible_mention_count": 2,
        },
        {
            "asset_id": "asset-b",
            "order": 1,
            "state": "eligible_under_manifest_declaration",
            "rights_state": "local_processing_allowed",
            "rights_evidence_state": "declared_unverified",
            "source_identity_state": "locally_verified",
            "eligible_mention_count": 1,
        },
        {
            "asset_id": "asset-unverified",
            "order": 2,
            "state": "not_locally_authorized",
            "rights_state": "unverified",
            "rights_evidence_state": "declared_unverified",
            "source_identity_state": "declared_unverified",
            "eligible_mention_count": 0,
        },
    ]




    # Read-only candidate discovery must resolve persisted Context from the
    # current graph without requiring an inference-provider configuration.
    monkeypatch.setenv(
        "PROJECT_LOCAL_VLM_DIGEST", configured_project_vlm_digest or "")

    def verify_runtime(self):
        self._runtime_binding_verified = True

    compare_calls = []
    compare_response = {"value": {"status": "INVALID"}}

    def compare(self, left_frames, right_frames, **kwargs):
        compare_calls.append((len(left_frames), len(right_frames), kwargs))
        return compare_response["value"]

    monkeypatch.setattr(OllamaVLMAdapter, "verify_runtime_binding", verify_runtime)
    monkeypatch.setattr(OllamaVLMAdapter, "compare_cross_asset_frames", compare)
    comparison_path = f"/v1/projects/{project_id}/story-link-comparisons"
    wrong_pair_request = dict(second["candidates"][0]["comparison_request"])
    wrong_pair_request["idempotency_key"] = first["candidates"][0]["candidate_id"]
    wrong_pair_request.pop("candidate_id")
    wrong_pair_request.pop("attempt_number")
    conflicting_record = client.post(comparison_path, json=wrong_pair_request)
    assert conflicting_record.status_code == 200, conflicting_record.text
    assert conflicting_record.json()["data"]["comparison"]["run_state"] == (
        "failed")

    conflict_progress = client.get(path, params={**query, "offset": 0}).json()[
        "data"]["candidate_page"]["candidates"][0]["comparison_progress"]
    assert conflict_progress["comparison_run_state"] == "idempotency_conflict"
    assert conflict_progress["comparison_id"] == (
        conflicting_record.json()["data"]["comparison"]["comparison_id"])

    failed_record = client.post(
        comparison_path, json=second["candidates"][0]["comparison_request"])
    assert failed_record.status_code == 200, failed_record.text
    assert failed_record.json()["data"]["comparison"]["failure_code"] == (
        "provider_invalid")
    failed_progress = client.get(path, params={
        **query,
        "offset": 1,
        "expected_candidate_set_id": first["candidate_set_id"],
    }).json()[
        "data"]["candidate_page"]["candidates"][0]["comparison_progress"]
    assert failed_progress["comparison_run_state"] == "failed"
    assert failed_progress["comparison_id"] == (
        failed_record.json()["data"]["comparison"]["comparison_id"])
    assert failed_progress["attempts"] == [{
        "attempt_number": 1,
        "comparison_id": failed_record.json()["data"]["comparison"][
            "comparison_id"],
        "run_state": "failed",
        "failure_code": "provider_invalid",
    }]
    candidate_two_retry = failed_progress["retry_comparison_request"]
    assert candidate_two_retry["candidate_id"] == second["candidates"][0][
        "candidate_id"]
    assert candidate_two_retry["attempt_number"] == 2
    assert candidate_two_retry["idempotency_key"] == (
        candidate_two_retry["candidate_id"] + ":retry:2")

    failed_retry = client.post(comparison_path, json=candidate_two_retry)
    assert failed_retry.status_code == 200, failed_retry.text
    assert failed_retry.json()["data"]["comparison"]["attempt_number"] == 2
    assert failed_retry.json()["data"]["comparison"]["run_state"] == "failed"
    failed_retry_progress = client.get(path, params={
        **query, "offset": 1,
        "expected_candidate_set_id": first["candidate_set_id"],
    }).json()["data"]["candidate_page"]["candidates"][0][
        "comparison_progress"]
    candidate_two_retry = failed_retry_progress["retry_comparison_request"]
    assert candidate_two_retry["attempt_number"] == 3
    assert candidate_two_retry["idempotency_key"].endswith(":retry:3")

    compare_response["value"] = {
        "status": "OBSERVED",
        "assessment": "possible_match",
        "evidence_for": ["synthetic retry evidence"],
        "evidence_against": [],
        "limitation": "synthetic footage; no identity-quality claim",
        "confidence_type": "UNCALIBRATED_MODEL_ASSESSMENT",
        "review_state": "unreviewed",
    }
    retried_record = client.post(
        comparison_path, json=candidate_two_retry)
    assert retried_record.status_code == 200, retried_record.text
    retried_comparison = retried_record.json()["data"]["comparison"]
    assert retried_comparison["run_state"] == "completed_unreviewed"
    assert retried_comparison["candidate_id"] == candidate_two_retry["candidate_id"]
    assert retried_comparison["attempt_number"] == 3
    assert retried_record.json()["data"]["reused"] is False
    compare_count_after_retry = len(compare_calls)
    replayed_retry = client.post(comparison_path, json=candidate_two_retry)
    assert replayed_retry.status_code == 200
    assert replayed_retry.json()["data"]["reused"] is True
    assert replayed_retry.json()["data"]["comparison"] == retried_comparison
    assert len(compare_calls) == compare_count_after_retry

    completed_candidate_two = client.get(path, params={
        **query, "offset": 1,
        "expected_candidate_set_id": first["candidate_set_id"],
    }).json()["data"]["candidate_page"]["candidates"][0][
        "comparison_progress"]
    assert completed_candidate_two["comparison_run_state"] == (
        "completed_unreviewed")
    assert [item["attempt_number"] for item in completed_candidate_two["attempts"]] == [1, 2, 3]
    assert [item["run_state"] for item in completed_candidate_two["attempts"]] == [
        "failed", "failed", "completed_unreviewed"]
    assert completed_candidate_two["retry_comparison_request"] is None

    restart_code = r"""
import json, sys
from fastapi.testclient import TestClient
from api.main import app
args = json.loads(sys.argv[1])
with TestClient(app, client=("127.0.0.1", 54325), headers=args["headers"]) as client:
    response = client.get(args["path"], params=args["params"])
    page = response.json().get("data", {}).get("candidate_page", {})
    print(json.dumps({"status": response.status_code,
                      "progress": page.get("candidates", [{}])[0].get("comparison_progress")}))
"""
    restarted = subprocess.run(
        [sys.executable, "-c", restart_code, json.dumps({
            "headers": {"Authorization": "Bearer comparison-test-token"},
            "path": path,
            "params": {
                **query,
                "offset": 1,
                "expected_candidate_set_id": first["candidate_set_id"],
            },
        })],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ.copy(),
    )
    restart_data = json.loads(restarted.stdout)
    assert restart_data["status"] == 200
    assert restart_data["progress"] == completed_candidate_two

    # The first candidate's canonical key was occupied by a different pair.
    # Its independent attempt 2 remains readable with that conflict preserved.
    candidate_one_conflict = client.get(path, params={
        **query, "offset": 0,
    }).json()["data"]["candidate_page"]["candidates"][0][
        "comparison_progress"]
    candidate_one_retry = candidate_one_conflict["retry_comparison_request"]
    recovered_candidate_one = client.post(
        comparison_path, json=candidate_one_retry)
    assert recovered_candidate_one.status_code == 200, recovered_candidate_one.text
    recovered_comparison = recovered_candidate_one.json()["data"]["comparison"]
    assert recovered_comparison["attempt_number"] == 2
    assert recovered_comparison["candidate_id"] == candidate_one_retry["candidate_id"]
    candidate_one_progress = client.get(path, params={
        **query, "offset": 0,
    }).json()["data"]["candidate_page"]["candidates"][0][
        "comparison_progress"]
    assert candidate_one_progress["comparison_run_state"] == (
        "completed_unreviewed")
    assert candidate_one_progress["attempts"] == [
        {
            "attempt_number": 1,
            "comparison_id": conflicting_record.json()["data"]["comparison"][
                "comparison_id"],
            "run_state": "idempotency_conflict",
            "failure_code": None,
        },
        {
            "attempt_number": 2,
            "comparison_id": recovered_comparison["comparison_id"],
            "run_state": "completed_unreviewed",
            "failure_code": None,
        },
    ]

    def link_group(candidate, comparison, link_id, label):
        anchors = [
            {
                "observation_id": candidate[side]["observation_id"],
                "story_graph_node_id": candidate[side]["story_graph_node_id"],
                "source_start": candidate[side]["source_start"],
                "source_end": candidate[side]["source_end"],
            }
            for side in ("left_anchor", "right_anchor")
        ]
        return {
            "link_id": link_id,
            "entity_kind": "person_identity",
            "display_label": label,
            "source_comparison_ids": [comparison["comparison_id"]],
            "anchors": anchors,
        }

    first_candidate = first["candidates"][0]
    second_candidate = second["candidates"][0]
    conflicting_review = client.put(
        f"/v1/projects/{project_id}/story-links",
        json={
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "expected_review_revision": 0,
            "idempotency_key": "conflicting-cross-asset-identities",
            "links": [
                link_group(
                    first_candidate, recovered_comparison,
                    "caller-person-group-1", "person group one"),
                link_group(
                    second_candidate, retried_comparison,
                    "caller-person-group-2", "person group two"),
            ],
            "comparison_dispositions": [
                {
                    "comparison_id": recovered_comparison["comparison_id"],
                    "decision": "accepted_as_caller_asserted",
                    "link_id": "caller-person-group-1",
                    "rationale": "caller test fixture",
                },
                {
                    "comparison_id": retried_comparison["comparison_id"],
                    "decision": "accepted_as_caller_asserted",
                    "link_id": "caller-person-group-2",
                    "rationale": "caller test fixture",
                },
            ],
        },
    )
    assert conflicting_review.status_code == 409, conflicting_review.text
    unchanged_review = client.get(
        f"/v1/projects/{project_id}/story-links")
    assert unchanged_review.status_code == 200
    assert unchanged_review.json()["data"]["review"] is None
    assert unchanged_review.json()["data"]["review_revision"] == 0


def test_candidate_read_defers_source_hash_check_but_comparison_fails_closed(
    tmp_path, monkeypatch,
):
    project_id, graph_id = _seed_project(
        tmp_path / "stale-source-db.sqlite", tmp_path, monkeypatch,
        people_by_asset={
            "asset-a": ["person in a red coat"],
            "asset-b": ["person near the doorway"],
        },
    )
    source_path = tmp_path / "comparison-a.mp4"
    with source_path.open("ab") as source:
        source.write(b"changed-after-manifest-registration")

    client = TestClient(
        app,
        client=("127.0.0.1", 54329),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    candidates_path = f"/v1/projects/{project_id}/story-link-candidates"
    page_response = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
    })
    assert page_response.status_code == 200, page_response.text
    page = page_response.json()["data"]["candidate_page"]
    assert page["source_hash_validation_state"] == "not_revalidated_by_read"
    assert page["total"] == 1

    provider_calls = []

    def verify_runtime(self):
        provider_calls.append("verify_runtime_binding")
        raise AssertionError("stale source must be rejected before provider access")

    def compare(self, *_args, **_kwargs):
        provider_calls.append("compare_cross_asset_frames")
        raise AssertionError("stale source must be rejected before frame comparison")

    monkeypatch.setattr(OllamaVLMAdapter, "verify_runtime_binding", verify_runtime)
    monkeypatch.setattr(OllamaVLMAdapter, "compare_cross_asset_frames", compare)
    compare_response = client.post(
        f"/v1/projects/{project_id}/story-link-comparisons",
        json=page["candidates"][0]["comparison_request"],
    )
    assert compare_response.status_code == 409, compare_response.text
    assert "no longer match the registered manifest" in compare_response.json()["detail"]
    assert provider_calls == []


def test_anchor_scopes_preserve_canonical_order_across_three_assets(
    tmp_path, monkeypatch,
):
    project_id, graph_id = _seed_project(
        tmp_path / "three-asset-db.sqlite",
        tmp_path,
        monkeypatch,
        include_third_asset=True,
        people_by_asset={
            "asset-a": ["mention-a-1", "mention-a-2"],
            "asset-b": ["mention-b"],
            "asset-c": ["mention-c"],
        },
    )
    client = TestClient(
        app,
        client=("127.0.0.1", 54325),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    path = f"/v1/projects/{project_id}/story-link-candidates"
    query = {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 10,
    }
    full_response = client.get(path, params=query)
    assert full_response.status_code == 200, full_response.text
    full = full_response.json()["data"]["candidate_page"]
    full_candidates = full["candidates"]
    assert full["total"] == 5
    assert [
        (item["left_person_description"], item["right_person_description"])
        for item in full_candidates
    ] == [("mention-a-1", "mention-b"), ("mention-a-2", "mention-b"),
          ("mention-a-1", "mention-c"), ("mention-a-2", "mention-c"),
          ("mention-b", "mention-c")]

    middle_anchor_id = full_candidates[0]["right_anchor"][
        "story_graph_node_id"]
    middle_response = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": middle_anchor_id,
    })
    assert middle_response.status_code == 200, middle_response.text
    middle = middle_response.json()["data"]["candidate_page"]
    assert middle["total"] == 3
    assert [item["candidate_id"] for item in middle["candidates"]] == [
        full_candidates[0]["candidate_id"],
        full_candidates[1]["candidate_id"],
        full_candidates[4]["candidate_id"],
    ]
    assert all(middle_anchor_id in {
        item["left_anchor"]["story_graph_node_id"],
        item["right_anchor"]["story_graph_node_id"],
    } for item in middle["candidates"])

    selected_counterpart_response = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": middle_anchor_id,
        "selected_counterpart_project_asset_id": "asset-a",
    })
    assert selected_counterpart_response.status_code == 200
    selected_counterpart = selected_counterpart_response.json()[
        "data"]["candidate_page"]
    assert selected_counterpart["total"] == 2
    assert selected_counterpart["selected_counterpart_project_asset_id"] == "asset-a"
    assert selected_counterpart["selection_semantics"] == (
        "all_eligible_pairs_for_anchor_and_counterpart_asset_in_manifest_order")
    assert selected_counterpart["candidate_set_id"] != middle["candidate_set_id"]
    assert [item["candidate_id"] for item in selected_counterpart["candidates"]] == [
        full_candidates[0]["candidate_id"],
        full_candidates[1]["candidate_id"],
    ]
    pair_assets = {
        selected_counterpart["candidates"][0]["left_anchor"]["project_asset_id"],
        selected_counterpart["candidates"][0]["right_anchor"]["project_asset_id"],
    }
    assert pair_assets == {"asset-a", "asset-b"}

    selected_asset_pair_response = client.get(path, params={
        **query,
        "selected_project_asset_pair": ["asset-c", "asset-a"],
    })
    assert selected_asset_pair_response.status_code == 200
    selected_asset_pair = selected_asset_pair_response.json()[
        "data"]["candidate_page"]
    assert selected_asset_pair["total"] == 2
    assert selected_asset_pair["selected_project_asset_pair"] == [
        "asset-a", "asset-c"]
    assert selected_asset_pair["selection_semantics"] == (
        "all_eligible_cross_asset_pairs_for_selected_asset_pair_in_manifest_order")
    assert selected_asset_pair["candidate_set_id"] != full["candidate_set_id"]
    assert [item["candidate_id"] for item in selected_asset_pair["candidates"]] == [
        full_candidates[2]["candidate_id"], full_candidates[3]["candidate_id"]]
    assert all({item["left_anchor"]["project_asset_id"],
                item["right_anchor"]["project_asset_id"]} == {"asset-a", "asset-c"}
               for item in selected_asset_pair["candidates"])
    selected_asset_pair_tail_response = client.get(path, params={
        **query,
        "selected_project_asset_pair": ["asset-a", "asset-c"],
        "offset": 1,
        "limit": 1,
        "expected_candidate_set_id": selected_asset_pair["candidate_set_id"],
    })
    assert selected_asset_pair_tail_response.status_code == 200
    selected_asset_pair_tail = selected_asset_pair_tail_response.json()[
        "data"]["candidate_page"]
    assert selected_asset_pair_tail["total"] == 2
    assert selected_asset_pair_tail["candidate_set_id"] == (
        selected_asset_pair["candidate_set_id"])
    assert [item["candidate_id"] for item in selected_asset_pair_tail[
        "candidates"]] == [full_candidates[3]["candidate_id"]]
    assert selected_asset_pair_tail["has_more"] is False
    assert client.get(path, params={
        **query,
        "selected_project_asset_pair": ["asset-a", "asset-a"],
    }).status_code == 422
    assert client.get(path, params={
        **query,
        "selected_project_asset_pair": ["asset-a", "asset-unknown"],
    }).status_code == 422
    assert client.get(path, params={
        **query,
        "selected_project_asset_pair": ["asset-a"],
    }).status_code == 422
    assert client.get(path, params={
        **query,
        "selected_project_asset_pair": ["asset-a", "asset-b", "asset-c"],
    }).status_code == 422
    assert client.get(path, params={
        **query,
        "selected_project_asset_pair": ["asset-a", "asset-c"],
        "selected_anchor_story_graph_node_id": middle_anchor_id,
    }).status_code == 422

    selected_counterpart_tail_response = client.get(path, params={
        **query,
        "offset": 1,
        "limit": 1,
        "expected_candidate_set_id": selected_counterpart["candidate_set_id"],
        "selected_anchor_story_graph_node_id": middle_anchor_id,
        "selected_counterpart_project_asset_id": "asset-a",
    })
    assert selected_counterpart_tail_response.status_code == 200
    selected_counterpart_tail = selected_counterpart_tail_response.json()[
        "data"]["candidate_page"]
    assert selected_counterpart_tail["candidate_set_id"] == (
        selected_counterpart["candidate_set_id"])
    assert selected_counterpart_tail["total"] == 2
    assert [item["candidate_id"] for item in selected_counterpart_tail["candidates"]] == [
        full_candidates[1]["candidate_id"]]
    assert selected_counterpart_tail["has_more"] is False

    other_counterpart_response = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": middle_anchor_id,
        "selected_counterpart_project_asset_id": "asset-c",
    })
    assert other_counterpart_response.status_code == 200
    other_counterpart = other_counterpart_response.json()["data"]["candidate_page"]
    assert other_counterpart["candidate_set_id"] != (
        selected_counterpart["candidate_set_id"])
    assert [item["candidate_id"] for item in other_counterpart["candidates"]] == [
        full_candidates[4]["candidate_id"]]

    same_asset_response = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": middle_anchor_id,
        "selected_counterpart_project_asset_id": "asset-b",
    })
    assert same_asset_response.status_code == 422
    unknown_asset_response = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": middle_anchor_id,
        "selected_counterpart_project_asset_id": "asset-unknown",
    })
    assert unknown_asset_response.status_code == 422
    counterpart_without_anchor = client.get(path, params={
        **query,
        "selected_counterpart_project_asset_id": "asset-a",
    })
    assert counterpart_without_anchor.status_code == 422


def test_event_comparison_uses_persisted_source_event_fields(
    tmp_path, monkeypatch,
):
    project_id, graph_id = _seed_project(
        tmp_path / "event-db.sqlite", tmp_path, monkeypatch,
        include_event_evidence=True,
    )
    calls = []

    def verify_runtime(self):
        self._runtime_binding_verified = True

    def compare(self, left_frames, right_frames, **kwargs):
        calls.append(kwargs)
        return {
            "status": "OBSERVED",
            "assessment": "insufficient_evidence",
            "evidence_for": [],
            "evidence_against": [],
            "limitation": "synthetic fixtures; no event-quality claim",
            "confidence_type": "UNCALIBRATED_MODEL_ASSESSMENT",
            "review_state": "unreviewed",
        }

    monkeypatch.setattr(OllamaVLMAdapter, "verify_runtime_binding", verify_runtime)
    monkeypatch.setattr(OllamaVLMAdapter, "compare_cross_asset_frames", compare)
    client = TestClient(
        app,
        client=("127.0.0.1", 54324),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    body = {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "idempotency_key": "event-comparison-001",
        "relation_kind": "event_identity",
        "left_observation_id": "obs-a",
        "right_observation_id": "obs-b",
    }
    path = f"/v1/projects/{project_id}/story-link-comparisons"
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    comparison = response.json()["data"]["comparison"]
    assert comparison["prompt_version"] == "project_cross_asset_pair_v2"
    expected_left = {
        "action_type": "action",
        "scene_description": "A couple enters the ceremony hall.",
        "temporal_notes": "The couple enters from the rear doors.",
    }
    expected_right = {
        "action_type": "action",
        "scene_description": "Guests stand as the couple reaches the stage.",
        "temporal_notes": "The guests rise before the couple reaches the stage.",
    }
    assert comparison["left_event_evidence"] == expected_left
    assert comparison["right_event_evidence"] == expected_right
    assert calls == [{
        "relation_kind": "event_identity",
        "left_person_description": None,
        "right_person_description": None,
        "left_event_evidence": expected_left,
        "right_event_evidence": expected_right,
    }]
    replay = client.post(path, json=body)
    assert replay.status_code == 200
    assert replay.json()["data"]["reused"] is True
    assert replay.json()["data"]["comparison"] == comparison
    assert client.get(path + "/" + comparison["comparison_id"]).json()[
        "data"]["comparison"] == comparison
    assert len(calls) == 1
    assert response.json()["data"]["automatic_link_created"] is False
    assert response.json()["data"]["relation_inference_state"] == "EXPERIMENTAL"


def test_event_comparison_fails_closed_without_specific_stored_event_evidence(
    tmp_path, monkeypatch,
):
    project_id, graph_id = _seed_project(
        tmp_path / "event-missing-db.sqlite", tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        OllamaVLMAdapter,
        "compare_cross_asset_frames",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    client = TestClient(
        app,
        client=("127.0.0.1", 54325),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    response = client.post(
        f"/v1/projects/{project_id}/story-link-comparisons",
        json={
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "idempotency_key": "event-comparison-missing-evidence",
            "relation_kind": "event_identity",
            "left_observation_id": "obs-a",
            "right_observation_id": "obs-b",
        },
    )
    assert response.status_code == 422
    assert calls == []


def test_source_mention_review_is_persisted_and_changes_candidates_and_comparisons(
    tmp_path, monkeypatch,
):
    from director_brain.models import ProjectStoryGraph

    db_path = tmp_path / "mention-review.sqlite"
    project_id, graph_id = _seed_project(
        db_path, tmp_path, monkeypatch, include_event_evidence=True)
    repo = SqliteRepository(str(db_path))
    manifest = repo.latest_project_manifest(project_id)
    graph = repo.get(ProjectStoryGraph, graph_id)
    assert manifest is not None and graph is not None
    context = repo.get(FilmContextSnapshot, graph.context_id)
    assert context is not None
    observations = {
        evidence_ref: repo.get(FilmObservation, evidence_ref)
        for evidence_ref in context.evidence_refs
    }
    assert all(observation is not None for observation in observations.values())
    mentions = {}
    for asset_graph in graph.assets:
        for node in asset_graph.story_graph.nodes:
            if node.node_type.value in {"person_mention", "event_mention"}:
                kind = (
                    "person_identity" if node.node_type.value == "person_mention"
                    else "event_identity")
                anchor = node.attributes["source_anchor"]
                mentions[(kind, asset_graph.asset_id)] = {
                    "relation_kind": kind,
                    "anchor": {
                        "project_asset_id": asset_graph.asset_id,
                        "story_graph_node_id": node.node_id,
                        "observation_id": node.ref_id,
                        "source_content_hash": anchor["source_content_hash"],
                        "source_start": anchor["source_start"],
                        "source_end": anchor["source_end"],
                        "timebase": anchor["timebase"],
                        "timebase_unit": anchor["timebase_unit"],
                    },
                    "original": node.attributes,
                }
    repo.close()

    person_a = mentions[("person_identity", "asset-a")]
    person_b = mentions[("person_identity", "asset-b")]
    event_a = mentions[("event_identity", "asset-a")]
    event_b = mentions[("event_identity", "asset-b")]
    path = f"/v1/projects/{project_id}/story-mention-reviews"
    client = TestClient(
        app,
        client=("127.0.0.1", 54326),
        headers={"Authorization": "Bearer comparison-test-token"},
    )

    def decision(target, disposition, **fields):
        return {
            "relation_kind": target["relation_kind"],
            "anchor": target["anchor"],
            "decision": disposition,
            "rationale": (
                "Synthetic fixture review rationale."
                if disposition != "accepted_as_observed" else ""),
            **fields,
        }

    review_one = {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "expected_review_revision": 0,
        "idempotency_key": "mention-review-1",
        "decisions": [
            decision(
                person_a, "corrected_by_caller",
                corrected_person_description="elderly person in a tan coat"),
            decision(person_b, "accepted_as_observed"),
            decision(
                event_a, "corrected_by_caller",
                corrected_event_evidence={
                    "action_type": "arrival",
                    "scene_description": "A couple arrives at the ceremony hall.",
                    "temporal_notes": "They enter through the rear doors.",
                }),
            decision(event_b, "accepted_as_observed"),
        ],
    }
    written = client.put(path, json=review_one)
    assert written.status_code == 200, written.text
    assert written.json()["data"]["review_revision"] == 1
    assert written.json()["data"]["source_graph_mutated"] is False
    replay = client.put(path, json=review_one)
    assert replay.status_code == 200, replay.text
    assert replay.json()["data"]["reused"] is True
    conflicting_replay = {
        **review_one,
        "decisions": [
            decision(
                person_a, "corrected_by_caller",
                corrected_person_description="different caller content"),
            *review_one["decisions"][1:],
        ],
    }
    assert client.put(path, json=conflicting_replay).status_code == 409
    changed_anchor = {
        **review_one,
        "expected_review_revision": 1,
        "idempotency_key": "mention-review-changed-anchor",
        "decisions": [dict(item) for item in review_one["decisions"]],
    }
    changed_anchor["decisions"][0]["anchor"] = dict(person_a["anchor"])
    changed_anchor["decisions"][0]["anchor"]["source_content_hash"] = "0" * 64
    assert client.put(path, json=changed_anchor).status_code == 409
    readback = client.get(path)
    assert readback.status_code == 200
    assert readback.json()["data"]["source_hash_validation_state"] == (
        "not_revalidated_by_read")
    stored_decisions = readback.json()["data"]["review"]["decisions"]
    assert [item["decision"] for item in stored_decisions] == [
        item["decision"] for item in review_one["decisions"]]
    assert stored_decisions[0]["corrected_person_description"] == (
        "elderly person in a tan coat")
    assert stored_decisions[2]["corrected_event_evidence"]["scene_description"] == (
        "A couple arrives at the ceremony hall.")

    candidates_path = f"/v1/projects/{project_id}/story-link-candidates"
    person_page = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
    }).json()["data"]["candidate_page"]
    assert person_page["total"] == 1
    assert person_page["source_mention_review_revision"] == 1
    assert person_page["source_mention_review_state"] == "caller_asserted"
    candidate = person_page["candidates"][0]
    assert candidate["left_person_description"] == "elderly person in a tan coat"

    event_page = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "event_identity",
    }).json()["data"]["candidate_page"]
    event_candidate = event_page["candidates"][0]
    assert event_candidate["left_event_evidence"]["scene_description"] == (
        "A couple arrives at the ceremony hall.")

    compared_inputs = []

    def verify_runtime(self):
        self._runtime_binding_verified = True

    def compare(self, left_frames, right_frames, **kwargs):
        compared_inputs.append(kwargs)
        return {
            "status": "OBSERVED",
            "assessment": "insufficient_evidence",
            "evidence_for": [],
            "evidence_against": [],
            "limitation": "synthetic fixture; no identity-quality claim",
            "confidence_type": "UNCALIBRATED_MODEL_ASSESSMENT",
            "review_state": "unreviewed",
        }

    monkeypatch.setattr(OllamaVLMAdapter, "verify_runtime_binding", verify_runtime)
    monkeypatch.setattr(OllamaVLMAdapter, "compare_cross_asset_frames", compare)
    unbound_request = dict(candidate["comparison_request"])
    unbound_request.pop("candidate_id")
    unbound_request.pop("attempt_number")
    unbound_request.pop("left_story_graph_node_id")
    unbound_request.pop("right_story_graph_node_id")
    unbound_request["idempotency_key"] = "manual-unbound-after-review"
    unbound_request["source_mention_review_id"] = None
    unbound_request["source_mention_review_revision"] = 0
    unbound_request["left_person_description"] = (
        person_a["original"]["description"])
    unbound_response = client.post(
        f"/v1/projects/{project_id}/story-link-comparisons",
        json=unbound_request,
    )
    assert unbound_response.status_code == 409, unbound_response.text
    assert compared_inputs == []

    explicitly_bound_unbound_request = {
        **unbound_request,
        "idempotency_key": "manual-unbound-with-review-after-review",
        "source_mention_review_id": written.json()["data"]["review"]["review_id"],
        "source_mention_review_revision": 1,
    }
    explicitly_bound_unbound_response = client.post(
        f"/v1/projects/{project_id}/story-link-comparisons",
        json=explicitly_bound_unbound_request,
    )
    assert explicitly_bound_unbound_response.status_code == 409
    assert compared_inputs == []

    comparison_response = client.post(
        f"/v1/projects/{project_id}/story-link-comparisons",
        json=candidate["comparison_request"],
    )
    assert comparison_response.status_code == 200, comparison_response.text
    comparison = comparison_response.json()["data"]["comparison"]
    assert comparison["left_person_description"] == "elderly person in a tan coat"
    assert compared_inputs[0]["left_person_description"] == "elderly person in a tan coat"
    event_comparison_response = client.post(
        f"/v1/projects/{project_id}/story-link-comparisons",
        json=event_candidate["comparison_request"],
    )
    assert event_comparison_response.status_code == 200, event_comparison_response.text
    event_comparison = event_comparison_response.json()["data"]["comparison"]
    assert event_comparison["left_event_evidence"]["scene_description"] == (
        "A couple arrives at the ceremony hall.")
    assert compared_inputs[1]["left_event_evidence"]["scene_description"] == (
        "A couple arrives at the ceremony hall.")

    def review_anchor(anchor):
        return {
            "observation_id": anchor["observation_id"],
            "story_graph_node_id": anchor["story_graph_node_id"],
            "source_start": anchor["source_start"],
            "source_end": anchor["source_end"],
        }

    link_id = "person-link-bound-to-mention-review-1"
    link_review_path = f"/v1/projects/{project_id}/story-links"
    accepted_link_body = {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "expected_review_revision": 0,
        "idempotency_key": "mention-review-link-1",
        "links": [{
            "link_id": link_id,
            "entity_kind": "person_identity",
            "display_label": "caller-reviewed person",
            "source_comparison_ids": [comparison["comparison_id"]],
            "anchors": [
                review_anchor(comparison["left_anchor"]),
                review_anchor(comparison["right_anchor"]),
            ],
        }],
        "comparison_dispositions": [{
            "comparison_id": comparison["comparison_id"],
            "decision": "accepted_as_caller_asserted",
            "link_id": link_id,
            "rationale": "The caller reviewed the exact source pair.",
        }],
    }
    accepted_link = client.put(link_review_path, json=accepted_link_body)
    assert accepted_link.status_code == 200, accepted_link.text
    assert accepted_link.json()["data"]["currentness_state"] == "current"
    assert accepted_link.json()["data"]["active_link_count"] == 1
    link_review = ProjectStoryLinkReview.model_validate(
        accepted_link.json()["data"]["review"])
    assert link_review.source_mention_review_revision == 1
    assert link_review.source_mention_review_id == readback.json()["data"][
        "review"]["review_id"]
    from director_brain.project_story_link_review import (
        validate_project_story_link_review_for_plan,
    )
    active_mention_review = ProjectStoryMentionReview.model_validate(
        readback.json()["data"]["review"])
    validate_project_story_link_review_for_plan(
        link_review, manifest, context, graph,
        list(observations.values()), active_mention_review)
    from director_brain.models import DirectorDecisionPlan, EditorialDecisionList
    plan_snapshot = DirectorDecisionPlan.model_construct(
        project_id=project_id,
        project_story_graph_id=graph.graph_id,
        project_story_link_review_id=link_review.review_id,
        project_story_link_review_revision=link_review.review_revision,
    )
    review_repo = SqliteRepository(str(db_path))
    review_repo._validate_project_plan_story_links_unlocked(
        plan_snapshot,
        EditorialDecisionList.model_construct(ordered_edits=[]),
        manifest,
        context,
        graph,
        {key: value for key, value in observations.items() if value is not None},
    )
    assert api_main._project_plan_link_currentness(
        review_repo, plan_snapshot) == "current"
    review_repo.close()
    current_graph_view = client.get(
        f"/v1/projects/{project_id}/story-graph").json()["data"]["story_graph_view"]
    assert current_graph_view["cross_asset_relation_state"] == "caller_asserted"
    assert current_graph_view["cross_asset_link_count"] == 1

    linked_plan_response = client.post(
        f"/v1/projects/{project_id}/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "Preserve the caller-reviewed cross-asset identity evidence.",
        },
    )
    assert linked_plan_response.status_code == 200, linked_plan_response.text
    linked_plan_data = linked_plan_response.json()["data"]
    assert linked_plan_data["plan"]["project_story_link_review_id"] == (
        link_review.review_id)
    assert linked_plan_data["plan"]["project_story_link_review_revision"] == (
        link_review.review_revision)
    assert linked_plan_data["evidence_scope"]["project_story_mention_review_id"] == (
        active_mention_review.review_id)
    assert linked_plan_data["evidence_scope"][
        "project_story_mention_review_revision"] == active_mention_review.review_revision
    assert linked_plan_data["evidence_scope"][
        "project_story_mention_review_state"] == "caller_asserted"
    assert linked_plan_data["evidence_scope"]["project_story_link_review_state"] == (
        "caller_asserted")

    shadow_inputs = {}

    def stop_shadow_at_provider_boundary(*args, **kwargs):
        shadow_inputs.update(kwargs)
        raise api_main.LLMTransportError("synthetic provider-unavailable check")

    monkeypatch.setattr(
        api_main,
        "generate_project_shadow_strategy_options",
        stop_shadow_at_provider_boundary,
    )
    shadow_response = client.post(
        f"/v1/projects/{project_id}/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "Verify the current caller review reaches SHADOW safely.",
            "idempotency_key": "project-shadow-provider-failure-001",
        },
    )
    assert shadow_response.status_code == 503
    assert shadow_response.headers["X-Director-Brain-Failure-Code"] == (
        "provider_transport_error")
    assert shadow_inputs["project_story_mention_review"].review_id == (
        active_mention_review.review_id)
    assert shadow_inputs["project_story_link_review"].review_id == (
        link_review.review_id)

    superseding_link_body = {
        **accepted_link_body,
        "expected_review_revision": 1,
        "idempotency_key": "mention-review-link-2",
    }
    superseding_link = client.put(
        link_review_path, json=superseding_link_body)
    assert superseding_link.status_code == 200, superseding_link.text
    superseding_review = ProjectStoryLinkReview.model_validate(
        superseding_link.json()["data"]["review"])
    assert superseding_review.review_revision == 2
    assert superseding_link.json()["data"]["currentness_state"] == "current"
    superseded_replay = client.put(link_review_path, json=accepted_link_body)
    assert superseded_replay.status_code == 200, superseded_replay.text
    superseded_data = superseded_replay.json()["data"]
    assert superseded_data["reused"] is True
    assert superseded_data["currentness_state"] == "superseded"
    assert superseded_data["active_link_count"] == 1
    assert superseded_data["active_link_review_id"] == superseding_review.review_id
    assert superseded_data["active_link_review_revision"] == 2
    plan_snapshot = plan_snapshot.model_copy(update={
        "project_story_link_review_id": superseding_review.review_id,
        "project_story_link_review_revision": superseding_review.review_revision,
    })

    review_two = {
        **review_one,
        "expected_review_revision": 1,
        "idempotency_key": "mention-review-2",
        "decisions": [
            decision(person_a, "rejected"),
            decision(person_b, "accepted_as_observed"),
            *review_one["decisions"][2:],
        ],
    }
    second = client.put(path, json=review_two)
    assert second.status_code == 200, second.text
    assert second.json()["data"]["currentness_state"] == "current"
    stale_linked_plan = client.post(
        f"/v1/projects/{project_id}/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "Reject links tied to an older source-mention review.",
        },
    )
    assert stale_linked_plan.status_code == 409
    stale_candidate_page = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "expected_candidate_set_id": person_page["candidate_set_id"],
    })
    assert stale_candidate_page.status_code == 409
    assert stale_candidate_page.json()["detail"] == (
        "candidate set changed; restart pagination from the first page")
    updated_mention_review = ProjectStoryMentionReview.model_validate(
        client.get(path).json()["data"]["review"])
    superseded_mention_replay = client.put(path, json=review_one)
    assert superseded_mention_replay.status_code == 200, superseded_mention_replay.text
    superseded_mention_data = superseded_mention_replay.json()["data"]
    assert superseded_mention_data["reused"] is True
    assert superseded_mention_data["currentness_state"] == "superseded"
    assert superseded_mention_data["review"]["review_revision"] == 1
    assert superseded_mention_data["active_review_revision"] == 2
    assert superseded_mention_data["active_review_id"] == updated_mention_review.review_id
    stale_graph_view = client.get(
        f"/v1/projects/{project_id}/story-graph").json()["data"]["story_graph_view"]
    assert stale_graph_view["cross_asset_link_review"]["review_id"] == (
        superseding_review.review_id)
    assert stale_graph_view["view_id"] != current_graph_view["view_id"]
    assert stale_graph_view["source_mention_review_revision"] == 2
    assert stale_graph_view["link_snapshot_state"] == "stale"
    assert stale_graph_view["cross_asset_relation_state"] == "needs_reconfirmation"
    assert stale_graph_view["cross_asset_link_count"] == 0
    assert stale_graph_view["cross_asset_relations"] == []
    link_readback = client.get(link_review_path).json()["data"]
    assert link_readback["currentness_state"] == "stale"
    assert link_readback["source_hash_validation_state"] == (
        "not_revalidated_by_read")
    assert link_readback["active_link_count"] == 0
    assert link_readback["source_mention_review_revision"] == 2
    stale_idempotent_replay = client.put(
        link_review_path, json=accepted_link_body)
    assert stale_idempotent_replay.status_code == 200, stale_idempotent_replay.text
    stale_idempotent_data = stale_idempotent_replay.json()["data"]
    assert stale_idempotent_data["reused"] is True
    assert stale_idempotent_data["currentness_state"] == "stale"
    assert stale_idempotent_data["active_link_count"] == 0
    assert stale_idempotent_data["source_mention_review_id"] == (
        updated_mention_review.review_id)
    assert stale_idempotent_data["source_mention_review_revision"] == 2
    assert stale_idempotent_data["review"]["review_id"] == link_review.review_id
    assert stale_idempotent_data["review"]["review_revision"] == 1
    conflicting_stale_replay = {
        **accepted_link_body,
        "links": [{
            **accepted_link_body["links"][0],
            "display_label": "different request under the original key",
        }],
    }
    stale_replay_conflict = client.put(
        link_review_path, json=conflicting_stale_replay)
    assert stale_replay_conflict.status_code == 409
    assert stale_replay_conflict.json()["detail"] == (
        "story link review idempotency conflict")
    stale_status_repo = SqliteRepository(str(db_path))
    assert api_main._project_plan_link_currentness(
        stale_status_repo, plan_snapshot) == "stale"
    stale_status_repo.close()
    with pytest.raises(ValueError, match="stale source mention review"):
        validate_project_story_link_review_for_plan(
            link_review, manifest, context, graph,
            list(observations.values()), updated_mention_review)
    stale_review_repo = SqliteRepository(str(db_path))
    with pytest.raises(ValueError, match="stale source mention review"):
        stale_review_repo._validate_project_plan_story_links_unlocked(
            plan_snapshot,
            EditorialDecisionList.model_construct(ordered_edits=[]),
            manifest,
            context,
            graph,
            {key: value for key, value in observations.items() if value is not None},
        )
    stale_review_repo.close()

    stale_comparison_link = client.put(link_review_path, json={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "expected_review_revision": 2,
        "idempotency_key": "mention-review-link-stale-comparison",
        "links": [{
            "link_id": "stale-comparison-link",
            "entity_kind": "person_identity",
            "display_label": "stale comparison link",
            "source_comparison_ids": [comparison["comparison_id"]],
            "anchors": [
                review_anchor(comparison["left_anchor"]),
                review_anchor(comparison["right_anchor"]),
            ],
        }],
        "comparison_dispositions": [{
            "comparison_id": comparison["comparison_id"],
            "decision": "accepted_as_caller_asserted",
            "link_id": "stale-comparison-link",
            "rationale": "This cites the prior mention review and must fail.",
        }],
    })
    assert stale_comparison_link.status_code == 409

    unbound_rejected_anchors = [
        review_anchor(comparison["left_anchor"]),
        review_anchor(comparison["right_anchor"]),
    ]
    for anchor in unbound_rejected_anchors:
        anchor.pop("story_graph_node_id")
    rejected_mention_link = client.put(link_review_path, json={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "expected_review_revision": 2,
        "idempotency_key": "mention-review-link-rejected-anchor",
        "links": [{
            "link_id": "rejected-mention-link",
            "entity_kind": "person_identity",
            "display_label": "link containing a rejected mention",
            "anchors": unbound_rejected_anchors,
        }],
        "comparison_dispositions": [],
    })
    assert rejected_mention_link.status_code == 409

    rejected_page = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
    }).json()["data"]["candidate_page"]
    assert rejected_page["total"] == 0
    assert rejected_page["source_mention_review_revision"] == 2
    stale_comparison = client.post(
        f"/v1/projects/{project_id}/story-link-comparisons",
        json=candidate["comparison_request"],
    )
    assert stale_comparison.status_code == 409
    assert len(compared_inputs) == 2

    stale_write = client.put(path, json={
        **review_two,
        "expected_review_revision": 1,
        "idempotency_key": "stale-mention-review",
    })
    assert stale_write.status_code == 409

    review_three = {
        **review_two,
        "expected_review_revision": 2,
        "idempotency_key": "mention-review-3",
        "decisions": [
            decision(person_a, "accepted_as_observed"),
            decision(person_b, "accepted_as_observed"),
            *review_one["decisions"][2:],
        ],
    }
    third = client.put(path, json=review_three)
    assert third.status_code == 200, third.text
    resumed_page = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
    }).json()["data"]["candidate_page"]
    assert resumed_page["total"] == 1
    resumed_candidate = resumed_page["candidates"][0]
    assert resumed_candidate["candidate_id"] != candidate["candidate_id"]
    assert resumed_candidate["comparison_progress"]["comparison_run_state"] == (
        "not_started")
    historical = client.get(path).json()["data"]
    assert [item["review_revision"] for item in historical["review_history"]] == [1, 2, 3]

    restarted_repo = SqliteRepository(str(db_path))
    restarted_graph = restarted_repo.get(ProjectStoryGraph, graph_id)
    assert restarted_graph is not None
    raw_person_a = next(
        node for asset_graph in restarted_graph.assets
        if asset_graph.asset_id == "asset-a"
        for node in asset_graph.story_graph.nodes
        if node.node_type.value == "person_mention"
    )
    assert raw_person_a.attributes["description"] == "red coat, short hair"
    persisted_revisions = restarted_repo.list_project_story_mention_reviews(
        project_id, graph_id)
    assert [item.review_revision for item in persisted_revisions] == [1, 2, 3]
    restarted_repo.close()


def test_caller_asserted_place_links_are_revisioned_and_projected(tmp_path, monkeypatch):
    project_id, graph_id = _seed_project(
        tmp_path / "place-links.sqlite", tmp_path, monkeypatch)
    client = TestClient(
        app,
        client=("127.0.0.1", 54344),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    path = f"/v1/projects/{project_id}/story-links"
    body = {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "expected_review_revision": 0,
        "idempotency_key": "place-link-review-001",
        "links": [{
            "link_id": "place-overlook",
            "entity_kind": "place_identity",
            "display_label": "north coast overlook",
            "anchors": [
                {
                    "observation_id": "obs-a",
                    "source_start": 100_000,
                    "source_end": 500_000,
                },
                {
                    "observation_id": "obs-b",
                    "source_start": 200_000,
                    "source_end": 700_000,
                },
            ],
        }],
    }

    created = client.put(path, json=body)
    assert created.status_code == 200, created.text
    first_review = created.json()["data"]["review"]
    assert first_review["links"][0]["entity_kind"] == "place_identity"
    assert first_review["links"][0]["source_comparison_ids"] == []
    assert first_review["automatic_inference_state"] == "not_attempted"
    assert first_review["actor_identity_state"] == "caller_asserted"

    graph_view = client.get(
        f"/v1/projects/{project_id}/story-graph").json()["data"][
            "story_graph_view"]
    assert graph_view["cross_asset_relations"][0]["relation_kind"] == (
        "place_identity")
    assert graph_view["cross_asset_relations"][0]["assertion_state"] == (
        "caller_asserted")
    assert [member["node_binding_state"]
            for member in graph_view["cross_asset_relations"][0]["members"]] == [
        "observation_interval_only", "observation_interval_only"]

    corrected = client.put(path, json={
        **body,
        "expected_review_revision": 1,
        "idempotency_key": "place-link-review-002",
        "links": [{
            **body["links"][0],
            "display_label": "harbor overlook",
        }],
    })
    assert corrected.status_code == 200, corrected.text
    corrected_review = corrected.json()["data"]["review"]
    assert corrected_review["review_revision"] == 2
    assert corrected_review["links"][0]["display_label"] == "harbor overlook"
    assert corrected.json()["data"]["review_history_revisions"] == [1, 2]
    corrected_view = client.get(
        f"/v1/projects/{project_id}/story-graph").json()["data"][
            "story_graph_view"]
    assert corrected_view["cross_asset_relations"][0]["display_label"] == (
        "harbor overlook")

    child_code = r"""
import json, sys
from fastapi.testclient import TestClient
from api.main import app
args = json.loads(sys.argv[1])
with TestClient(app, client=("127.0.0.1", 54345), headers=args["headers"]) as client:
    graph = client.get(args["graph_path"])
    links = client.get(args["links_path"])
    print(json.dumps({"graph": graph.json()["data"], "links": links.json()["data"]}))
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, json.dumps({
            "headers": {"Authorization": "Bearer comparison-test-token"},
            "graph_path": f"/v1/projects/{project_id}/story-graph",
            "links_path": path,
        })],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ.copy(),
    )
    restarted = json.loads(child.stdout)
    assert restarted["graph"]["story_graph_view"] == corrected_view
    assert restarted["links"]["review"]["review_revision"] == 2
    assert restarted["links"]["review_history_revisions"] == [1, 2]
    assert restarted["links"]["source_hash_validation_state"] == (
        "not_revalidated_by_read")


def test_candidate_paging_counts_and_skips_duplicate_observation_pairs(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "duplicate-observation-candidates.sqlite"
    project_id, graph_id = _seed_project(
        db_path,
        tmp_path,
        monkeypatch,
        people_by_asset={
            "asset-a": ["Ada", "Grace"],
            "asset-b": ["Lin", "Turing"],
        },
    )

    from director_brain.models.project_story_graph import ProjectStoryGraph

    repo = SqliteRepository(str(db_path))
    graph = repo.get(ProjectStoryGraph, graph_id)
    assert graph is not None
    asset_graphs = []
    for asset_graph in graph.assets:
        if asset_graph.asset_id == "asset-b":
            source_graph = asset_graph.story_graph
            assert source_graph is not None
            first_person_node = next(
                node for node in source_graph.nodes
                if node.node_type.value == "person_mention"
            )
            nodes = [
                node.model_copy(update={"ref_id": "obs-a"})
                if node.node_id == first_person_node.node_id else node
                for node in source_graph.nodes
            ]
            asset_graph = asset_graph.model_copy(update={
                "story_graph": source_graph.model_copy(update={"nodes": nodes}),
            })
        asset_graphs.append(asset_graph)
    graph = ProjectStoryGraph.model_validate(graph.model_dump(mode="python") | {
        "assets": [item.model_dump(mode="python") for item in asset_graphs],
    })
    repo.save(graph)
    repo.close()

    client = TestClient(
        app,
        client=("127.0.0.1", 54346),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    path = f"/v1/projects/{project_id}/story-link-candidates"
    first_response = client.get(path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 0,
    })
    assert first_response.status_code == 200, first_response.text
    first_page = first_response.json()["data"]["candidate_page"]
    assert first_page["total"] == 2
    assert first_page["has_more"] is True
    first_candidate = first_page["candidates"][0]

    second_response = client.get(path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 1,
        "expected_candidate_set_id": first_page["candidate_set_id"],
    })
    assert second_response.status_code == 200, second_response.text
    second_page = second_response.json()["data"]["candidate_page"]
    assert second_page["total"] == 2
    assert second_page["has_more"] is False
    second_candidate = second_page["candidates"][0]
    assert second_candidate["candidate_id"] != first_candidate["candidate_id"]

    terminal_response = client.get(path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 2,
        "expected_candidate_set_id": first_page["candidate_set_id"],
    })
    assert terminal_response.status_code == 200, terminal_response.text
    terminal_page = terminal_response.json()["data"]["candidate_page"]
    assert terminal_page["candidates"] == []
    assert terminal_page["total"] == 2
    assert terminal_page["has_more"] is False

    pair_first_response = client.get(path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "selected_project_asset_pair": ["asset-b", "asset-a"],
        "limit": 1,
        "offset": 0,
    })
    assert pair_first_response.status_code == 200, pair_first_response.text
    pair_first_page = pair_first_response.json()["data"]["candidate_page"]
    assert pair_first_page["selected_project_asset_pair"] == [
        "asset-a", "asset-b"]
    assert pair_first_page["total"] == 2
    assert pair_first_page["candidates"][0]["candidate_id"] == (
        first_candidate["candidate_id"])

    pair_second_response = client.get(path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "selected_project_asset_pair": ["asset-a", "asset-b"],
        "limit": 1,
        "offset": 1,
        "expected_candidate_set_id": pair_first_page["candidate_set_id"],
    })
    assert pair_second_response.status_code == 200, pair_second_response.text
    pair_second_page = pair_second_response.json()["data"]["candidate_page"]
    assert pair_second_page["total"] == 2
    assert pair_second_page["candidates"][0]["candidate_id"] == (
        second_candidate["candidate_id"])
    assert pair_second_page["has_more"] is False


@pytest.mark.parametrize(
    ("unverified_order", "people_by_asset", "selected_pair", "counterpart_asset",
     "candidate_total"),
    [
        (0, None, ("asset-a", "asset-c"), "asset-c", 3),
        (1, None, ("asset-a", "asset-b"), "asset-b", 3),
        (None, {"asset-b": []}, ("asset-a", "asset-c"), "asset-c", 1),
    ],
)
def test_candidate_scopes_keep_manifest_positions_for_ineligible_assets(
    tmp_path, monkeypatch, unverified_order, people_by_asset, selected_pair,
    counterpart_asset, candidate_total,
):
    project_id, graph_id = _seed_project(
        tmp_path / "candidate-index-gap.sqlite",
        tmp_path,
        monkeypatch,
        include_third_asset=True,
        people_by_asset=people_by_asset,
        include_unverified_asset=unverified_order is not None,
        include_unverified_asset_order=unverified_order,
    )
    client = TestClient(
        app,
        client=("127.0.0.1", 54325),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    path = f"/v1/projects/{project_id}/story-link-candidates"
    base = client.get(path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 20,
        "offset": 0,
    })
    assert base.status_code == 200, base.text
    base_page = base.json()["data"]["candidate_page"]
    candidates = base_page["candidates"]
    assert base_page["total"] == candidate_total
    assert len(candidates) == candidate_total

    asset_pair = client.get(path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "selected_project_asset_pair": list(selected_pair),
        "limit": 20,
        "offset": 0,
    })
    assert asset_pair.status_code == 200, asset_pair.text
    pair_page = asset_pair.json()["data"]["candidate_page"]
    assert pair_page["total"] == 1
    assert len(pair_page["candidates"]) == 1
    assert {
        pair_page["candidates"][0]["left_anchor"]["project_asset_id"],
        pair_page["candidates"][0]["right_anchor"]["project_asset_id"],
    } == set(selected_pair)

    anchor = candidates[0]["left_anchor"]
    assert anchor["project_asset_id"] == "asset-a"
    counterpart = client.get(path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "selected_anchor_story_graph_node_id": anchor["story_graph_node_id"],
        "selected_counterpart_project_asset_id": counterpart_asset,
        "limit": 20,
        "offset": 0,
    })
    assert counterpart.status_code == 200, counterpart.text
    counterpart_page = counterpart.json()["data"]["candidate_page"]
    assert counterpart_page["total"] == 1
    assert len(counterpart_page["candidates"]) == 1
    assert {
        counterpart_page["candidates"][0]["left_anchor"]["project_asset_id"],
        counterpart_page["candidates"][0]["right_anchor"]["project_asset_id"],
    } == {"asset-a", counterpart_asset}


@pytest.mark.parametrize(
    ("failure_type", "expected_failure_code"),
    [
        ("RATE_LIMIT", "provider_rate_limited"),
        ("REQUEST", "provider_request_rejected"),
    ],
)
def test_provider_failure_category_survives_candidate_readback(
    tmp_path, monkeypatch, failure_type, expected_failure_code,
):
    project_id, graph_id = _seed_project(
        tmp_path / "provider-failure.sqlite",
        tmp_path,
        monkeypatch,
        people_by_asset={
            "asset-a": ["red coat, short hair"],
            "asset-b": ["red coat, short hair"],
        },
    )
    provider_calls = []

    def verify_runtime(self):
        self._runtime_binding_verified = True

    def compare(self, left_frames, right_frames, **kwargs):
        provider_calls.append((len(left_frames), len(right_frames)))
        return {
            "status": "FAILED",
            "failure_type": failure_type,
            "diagnostic": "private provider response must not be persisted",
        }

    monkeypatch.setattr(OllamaVLMAdapter, "verify_runtime_binding", verify_runtime)
    monkeypatch.setattr(OllamaVLMAdapter, "compare_cross_asset_frames", compare)
    client = TestClient(
        app,
        client=("127.0.0.1", 54326),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    candidates_path = f"/v1/projects/{project_id}/story-link-candidates"
    page_response = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 0,
    })
    assert page_response.status_code == 200, page_response.text
    page = page_response.json()["data"]["candidate_page"]
    comparison_path = f"/v1/projects/{project_id}/story-link-comparisons"
    recorded = client.post(
        comparison_path,
        json=page["candidates"][0]["comparison_request"],
    )
    assert recorded.status_code == 200, recorded.text
    response_body = recorded.json()
    assert response_body["data"]["comparison"]["failure_code"] == (
        expected_failure_code)
    assert "private provider response" not in json.dumps(response_body)
    assert provider_calls == [(3, 3)]

    readback = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 0,
        "expected_candidate_set_id": page["candidate_set_id"],
    })
    assert readback.status_code == 200, readback.text
    progress = readback.json()["data"]["candidate_page"]["candidates"][0][
        "comparison_progress"]
    assert progress["attempts"][0]["failure_code"] == expected_failure_code
    assert progress["retry_comparison_request"]["attempt_number"] == 2
    assert "private provider response" not in json.dumps(readback.json())


def test_local_sampling_oserror_survives_candidate_readback(
    tmp_path, monkeypatch,
):
    project_id, graph_id = _seed_project(
        tmp_path / "local-sampling-failure.sqlite",
        tmp_path,
        monkeypatch,
        people_by_asset={
            "asset-a": ["red coat, short hair"],
            "asset-b": ["red coat, short hair"],
        },
    )
    provider_calls = []

    def verify_runtime(self):
        self._runtime_binding_verified = True

    def compare(self, left_frames, right_frames, **kwargs):
        provider_calls.append((len(left_frames), len(right_frames)))
        return {"status": "OBSERVED", "relation_probability": 0.9}

    class FailingKeyframeContext:
        def __enter__(self):
            raise OSError("private local sampler detail")

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(OllamaVLMAdapter, "verify_runtime_binding", verify_runtime)
    monkeypatch.setattr(OllamaVLMAdapter, "compare_cross_asset_frames", compare)
    monkeypatch.setattr(
        keyframe_module,
        "extract_keyframes",
        lambda *args, **kwargs: FailingKeyframeContext(),
    )
    client = TestClient(
        app,
        client=("127.0.0.1", 54327),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    candidates_path = f"/v1/projects/{project_id}/story-link-candidates"
    page_response = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 0,
    })
    assert page_response.status_code == 200, page_response.text
    page = page_response.json()["data"]["candidate_page"]
    comparison_path = f"/v1/projects/{project_id}/story-link-comparisons"
    recorded = client.post(
        comparison_path,
        json=page["candidates"][0]["comparison_request"],
    )
    assert recorded.status_code == 200, recorded.text
    response_body = recorded.json()
    assert response_body["data"]["comparison"]["failure_code"] == (
        "sampling_failed")
    assert "private local sampler detail" not in json.dumps(response_body)
    assert provider_calls == []

    readback = client.get(candidates_path, params={
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "offset": 0,
        "expected_candidate_set_id": page["candidate_set_id"],
    })
    assert readback.status_code == 200, readback.text
    progress = readback.json()["data"]["candidate_page"]["candidates"][0][
        "comparison_progress"]
    assert progress["attempts"][0]["failure_code"] == "sampling_failed"
    assert progress["retry_comparison_request"]["attempt_number"] == 2
    assert "private local sampler detail" not in json.dumps(readback.json())

def test_opt_in_shadow_ranking_is_anchor_scoped_exhaustive_and_paginates(
    tmp_path, monkeypatch
):
    from director_brain.project_story_link_ranking import (
        ProjectStoryLinkRankingResult,
        ranked_candidate_set_id,
    )

    project_id, graph_id = _seed_project(
        tmp_path / "ranked-candidate-db.sqlite",
        tmp_path,
        monkeypatch,
        people_by_asset={
            "asset-a": ["anchor"],
            "asset-b": ["same description", "unrelated description"],
        },
    )
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    client = TestClient(
        app,
        client=("127.0.0.1", 54324),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    path = f"/v1/projects/{project_id}/story-link-candidates"
    query = {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "relation_kind": "person_identity",
        "limit": 1,
        "ranking_profile": "bge_m3_shadow_v1",
    }
    missing_anchor = client.get(path, params=query)
    assert missing_anchor.status_code == 422
    assert missing_anchor.json()["detail"] == (
        "shadow ranking requires selected_anchor_story_graph_node_id")

    unranked_response = client.get(path, params={
        **{key: value for key, value in query.items()
           if key != "ranking_profile"},
        "limit": 2,
    })
    assert unranked_response.status_code == 200, unranked_response.text
    unranked_page = unranked_response.json()["data"]["candidate_page"]
    anchor_node_id = unranked_page["candidates"][0]["left_anchor"][
        "story_graph_node_id"]
    digest = "d" * 64
    fake_calls = []

    def fake_binding(base_url):
        assert base_url == "http://127.0.0.1:11434"
        return type("Binding", (), {"digest": digest, "dimensions": 2})()

    def fake_rank(candidates, *, base_url, binding):
        fake_calls.append((len(candidates), base_url, binding.digest))
        score_by_description = {
            "same description": 0.91,
            "unrelated description": 0.12,
        }
        ordered = sorted(
            candidates,
            key=lambda item: (
                -score_by_description[item.right_person_description],
                item.candidate_id,
            ),
        )
        ranked = [
            item.model_copy(update={
                "ranking_state": "shadow_ranked_unadmitted",
                "ranking_position": position,
                "ranking_score": score_by_description[item.right_person_description],
            })
            for position, item in enumerate(ordered, start=1)
        ]
        return ProjectStoryLinkRankingResult(
            candidates=ranked,
            model_digest=digest,
            provider_invocation_count=2,
            embedding_invocation_count=1,
        )

    monkeypatch.setattr(
        api_main, "resolve_project_story_link_ranking_binding", fake_binding)
    monkeypatch.setattr(api_main, "rank_project_story_link_candidates", fake_rank)
    ranked_first_response = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": anchor_node_id,
        "offset": 0,
    })
    assert ranked_first_response.status_code == 200, ranked_first_response.text
    ranked_first_data = ranked_first_response.json()["data"]
    ranked_first = ranked_first_data["candidate_page"]
    assert ranked_first_data["provider_invocation_count"] == 2
    assert ranked_first_data["embedding_invocation_count"] == 1
    assert ranked_first_data["automatic_link_created"] is False
    assert ranked_first_data["identity_decision_state"] == "not_attempted"
    assert ranked_first_data["quality_acceptance"] == "not_proven"
    assert ranked_first["ranking_state"] == "shadow_ranked_unadmitted"
    assert ranked_first["ranking_profile_id"] == "bge_m3_shadow_v1"
    assert ranked_first["ranking_model_digest"] == digest
    assert ranked_first["ranking_score_semantics"] == "uncalibrated_similarity_only"
    assert ranked_first["total"] == 2
    assert ranked_first["candidates"][0]["right_person_description"] == (
        "same description")
    assert ranked_first["candidates"][0]["ranking_position"] == 1
    assert ranked_first["candidates"][0]["ranking_score"] == 0.91
    assert ranked_first["candidates"][0]["automatic_inference_state"] == (
        "not_attempted")
    anchor_unranked_response = client.get(path, params={
        **{key: value for key, value in query.items()
           if key != "ranking_profile"},
        "selected_anchor_story_graph_node_id": anchor_node_id,
        "offset": 0,
    })
    assert anchor_unranked_response.status_code == 200
    anchor_unranked_page = anchor_unranked_response.json()["data"]["candidate_page"]
    from storage.sqlite_repository import SqliteRepository

    restarted_repo = SqliteRepository(str(tmp_path / "ranked-candidate-db.sqlite"))
    try:
        persisted_snapshot = restarted_repo.get_project_story_link_ranking_snapshot(
            project_id,
            anchor_unranked_page["candidate_set_id"],
            "bge_m3_shadow_v1",
            "bge-m3:latest",
            digest,
            "cosine_similarity",
        )
    finally:
        restarted_repo.close()
    assert persisted_snapshot is not None
    assert len(persisted_snapshot) == 2
    expected_ranked_set_id = ranked_candidate_set_id(
        anchor_unranked_page["candidate_set_id"], digest)
    assert ranked_first["candidate_set_id"] == expected_ranked_set_id

    ranked_second_response = client.get(path, params={
        **query,
        "selected_anchor_story_graph_node_id": anchor_node_id,
        "offset": 1,
        "expected_candidate_set_id": expected_ranked_set_id,
    })
    assert ranked_second_response.status_code == 200, ranked_second_response.text
    ranked_second = ranked_second_response.json()["data"]["candidate_page"]
    assert ranked_second["candidate_set_id"] == expected_ranked_set_id
    assert ranked_second["candidates"][0]["right_person_description"] == (
        "unrelated description")
    assert ranked_second["candidates"][0]["ranking_position"] == 2
    assert ranked_second["has_more"] is False
    ranked_second_data = ranked_second_response.json()["data"]
    assert ranked_second_data["provider_invocation_count"] == 1
    assert ranked_second_data["embedding_invocation_count"] == 0
    assert len(fake_calls) == 1


@pytest.mark.parametrize("review_kind", ["mention", "link"])
def test_project_story_review_rechecks_source_hash_before_persisting(
    tmp_path, monkeypatch, review_kind,
):
    """Reject review writes when a registered source drifts during the request."""
    db_path = tmp_path / f"{review_kind}-source-drift.sqlite"
    project_id, graph_id = _seed_project(
        db_path, tmp_path, monkeypatch, include_event_evidence=True)
    repo = SqliteRepository(str(db_path))
    manifest = repo.latest_project_manifest(project_id)
    graph = repo.get(ProjectStoryGraph, graph_id)
    assert manifest is not None and graph is not None

    person_mentions = []
    for asset_graph in graph.assets:
        if asset_graph.story_graph is None:
            continue
        for node in asset_graph.story_graph.nodes:
            if node.node_type.value != "person_mention":
                continue
            source_anchor = node.attributes["source_anchor"]
            person_mentions.append({
                "project_asset_id": asset_graph.asset_id,
                "story_graph_node_id": node.node_id,
                "observation_id": node.ref_id,
                "source_content_hash": source_anchor["source_content_hash"],
                "source_start": source_anchor["source_start"],
                "source_end": source_anchor["source_end"],
                "timebase": source_anchor["timebase"],
                "timebase_unit": source_anchor["timebase_unit"],
            })
    repo.close()
    assert len(person_mentions) >= 2

    if review_kind == "mention":
        body = {
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "expected_review_revision": 0,
            "idempotency_key": "mention-source-drift-before-write",
            "decisions": [{
                "relation_kind": "person_identity",
                "anchor": person_mentions[0],
                "decision": "rejected",
                "rationale": "Reject this exact source-local mention.",
            }],
        }
        path = f"/v1/projects/{project_id}/story-mention-reviews"
    else:
        body = {
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "expected_review_revision": 0,
            "idempotency_key": "link-source-drift-before-write",
            "links": [{
                "link_id": "source-drift-person-link",
                "entity_kind": "person_identity",
                "display_label": "caller-reviewed person",
                "anchors": [{
                    key: mention[key]
                    for key in (
                        "observation_id", "story_graph_node_id",
                        "source_start", "source_end",
                    )
                } for mention in person_mentions[:2]],
            }],
            "comparison_dispositions": [],
        }
        path = f"/v1/projects/{project_id}/story-links"

    source_path = tmp_path / "comparison-a.mp4"
    original_source_check = api_main._require_current_project_sources
    source_check_count = 0

    def drift_after_initial_check(current_manifest):
        nonlocal source_check_count
        source_check_count += 1
        if source_check_count == 2:
            source_path.write_bytes(source_path.read_bytes() + b"source-drift")
        return original_source_check(current_manifest)

    monkeypatch.setattr(
        api_main, "_require_current_project_sources", drift_after_initial_check)
    client = TestClient(
        app,
        client=("127.0.0.1", 54327),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    response = client.put(path, json=body)
    assert response.status_code == 409, response.text
    assert source_check_count == 2

    repo = SqliteRepository(str(db_path))
    if review_kind == "mention":
        persisted = repo.list_project_story_mention_reviews(project_id, graph_id)
    else:
        persisted = repo.list_project_story_link_reviews(project_id, graph_id)
    repo.close()
    assert persisted == []


@pytest.mark.parametrize("relation_kind", ["person_identity", "event_identity"])
def test_candidate_pair_preview_returns_both_current_unranked_source_sides(
    relation_kind, tmp_path, monkeypatch,
):
    project_id, graph_id = _seed_project(
        tmp_path / "candidate-preview.sqlite",
        tmp_path,
        monkeypatch,
        include_event_evidence=True,
        people_by_asset={
            "asset-a": ["red coat, short hair"],
            "asset-b": ["blue jacket, long hair"],
        },
    )
    monkeypatch.setattr(
        OllamaVLMAdapter,
        "compare_cross_asset_frames",
        lambda *args, **kwargs: pytest.fail("preview must not call a model"),
    )
    client = TestClient(
        app,
        client=("127.0.0.1", 54328),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    candidate_page = client.get(
        f"/v1/projects/{project_id}/story-link-candidates",
        params={
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "relation_kind": relation_kind,
            "limit": 10,
            "offset": 0,
        },
    )
    assert candidate_page.status_code == 200, candidate_page.text
    candidate = candidate_page.json()["data"]["candidate_page"]["candidates"][0]
    preview_path = (
        f"/v1/projects/{project_id}/story-graph/{graph_id}/link-candidates/"
        f"{candidate['candidate_id']}/preview"
    )
    params = {
        "manifest_revision": 1,
        "relation_kind": relation_kind,
        "left_story_graph_node_id": (
            candidate["left_anchor"]["story_graph_node_id"]),
        "right_story_graph_node_id": (
            candidate["right_anchor"]["story_graph_node_id"]),
    }

    anonymous = TestClient(app, client=("127.0.0.1", 54329))
    assert anonymous.get(preview_path, params=params).status_code == 401
    response = client.get(preview_path, params=params)
    assert response.status_code == 200, response.text
    assert "no-store" in response.headers["cache-control"]
    preview = response.json()["data"]
    assert preview["candidate_id"] == candidate["candidate_id"]
    assert preview["source_hash_validation_state"] == "verified_before_and_after"
    assert preview["source_time_alignment_state"] == "not_assumed"
    assert preview["left"]["anchor"]["story_graph_node_id"] == (
        candidate["left_anchor"]["story_graph_node_id"])
    assert preview["right"]["anchor"]["story_graph_node_id"] == (
        candidate["right_anchor"]["story_graph_node_id"])
    assert preview["left"]["anchor"]["project_asset_id"] != (
        preview["right"]["anchor"]["project_asset_id"])
    assert all(
        side["source_hash_validation_state"] == "verified_before_and_after"
        and [frame["relative_position"] for frame in side["frames"]]
        == [0.15, 0.5, 0.85]
        for side in (preview["left"], preview["right"])
    )
    import base64
    import hashlib
    for side in (preview["left"], preview["right"]):
        for frame in side["frames"]:
            jpeg = base64.b64decode(frame["jpeg_base64"], validate=True)
            assert jpeg.startswith(b"\xff\xd8\xff")
            assert hashlib.sha256(jpeg).hexdigest() == frame["sha256"]
            assert max(frame["width"], frame["height"]) <= 1280
    assert str(tmp_path / "comparison-a.mp4") not in response.text
    assert str(tmp_path / "comparison-b.mp4") not in response.text

    stale = client.get(
        preview_path.replace(candidate["candidate_id"], "stale-candidate"),
        params=params,
    )
    assert stale.status_code == 409


def test_candidate_pair_preview_rechecks_both_sources_after_extraction(
    tmp_path, monkeypatch,
):
    project_id, graph_id = _seed_project(
        tmp_path / "candidate-preview-drift.sqlite",
        tmp_path,
        monkeypatch,
    )
    client = TestClient(
        app,
        client=("127.0.0.1", 54330),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    page = client.get(
        f"/v1/projects/{project_id}/story-link-candidates",
        params={
            "manifest_revision": 1,
            "story_graph_id": graph_id,
            "relation_kind": "person_identity",
            "limit": 10,
            "offset": 0,
        },
    )
    assert page.status_code == 200, page.text
    candidate = page.json()["data"]["candidate_page"]["candidates"][0]
    path = (
        f"/v1/projects/{project_id}/story-graph/{graph_id}/link-candidates/"
        f"{candidate['candidate_id']}/preview"
    )
    real_builder = api_main._build_project_story_mention_preview

    def drift_first_source_after_its_preview(**kwargs):
        preview = real_builder(**kwargs)
        if preview.anchor.project_asset_id == "asset-a":
            source = tmp_path / "comparison-a.mp4"
            source.write_bytes(source.read_bytes() + b"drift-during-pair-preview")
        return preview

    monkeypatch.setattr(
        api_main,
        "_build_project_story_mention_preview",
        drift_first_source_after_its_preview,
    )
    response = client.get(path, params={
        "manifest_revision": 1,
        "relation_kind": "person_identity",
        "left_story_graph_node_id": (
            candidate["left_anchor"]["story_graph_node_id"]),
        "right_story_graph_node_id": (
            candidate["right_anchor"]["story_graph_node_id"]),
    })
    assert response.status_code == 409, response.text


def test_caller_asserted_story_links_are_isolated_between_projects(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "two-project-link-isolation.sqlite"
    project_a, graph_a = _seed_project(
        db_path,
        tmp_path,
        monkeypatch,
        project_id="link-isolation-alpha",
        media_prefix="alpha-",
        observation_prefix="alpha-",
    )
    project_b, graph_b = _seed_project(
        db_path,
        tmp_path,
        monkeypatch,
        project_id="link-isolation-beta",
        media_prefix="beta-",
        observation_prefix="beta-",
    )
    assert graph_a != graph_b

    client = TestClient(
        app,
        client=("127.0.0.1", 54331),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    candidate_response = client.get(
        f"/v1/projects/{project_a}/story-link-candidates",
        params={
            "manifest_revision": 1,
            "story_graph_id": graph_a,
            "relation_kind": "person_identity",
            "limit": 1,
            "offset": 0,
        },
    )
    assert candidate_response.status_code == 200, candidate_response.text
    candidate_page = candidate_response.json()["data"]["candidate_page"]
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
    review_response = client.put(
        f"/v1/projects/{project_a}/story-links",
        json={
            "manifest_revision": 1,
            "story_graph_id": graph_a,
            "expected_review_revision": 0,
            "idempotency_key": "alpha-caller-link-v1",
            "links": [{
                "link_id": "alpha-private-person-link",
                "entity_kind": "person_identity",
                "display_label": "alpha caller-reviewed person",
                "anchors": anchors,
            }],
        },
    )
    assert review_response.status_code == 200, review_response.text
    saved_review = review_response.json()["data"]["review"]
    assert saved_review["project_id"] == project_a

    alpha_read = client.get(f"/v1/projects/{project_a}/story-links")
    beta_read = client.get(f"/v1/projects/{project_b}/story-links")
    assert alpha_read.status_code == beta_read.status_code == 200
    assert alpha_read.json()["data"]["review"] == saved_review
    beta_data = beta_read.json()["data"]
    assert beta_data["review"] is None
    assert beta_data["review_history_revisions"] == []
    assert beta_data["active_link_count"] == 0

    beta_graph = client.get(f"/v1/projects/{project_b}/story-graph")
    assert beta_graph.status_code == 200, beta_graph.text
    beta_view = beta_graph.json()["data"]["story_graph_view"]
    assert beta_view["source_graph"]["graph_id"] == graph_b
    assert beta_view["cross_asset_link_review"] is None
    assert beta_view["cross_asset_relations"] == []

    foreign_candidate_request = client.get(
        f"/v1/projects/{project_b}/story-link-candidates",
        params={
            "manifest_revision": 1,
            "story_graph_id": graph_a,
            "relation_kind": "person_identity",
            "limit": 1,
            "offset": 0,
        },
    )
    assert foreign_candidate_request.status_code == 409


def test_story_link_put_rejects_interval_anchor_assigned_to_two_groups(
    tmp_path, monkeypatch,
):
    project_id, graph_id = _seed_project(
        tmp_path / "duplicate-interval-links.sqlite", tmp_path, monkeypatch)
    client = TestClient(
        app,
        client=("127.0.0.1", 54346),
        headers={"Authorization": "Bearer comparison-test-token"},
    )
    path = f"/v1/projects/{project_id}/story-links"
    shared_anchor = {
        "observation_id": "obs-a",
        "source_start": 100_000,
        "source_end": 500_000,
    }
    body = {
        "manifest_revision": 1,
        "story_graph_id": graph_id,
        "expected_review_revision": 0,
        "idempotency_key": "duplicate-interval-anchor-groups",
        "links": [
            {
                "link_id": "place-group-a",
                "entity_kind": "place_identity",
                "display_label": "place one",
                "anchors": [
                    shared_anchor,
                    {
                        "observation_id": "obs-b",
                        "source_start": 100_000,
                        "source_end": 500_000,
                    },
                ],
            },
            {
                "link_id": "place-group-b",
                "entity_kind": "place_identity",
                "display_label": "place two",
                "anchors": [
                    shared_anchor,
                    {
                        "observation_id": "obs-b",
                        "source_start": 600_000,
                        "source_end": 900_000,
                    },
                ],
            },
        ],
    }

    rejected = client.put(path, json=body)
    assert rejected.status_code == 409, rejected.text
    readback = client.get(path)
    assert readback.status_code == 200, readback.text
    assert readback.json()["data"]["review"] is None
    assert readback.json()["data"]["review_history_revisions"] == []


def test_project_shadow_same_key_concurrent_failure_runs_once(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    project_id, _graph_id = _seed_project(
        tmp_path / "shadow-concurrent.sqlite3", tmp_path, monkeypatch,
        include_event_evidence=True,
    )
    call_lock = Lock()
    first_call_entered = Event()
    second_call_entered = Event()
    second_request_submitted = Event()
    release_failure = Event()
    generator_calls = 0

    def blocked_failure(*args, **kwargs):
        nonlocal generator_calls
        with call_lock:
            generator_calls += 1
            call_number = generator_calls
        if call_number == 1:
            first_call_entered.set()
        else:
            second_call_entered.set()
        if not release_failure.wait(timeout=5):
            raise AssertionError("test did not release the blocked generator")
        from director_brain.llm_adapter import LLMStructuredOutputError

        raise LLMStructuredOutputError(
            "SYNTHETIC_CONCURRENT_FAILURE_DETAIL",
            failure_code="segment_emotions",
        )

    monkeypatch.setattr(
        api_main, "generate_project_shadow_strategy_options", blocked_failure)
    path = f"/v1/projects/{project_id}/director-plans:compare-shadow-strategies"
    payload = {
        "manifest_revision": 1,
        "target_duration_us": 1_600_000,
        "intent_text": "Synthetic concurrent idempotency check.",
        "idempotency_key": "shadow-concurrent-001",
    }

    with (
        TestClient(
            app,
            client=("127.0.0.1", 54340),
            headers={"Authorization": "Bearer comparison-test-token"},
        ) as first_client,
        TestClient(
            app,
            client=("127.0.0.1", 54341),
            headers={"Authorization": "Bearer comparison-test-token"},
        ) as second_client,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(first_client.post, path, json=payload)
        assert first_call_entered.wait(timeout=5)

        def send_second_request():
            second_request_submitted.set()
            return second_client.post(path, json=payload)

        second = pool.submit(send_second_request)
        assert second_request_submitted.wait(timeout=5)
        duplicate_generator_started = second_call_entered.wait(timeout=1)
        release_failure.set()
        first_response = first.result(timeout=10)
        second_response = second.result(timeout=10)

        assert not duplicate_generator_started
        assert generator_calls == 1
        assert first_response.status_code == 502
        assert second_response.status_code == 502
        assert first_response.headers["X-Director-Brain-Failure-Code"] == (
            "segment_emotions")
        assert second_response.headers["X-Director-Brain-Failure-Code"] == (
            "segment_emotions")
        assert "SYNTHETIC_CONCURRENT_FAILURE_DETAIL" not in first_response.text
        assert "SYNTHETIC_CONCURRENT_FAILURE_DETAIL" not in second_response.text

        ledger = first_client.get(f"/v1/projects/{project_id}/decision-ledger")
        assert ledger.status_code == 200, ledger.text
        failures = [
            item for item in ledger.json()["data"]["entries"]
            if item["action"] == "director_strategy_comparison_failed"
        ]
        assert len(failures) == 1
        assert failures[0]["detail"]["failure_code"] == (
            "segment_emotions")


def test_project_shadow_same_key_cross_process_failure_runs_once(
    tmp_path, monkeypatch,
):
    context = multiprocessing.get_context("spawn")
    db_path = tmp_path / "shadow-cross-process.sqlite3"
    project_id, _graph_id = _seed_project(
        db_path, tmp_path, monkeypatch, include_event_evidence=True,
    )
    payload = {
        "manifest_revision": 1,
        "target_duration_us": 1_600_000,
        "intent_text": "Synthetic cross-process idempotency check.",
        "idempotency_key": "shadow-cross-process-001",
    }
    first_request_started = context.Event()
    first_generator_entered = context.Event()
    second_request_started = context.Event()
    second_generator_entered = context.Event()
    release_generator = context.Event()
    result_queue = context.Queue()
    first = context.Process(
        target=_run_shadow_failure_request_in_process,
        args=(
            str(db_path), project_id, payload, first_request_started,
            first_generator_entered, release_generator, result_queue,
        ),
    )
    second = context.Process(
        target=_run_shadow_failure_request_in_process,
        args=(
            str(db_path), project_id, payload, second_request_started,
            second_generator_entered, release_generator, result_queue,
        ),
    )
    first.start()
    try:
        assert first_request_started.wait(timeout=10)
        assert first_generator_entered.wait(timeout=10)
        second.start()
        assert second_request_started.wait(timeout=10)
        assert not second_generator_entered.wait(timeout=1)
    finally:
        release_generator.set()
        first.join(timeout=20)
        if second.pid is not None:
            second.join(timeout=20)
        for process in (first, second):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert first_generator_entered.is_set()
    assert not second_generator_entered.is_set()
    outcomes = [result_queue.get(timeout=3), result_queue.get(timeout=3)]
    expected_outcome = {
        "status_code": 502,
        "failure_code": "segment_emotions",
        "leaked_detail": False,
    }
    assert len(outcomes) == 2
    assert all(outcome == expected_outcome for outcome in outcomes)

    with TestClient(
        app,
        client=("127.0.0.1", 54343),
        headers={"Authorization": "Bearer comparison-test-token"},
    ) as client:
        ledger = client.get(f"/v1/projects/{project_id}/decision-ledger")
    assert ledger.status_code == 200, ledger.text
    failures = [
        item for item in ledger.json()["data"]["entries"]
        if item["action"] == "director_strategy_comparison_failed"
    ]
    assert len(failures) == 1
    assert failures[0]["detail"]["failure_code"] == "segment_emotions"


def test_project_shadow_run_readback_reports_running_and_unknown(
    tmp_path, monkeypatch,
):
    from storage.project_director_shadow_runs import (
        SqliteProjectDirectorShadowRunStore,
    )

    db_path = tmp_path / "shadow-run-readback.sqlite3"
    project_id, _graph_id = _seed_project(
        db_path, tmp_path, monkeypatch, include_event_evidence=True,
    )
    payload = {
        "manifest_revision": 1,
        "target_duration_us": 1_600_000,
        "intent_text": "Synthetic run-state readback check.",
        "idempotency_key": "shadow-readback-001",
    }
    request_model = api_main.ProjectShadowStrategyComparisonRequest(**payload)
    comparison_id = api_main._project_director_shadow_comparison_id(
        project_id, payload["idempotency_key"],
    )
    fingerprint = api_main._project_director_shadow_request_fingerprint(
        project_id, request_model,
    )
    store = SqliteProjectDirectorShadowRunStore(str(db_path))
    readback_path = (
        f"/v1/projects/{project_id}/director-plan-shadow-comparisons/"
        f"{comparison_id}"
    )
    compare_path = (
        f"/v1/projects/{project_id}/director-plans:compare-shadow-strategies"
    )

    with TestClient(
        app,
        client=("127.0.0.1", 54344),
        headers={"Authorization": "Bearer comparison-test-token"},
    ) as client:
        with store.acquire(
            comparison_id=comparison_id,
            project_id=project_id,
            request_fingerprint=fingerprint,
        ) as claim:
            claim.mark_provider_started()
            running = client.get(readback_path)
            assert running.status_code == 202, running.text
            assert running.json()["data"]["run_state"] == "running"
            assert running.headers["Cache-Control"] == "no-store"
            assert running.headers["Retry-After"] == "2"
            assert running.headers["X-Director-Brain-Run-State"] == "running"

        unknown = client.get(readback_path)
        assert unknown.status_code == 409, unknown.text
        assert unknown.json()["data"]["run_state"] == "outcome_unknown"
        assert unknown.json()["data"]["retry_action"] == (
            "new_idempotency_key_required")
        assert unknown.headers["X-Director-Brain-Run-State"] == (
            "outcome_unknown")

        monkeypatch.setattr(
            api_main,
            "generate_project_shadow_strategy_options",
            lambda *args, **kwargs: pytest.fail(
                "uncertain same-key replay must not call the Reasoner"),
        )
        replay = client.post(compare_path, json=payload)
        assert replay.status_code == 409, replay.text
        assert "uncertain prior outcome" in replay.json()["detail"]
        assert comparison_id in replay.json()["detail"]
        assert replay.headers["X-Director-Brain-Run-State"] == (
            "outcome_unknown")

        openapi = client.get("/openapi.json").json()
        compare_responses = openapi["paths"][
            "/v1/projects/{project_id}/director-plans:compare-shadow-strategies"
        ]["post"]["responses"]
        readback_responses = openapi["paths"][
            "/v1/projects/{project_id}/director-plan-shadow-comparisons/"
            "{comparison_id}"
        ]["get"]["responses"]
        assert "409" in compare_responses
        assert "202" in readback_responses
        assert "409" in readback_responses
        assert compare_responses["409"]["headers"][
            "X-Director-Brain-Run-State"]["schema"]["enum"] == [
                "running", "recoverable", "outcome_unknown",
                "idempotency_conflict", "ownership_lost",
            ]
        assert readback_responses["202"]["headers"][
            "X-Director-Brain-Run-State"]["schema"]["enum"] == [
                "running", "recoverable",
            ]
        assert readback_responses["409"]["headers"][
            "X-Director-Brain-Run-State"]["schema"]["enum"] == [
                "outcome_unknown",
            ]


def test_project_shadow_distinct_keys_are_not_globally_serialized(
    tmp_path, monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, BrokenBarrierError, Lock

    project_id, _graph_id = _seed_project(
        tmp_path / "shadow-distinct-keys.sqlite3", tmp_path, monkeypatch,
        include_event_evidence=True,
    )
    generator_barrier = Barrier(2, timeout=5)
    start_barrier = Barrier(3, timeout=5)
    guard = Lock()
    generator_calls = 0
    concurrent_calls_passed = []

    def concurrent_failure(*args, **kwargs):
        nonlocal generator_calls
        with guard:
            generator_calls += 1
        try:
            generator_barrier.wait()
            passed = True
        except BrokenBarrierError:
            passed = False
        with guard:
            concurrent_calls_passed.append(passed)
        from director_brain.llm_adapter import LLMStructuredOutputError

        raise LLMStructuredOutputError(
            "SYNTHETIC_DISTINCT_KEY_FAILURE",
            failure_code="project_strategy_schema_invalid",
        )

    monkeypatch.setattr(
        api_main, "generate_project_shadow_strategy_options",
        concurrent_failure,
    )
    path = f"/v1/projects/{project_id}/director-plans:compare-shadow-strategies"
    payload = {
        "manifest_revision": 1,
        "target_duration_us": 1_600_000,
        "intent_text": "Synthetic independent-key concurrency check.",
    }

    with (
        TestClient(
            app,
            client=("127.0.0.1", 54342),
            headers={"Authorization": "Bearer comparison-test-token"},
        ) as first_client,
        TestClient(
            app,
            client=("127.0.0.1", 54343),
            headers={"Authorization": "Bearer comparison-test-token"},
        ) as second_client,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        def post(client, key):
            start_barrier.wait()
            return client.post(
                path,
                json={**payload, "idempotency_key": key},
            )

        first = pool.submit(post, first_client, "shadow-distinct-key-a")
        second = pool.submit(post, second_client, "shadow-distinct-key-b")
        start_barrier.wait()
        first_response = first.result(timeout=10)
        second_response = second.result(timeout=10)

        assert generator_calls == 2
        assert concurrent_calls_passed == [True, True]
        assert first_response.status_code == 502
        assert second_response.status_code == 502
        ledger = first_client.get(f"/v1/projects/{project_id}/decision-ledger")
        assert ledger.status_code == 200, ledger.text
        failures = [
            item for item in ledger.json()["data"]["entries"]
            if item["action"] == "director_strategy_comparison_failed"
        ]
        assert len(failures) == 2
        assert len({item["decision_id"] for item in failures}) == 2
