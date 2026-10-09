"""REST API 全端点集成测试（方案 §6 · 8 端点）。

S6 复验发现 api/main.py 无任何测试覆盖（四支柱之 REST API 此前仅验证
"可 import"）。本文件补齐：真实 FastAPI TestClient + ffmpeg testsrc
真实视频 + 合法 FilmObservation，逐端点验证 200/400/404 行为与信封结构。

覆盖：
1.  POST /v1/film-context:snapshot   asset 层 → 200 快照结构
2.  同请求重发 → cache_hit=True（同 context_id 复用）
3.  POST layer=project（带 intent）→ 200 project 层
4.  POST layer=scene → 200（管线编译 brief+graph+plan）
5.  POST layer=evidence 无 reason → 400（审计留痕强制）
6.  POST layer=evidence 带 shot_ids+reason → 200
7.  POST layer=evidence 用 window_us 定位 → 200
8.  POST 非法 layer → 400
9.  GET  /v1/film-context/{id} → 200；未知 id → 404；evidence 层缺 reason → 400
10. POST /v1/briefs:compile → 200
11. POST /v1/director-plans:generate → 200 含 plan/edl/hash
12. POST /v1/director-plans/{id}:confirm-strategy → 200 STRATEGY_CONFIRMED
13. POST /v1/director-plans/{id}:validate → 200
14. POST /v1/revisions:propose → 200 DRAFT
15. GET  /v1/projects/{id}/decision-ledger → 200 信封
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from director_brain.models import ClaimKind, FilmObservation, TimebaseUnit

client = TestClient(app)


def test_strategy_selected_audio_is_opt_in_for_shadow_or_bound_llm_plan():
    inputs = {
        "manifest_revision": 1,
        "target_duration_us": 5_000_000,
        "intent_text": "Compare two editorial approaches.",
        "audio_style": "strategy",
    }
    comparison = api_main.ProjectShadowStrategyComparisonRequest(
        **inputs, idempotency_key="typed-shadow-audio-001")
    assert comparison.audio_style == "strategy"

    with pytest.raises(ValueError, match="only for LLM strategy Plans"):
        api_main.ProjectPlanRequest(**inputs)

    selected = api_main.ProjectPlanRequest(
        **inputs,
        reasoner_strategy="llm",
        selected_strategy_hypothesis_id="candidate-a",
        selected_candidate_binding_digest="0" * 64,
        selected_comparison_id="comparison-a",
    )
    assert selected.audio_style == "strategy"


def test_strategy_selected_pacing_is_opt_in_for_shadow_or_bound_llm_plan():
    inputs = {
        "manifest_revision": 1,
        "target_duration_us": 5_000_000,
        "intent_text": "Compare supported pacing approaches.",
        "pacing_style": "strategy",
    }
    comparison = api_main.ProjectShadowStrategyComparisonRequest(
        **inputs, idempotency_key="typed-shadow-pacing-001")
    assert comparison.pacing_style == "strategy"

    with pytest.raises(ValueError, match="only for LLM strategy Plans"):
        api_main.ProjectPlanRequest(**inputs)

    selected = api_main.ProjectPlanRequest(
        **inputs,
        reasoner_strategy="llm",
        selected_strategy_hypothesis_id="candidate-a",
        selected_candidate_binding_digest="0" * 64,
        selected_comparison_id="comparison-a",
    )
    assert selected.pacing_style == "strategy"


def test_strategy_selected_transition_policy_requires_bound_llm_plan():
    from types import SimpleNamespace

    inputs = {
        "manifest_revision": 1,
        "target_duration_us": 5_000_000,
        "intent_text": "Compare evidence-bound transition choices.",
        "transition_policy": "strategy",
    }
    comparison = api_main.ProjectShadowStrategyComparisonRequest(
        **inputs, idempotency_key="typed-shadow-transition-001")
    assert comparison.transition_policy == "strategy"

    with pytest.raises(ValueError, match="only for LLM strategy Plans"):
        api_main.ProjectPlanRequest(**inputs)

    selected = api_main.ProjectPlanRequest(
        **inputs,
        reasoner_strategy="llm",
        selected_strategy_hypothesis_id="candidate-a",
        selected_candidate_binding_digest="0" * 64,
        selected_comparison_id="comparison-a",
    )
    assert selected.transition_policy == "strategy"

    identity_args = {
        "project_id": "project-01",
        "manifest": SimpleNamespace(manifest_id="manifest-01", revision=1),
        "context": SimpleNamespace(context_id="context-01"),
        "graph": SimpleNamespace(graph_id="graph-01"),
        "active_link_review": None,
        "active_mention_review": None,
        "brief": SimpleNamespace(producer="brief-compiler"),
    }
    request_fields = {
        "target_duration_us": 5_000_000,
        "intent_text": "Compare evidence-bound transition choices.",
        "voice_led": False,
        "audio_style": "none",
        "pacing_style": "brief",
    }
    legacy_identity = api_main._project_director_candidate_request_identity(
        **identity_args,
        req=SimpleNamespace(**request_fields, transition_policy="none"),
    )
    strategy_identity = api_main._project_director_candidate_request_identity(
        **identity_args,
        req=SimpleNamespace(**request_fields, transition_policy="strategy"),
    )
    assert "transition_policy" not in legacy_identity
    assert strategy_identity["transition_policy"] == "strategy"


# ---------------------------------------------------------------------------
# 真实测试视频 + 合法观测
# ---------------------------------------------------------------------------

def _make_test_video(path: str, duration_sec: int = 6) -> None:
    cmd = [
        "ffmpeg", "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=320x240:rate=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path, "-y",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def _make_obs(obs_id: str, asset_id: str, start_us: int, end_us: int) -> FilmObservation:
    return FilmObservation(
        observation_id=obs_id,
        media_asset_id=asset_id,
        media_hash="hash_" + asset_id,
        start_frame=start_us,
        end_frame=end_us,
        timebase=1_000_000,
        timebase_unit=TimebaseUnit.MICROSECONDS,
        observation_type="deterministic_technical",
        claim=json.dumps({
            "blur_score": 12.5, "exposure_ok": True,
            "brightness_mean": 120, "shake_score": 0.05,
        }),
        provider="deterministic_opencv",
        model_version="v1",
        prompt_version="n/a",
        confidence=1.0,
        review_state="final",
        claim_kind=ClaimKind.MEASURED,
        project_id="unknown",
        created_at=1_700_000_000,
        producer="test",
        source_ref="api_test.mp4",
    )


def _make_fake_project_shadow_comparison(
    brief, manifest, context, graph, observations, **kwargs,
):
    """Build a typed synthetic comparison for local lifecycle/API coverage."""
    from director_brain.models.director_plan import (
        ProjectDirectorShadowComparison,
        ProjectDirectorSourceSelectionAuditEntry,
        ProjectNarrativeEvidenceClaim,
        ProjectNarrativeEvidenceRef,
        ProjectNarrativeReasoningTrace,
        ProjectNarrativeShotHypothesis,
        ProjectNarrativeSourceRationale,
        ProjectNarrativeConstraintAssessment,
        ProjectNarrativeStrategyHypothesis,
        ProjectDirectorStrategyCandidate,
    )
    from director_brain.intent_constraints import unresolved_constraint_review_items

    baseline_edl, baseline_plan = api_main.generate_project_plan(
        brief,
        manifest,
        context,
        graph,
        observations,
        strategy="heuristic",
        voice_led=kwargs.get("voice_led", False),
        audio_style=kwargs.get("audio_style", "none"),
        project_story_link_review=kwargs.get("project_story_link_review"),
        project_story_mention_review=kwargs.get("project_story_mention_review"),
    )
    refs = []
    for edit in baseline_edl.ordered_edits:
        source = next((
            item for item in observations
            if item.observation_id in edit.source_observation_refs
            and item.project_asset_id == edit.project_asset_id
            and item.media_hash.lower() == edit.source_media_hash.lower()
        ), None)
        if source is None:
            continue
        ref = ProjectNarrativeEvidenceRef(
            project_asset_id=edit.project_asset_id,
            source_media_hash=edit.source_media_hash,
            source_asset_id=edit.source_asset_id,
            observation_id=source.observation_id,
        )
        identity = (ref.project_asset_id, ref.source_media_hash.lower(), ref.source_asset_id)
        if identity not in {
            (item.project_asset_id, item.source_media_hash.lower(), item.source_asset_id)
            for item in refs
        }:
            refs.append(ref)
    if len(refs) < 2:
        raise AssertionError("API lifecycle fixture must select at least two sources")

    def claim(statement, source):
        return ProjectNarrativeEvidenceClaim(
            statement=statement, source_refs=[source])

    open_constraints = unresolved_constraint_review_items(brief)
    strategies = []
    for hypothesis_id, order in (("A", refs), ("B", list(reversed(refs)))):
        hypothesis = ProjectNarrativeStrategyHypothesis(
            hypothesis_id=hypothesis_id,
            label=f"synthetic option {hypothesis_id}",
            editorial_intent="An unverified synthetic strategy.",
            emotional_arc=claim("Synthetic emotional arc hypothesis.", order[0]),
            ordered_sources=order,
            source_rationales=[
                ProjectNarrativeSourceRationale(
                    focus_source=source,
                    disposition="include",
                    statement="Synthetic source inclusion.",
                    source_refs=[source],
                )
                for source in order
            ],
            tradeoffs=[claim("Synthetic tradeoff.", order[0])],
            uncertainties=[claim("Synthetic uncertainty.", order[-1])],
            constraint_assessments=[
                ProjectNarrativeConstraintAssessment(
                    constraint_kind=item["kind"],
                    brief_index=item["brief_index"],
                    constraint_text=item["text"],
                    assessment=(
                        "candidate_supported" if hypothesis_id == "A"
                        else "candidate_conflicted"),
                    statement="Synthetic assessment; not quality evidence.",
                    source_refs=[order[0]],
                    candidate_edl_source_asset_alignment=(
                        "all_cited_assets_selected"),
                )
                for item in open_constraints
            ],
        )
        strategies.append(hypothesis)
    candidates = []
    for hypothesis in strategies:
        hypothesis_id = hypothesis.hypothesis_id
        order = hypothesis.ordered_sources
        edl = baseline_edl.model_copy(deep=True, update={
            "edl_id": f"{baseline_edl.edl_id}_{hypothesis_id}",
            "producer": "test_project_reasoner_shadow",
        })
        plan = baseline_plan.model_copy(deep=True, update={
            "plan_id": f"{baseline_plan.plan_id}_{hypothesis_id}",
            "edl_id": edl.edl_id,
            "producer": "test_project_reasoner_shadow",
            "state": "draft",
            "project_narrative_reasoning": None,
            "constraints": list(baseline_plan.constraints)
            + ["director_reasoner_shadow_candidate=not_confirmable"],
        })
        trace = ProjectNarrativeReasoningTrace(
            brief_id=brief.brief_id,
            model="synthetic-test-only",
            prompt_version="synthetic-test-only",
            temperature=0,
            story_arc="Synthetic unverified project arc.",
            input_evidence_refs=refs,
            emotional_hypotheses=[
                ProjectNarrativeShotHypothesis(
                    source=source, label="synthetic unverified label")
                for source in refs
            ],
            suggested_order=order,
            strategy_hypotheses=strategies,
            active_strategy_hypothesis_id=hypothesis_id,
            limitations=["Synthetic fixture; no Director Quality evidence."],
        )
        plan.project_narrative_reasoning = trace
        selected = {
            (item.project_asset_id, item.source_media_hash.lower(), item.source_asset_id)
            for item in refs
        }
        selection_audit = [
            ProjectDirectorSourceSelectionAuditEntry(
                strategy_rank=rank,
                strategy_disposition="include",
                selected_in_edl=(
                    (source.project_asset_id, source.source_media_hash.lower(),
                     source.source_asset_id) in selected
                ),
                reason_codes=["present_in_candidate_edl"],
            )
            for rank, source in enumerate(order)
        ]
        candidates.append(ProjectDirectorStrategyCandidate(
            hypothesis=hypothesis,
            candidate_binding_digest="0" * 64,
            edl=edl,
            plan=plan,
            selection_audit=selection_audit,
            sequence_changed_vs_heuristic=False,
            distinct_edl_sequence_from_baseline=False,
            distinct_edl_sequence_from_other_candidates=False,
        ))
    return ProjectDirectorShadowComparison(
        baseline_edl=baseline_edl,
        baseline_plan=baseline_plan,
        strategy_candidates=candidates,
    )


def _observations() -> list[dict]:
    obs = [
        _make_obs("obs_001", "shot_a", 0, 3_000_000),
        _make_obs("obs_002", "shot_b", 3_000_000, 6_000_000),
    ]
    return [json.loads(o.model_dump_json()) for o in obs]


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("api_videos") / "api_test.mp4")
    _make_test_video(path, duration_sec=6)
    return path


def _snapshot_request(video_path: str, **overrides) -> dict:
    body = {
        "project_id": "proj_api",
        "video_path": video_path,
        "layer": "asset",
        "observations_json": _observations(),
    }
    body.update(overrides)
    return body


def _envelope_data(resp) -> dict:
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["correlation_id"].startswith("corr_")
    assert "schema_version" in body and "timestamp" in body
    return body["data"]


def _fresh_process_project_readback(
    *, db_path: Path, media_root: Path, token: str, project_id: str,
    plan_id: str, confirmation_request: dict, read_project_links: bool = False,
) -> dict:
    """Read persisted project state through a newly imported API process."""
    child_code = r"""
import json
import sys
from fastapi.testclient import TestClient
from api.main import app

payload = json.loads(sys.argv[1])
headers = {"Authorization": "Bearer " + payload["token"]}
with TestClient(app, client=("127.0.0.1", 54127), headers=headers) as client:
    plan_path = (
        f"/v1/projects/{payload['project_id']}/director-plans/"
        f"{payload['plan_id']}"
    )
    readback = client.get(plan_path)
    replay = client.post(
        plan_path + ":confirm-strategy",
        json=payload["confirmation_request"],
    )
    ledger = client.get(
        f"/v1/projects/{payload['project_id']}/decision-ledger"
    )
    result = {
        "readback_status": readback.status_code,
        "readback": readback.json().get("data"),
        "replay_status": replay.status_code,
        "replay": replay.json().get("data"),
        "ledger_status": ledger.status_code,
        "ledger": ledger.json().get("data"),
    }
    if payload.get("read_project_links"):
        link_review = client.get(
            f"/v1/projects/{payload['project_id']}/story-links"
        )
        story_graph = client.get(
            f"/v1/projects/{payload['project_id']}/story-graph"
        )
        result.update({
            "link_review_status": link_review.status_code,
            "link_review": link_review.json().get("data"),
            "story_graph_status": story_graph.status_code,
            "story_graph": story_graph.json().get("data"),
        })
print("P1_FRESH_PROCESS_READBACK=" + json.dumps(result, separators=(",", ":")))
"""
    payload = {
        "token": token,
        "project_id": project_id,
        "plan_id": plan_id,
        "confirmation_request": confirmation_request,
        "read_project_links": read_project_links,
    }
    env = os.environ.copy()
    env["SQLITE_PATH"] = str(db_path)
    env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(media_root)
    env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = token
    result = subprocess.run(
        [sys.executable, "-c", child_code, json.dumps(payload)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    marker = "P1_FRESH_PROCESS_READBACK="
    line = next(
        (item for item in result.stdout.splitlines() if item.startswith(marker)),
        None,
    )
    assert line is not None, result.stdout
    return json.loads(line[len(marker):])


# ---------------------------------------------------------------------------
# film-context:snapshot（POST）
# ---------------------------------------------------------------------------

def test_snapshot_asset_layer(video_path):
    data = _envelope_data(client.post("/v1/film-context:snapshot",
                                      json=_snapshot_request(video_path)))
    assert data["context_id"].startswith("ctx_asset_")
    assert data["snapshot"]["layers"] == ["asset"]
    assert data["coverage"] == "asset_level"
    assert len(data["evidence_refs"]) == 2
    assert data["snapshot"]["sampling_config"]["observation_count"] == 2


def test_snapshot_cache_hit_on_repeat(video_path):
    # 独立观测集隔离 context_id（asset 层 id 不含 project_id）
    own_obs = [
        json.loads(o.model_dump_json()) for o in [
            _make_obs("obs_cache_1", "shot_c1", 0, 3_000_000),
            _make_obs("obs_cache_2", "shot_c2", 3_000_000, 6_000_000),
        ]
    ]
    body = _snapshot_request(video_path, observations_json=own_obs)
    first = _envelope_data(client.post("/v1/film-context:snapshot", json=body))
    second = _envelope_data(client.post("/v1/film-context:snapshot", json=body))
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["context_id"] == first["context_id"]


def test_snapshot_project_layer(video_path):
    data = _envelope_data(client.post("/v1/film-context:snapshot", json=_snapshot_request(
        video_path, layer="project", intent_text="快剪风格")))
    assert data["snapshot"]["layers"] == ["project"]
    assert data["snapshot"]["sampling_config"]["brief_id"]
    assert data["coverage"] == "project_summary"


def test_snapshot_scene_layer(video_path):
    data = _envelope_data(client.post("/v1/film-context:snapshot", json=_snapshot_request(
        video_path, layer="scene")))
    assert data["snapshot"]["layers"] == ["scene"]
    assert data["snapshot"]["sampling_config"]["act_count"] == 4


def test_snapshot_evidence_requires_reason(video_path):
    resp = client.post("/v1/film-context:snapshot", json=_snapshot_request(
        video_path, layer="evidence", shot_ids=["shot_a"]))
    assert resp.status_code == 400
    assert "reason" in resp.json()["detail"]


def test_snapshot_evidence_with_shot_ids(video_path):
    data = _envelope_data(client.post("/v1/film-context:snapshot", json=_snapshot_request(
        video_path, layer="evidence", shot_ids=["shot_a"],
        reason="歧义镜头复核")))
    assert data["snapshot"]["layers"] == ["evidence"]
    assert data["snapshot"]["sampling_config"]["expansion_reason"] == "歧义镜头复核"
    assert data["snapshot"]["sampling_config"]["target_shot_ids"] == ["shot_a"]


def test_snapshot_evidence_window_mapping(video_path):
    data = _envelope_data(client.post("/v1/film-context:snapshot", json=_snapshot_request(
        video_path, layer="evidence", window_us=[2_500_000, 4_000_000],
        reason="时间窗复核")))
    assert data["snapshot"]["sampling_config"]["target_shot_ids"] == ["shot_a", "shot_b"]


def test_window_us_rejects_unknown_or_frame_timebases(video_path):
    frame_observation = _make_obs(
        "obs_frames", "shot_frames", 0, 75
    ).model_copy(update={
        "timebase": 25,
        "timebase_unit": TimebaseUnit.FRAMES,
    })
    response = client.post(
        "/v1/film-context:snapshot",
        json=_snapshot_request(
            video_path,
            layer="evidence",
            observations_json=[json.loads(frame_observation.model_dump_json())],
            window_us=[0, 1_000_000],
            reason="timebase validation",
        ),
    )
    assert response.status_code == 422
    assert "declare microseconds" in response.json()["detail"]


def test_window_us_rejects_invalid_bounds(video_path):
    response = client.post(
        "/v1/film-context:snapshot",
        json=_snapshot_request(
            video_path,
            layer="evidence",
            window_us=[4_000_000, 2_500_000],
            reason="invalid window",
        ),
    )
    assert response.status_code == 422
    assert "increasing" in response.json()["detail"]


def test_invalid_window_is_rejected_before_local_analysis(video_path, monkeypatch):
    from observation_service import pipeline

    def forbidden(*_args, **_kwargs):
        raise AssertionError("malformed request must not start media analysis")

    monkeypatch.setattr(pipeline, "analyze_media", forbidden)
    response = client.post(
        "/v1/film-context:snapshot",
        json=_snapshot_request(
            video_path,
            layer="evidence",
            observations_json=None,
            window_us=[0],
            reason="malformed window must short-circuit",
        ),
    )

    assert response.status_code == 422
    assert "exactly" in response.json()["detail"]


def test_snapshot_invalid_layer(video_path):
    resp = client.post("/v1/film-context:snapshot", json=_snapshot_request(
        video_path, layer="galaxy"))
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# film-context/{context_id}（GET）
# ---------------------------------------------------------------------------

def test_get_film_context_roundtrip(video_path):
    data = _envelope_data(client.post("/v1/film-context:snapshot",
                                      json=_snapshot_request(video_path)))
    got = _envelope_data(client.get(
        f"/v1/film-context/{data['context_id']}"))
    assert got["snapshot"]["context_id"] == data["context_id"]


def test_get_film_context_unknown_id():
    resp = client.get("/v1/film-context/ctx_never_built")
    assert resp.status_code == 404


def test_get_film_context_evidence_requires_reason(video_path):
    data = _envelope_data(client.post("/v1/film-context:snapshot", json=_snapshot_request(
        video_path, layer="evidence", shot_ids=["shot_a"], reason="审计留痕")))
    resp = client.get(f"/v1/film-context/{data['context_id']}?layer=evidence")
    assert resp.status_code == 400
    ok = _envelope_data(client.get(
        f"/v1/film-context/{data['context_id']}?layer=evidence&reason=复查"))
    assert ok["requested_layer"] == "evidence"


# ---------------------------------------------------------------------------
# 原有 6 端点（S3 建成、此前零测试——本组补齐）
# ---------------------------------------------------------------------------

def test_compile_brief_endpoint(video_path):
    data = _envelope_data(client.post("/v1/briefs:compile", json=_snapshot_request(
        video_path, intent_text="快剪风格，避免模糊镜头")))
    assert data["brief_id"]
    # DirectorBrief.intent 是分类标记；原文被规则提取到语义字段
    # （source_text 保留给 ASR 转写，本测试无语音 → no_speech_detected）
    assert data["brief"]["intent"] == "user_provided"
    assert data["brief"]["editing_language"] == "fast_cut"


def test_generate_plan_endpoint(video_path):
    data = _envelope_data(client.post("/v1/director-plans:generate", json=_snapshot_request(
        video_path, intent_text="快剪风格")))
    assert data["plan_hash"] and data["edl_hash"]
    # plan↔edl 双绑定字段（Plan 状态机/策略确认共用）
    assert data["plan"]["brief_version"] == data["edl"]["brief_version"]


def _plan_and_edl(video_path) -> tuple[dict, dict]:
    data = _envelope_data(client.post("/v1/director-plans:generate", json=_snapshot_request(
        video_path, intent_text="快剪风格")))
    return data["plan"], data["edl"]


def test_confirm_strategy_endpoint(video_path, tmp_path, monkeypatch):
    db_path = tmp_path / "single-source-confirmation.db"
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "ledger-test-token")
    plan, edl = _plan_and_edl(video_path)
    # 候选③执法：确认前 plan 须先到 ready_for_strategy_confirmation
    # （生成端点产出的 plan 处于 draft，直接确认会被 409 拒绝）
    pre = client.post(
        f"/v1/director-plans/{plan['plan_id']}:confirm-strategy",
        json={"plan_id": plan["plan_id"], "edl_id": edl["edl_id"],
              "plan_json": plan, "edl_json": edl, "confirmed_by": "owner"})
    assert pre.status_code == 409
    plan["state"] = "ready_for_strategy_confirmation"
    data = _envelope_data(client.post(
        f"/v1/director-plans/{plan['plan_id']}:confirm-strategy",
        json={"plan_id": plan["plan_id"], "edl_id": edl["edl_id"],
              "plan_json": plan, "edl_json": edl, "confirmed_by": "owner"}))
    assert data["state"] == "dispatch_eligible"
    assert data["confirmation"]["plan_hash"]
    ledger_client = TestClient(
        app, client=("127.0.0.1", 54132),
        headers={"Authorization": "Bearer ledger-test-token"},
    )
    ledger = _envelope_data(ledger_client.get(
        f"/v1/projects/{plan['project_id']}/decision-ledger"))
    assert len(ledger["entries"]) == 1
    assert ledger["entries"][0]["project_id"] == plan["project_id"]
    assert ledger["entries"][0]["decision_id"] == plan["plan_id"]


def test_validate_endpoint(video_path):
    plan, edl = _plan_and_edl(video_path)
    data = _envelope_data(client.post(
        f"/v1/director-plans/{plan['plan_id']}:validate",
        json={"plan_json": plan, "edl_json": edl,
              "observations_json": _observations()}))
    assert "valid" in data and "errors" in data


def test_propose_revision_endpoint(video_path):
    """候选⑦：端点产出真实修订提案（不再硬编码 DRAFT）。"""
    plan, edl = _plan_and_edl(video_path)
    obs = _observations()
    data = _envelope_data(client.post("/v1/revisions:propose", json={
        "finding_ids": ["fql_001"],
        "revision_type": "adjust_duration",
        "change_summary": "黑帧段调整",
        "plan_json": plan,
        "edl_json": edl,
        "observations_json": obs,
    }))
    assert data["status"] == "pending"
    assert data["revision_type"] == "adjust_duration"
    assert data["source_finding_ids"] == ["fql_001"]
    assert data["proposal"]["proposal_id"].startswith("rev_")
    # 不直接执行：只返回提案（spec §6）
    assert "直接执行" in data["note"] or "未执行" in data["note"]


def test_decision_ledger_endpoint(tmp_path, monkeypatch):
    db_path = tmp_path / "decision-ledger.db"
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "ledger-test-token")
    from director_brain.audit_trail import log_decision
    from storage.sqlite_repository import SqliteRepository

    repo = SqliteRepository(str(db_path))
    log_decision(repo, "plan-in-scope", "strategy_confirmed", {},
                 project_id="proj_api")
    log_decision(repo, "plan-out-of-scope", "strategy_confirmed", {},
                 project_id="another-project")
    repo.close()
    ledger_client = TestClient(
        app,
        client=("127.0.0.1", 54131),
        headers={"Authorization": "Bearer ledger-test-token"},
    )
    resp = ledger_client.get("/v1/projects/proj_api/decision-ledger")
    body = resp.json()
    assert resp.status_code == 200
    assert body["data"]["project_id"] == "proj_api"
    assert len(body["data"]["entries"]) == 1
    assert body["data"]["entries"][0]["project_id"] == "proj_api"
    assert body["data"]["entries"][0]["decision_id"] == "plan-in-scope"


def test_project_manifest_and_multi_asset_context_persist_with_source_bindings(
    tmp_path, monkeypatch
):
    """Exercise the local multi-asset path; synthetic media is plumbing evidence only."""
    db_path = tmp_path / "director_brain.db"
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-test-token")
    local_client = TestClient(
        app, client=("127.0.0.1", 54123),
        headers={"Authorization": "Bearer p1-test-token"},
    )
    media_a = tmp_path / "asset_a.mkv"
    media_b = tmp_path / "asset_b.mp4"
    subprocess.run([
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
        "-disposition:a:1", "0", str(media_a),
    ], capture_output=True, text=True, check=True)
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i",
        "color=c=blue:s=320x240:r=30000/1001:d=2", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(media_b), "-y",
    ], capture_output=True, text=True, check=True)

    def asset(asset_id: str, path: str, order: int) -> dict:
        return {
            "asset_id": asset_id,
            "source_ref": path,
            "order": order,
            "rights": {
                "state": "local_processing_allowed",
                "basis": "owner_permission",
                "evidence_ref": f"test-fixture://rights/{asset_id}",
            },
        }

    manifest_body = {
        "expected_revision": 0,
        "boundary_basis": "user_project_manifest",
        "boundary_source_ref": "test-fixture://project/project-01",
        "boundary_evidence_refs": ["test-fixture://project/manifest.json"],
        "assets": [
            asset("asset-a", str(media_a), 0),
            asset("asset-b", str(media_b), 1),
        ],
    }
    saved = _envelope_data(local_client.put(
        "/v1/projects/project-01/film-manifest", json=manifest_body))
    assert saved["current_revision"] == 1
    assert saved["boundary_state"] == "declared_unverified"
    assert len(saved["manifest"]["assets"]) == 2
    assert saved["manifest"]["assets"][0]["source_ref"] == "local-media://asset-a"
    assert saved["manifest"]["assets"][1]["r_frame_rate"] == "30000/1001"
    assert saved["manifest"]["assets"][1]["stream_time_base"]
    audio_time_map = saved["manifest"]["assets"][0]["time_map"]
    assert audio_time_map["mapping_state"] == "complete"
    assert [
        (item["source_start_offset_numerator"],
         item["source_start_offset_denominator"])
        for item in audio_time_map["audio_streams"]
    ] == [(1, 4), (3, 4)]
    assert [item["sample_rate"] for item in audio_time_map["audio_streams"]] == [
        48000, 44100]
    assert saved["manifest"]["assets"][0]["rights"]["evidence_state"] == (
        "declared_unverified")

    snapshot_req = {"manifest_revision": 1}
    first = _envelope_data(local_client.post(
        "/v1/projects/project-01/film-context:snapshot", json=snapshot_req))
    assert first["cache_hit"] is False
    assert first["coverage"] == "project_multi_asset_observed"
    snap = first["snapshot"]
    assert snap["project_manifest_id"] == saved["manifest"]["manifest_id"]
    assert snap["asset_refs"] == ["asset-a", "asset-b"]
    assert len(snap["source_content_hashes"]) == 2
    assert snap["sampling_config"]["observation_scope"] == (
        "per_asset_preserving_explicit_timebase_units")
    assert snap["timeline_scope"] == "project_per_asset"
    assert snap["timebase"] is None
    assert snap["timebase_unit"] is None
    assert [item["asset_id"] for item in snap["asset_coverage"]] == [
        "asset-a", "asset-b"]
    assert [item["r_frame_rate"] for item in snap["asset_coverage"]] == [
        "25/1", "30000/1001"]
    assert snap["asset_coverage"][0]["time_map"] == audio_time_map
    assert snap["asset_coverage"][1]["time_map"]["audio_streams"] == []
    assert all(item["analysis_state"] == "observed"
               for item in snap["asset_coverage"])
    assert snap["asset_coverage"][0]["observation_timebases"] == [
        {"value": 1_000_000, "unit": "microseconds"}]
    assert len(snap["evidence_refs"]) >= 2
    first_asset_evidence = snap["asset_coverage"][0]["evidence_refs"][0]
    observed = _envelope_data(local_client.get(
        f"/v1/projects/project-01/observations/{first_asset_evidence}"))
    assert observed["observation"]["project_asset_id"] == "asset-a"
    assert observed["observation"]["timebase_unit"] == "microseconds"

    # Read through the public endpoints again after the first request's local
    # repository connection has closed: persisted state, not a process dict.
    second = _envelope_data(local_client.post(
        "/v1/projects/project-01/film-context:snapshot", json=snapshot_req))
    assert second["cache_hit"] is True
    assert second["snapshot"] == snap
    get_manifest = _envelope_data(local_client.get(
        "/v1/projects/project-01/film-manifest?revision=1"))
    assert get_manifest["manifest"]["manifest_id"] == snap["project_manifest_id"]
    assert get_manifest["manifest"]["assets"][0]["time_map"] == audio_time_map
    get_context = _envelope_data(local_client.get(
        f"/v1/film-context/{first['context_id']}"))
    assert get_context["snapshot"] == snap

    built_graph = _envelope_data(local_client.post(
        "/v1/projects/project-01/story-graph:build"))
    project_graph = built_graph["story_graph"]
    assert built_graph["persisted"] is True
    assert built_graph["reused"] is False
    assert project_graph["context_id"] == snap["context_id"]
    assert project_graph["timeline_scope"] == "project_per_asset"
    assert project_graph["cross_asset_relations_state"] == "not_attempted"
    assert [item["asset_id"] for item in project_graph["assets"]] == [
        "asset-a", "asset-b"]
    assert all(item["story_graph"] is not None
               for item in project_graph["assets"])
    assert [item["story_graph"]["project_asset_id"]
            for item in project_graph["assets"]] == ["asset-a", "asset-b"]
    assert str(media_a) not in json.dumps(project_graph)
    readback = _envelope_data(local_client.get(
        "/v1/projects/project-01/story-graph"))
    assert readback["story_graph"] == project_graph
    assert readback["currentness"] == "current"
    assert readback["stale_reason"] is None
    assert readback["usable_for_new_plan"] is True
    assert readback["source_hash_validation_state"] == "not_revalidated_by_read"
    repeated = _envelope_data(local_client.post(
        "/v1/projects/project-01/story-graph:build"))
    assert repeated["reused"] is True
    assert repeated["story_graph"] == project_graph

    # Caller-reviewed links live in a revisioned overlay. They do not mutate
    # the source-local graph or imply automatic cross-asset inference.
    second_asset_evidence = snap["asset_coverage"][1]["evidence_refs"][0]
    second_observed = _envelope_data(local_client.get(
        f"/v1/projects/project-01/observations/{second_asset_evidence}"
    ))["observation"]
    link_body = {
        "manifest_revision": 1,
        "story_graph_id": project_graph["graph_id"],
        "expected_review_revision": 0,
        "idempotency_key": "story-links-review-001",
        "links": [{
            "link_id": "subject-1",
            "entity_kind": "person_identity",
            "display_label": "caller-reviewed subject",
            "anchors": [
                {
                    "observation_id": first_asset_evidence,
                    "source_start": observed["observation"]["start_frame"],
                    "source_end": observed["observation"]["end_frame"],
                },
                {
                    "observation_id": second_asset_evidence,
                    "source_start": second_observed["start_frame"],
                    "source_end": second_observed["end_frame"],
                },
            ],
        }],
    }
    first_link_review = _envelope_data(local_client.put(
        "/v1/projects/project-01/story-links", json=link_body))
    assert first_link_review["review_revision"] == 1
    assert first_link_review["actor_identity_state"] == "caller_asserted"
    assert first_link_review["automatic_inference_state"] == "not_attempted"
    assert first_link_review["quality_acceptance"] == "not_proven"
    assert len(first_link_review["review"]["links"][0]["anchors"]) == 2
    for anchor, source_observation in zip(
        first_link_review["review"]["links"][0]["anchors"],
        (observed["observation"], second_observed), strict=True,
    ):
        assert anchor["timebase"] == source_observation["timebase"]
        assert anchor["timebase_unit"] == source_observation["timebase_unit"]
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/story-links"))["review"] == (
            first_link_review["review"])
    replay = _envelope_data(local_client.put(
        "/v1/projects/project-01/story-links", json=link_body))
    assert replay["reused"] is True
    assert replay["review"] == first_link_review["review"]

    stale_link_review = local_client.put(
        "/v1/projects/project-01/story-links",
        json={**link_body, "idempotency_key": "story-links-stale-001",
              "expected_review_revision": 0},
    )
    assert stale_link_review.status_code == 409

    correction_body = {
        **link_body,
        "expected_review_revision": 1,
        "idempotency_key": "story-links-review-002",
        "links": [{**link_body["links"][0],
                   "display_label": "corrected caller label"}],
    }
    corrected_review = _envelope_data(local_client.put(
        "/v1/projects/project-01/story-links", json=correction_body))
    assert corrected_review["review_revision"] == 2
    assert corrected_review["review_history_revisions"] == [1, 2]
    clear_review = _envelope_data(local_client.put(
        "/v1/projects/project-01/story-links",
        json={**link_body, "expected_review_revision": 2,
              "idempotency_key": "story-links-review-003", "links": []},
    ))
    assert clear_review["active_link_count"] == 0
    assert clear_review["review_history_revisions"] == [1, 2, 3]

    shadow_comparison_calls = 0
    shadow_failure_generator_calls = 0

    def fail_shadow_comparison(
        brief, manifest, context, graph, observations, **kwargs,
    ):
        nonlocal shadow_failure_generator_calls
        shadow_failure_generator_calls += 1
        from director_brain.llm_adapter import LLMStructuredOutputError

        failure = LLMStructuredOutputError(
            "SENSITIVE_PROVIDER_DETAIL_MUST_NOT_BE_RETAINED",
            failure_code="project_strategy_schema_invalid",
        )
        failure.provider_call_count = 2
        failure.failure_stage = "segment"
        failure.provider_response_metadata = {
            "model": "provider/qwen-build-42",
            "system_fingerprint": "fp_build_42",
            "finish_reason": "length",
            "prompt_tokens": 1270,
            "completion_tokens": 4096,
            "total_tokens": 5366,
            "private_detail": "SENSITIVE_ENVELOPE_DETAIL_MUST_NOT_BE_RETAINED",
        }
        raise failure

    monkeypatch.setattr(
        api_main, "generate_project_shadow_strategy_options",
        fail_shadow_comparison,
    )
    failed_shadow_request = {
        "manifest_revision": 1,
        "target_duration_us": 1_600_000,
        "intent_text": "PRIVATE_INTENT_SENTINEL_MUST_NOT_BE_RETAINED",
        "idempotency_key": "project-shadow-comparison-failure-001",
    }
    failed_shadow_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json=failed_shadow_request,
    )
    assert failed_shadow_response.status_code == 502
    assert failed_shadow_response.headers["X-Director-Brain-Failure-Code"] == (
        "project_strategy_schema_invalid")
    failure_ledger = _envelope_data(local_client.get(
        "/v1/projects/project-01/decision-ledger"))
    failed_attempts = [
        item for item in failure_ledger["entries"]
        if item["action"] == "director_strategy_comparison_failed"
    ]
    assert len(failed_attempts) == 1
    failure_entry = failed_attempts[0]
    assert failure_entry["decision_id"] == (
        api_main._project_director_shadow_comparison_id(
            "project-01", failed_shadow_request["idempotency_key"]))
    failure_detail = failure_entry["detail"]
    assert set(failure_detail) == {
        "stage", "failure_code", "request_fingerprint",
        "idempotency_key_sha256", "manifest_id", "manifest_revision",
        "context_id", "story_graph_id", "provider_failure_stage",
        "provider_call_count", "provider_response_metadata",
    }
    assert failure_detail["stage"] == "reasoner_generation"
    assert failure_detail["failure_code"] == "project_strategy_schema_invalid"
    assert failure_detail["manifest_revision"] == 1
    assert failure_detail["provider_failure_stage"] == "segment"
    assert failure_detail["provider_call_count"] == 2
    assert failure_detail["provider_response_metadata"] == {
        "model": "provider/qwen-build-42",
        "system_fingerprint": "fp_build_42",
        "finish_reason": "length",
        "prompt_tokens": 1270,
        "completion_tokens": 4096,
        "total_tokens": 5366,
    }
    assert all(
        isinstance(failure_detail[key], str) and failure_detail[key]
        for key in ("manifest_id", "context_id", "story_graph_id")
    )
    assert len(failure_detail["request_fingerprint"]) == 64
    assert len(failure_detail["idempotency_key_sha256"]) == 64
    serialized_failure_entry = json.dumps(failure_entry, ensure_ascii=False)
    assert "PRIVATE_INTENT_SENTINEL_MUST_NOT_BE_RETAINED" not in serialized_failure_entry
    assert "SENSITIVE_PROVIDER_DETAIL_MUST_NOT_BE_RETAINED" not in serialized_failure_entry
    assert "SENSITIVE_ENVELOPE_DETAIL_MUST_NOT_BE_RETAINED" not in serialized_failure_entry
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plan-shadow-comparisons"))["count"] == 0
    replayed_failure = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json=failed_shadow_request,
    )
    assert replayed_failure.status_code == 502
    assert replayed_failure.headers["X-Director-Brain-Failure-Code"] == (
        "project_strategy_schema_invalid")
    assert "PRIVATE_INTENT_SENTINEL_MUST_NOT_BE_RETAINED" not in (
        replayed_failure.text)
    assert shadow_failure_generator_calls == 1
    changed_failed_key = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={**failed_shadow_request,
              "intent_text": "A failed idempotency key cannot change requests."},
    )
    assert changed_failed_key.status_code == 409
    assert shadow_failure_generator_calls == 1
    retry_with_new_key = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={**failed_shadow_request,
              "idempotency_key": "project-shadow-comparison-failure-002"},
    )
    assert retry_with_new_key.status_code == 502
    assert shadow_failure_generator_calls == 2
    failure_ledger = _envelope_data(local_client.get(
        "/v1/projects/project-01/decision-ledger"))
    failed_attempts = [
        item for item in failure_ledger["entries"]
        if item["action"] == "director_strategy_comparison_failed"
    ]
    assert len(failed_attempts) == 2
    assert len({item["decision_id"] for item in failed_attempts}) == 2
    assert all(
        "PRIVATE_INTENT_SENTINEL_MUST_NOT_BE_RETAINED" not in json.dumps(item)
        for item in failed_attempts
    )

    unclassified_generator_calls = 0
    private_value_error_marker = "PRIVATE_VALUE_ERROR_DETAIL_MUST_NOT_LEAK"

    def fail_with_unclassified_value_error(*args, **kwargs):
        nonlocal unclassified_generator_calls
        unclassified_generator_calls += 1
        raise ValueError(private_value_error_marker)

    monkeypatch.setattr(
        api_main, "generate_project_shadow_strategy_options",
        fail_with_unclassified_value_error,
    )
    unclassified_failure_request = {
        "manifest_revision": 1,
        "target_duration_us": 1_600_000,
        "intent_text": "PRIVATE_UNCLASSIFIED_INTENT_MUST_NOT_LEAK",
        "idempotency_key": "project-shadow-comparison-unclassified-value-001",
    }
    unclassified_failure_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json=unclassified_failure_request,
    )
    assert unclassified_failure_response.status_code == 500
    unclassified_failure_code = "reasoner_value_error_unclassified"
    assert unclassified_failure_response.headers[
        "X-Director-Brain-Failure-Code"] == unclassified_failure_code
    assert private_value_error_marker not in unclassified_failure_response.text
    assert "PRIVATE_UNCLASSIFIED_INTENT_MUST_NOT_LEAK" not in (
        unclassified_failure_response.text)

    failure_ledger = _envelope_data(local_client.get(
        "/v1/projects/project-01/decision-ledger"))
    unclassified_entries = [
        item for item in failure_ledger["entries"]
        if item["action"] == "director_strategy_comparison_failed"
        and item["decision_id"] == api_main._project_director_shadow_comparison_id(
            "project-01", unclassified_failure_request["idempotency_key"])
    ]
    assert len(unclassified_entries) == 1
    unclassified_detail = unclassified_entries[0]["detail"]
    assert unclassified_detail["failure_code"] == unclassified_failure_code
    assert unclassified_detail["stage"] == "reasoner_generation"
    assert unclassified_detail["attribution_state"] == "unclassified"
    assert unclassified_detail["exception_type"] == "ValueError"
    assert "provider_failure_stage" not in unclassified_detail
    assert private_value_error_marker not in json.dumps(unclassified_entries[0])
    assert "PRIVATE_UNCLASSIFIED_INTENT_MUST_NOT_LEAK" not in json.dumps(
        unclassified_entries[0])

    unclassified_replay = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json=unclassified_failure_request,
    )
    assert unclassified_replay.status_code == 500
    assert unclassified_replay.headers[
        "X-Director-Brain-Failure-Code"] == unclassified_failure_code
    assert private_value_error_marker not in unclassified_replay.text
    assert unclassified_generator_calls == 1
    documented_unclassified_header = api_main.app.openapi()["paths"][
        "/v1/projects/{project_id}/director-plans:compare-shadow-strategies"
    ]["post"]["responses"]["500"]["headers"][
        "X-Director-Brain-Failure-Code"]
    assert documented_unclassified_header["schema"]["enum"] == [
        unclassified_failure_code]

    def fake_shadow_comparison(
        brief, manifest, context, graph, observations, **kwargs,
    ):
        nonlocal shadow_comparison_calls
        shadow_comparison_calls += 1
        return _make_fake_project_shadow_comparison(
            brief, manifest, context, graph, observations, **kwargs)

    monkeypatch.setattr(
        api_main, "generate_project_shadow_strategy_options", fake_shadow_comparison)
    before_shadow_preview = _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))
    shadow_preview_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
            "idempotency_key": "project-shadow-comparison-001",
        },
    )
    assert shadow_preview_response.status_code == 200
    shadow_preview = _envelope_data(shadow_preview_response)
    assert shadow_preview["persisted"] is True
    assert shadow_preview["confirmable"] is False
    assert shadow_preview["quality_acceptance"] == "NOT_PROVEN"
    assert shadow_preview["provider_scope"] == "local_ollama_loopback_only"
    assert shadow_preview["comparison"]["state"] == "SHADOW_UNVERIFIED"
    assert shadow_preview["comparison"]["persistence_state"] == "PERSISTED_LOCAL"
    assert shadow_preview["comparison"]["quality_acceptance"] == "NOT_PROVEN"
    assert len(shadow_preview["comparison"]["strategy_candidates"]) == 2
    selected_shadow_candidate = shadow_preview[
        "comparison"]["strategy_candidates"][0]
    assert len(selected_shadow_candidate["candidate_binding_digest"]) == 64
    asset_coverage = shadow_preview["strategy_asset_coverage"]
    candidates = shadow_preview["comparison"]["strategy_candidates"]
    assert len(asset_coverage) == len(candidates) == 2
    for coverage, candidate in zip(asset_coverage, candidates, strict=True):
        assert coverage["hypothesis_id"] == candidate["hypothesis"]["hypothesis_id"]
        asset_rows = coverage["assets"]
        ordered_sources = candidate["hypothesis"]["ordered_sources"]
        selection_audit = candidate["selection_audit"]
        source_asset_ids = sorted({
            source["project_asset_id"] for source in ordered_sources
        })
        assert [row["project_asset_id"] for row in asset_rows] == source_asset_ids
        assert sum(row["source_count"] for row in asset_rows) == len(
            ordered_sources)
        for row in asset_rows:
            asset_entries = [
                audit for source, audit in zip(
                    ordered_sources, selection_audit, strict=True)
                if source["project_asset_id"] == row["project_asset_id"]
            ]
            assert row["source_count"] == len(asset_entries)
            assert row["strategy_include_count"] == sum(
                audit["strategy_disposition"] == "include"
                for audit in asset_entries)
            assert row["strategy_exclude_count"] == sum(
                audit["strategy_disposition"] == "exclude"
                for audit in asset_entries)
            assert row["selected_in_edl_count"] == sum(
                audit["selected_in_edl"] for audit in asset_entries)
            assert row["included_but_not_selected_count"] == sum(
                audit["strategy_disposition"] == "include"
                and not audit["selected_in_edl"]
                for audit in asset_entries)
        assert sum(row["selected_in_edl_count"] for row in asset_rows) == sum(
            item["selected_in_edl"] for item in selection_audit)
    assert shadow_comparison_calls == 1
    replay = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
            "idempotency_key": "project-shadow-comparison-001",
        },
    ))
    assert replay["comparison_id"] == shadow_preview["comparison_id"]
    assert replay["idempotent_replay"] is True
    assert replay["strategy_asset_coverage"] == asset_coverage
    assert shadow_comparison_calls == 1
    changed_key_reuse = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "A changed request cannot reuse the prior comparison key.",
            "idempotency_key": "project-shadow-comparison-001",
        },
    )
    assert changed_key_reuse.status_code == 409
    assert shadow_comparison_calls == 1
    listed_shadow_response = local_client.get(
        "/v1/projects/project-01/director-plan-shadow-comparisons")
    assert listed_shadow_response.headers["cache-control"] == "no-store"
    listed_shadow = _envelope_data(listed_shadow_response)
    assert listed_shadow["count"] == 1
    assert listed_shadow["items"][0]["comparison_id"] == (
        shadow_preview["comparison_id"])
    assert listed_shadow["items"][0]["source_currentness"] == "current"
    assert listed_shadow["items"][0]["stale_reasons"] == []
    recovered_shadow = _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plan-shadow-comparisons/"
        f"{shadow_preview['comparison_id']}"))
    assert recovered_shadow["persisted"] is True
    assert recovered_shadow["source_currentness"] == "current"
    assert recovered_shadow["strategy_asset_coverage"] == asset_coverage
    local_client.close()

    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        restart_port = int(port_probe.getsockname()[1])
    restart_env = os.environ.copy()
    restart_env["SQLITE_PATH"] = str(db_path)
    restart_env["DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT"] = str(tmp_path)
    restart_env["DIRECTOR_BRAIN_PROJECT_API_TOKEN"] = "p1-test-token"
    restart_process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "127.0.0.1", "--port", str(restart_port),
            "--log-level", "critical", "--no-access-log",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=restart_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    restart_url = f"http://127.0.0.1:{restart_port}"
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if restart_process.poll() is not None:
                raise AssertionError(
                    "Uvicorn exited before successful comparison readback")
            try:
                with urlopen(restart_url + "/openapi.json", timeout=1) as ready:
                    if ready.status == 200:
                        break
            except (OSError, URLError):
                time.sleep(0.1)
        else:
            raise AssertionError("restarted Uvicorn did not become ready")

        readback_request = Request(
            restart_url
            + "/v1/projects/project-01/director-plan-shadow-comparisons/"
            + shadow_preview["comparison_id"],
            headers={"Authorization": "Bearer p1-test-token"},
        )
        with urlopen(readback_request, timeout=10) as readback_response:
            restarted_shadow = json.loads(readback_response.read())["data"]
        assert restarted_shadow["persisted"] is True
        assert restarted_shadow["comparison_record"] == (
            recovered_shadow["comparison_record"])
        assert restarted_shadow["strategy_asset_coverage"] == asset_coverage
        assert restarted_shadow["quality_acceptance"] == "NOT_PROVEN"
    finally:
        if restart_process.poll() is None:
            restart_process.terminate()
        try:
            restart_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            restart_process.kill()
            restart_process.wait(timeout=10)
    local_client = TestClient(
        app,
        client=("127.0.0.1", 54123),
        headers={"Authorization": "Bearer p1-test-token"},
    )
    assert recovered_shadow["comparison_record"]["comparison_id"] == (
        shadow_preview["comparison_id"])
    source_bytes = media_a.read_bytes()
    try:
        media_a.write_bytes(source_bytes + b"source drift")
        stale_list = _envelope_data(local_client.get(
            "/v1/projects/project-01/director-plan-shadow-comparisons"))
        assert stale_list["items"][0]["source_currentness"] == "stale"
        assert stale_list["items"][0]["stale_reasons"] == [
            "project_source_changed"]
    finally:
        media_a.write_bytes(source_bytes)
    after_shadow_preview = _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))
    assert after_shadow_preview["count"] == before_shadow_preview["count"]

    unresolved_shadow = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的旅行记录，必须包含日出镜头",
            "idempotency_key": "project-shadow-needs-input-001",
        },
    ))
    assert unresolved_shadow["confirmable"] is False
    assert unresolved_shadow["comparison"]["baseline_plan"][
        "validation_status"] == "needs_input"
    for candidate in unresolved_shadow["comparison"]["strategy_candidates"]:
        assert candidate["plan"]["validation_status"] == "needs_input"
        assert any("constraint_unverifiable" in item
                   for item in candidate["plan"]["open_questions"])
    review_candidate = unresolved_shadow["comparison"]["strategy_candidates"][0]
    assessment_suggestion = review_candidate["hypothesis"][
        "constraint_assessments"][0]
    assert assessment_suggestion["candidate_edl_source_asset_alignment"] == (
        "all_cited_assets_selected")
    rejection_body = {
        "candidate_binding_digest": review_candidate["candidate_binding_digest"],
        "hypothesis_id": review_candidate["hypothesis"]["hypothesis_id"],
        "constraint_kind": assessment_suggestion["constraint_kind"],
        "brief_index": assessment_suggestion["brief_index"],
        "expected_assessment": assessment_suggestion["assessment"],
        "idempotency_key": "project-constraint-rejection-001",
        "rejected_by": "local-reviewer",
        "reason_code": "source_evidence_insufficient",
    }
    rejected_assessment_response = local_client.post(
        "/v1/projects/project-01/director-plan-shadow-comparisons/"
        f"{unresolved_shadow['comparison_id']}:reject-constraint-assessment",
        json=rejection_body,
    )
    assert rejected_assessment_response.status_code == 200, (
        rejected_assessment_response.text)
    rejected_assessment = rejected_assessment_response.json()["data"]
    assert rejected_assessment["persisted"] is True
    assert rejected_assessment["reused"] is False
    assert rejected_assessment["actor_identity_state"] == "caller_asserted"
    assert rejected_assessment["candidate_mutated"] is False
    assert rejected_assessment["plan_state_unchanged"] is True
    assert rejected_assessment["quality_acceptance"] == "NOT_PROVEN"
    rejection_receipt = rejected_assessment["assessment_rejection"]
    assert rejection_receipt["comparison_id"] == unresolved_shadow["comparison_id"]
    assert rejection_receipt["constraint_ref"] == (
        f"{assessment_suggestion['constraint_kind']}:{assessment_suggestion['brief_index']}")
    assert rejection_receipt["expected_assessment"] == (
        assessment_suggestion["assessment"])
    assert rejection_receipt["reason_code"] == "source_evidence_insufficient"
    assert "constraint_text" not in rejection_receipt
    rejection_replay = local_client.post(
        "/v1/projects/project-01/director-plan-shadow-comparisons/"
        f"{unresolved_shadow['comparison_id']}:reject-constraint-assessment",
        json=rejection_body,
    )
    assert rejection_replay.status_code == 200
    assert rejection_replay.json()["data"]["reused"] is True
    changed_rejection_replay = local_client.post(
        "/v1/projects/project-01/director-plan-shadow-comparisons/"
        f"{unresolved_shadow['comparison_id']}:reject-constraint-assessment",
        json={**rejection_body, "reason_code": "misread_constraint"},
    )
    assert changed_rejection_replay.status_code == 409
    unresolved_rejection = local_client.post(
        "/v1/projects/project-01/director-plan-shadow-comparisons/"
        f"{unresolved_shadow['comparison_id']}:reject-constraint-assessment",
        json={**rejection_body,
              "idempotency_key": "project-constraint-rejection-unresolved",
              "expected_assessment": "unresolved"},
    )
    assert unresolved_rejection.status_code == 422
    wrong_binding_rejection = local_client.post(
        "/v1/projects/project-01/director-plan-shadow-comparisons/"
        f"{unresolved_shadow['comparison_id']}:reject-constraint-assessment",
        json={**rejection_body,
              "idempotency_key": "project-constraint-rejection-wrong-binding",
              "candidate_binding_digest": "f" * 64},
    )
    assert wrong_binding_rejection.status_code == 409
    rejection_ledger = _envelope_data(local_client.get(
        "/v1/projects/project-01/decision-ledger"))["entries"]
    rejection_events = [
        item for item in rejection_ledger
        if item["action"] == "director_constraint_assessment_rejected"
    ]
    assert len(rejection_events) == 1
    assert set(rejection_events[0]["detail"]) == {
        "assessment_rejection", "actor_identity_state", "candidate_mutated",
        "plan_state_unchanged", "quality_acceptance",
        "idempotency_key_sha256", "request_fingerprint",
    }
    serialized_rejection = json.dumps(rejection_events[0], ensure_ascii=False)
    assert "必须包含日出镜头" not in serialized_rejection
    assert "Synthetic assessment; not quality evidence." not in serialized_rejection

    unresolved_readback = _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plan-shadow-comparisons/"
        f"{unresolved_shadow['comparison_id']}"))
    assert unresolved_readback["comparison_record"]["comparison"][
        "baseline_plan"]["validation_status"] == "needs_input"
    readback_candidate = unresolved_readback["comparison_record"][
        "comparison"]["strategy_candidates"][0]
    assert readback_candidate["hypothesis"]["constraint_assessments"][0] == (
        assessment_suggestion)
    assert all(
        candidate["plan"]["validation_status"] == "needs_input"
        for candidate in unresolved_readback["comparison_record"][
            "comparison"]["strategy_candidates"])
    assert shadow_comparison_calls == 2

    from director_brain.pathway_protocol import (
        PathwayStatus,
        get_pathway_status,
        set_pathway_status,
    )

    previous_director_status = get_pathway_status("director_strategy_reasoning")
    set_pathway_status("director_strategy_reasoning", PathwayStatus.SHADOW)
    try:
        blocked_formal_plan = local_client.post(
            "/v1/projects/project-01/director-plans:generate",
            json={
                "manifest_revision": 1,
                "target_duration_us": 1_600_000,
                "intent_text": "制作一段温暖的家庭旅行记录",
                "reasoner_strategy": "llm",
                "selected_comparison_id": shadow_preview["comparison_id"],
                "selected_strategy_hypothesis_id": (
                    selected_shadow_candidate["hypothesis"]["hypothesis_id"]),
                "selected_candidate_binding_digest": (
                    selected_shadow_candidate["candidate_binding_digest"]),
            },
        )
    finally:
        set_pathway_status("director_strategy_reasoning", previous_director_status)
    assert blocked_formal_plan.status_code == 409
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))["count"] == (
            before_shadow_preview["count"])
    from director_brain.pathway_protocol import PathwayStatus
    set_pathway_status("director_strategy_reasoning", PathwayStatus.ACTIVE)
    try:
        materialized_response = local_client.post(
            "/v1/projects/project-01/director-plans:generate",
            json={
                "manifest_revision": 1,
                "target_duration_us": 1_600_000,
                "intent_text": "制作一段温暖的家庭旅行记录",
                "reasoner_strategy": "llm",
                "selected_comparison_id": shadow_preview["comparison_id"],
                "selected_strategy_hypothesis_id": (
                    selected_shadow_candidate["hypothesis"]["hypothesis_id"]),
                "selected_candidate_binding_digest": (
                    selected_shadow_candidate["candidate_binding_digest"]),
            },
        )
    finally:
        set_pathway_status("director_strategy_reasoning", previous_director_status)
    assert materialized_response.status_code == 409
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))["count"] == (
            before_shadow_preview["count"])
    assert shadow_comparison_calls == 2
    before_shadow_preview = _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))
    from director_brain.llm_adapter import LLMTransportError
    transport_failure_code = "provider_connection_error"

    def unavailable_shadow_provider(*args, **kwargs):
        raise LLMTransportError(
            "synthetic provider failure",
            failure_code=transport_failure_code,
        )

    monkeypatch.setattr(
        api_main, "generate_project_shadow_strategy_options",
        unavailable_shadow_provider,
    )
    unavailable_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
            "idempotency_key": "shadow-provider-connection-001",
        },
    )
    assert unavailable_response.status_code == 503
    assert unavailable_response.headers[
        "X-Director-Brain-Failure-Code"] == "provider_connection_error"
    assert "synthetic provider failure" not in unavailable_response.text
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))["count"] == (
            before_shadow_preview["count"])

    transport_failure_code = "provider_model_binding_error"
    binding_failure_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
            "idempotency_key": "shadow-provider-binding-change-001",
        },
    )
    assert binding_failure_response.status_code == 503
    assert binding_failure_response.headers[
        "X-Director-Brain-Failure-Code"] == "provider_model_binding_error"
    assert "synthetic provider failure" not in binding_failure_response.text
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))["count"] == (
            before_shadow_preview["count"])

    transport_failure_code = "provider_configuration_error"
    configuration_failure_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
            "idempotency_key": "shadow-provider-config-001",
        },
    )
    assert configuration_failure_response.status_code == 503
    assert configuration_failure_response.headers[
        "X-Director-Brain-Failure-Code"] == "provider_configuration_error"
    assert "provider credentials" not in configuration_failure_response.text
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))["count"] == (
            before_shadow_preview["count"])

    transport_failure_code = "PRIVATE_PROVIDER_TRANSPORT_MARKER"
    unknown_transport_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
            "idempotency_key": "shadow-provider-unknown-001",
        },
    )
    assert unknown_transport_response.status_code == 503
    assert "PRIVATE_PROVIDER_TRANSPORT_MARKER" not in unknown_transport_response.text
    assert "X-Director-Brain-Failure-Code" not in unknown_transport_response.headers
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))["count"] == (
            before_shadow_preview["count"])

    from director_brain.llm_adapter import LLMStructuredOutputError
    private_response_marker = "PRIVATE_PROVIDER_RESPONSE_MARKER"
    structured_failure_code = "segment_emotions"

    def invalid_shadow_output(*args, **kwargs):
        raise LLMStructuredOutputError(
            private_response_marker,
            failure_code=structured_failure_code,
        )

    monkeypatch.setattr(
        api_main, "generate_project_shadow_strategy_options",
        invalid_shadow_output,
    )
    invalid_output_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
            "idempotency_key": "shadow-provider-invalid-output-001",
        },
    )
    assert invalid_output_response.status_code == 502
    assert private_response_marker not in invalid_output_response.text
    assert "segment_emotions" not in invalid_output_response.text
    assert invalid_output_response.headers[
        "X-Director-Brain-Failure-Code"] == "segment_emotions"
    openapi = api_main.app.openapi()
    for path in (
        "/v1/projects/{project_id}/director-plans:generate",
        "/v1/projects/{project_id}/director-plans:compare-shadow-strategies",
    ):
        responses = openapi["paths"][path]["post"]["responses"]
        documented_header = responses[
            "502"]["headers"]["X-Director-Brain-Failure-Code"]
        assert "segment_emotions" in documented_header["schema"]["enum"]
        assert "project_constraint_assessment_evidence" in (
            documented_header["schema"]["enum"])
        assert "project_strategy_schema_invalid" in (
            documented_header["schema"]["enum"])
        assert "project_strategies_identical" in (
            documented_header["schema"]["enum"])
        assert "segment_strategies_identical" in (
            documented_header["schema"]["enum"])
        assert "project_reference_binding_invalid" in (
            documented_header["schema"]["enum"])
        documented_transport_header = responses[
            "503"]["headers"]["X-Director-Brain-Failure-Code"]
        assert "provider_connection_error" in documented_transport_header[
            "schema"]["enum"]
        assert "provider_configuration_error" in documented_transport_header[
            "schema"]["enum"]
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))["count"] == (
            before_shadow_preview["count"])

    structured_failure_code = private_response_marker
    unknown_code_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
            "idempotency_key": "shadow-provider-unknown-code-001",
        },
    )
    assert unknown_code_response.status_code == 502
    assert private_response_marker not in unknown_code_response.text
    assert "X-Director-Brain-Failure-Code" not in unknown_code_response.headers
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))["count"] == (
            before_shadow_preview["count"])

    from director_brain.llm_adapter import LLMInputCapacityError

    def oversized_shadow_input(*args, **kwargs):
        raise LLMInputCapacityError("private capacity detail")

    monkeypatch.setattr(
        api_main, "generate_project_shadow_strategy_options",
        oversized_shadow_input,
    )
    oversized_response = local_client.post(
        "/v1/projects/project-01/director-plans:compare-shadow-strategies",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
            "idempotency_key": "shadow-provider-capacity-001",
        },
    )
    assert oversized_response.status_code == 413
    assert "private capacity detail" not in oversized_response.text
    assert _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))["count"] == (
            before_shadow_preview["count"])

    invalid_same_asset_link = local_client.put(
        "/v1/projects/project-01/story-links",
        json={
            **link_body,
            "expected_review_revision": 3,
            "idempotency_key": "story-links-invalid-001",
            "links": [{
                **link_body["links"][0],
                "anchors": [link_body["links"][0]["anchors"][0], {
                    **link_body["links"][0]["anchors"][0],
                    "source_start": observed["observation"]["start_frame"] + 1,
                }],
            }],
        },
    )
    assert invalid_same_asset_link.status_code == 409
    unchanged_links = _envelope_data(local_client.get(
        "/v1/projects/project-01/story-links"))
    assert unchanged_links["review_revision"] == 3
    assert unchanged_links["review"]["links"] == []
    graph_after_link_review = _envelope_data(local_client.get(
        "/v1/projects/project-01/story-graph"))["story_graph"]
    assert graph_after_link_review == project_graph
    assert graph_after_link_review["cross_asset_relations_state"] == "not_attempted"

    failed_plan = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 2_000_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
        },
    ))
    assert failed_plan["persisted"] is True
    assert failed_plan["validation"]["valid"] is False
    assert failed_plan["plan"]["state"] == "failed_validation"
    assert any("outside target range" in error
               for error in failed_plan["validation"]["errors"])
    failed_confirmation = local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{failed_plan['plan']['plan_id']}:confirm-strategy",
        json={
            "expected_plan_hash": failed_plan["plan_hash"],
            "expected_edl_hash": failed_plan["edl_hash"],
            "idempotency_key": "project-01-invalid-confirmation-001",
            "confirmed_by": "project-owner",
        },
    )
    assert failed_confirmation.status_code == 409

    # Mechanical plan validation does not establish that a semantic hard
    # constraint was satisfied. Such a Plan must remain durable but cannot be
    # promoted to the user-confirmation state.
    needs_input_plan = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的旅行记录，必须包含日出镜头",
        },
    ))
    assert needs_input_plan["persisted"] is True
    assert needs_input_plan["validation"]["valid"] is True
    assert needs_input_plan["validation"]["requires_input"] is True
    assert needs_input_plan["plan"]["validation_status"] == "needs_input"
    assert needs_input_plan["plan"]["state"] == "needs_input"
    assert any("constraint_unverifiable" in item
               for item in needs_input_plan["plan"]["open_questions"])
    needs_input_readback = _envelope_data(local_client.get(
        f"/v1/projects/project-01/director-plans/"
        f"{needs_input_plan['plan']['plan_id']}"))
    assert needs_input_readback["plan"] == needs_input_plan["plan"]
    needs_input_replay = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的旅行记录，必须包含日出镜头",
        },
    ))
    assert needs_input_replay["reused"] is True
    assert needs_input_replay["plan"] == needs_input_plan["plan"]

    guarded_repo = api_main._project_manifest_repository()
    try:
        with pytest.raises(ValueError, match="unresolved semantic constraints"):
            guarded_repo.confirm_project_strategy(
                "project-01",
                needs_input_plan["plan"]["plan_id"],
                expected_plan_hash=needs_input_plan["plan_hash"],
                expected_edl_hash=needs_input_plan["edl_hash"],
                idempotency_key="project-01-unverifiable-confirmation-001",
                confirmed_by="project-owner",
            )
    finally:
        guarded_repo.close()

    needs_input_confirmation = local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{needs_input_plan['plan']['plan_id']}:confirm-strategy",
        json={
            "expected_plan_hash": needs_input_plan["plan_hash"],
            "expected_edl_hash": needs_input_plan["edl_hash"],
            "idempotency_key": "project-01-unverifiable-confirmation-002",
            "confirmed_by": "project-owner",
        },
    )
    assert needs_input_confirmation.status_code == 409

    rejection_request = {
        "expected_plan_hash": needs_input_plan["plan_hash"],
        "expected_edl_hash": needs_input_plan["edl_hash"],
        "idempotency_key": "project-01-unverifiable-rejection-001",
        "rejected_by": "project-owner",
        "reason": "Required semantic content cannot be verified from current evidence",
    }
    blank_rejection_reason = local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{needs_input_plan['plan']['plan_id']}:reject-strategy",
        json={**rejection_request, "reason": "   "},
    )
    assert blank_rejection_reason.status_code == 422
    unexpected_rejection_field = local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{needs_input_plan['plan']['plan_id']}:reject-strategy",
        json={**rejection_request, "force": True},
    )
    assert unexpected_rejection_field.status_code == 422

    rejected = _envelope_data(local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{needs_input_plan['plan']['plan_id']}:reject-strategy",
        json=rejection_request,
    ))
    assert rejected["persisted"] is True
    assert rejected["reused"] is False
    assert rejected["state"] == "rejected"
    assert rejected["plan"]["state"] == "rejected"
    assert rejected["rejection"]["prior_state"] == "needs_input"
    assert rejected["rejection_receipt"]["receipt_consistency_state"] == "consistent"
    assert rejected["rejection_receipt"]["reason"] == rejection_request["reason"]

    rejection_replay = _envelope_data(local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{needs_input_plan['plan']['plan_id']}:reject-strategy",
        json=rejection_request,
    ))
    assert rejection_replay["reused"] is True
    assert rejection_replay["rejection_receipt"] == rejected["rejection_receipt"]

    reused_key_with_changed_reason = local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{needs_input_plan['plan']['plan_id']}:reject-strategy",
        json={**rejection_request, "reason": "Changed after rejection"},
    )
    assert reused_key_with_changed_reason.status_code == 409

    rejected_readback = _envelope_data(local_client.get(
        f"/v1/projects/project-01/director-plans/"
        f"{needs_input_plan['plan']['plan_id']}"))
    assert rejected_readback["plan"]["state"] == "rejected"
    assert rejected_readback["rejection_receipt"] == rejected["rejection_receipt"]
    confirmed_after_rejection = local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{needs_input_plan['plan']['plan_id']}:confirm-strategy",
        json={
            "expected_plan_hash": needs_input_plan["plan_hash"],
            "expected_edl_hash": needs_input_plan["edl_hash"],
            "idempotency_key": "project-01-rejected-confirmation-001",
            "confirmed_by": "project-owner",
        },
    )
    assert confirmed_after_rejection.status_code == 409

    rejection_ledger = _envelope_data(local_client.get(
        "/v1/projects/project-01/decision-ledger"))["entries"]
    rejected_entries = [
        entry for entry in rejection_ledger
        if entry["decision_id"] == needs_input_plan["plan"]["plan_id"]
        and entry["action"] == "strategy_rejected"
    ]
    assert len(rejected_entries) == 1

    # A separately clarified Plan must bind the exact needs-input predecessor,
    # supersede it atomically, and remain idempotent after the first commit.
    predecessor_needs_input = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段旅行记录，必须包含篝火镜头",
        },
    ))
    assert predecessor_needs_input["plan"]["state"] == "needs_input"
    predecessor_id = predecessor_needs_input["plan"]["plan_id"]
    clarified_request = {
        "manifest_revision": 1,
        "target_duration_us": 1_600_000,
        "intent_text": "制作一段宁静的旅行记录，若有篝火镜头可优先考虑",
        "supersedes_plan_id": predecessor_id,
        "expected_superseded_plan_hash": predecessor_needs_input["plan_hash"],
        "expected_superseded_edl_hash": predecessor_needs_input["edl_hash"],
        "clarified_by": "project-owner",
    }
    partial_supersession = local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={**clarified_request, "expected_superseded_edl_hash": None},
    )
    assert partial_supersession.status_code == 422
    stale_supersession = local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={**clarified_request, "expected_superseded_plan_hash": "0" * 16},
    )
    assert stale_supersession.status_code == 409
    predecessor_after_stale_attempt = _envelope_data(local_client.get(
        f"/v1/projects/project-01/director-plans/{predecessor_id}"))
    assert predecessor_after_stale_attempt["plan"]["state"] == "needs_input"

    clarified_plan = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json=clarified_request,
    ))
    assert clarified_plan["persisted"] is True
    assert clarified_plan["reused"] is False
    assert clarified_plan["plan"]["state"] == "ready_for_strategy_confirmation"
    assert clarified_plan["plan"]["supersedes_plan_id"] == predecessor_id
    predecessor_readback = _envelope_data(local_client.get(
        f"/v1/projects/project-01/director-plans/{predecessor_id}"))
    assert predecessor_readback["plan"]["state"] == "superseded"
    clarified_replay = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json=clarified_request,
    ))
    assert clarified_replay["reused"] is True
    assert clarified_replay["plan"] == clarified_plan["plan"]
    supersession_ledger = _envelope_data(local_client.get(
        "/v1/projects/project-01/decision-ledger"))["entries"]
    supersession_entries = [
        entry for entry in supersession_ledger
        if entry["decision_id"] == predecessor_id
        and entry["action"] == "plan_superseded"
    ]
    assert len(supersession_entries) == 1
    assert supersession_entries[0]["detail"]["supersession"][
        "successor_plan_id"] == clarified_plan["plan"]["plan_id"]
    assert supersession_entries[0]["detail"]["supersession"][
        "clarified_by"] == "project-owner"
    assert supersession_entries[0]["detail"]["supersession"][
        "actor_identity_state"] == "caller_asserted"

    # Readback must fail closed if either side of the persisted receipt drifts.
    repo = api_main._project_manifest_repository()
    try:
        ledger_row = repo._conn.execute(
            "SELECT data FROM decision_ledger WHERE ledger_id = ?",
            (supersession_entries[0]["ledger_id"],),
        ).fetchone()
        original_ledger_payload = ledger_row["data"]
        tampered_ledger = json.loads(original_ledger_payload)
        tampered_ledger["detail"]["supersession"]["successor_edl_hash"] = "0" * 16
        repo._conn.execute(
            "UPDATE decision_ledger SET data = ? WHERE ledger_id = ?",
            (json.dumps(tampered_ledger, ensure_ascii=False),
             supersession_entries[0]["ledger_id"]),
        )
        repo._conn.commit()
    finally:
        repo.close()
    inconsistent_successor = local_client.get(
        f"/v1/projects/project-01/director-plans/"
        f"{clarified_plan['plan']['plan_id']}"
    )
    assert inconsistent_successor.status_code == 409
    repo = api_main._project_manifest_repository()
    try:
        repo._conn.execute(
            "UPDATE decision_ledger SET data = ? WHERE ledger_id = ?",
            (original_ledger_payload, supersession_entries[0]["ledger_id"]),
        )
        repo._conn.commit()
    finally:
        repo.close()

    # The synthetic fixture's verified candidate intervals total 1.6s. Use
    # that feasible requested duration to exercise successful persistence and
    # confirmation; the 2.0s request above remains a fail-closed regression.
    draft_plan = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
        },
    ))
    assert draft_plan["persisted"] is True
    assert draft_plan["reused"] is False
    assert draft_plan["validation"]["valid"] is True
    assert draft_plan["plan"]["state"] == "ready_for_strategy_confirmation"
    assert "caller_asserted_project_links_not_independently_verified" not in (
        draft_plan["plan"]["open_questions"])
    assert "automatic_cross_asset_person_event_inference_not_attempted" in (
        draft_plan["plan"]["open_questions"])
    assert draft_plan["evidence_scope"] == {
        "manifest_id": saved["manifest"]["manifest_id"],
        "manifest_revision": 1,
        "context_id": snap["context_id"],
        "story_graph_id": project_graph["graph_id"],
        "timeline_scope": "project_per_asset",
        "cross_asset_relations_state": "not_attempted",
        "project_story_mention_review_id": None,
        "project_story_mention_review_revision": 0,
        "project_story_mention_review_state": "not_provided",
        "project_story_link_review_id": clear_review["review"]["review_id"],
        "project_story_link_review_revision": 3,
        "project_story_link_review_state": "caller_asserted",
        "automatic_cross_asset_inference_state": "not_attempted",
        "quality_acceptance": "not_proven",
    }
    assert draft_plan["plan"]["project_story_link_review_id"] == (
        clear_review["review"]["review_id"])
    assert draft_plan["plan"]["project_story_link_review_revision"] == 3
    assert all(not edit["project_story_link_refs"]
               for edit in draft_plan["edl"]["ordered_edits"])
    assert draft_plan["plan"]["sequence_project_asset_ids"] == [
        edit["project_asset_id"] for edit in draft_plan["edl"]["ordered_edits"]
    ]
    assert draft_plan["plan"]["brief_id"] == draft_plan["brief"]["brief_id"]
    assert draft_plan["plan"]["edl_id"] == draft_plan["edl"]["edl_id"]
    assert draft_plan["plan"]["project_manifest_id"] == (
        saved["manifest"]["manifest_id"])
    assert draft_plan["plan"]["project_revision"] == 1
    assert draft_plan["plan"]["project_context_id"] == snap["context_id"]
    assert draft_plan["plan"]["project_story_graph_id"] == project_graph["graph_id"]
    assert draft_plan["edl"]["context_id"] == project_graph["graph_id"]
    assert all(edit["project_asset_id"] for edit in draft_plan["edl"]["ordered_edits"])
    assert "director_quality_not_established_by_planner_execution" in (
        draft_plan["plan"]["open_questions"])

    restored_plan = _envelope_data(local_client.get(
        f"/v1/projects/project-01/director-plans/{draft_plan['plan']['plan_id']}"))
    assert restored_plan["persisted"] is True
    assert restored_plan["plan"] == draft_plan["plan"]
    assert restored_plan["edl"] == draft_plan["edl"]
    assert restored_plan["brief"] == draft_plan["brief"]
    generic_confirmation = local_client.post(
        f"/v1/director-plans/{draft_plan['plan']['plan_id']}:confirm-strategy",
        json={
            "plan_id": draft_plan["plan"]["plan_id"],
            "edl_id": draft_plan["edl"]["edl_id"],
            "plan_json": draft_plan["plan"],
            "edl_json": draft_plan["edl"],
            "confirmed_by": "project-owner",
        },
    )
    assert generic_confirmation.status_code == 409
    plan_index = _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))
    assert plan_index["count"] == 5
    draft_index_entry = next(
        item for item in plan_index["plans"]
        if item["plan_id"] == draft_plan["plan"]["plan_id"])
    assert draft_index_entry["revision_state"] == "latest_manifest_revision"
    failed_index_entry = next(
        item for item in plan_index["plans"]
        if item["plan_id"] == failed_plan["plan"]["plan_id"])
    assert failed_index_entry["state"] == "failed_validation"
    needs_input_index_entry = next(
        item for item in plan_index["plans"]
        if item["plan_id"] == needs_input_plan["plan"]["plan_id"])
    assert needs_input_index_entry["state"] == "rejected"
    superseded_index_entry = next(
        item for item in plan_index["plans"]
        if item["plan_id"] == predecessor_id)
    assert superseded_index_entry["state"] == "superseded"
    successor_index_entry = next(
        item for item in plan_index["plans"]
        if item["plan_id"] == clarified_plan["plan"]["plan_id"])
    assert successor_index_entry["supersedes_plan_id"] == predecessor_id
    assert _envelope_data(local_client.get(
        "/v1/projects/another-project/director-plans"))["count"] == 0
    assert local_client.get(
        f"/v1/projects/another-project/director-plans/{draft_plan['plan']['plan_id']}"
    ).status_code == 404

    repeated_plan = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
        },
    ))
    assert repeated_plan["persisted"] is True
    assert repeated_plan["reused"] is True
    assert repeated_plan["plan"] == draft_plan["plan"]

    confirmation_request = {
        "expected_plan_hash": draft_plan["plan_hash"],
        "expected_edl_hash": draft_plan["edl_hash"],
        "idempotency_key": "project-01-confirmation-001",
        "confirmed_by": "project-owner",
        "output_target": "delivery",
        "notes": "reviewed exact Plan and EDL hashes",
    }
    stale_confirmation = local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{draft_plan['plan']['plan_id']}:confirm-strategy",
        json={**confirmation_request, "expected_plan_hash": "0" * 16},
    )
    assert stale_confirmation.status_code == 409

    confirmed = _envelope_data(local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{draft_plan['plan']['plan_id']}:confirm-strategy",
        json=confirmation_request,
    ))
    assert confirmed["persisted"] is True
    assert confirmed["reused"] is False
    assert confirmed["state"] == "strategy_confirmed"
    assert confirmed["plan"]["approval_state"] == "approved"
    assert confirmed["edl"]["approval_state"] == "approved"
    assert confirmed["dispatch_eligible"] is False
    assert confirmed["actor_identity_state"] == "caller_asserted"
    assert confirmed["quality_acceptance"] == "not_proven"

    replayed_confirmation = _envelope_data(local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{draft_plan['plan']['plan_id']}:confirm-strategy",
        json=confirmation_request,
    ))
    assert replayed_confirmation["reused"] is True
    assert replayed_confirmation["confirmation"] == confirmed["confirmation"]

    confirmed_readback = _envelope_data(local_client.get(
        f"/v1/projects/project-01/director-plans/"
        f"{draft_plan['plan']['plan_id']}"))
    assert confirmed_readback["plan"]["state"] == "strategy_confirmed"
    assert confirmed_readback["confirmation_receipt"][
        "plan_edl_hash_binding_valid"] is True
    assert confirmed_readback["confirmation_receipt"]["confirmation"] == (
        confirmed["confirmation"])
    confirmed_index = _envelope_data(local_client.get(
        "/v1/projects/project-01/director-plans"))
    confirmed_index_entry = next(
        item for item in confirmed_index["plans"]
        if item["plan_id"] == draft_plan["plan"]["plan_id"])
    assert confirmed_index_entry["state"] == "strategy_confirmed"

    confirmed_generation_replay = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
        },
    ))
    assert confirmed_generation_replay["reused"] is True
    assert confirmed_generation_replay["plan"]["state"] == "strategy_confirmed"

    project_ledger = _envelope_data(local_client.get(
        "/v1/projects/project-01/decision-ledger"))
    confirmation_entries = [
        entry for entry in project_ledger["entries"]
        if entry["decision_id"] == draft_plan["plan"]["plan_id"]
        and entry["action"] == "strategy_confirmed"
    ]
    assert len(confirmation_entries) == 1
    assert confirmation_entries[0]["project_id"] == "project-01"

    fresh_process = _fresh_process_project_readback(
        db_path=db_path,
        media_root=tmp_path,
        token="p1-test-token",
        project_id="project-01",
        plan_id=draft_plan["plan"]["plan_id"],
        confirmation_request=confirmation_request,
    )
    assert fresh_process["readback_status"] == 200
    assert fresh_process["readback"]["plan"]["state"] == "strategy_confirmed"
    assert fresh_process["readback"]["confirmation_receipt"][
        "plan_edl_hash_binding_valid"] is True
    assert fresh_process["replay_status"] == 200
    assert fresh_process["replay"]["reused"] is True
    assert fresh_process["replay"]["confirmation"] == confirmed["confirmation"]
    fresh_confirmation_entries = [
        entry for entry in fresh_process["ledger"]["entries"]
        if entry["decision_id"] == draft_plan["plan"]["plan_id"]
        and entry["action"] == "strategy_confirmed"
    ]
    assert len(fresh_confirmation_entries) == 1
    assert fresh_confirmation_entries[0]["project_id"] == "project-01"

    # A later Plan binds the current caller-asserted identity overlay to exact
    # selected source intervals. Updating the complete link set makes that
    # draft stale and blocks strategy confirmation.
    linked_review = _envelope_data(local_client.put(
        "/v1/projects/project-01/story-links",
        json={**link_body, "expected_review_revision": 3,
              "idempotency_key": "story-links-review-004"},
    ))
    linked_plan = _envelope_data(local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
        },
    ))
    assert linked_plan["plan"]["project_story_link_review_id"] == (
        linked_review["review"]["review_id"])
    assert linked_plan["plan"]["project_story_link_review_revision"] == 4
    assert linked_plan["evidence_scope"]["project_story_link_review_state"] == (
        "caller_asserted")
    assert linked_plan["evidence_scope"]["cross_asset_relations_state"] == (
        "not_attempted")
    assert linked_plan["evidence_scope"]["automatic_cross_asset_inference_state"] == (
        "not_attempted")
    assert "automatic_cross_asset_person_event_inference_not_attempted" in (
        linked_plan["plan"]["open_questions"])
    assert "cross_asset_person_event_relations_not_attempted" not in (
        linked_plan["plan"]["open_questions"])
    assert "caller_asserted_project_links_not_independently_verified" in (
        linked_plan["plan"]["open_questions"])
    assert any("subject-1" in edit["project_story_link_refs"]
               for edit in linked_plan["edl"]["ordered_edits"])

    revised_link = {**link_body["links"][0],
                    "display_label": "corrected after plan generation"}
    newer_link_review = _envelope_data(local_client.put(
        "/v1/projects/project-01/story-links",
        json={**link_body, "expected_review_revision": 4,
              "idempotency_key": "story-links-review-005",
              "links": [revised_link]},
    ))
    assert newer_link_review["review_revision"] == 5
    stale_link_confirmation = local_client.post(
        f"/v1/projects/project-01/director-plans/"
        f"{linked_plan['plan']['plan_id']}:confirm-strategy",
        json={
            "expected_plan_hash": linked_plan["plan_hash"],
            "expected_edl_hash": linked_plan["edl_hash"],
            "idempotency_key": "project-01-stale-link-confirmation-001",
            "confirmed_by": "project-owner",
        },
    )
    assert stale_link_confirmation.status_code == 409

    # A new API process must preserve both snapshots independently: the old
    # Plan/EDL retain the review revision they were built from, while the
    # project graph exposes the corrected current review. The old Plan must
    # still fail confirmation after restart.
    restarted_link_state = _fresh_process_project_readback(
        db_path=db_path,
        media_root=tmp_path,
        token="p1-test-token",
        project_id="project-01",
        plan_id=linked_plan["plan"]["plan_id"],
        read_project_links=True,
        confirmation_request={
            "expected_plan_hash": linked_plan["plan_hash"],
            "expected_edl_hash": linked_plan["edl_hash"],
            "idempotency_key": "project-01-stale-link-confirmation-after-restart-001",
            "confirmed_by": "project-owner",
        },
    )
    assert restarted_link_state["readback_status"] == 200
    assert restarted_link_state["readback"]["plan"][
        "project_story_link_review_revision"] == 4
    assert any(
        "subject-1" in edit["project_story_link_refs"]
        for edit in restarted_link_state["readback"]["edl"]["ordered_edits"]
    )
    assert restarted_link_state["replay_status"] == 409

    current_review = restarted_link_state["link_review"]["review"]
    assert restarted_link_state["link_review_status"] == 200
    assert current_review["review_revision"] == 5
    assert current_review["links"][0]["display_label"] == (
        "corrected after plan generation")
    graph_view = restarted_link_state["story_graph"]["story_graph_view"]
    assert restarted_link_state["story_graph_status"] == 200
    assert graph_view["cross_asset_link_review"] == current_review

    # Persisted evidence remains tied to the registered bytes. A file change
    # after context/graph creation must block a newly generated project Plan.
    original_media_a = media_a.read_bytes()
    try:
        media_a.write_bytes(original_media_a + b"changed-after-context")
        changed_source = local_client.post(
            "/v1/projects/project-01/director-plans:generate",
            json={
                "manifest_revision": 1,
                "target_duration_us": 1_600_000,
                "intent_text": "制作一段温暖的家庭旅行记录",
            },
        )
        assert changed_source.status_code == 409
        assert "asset-a" in changed_source.json()["detail"]
        assert str(media_a) not in changed_source.text
    finally:
        media_a.write_bytes(original_media_a)

    stale = local_client.put(
        "/v1/projects/project-01/film-manifest", json=manifest_body)
    assert stale.status_code == 409
    revised_body = {**manifest_body, "expected_revision": 1}
    revised = _envelope_data(local_client.put(
        "/v1/projects/project-01/film-manifest", json=revised_body))
    assert revised["current_revision"] == 2
    invalidated = local_client.get(f"/v1/film-context/{first['context_id']}")
    assert invalidated.status_code == 410
    project_scoped_invalidated = local_client.get(
        f"/v1/projects/project-01/film-context/{first['context_id']}"
    )
    assert project_scoped_invalidated.status_code == 410
    historical_context = local_client.get(
        f"/v1/projects/project-01/film-context/{first['context_id']}",
        params={"include_stale": "true"},
    )
    assert historical_context.status_code == 200, historical_context.text
    historical_data = historical_context.json()["data"]
    assert historical_data["currentness"] == "stale"
    assert historical_data["stale_reason"] == "invalidated"
    assert historical_data["usable_for_new_plan"] is False
    assert historical_data["snapshot"]["context_id"] == first["context_id"]
    stale_plan = local_client.post(
        "/v1/projects/project-01/director-plans:generate",
        json={
            "manifest_revision": 1,
            "target_duration_us": 1_600_000,
            "intent_text": "制作一段温暖的家庭旅行记录",
        },
    )
    assert stale_plan.status_code == 409
    second_revision = _envelope_data(local_client.post(
        "/v1/projects/project-01/film-context:snapshot",
        json={"manifest_revision": 2},
    ))
    assert second_revision["cache_hit"] is False
    assert second_revision["context_id"] != first["context_id"]
    assert [item["analysis_cache_state"] for item in
            second_revision["snapshot"]["asset_coverage"]] == [
                "reused", "reused"]
    assert local_client.get(
        "/v1/projects/project-01/story-graph").status_code == 404
    historical_graph = _envelope_data(local_client.get(
        "/v1/projects/project-01/story-graph?revision=1"))
    assert historical_graph["story_graph"] == project_graph
    assert historical_graph["currentness"] == "stale"
    assert historical_graph["stale_reason"] == "manifest_revision_changed"
    assert historical_graph["usable_for_new_plan"] is False


def test_project_context_preserves_partial_coverage_and_retries_incomplete_assets(
    tmp_path, monkeypatch
):
    """One unavailable or failed asset must not erase evidence from other assets."""
    from pathlib import Path

    from director_brain.models import TimebaseUnit
    from director_brain.utils import file_sha256
    from observation_service import pipeline

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "partial_project.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-partial-token")
    media_a = tmp_path / "asset_a.mp4"
    media_b = tmp_path / "asset_b.mp4"
    _make_test_video(str(media_a), duration_sec=2)
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i",
        "color=c=blue:s=320x240:r=25:d=2", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(media_b), "-y",
    ], capture_output=True, text=True, check=True)
    saved_b = media_b.read_bytes()

    def asset(asset_id: str, path: Path, order: int) -> dict:
        return {
            "asset_id": asset_id,
            "source_ref": str(path),
            "order": order,
            "rights": {
                "state": "local_processing_allowed",
                "basis": "owner_permission",
                "evidence_ref": f"test-fixture://rights/{asset_id}",
            },
        }

    local_client = TestClient(
        app, client=("127.0.0.1", 54131),
        headers={"Authorization": "Bearer p1-partial-token"},
    )
    manifest = _envelope_data(local_client.put(
        "/v1/projects/partial-01/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/partial-01",
            "boundary_evidence_refs": ["test-fixture://project/manifest.json"],
            "assets": [asset("asset-a", media_a, 0), asset("asset-b", media_b, 1)],
        }))
    assert manifest["current_revision"] == 1

    original_analyze = pipeline.analyze_media
    calls: list[str] = []
    fail_b = {"enabled": True}

    def controlled_analyze(path: str, *, cache=None):
        asset_name = Path(path).stem
        calls.append(asset_name)
        if asset_name == "asset_b" and fail_b["enabled"]:
            raise RuntimeError("must not appear in public evidence: C:\\private\\media")
        return [_make_obs(
            f"source-{asset_name}", asset_name, 0, 1_000_000
        ).model_copy(update={
            "media_hash": file_sha256(path),
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        })]

    monkeypatch.setattr(pipeline, "analyze_media", controlled_analyze)

    media_b.unlink()
    first = _envelope_data(local_client.post(
        "/v1/projects/partial-01/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert first["coverage"] == "project_multi_asset_partial"
    assert [item["analysis_state"] for item in
            first["snapshot"]["asset_coverage"]] == [
                "observed", "source_unavailable"]
    assert first["snapshot"]["asset_coverage"][1]["failure_code"] == (
        "source_unavailable")
    assert len(first["evidence_refs"]) == 1
    assert calls == ["asset_a"]

    media_b.write_bytes(saved_b)
    second = _envelope_data(local_client.post(
        "/v1/projects/partial-01/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert second["coverage"] == "project_multi_asset_partial"
    assert [item["analysis_state"] for item in
            second["snapshot"]["asset_coverage"]] == ["observed", "failed"]
    assert second["snapshot"]["asset_coverage"][0]["analysis_cache_state"] == (
        "reused")
    assert second["snapshot"]["asset_coverage"][1]["failure_code"] == (
        "analyzer_failed")
    assert "C:\\private\\media" not in json.dumps(second["snapshot"])
    assert calls == ["asset_a", "asset_b"]

    fail_b["enabled"] = False
    third = _envelope_data(local_client.post(
        "/v1/projects/partial-01/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert third["coverage"] == "project_multi_asset_observed"
    assert [item["analysis_state"] for item in
            third["snapshot"]["asset_coverage"]] == ["observed", "observed"]
    assert [item["analysis_cache_state"] for item in
            third["snapshot"]["asset_coverage"]] == ["reused", "computed"]
    assert calls == ["asset_a", "asset_b", "asset_b"]

    cached = _envelope_data(local_client.post(
        "/v1/projects/partial-01/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert cached["cache_hit"] is True
    assert calls == ["asset_a", "asset_b", "asset_b"]
    monkeypatch.setattr(pipeline, "analyze_media", original_analyze)


def test_project_media_endpoints_require_explicit_bearer_token(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "project_auth.db"))
    monkeypatch.delenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", raising=False)
    path = str(tmp_path / "unused.mp4")
    body = {
        "expected_revision": 0,
        "boundary_basis": "user_project_manifest",
        "boundary_source_ref": "test-fixture://project/auth",
        "boundary_evidence_refs": ["test-fixture://project/manifest"],
        "assets": [{
            "asset_id": "asset-a",
            "source_ref": path,
            "order": 0,
            "rights": {
                "state": "local_processing_allowed",
                "basis": "owner_permission",
                "evidence_ref": "test-fixture://rights/asset-a",
            },
        }],
    }
    client_without_config = TestClient(app, client=("127.0.0.1", 54127))
    disabled = client_without_config.put(
        "/v1/projects/project-auth/film-manifest", json=body)
    assert disabled.status_code == 503
    ledger_disabled = client_without_config.get(
        "/v1/projects/project-auth/decision-ledger")
    assert ledger_disabled.status_code == 503
    assert not (tmp_path / "project_auth.db").exists()

    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "expected-token")
    client_with_bad_token = TestClient(
        app,
        client=("127.0.0.1", 54128),
        headers={"Authorization": "Bearer wrong-token"},
    )
    rejected = client_with_bad_token.put(
        "/v1/projects/project-auth/film-manifest", json=body)
    assert rejected.status_code == 401
    ledger_rejected = client_with_bad_token.get(
        "/v1/projects/project-auth/decision-ledger")
    assert ledger_rejected.status_code == 401
    assert not (tmp_path / "project_auth.db").exists()

    client_with_token = TestClient(
        app,
        client=("127.0.0.1", 54129),
        headers={"Authorization": "Bearer expected-token"},
    )
    absent = client_with_token.get("/v1/projects/project-auth/film-manifest")
    assert absent.status_code == 404
    assert (tmp_path / "project_auth.db").exists()
    ledger_response = client_with_token.get(
        "/v1/projects/project-auth/decision-ledger")
    assert ledger_response.status_code == 200
    ledger = _envelope_data(ledger_response)
    assert ledger["project_id"] == "project-auth"
    assert ledger["entries"] == []

def test_project_inventory_only_never_opens_unverified_source(tmp_path, monkeypatch):
    from observation_service import pipeline

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "inventory_only.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-test-token")
    monkeypatch.delenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", raising=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an unverified source must never be opened or analyzed")

    monkeypatch.setattr("api.main._authorized_local_media_path", forbidden)
    monkeypatch.setattr("director_brain.utils.file_sha256", forbidden)
    monkeypatch.setattr("director_brain.context_gateway.file_sha256", forbidden)
    monkeypatch.setattr("director_brain.context_gateway._probe_video_meta", forbidden)
    monkeypatch.setattr(pipeline, "analyze_media", forbidden)
    local_client = TestClient(
        app, client=("127.0.0.1", 54124),
        headers={"Authorization": "Bearer p1-test-token"},
    )
    response = local_client.put("/v1/projects/project-denied/film-manifest", json={
        "expected_revision": 0,
        "boundary_basis": "dataset_project_id",
        "boundary_source_ref": "dataset://project-1",
        "boundary_evidence_refs": ["dataset://manifest"],
        "assets": [{
            "asset_id": "asset-unverified",
            "source_ref": "dataset://muvy/event-01/recording-01",
            "order": 0,
            "rights": {"state": "unverified"},
        }],
    })
    saved = _envelope_data(response)
    asset = saved["manifest"]["assets"][0]
    assert asset["source_ref"] == "source-asset://asset-unverified"
    assert asset["source_identity_state"] == "declared_unverified"
    assert asset["source_content_hash"] is None
    assert asset["size_bytes"] is None

    snapshot = _envelope_data(local_client.post(
        "/v1/projects/project-denied/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert snapshot["coverage"] == "project_single_asset_inventory_only"
    coverage = snapshot["snapshot"]["asset_coverage"][0]
    assert coverage["analysis_state"] == "not_authorized"
    assert coverage["analysis_cache_state"] == "not_run"
    assert coverage["blocked_reason"] == "rights_unverified"
    assert coverage["source_content_hash"] is None
    assert coverage["observation_count"] == 0
    assert coverage["evidence_refs"] == []
    assert snapshot["snapshot"]["evidence_refs"] == []

    project_graph = _envelope_data(local_client.post(
        "/v1/projects/project-denied/story-graph:build"))["story_graph"]
    assert project_graph["cross_asset_relations_state"] == "not_attempted"
    assert project_graph["assets"][0]["story_graph_state"] == "not_authorized"
    assert project_graph["assets"][0]["story_graph"] is None
    assert project_graph["assets"][0]["evidence_refs"] == []

    cached = _envelope_data(local_client.post(
        "/v1/projects/project-denied/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert cached["cache_hit"] is True

    injected = local_client.post(
        "/v1/projects/project-denied/film-context:snapshot",
        json={"manifest_revision": 1, "observations_by_asset": {
            "asset-unverified": [],
        }},
    )
    assert injected.status_code == 409


def test_comind_recording_boundary_is_persisted_without_claiming_a_film_project(
    tmp_path, monkeypatch
):
    """Dataset recording/session identity must remain distinct from event/project IDs."""
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "comind_boundary.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-comind-token")
    monkeypatch.delenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", raising=False)
    client = TestClient(
        app,
        client=("127.0.0.1", 54126),
        headers={"Authorization": "Bearer p1-comind-token"},
    )
    recording_id = "01d63ce5-f734-4647-93c8-ea4af07b8c7f"
    project_id = f"dataset:comind-v1:recording:{recording_id}"
    manifest_url = f"https://comind.ethz.ch/dataset/{recording_id}/healthcheck.json"
    response = client.put(
        f"/v1/projects/{project_id}/film-manifest",
        json={
            "expected_revision": 0,
            "boundary_basis": "dataset_recording_id",
            "boundary_source_ref": manifest_url,
            "boundary_evidence_refs": [
                manifest_url,
                "https://comind.ethz.ch/",
                "https://arxiv.org/html/2607.06691v1#S9",
            ],
            "assets": [
                {
                    "asset_id": "leader",
                    "source_ref": (
                        f"dataset://comind-v1/{recording_id}/"
                        "mp4s/leader_trimmed_sync.mp4"
                    ),
                    "order": 0,
                    "rights": {"state": "unverified"},
                },
                {
                    "asset_id": "helper",
                    "source_ref": (
                        f"dataset://comind-v1/{recording_id}/"
                        "mp4s/helper_trimmed_sync.mp4"
                    ),
                    "order": 1,
                    "rights": {"state": "unverified"},
                },
            ],
        },
    )
    saved = _envelope_data(response)["manifest"]
    assert saved["boundary_basis"] == "dataset_recording_id"
    assert saved["boundary_source_ref"] == manifest_url
    assert saved["boundary_state"] == "declared_unverified"
    assert [asset["asset_id"] for asset in saved["assets"]] == ["leader", "helper"]
    assert [asset["source_ref"] for asset in saved["assets"]] == [
        "source-asset://leader",
        "source-asset://helper",
    ]
    assert all(asset["source_content_hash"] is None for asset in saved["assets"])
    assert all(asset["rights"]["state"] == "unverified" for asset in saved["assets"])

    readback = _envelope_data(client.get(
        f"/v1/projects/{project_id}/film-manifest"))["manifest"]
    assert readback["manifest_id"] == saved["manifest_id"]
    assert readback["boundary_basis"] == "dataset_recording_id"
    assert readback["project_id"] == project_id


def test_project_context_analyzes_only_assets_with_local_rights(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from director_brain.models import TimebaseUnit
    from director_brain.utils import file_sha256
    from observation_service import pipeline

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "mixed_rights.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-mixed-token")
    media = tmp_path / "locally-authorized.mp4"
    _make_test_video(str(media), duration_sec=2)
    local_client = TestClient(
        app, client=("127.0.0.1", 54125),
        headers={"Authorization": "Bearer p1-mixed-token"},
    )
    manifest = _envelope_data(local_client.put(
        "/v1/projects/mixed-rights/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "dataset_event_id",
            "boundary_source_ref": "dataset://muvy/event-01",
            "boundary_evidence_refs": ["dataset://muvy/events"],
            "assets": [
                {
                    "asset_id": "asset-local",
                    "source_ref": str(media),
                    "order": 0,
                    "rights": {
                        "state": "local_processing_allowed",
                        "basis": "owner_permission",
                        "evidence_ref": "test-fixture://rights/asset-local",
                    },
                },
                {
                    "asset_id": "asset-unverified",
                    "source_ref": "dataset://muvy/event-01/recording-02",
                    "order": 1,
                    "rights": {"state": "unverified"},
                },
            ],
        }))
    assert manifest["manifest"]["assets"][1]["source_ref"] == (
        "source-asset://asset-unverified")

    calls: list[str] = []

    def analyze_authorized(path: str, *, cache=None):
        calls.append(Path(path).name)
        return [_make_obs(
            "source-local", "asset-local", 0, 1_000_000
        ).model_copy(update={
            "media_hash": file_sha256(path),
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        })]

    monkeypatch.setattr(pipeline, "analyze_media", analyze_authorized)
    first = _envelope_data(local_client.post(
        "/v1/projects/mixed-rights/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert first["coverage"] == "project_multi_asset_partial"
    coverage = first["snapshot"]["asset_coverage"]
    assert [item["analysis_state"] for item in coverage] == [
        "observed", "not_authorized"]
    assert coverage[0]["source_content_hash"] == file_sha256(str(media))
    assert coverage[1]["source_content_hash"] is None
    assert coverage[1]["blocked_reason"] == "rights_unverified"
    assert coverage[1]["failure_code"] is None
    assert coverage[1]["observation_count"] == 0
    assert first["snapshot"]["evidence_refs"] == coverage[0]["evidence_refs"]
    assert calls == ["locally-authorized.mp4"]

    cached = _envelope_data(local_client.post(
        "/v1/projects/mixed-rights/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert cached["cache_hit"] is True
    assert cached["snapshot"] == first["snapshot"]
    assert calls == ["locally-authorized.mp4"]


def test_third_party_only_rights_do_not_authorize_local_processing(
    tmp_path, monkeypatch
):
    from observation_service import pipeline

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "third_party_only.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-third-party-token")
    monkeypatch.delenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", raising=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("third-party-only rights do not authorize local access")

    monkeypatch.setattr("api.main._authorized_local_media_path", forbidden)
    monkeypatch.setattr("director_brain.utils.file_sha256", forbidden)
    monkeypatch.setattr("director_brain.context_gateway.file_sha256", forbidden)
    monkeypatch.setattr("director_brain.context_gateway._probe_video_meta", forbidden)
    monkeypatch.setattr(pipeline, "analyze_media", forbidden)
    local_client = TestClient(
        app, client=("127.0.0.1", 54126),
        headers={"Authorization": "Bearer p1-third-party-token"},
    )
    manifest = _envelope_data(local_client.put(
        "/v1/projects/third-party-only/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "dataset_project_id",
            "boundary_source_ref": "dataset://licensed-project/project-01",
            "boundary_evidence_refs": ["dataset://licensed-project/manifest"],
            "assets": [{
                "asset_id": "asset-third-party-only",
                "source_ref": "dataset://licensed-project/asset-01",
                "order": 0,
                "rights": {
                    "state": "third_party_processing_allowed",
                    "basis": "public_license",
                    "evidence_ref": "test-fixture://license/asset-01",
                    "license_id": "CC-BY-4.0",
                },
            }],
        }))
    assert manifest["manifest"]["assets"][0]["source_ref"] == (
        "source-asset://asset-third-party-only")
    assert manifest["manifest"]["assets"][0]["source_content_hash"] is None

    result = _envelope_data(local_client.post(
        "/v1/projects/third-party-only/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert result["coverage"] == "project_single_asset_inventory_only"
    coverage = result["snapshot"]["asset_coverage"][0]
    assert coverage["analysis_state"] == "not_authorized"
    assert coverage["blocked_reason"] == "local_processing_not_authorized"
    assert coverage["source_content_hash"] is None
    assert result["snapshot"]["evidence_refs"] == []


def test_project_cache_recomputes_when_analysis_profile_changes(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from director_brain.models import TimebaseUnit
    from director_brain.utils import file_sha256
    from observation_service import pipeline

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "profile_change.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-profile-token")
    monkeypatch.setenv("SHOT_DISCOVERY_BACKEND", "scenedetect")
    media = tmp_path / "profile-asset.mp4"
    _make_test_video(str(media), duration_sec=2)
    local_client = TestClient(
        app, client=("127.0.0.1", 54130),
        headers={"Authorization": "Bearer p1-profile-token"},
    )
    manifest = _envelope_data(local_client.put(
        "/v1/projects/profile-change/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/profile-change",
            "boundary_evidence_refs": ["test-fixture://project/manifest"],
            "assets": [{
                "asset_id": "asset-a",
                "source_ref": str(media),
                "order": 0,
                "rights": {
                    "state": "local_processing_allowed",
                    "basis": "owner_permission",
                    "evidence_ref": "test-fixture://rights/asset-a",
                },
            }],
        }))
    assert manifest["current_revision"] == 1

    calls: list[str] = []

    def analyze_profiled_asset(path: str, *, cache=None):
        calls.append(Path(path).name)
        return [_make_obs("profile-source-a", "asset-a", 0, 1_000_000)
                .model_copy(update={
                    "media_hash": file_sha256(path),
                    "timebase_unit": TimebaseUnit.MICROSECONDS,
                })]

    monkeypatch.setattr(pipeline, "analyze_media", analyze_profiled_asset)
    first = _envelope_data(local_client.post(
        "/v1/projects/profile-change/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    cached_same_profile = _envelope_data(local_client.post(
        "/v1/projects/profile-change/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert first["cache_hit"] is False
    assert cached_same_profile["cache_hit"] is True
    assert calls == [media.name]

    monkeypatch.setenv("SHOT_DISCOVERY_BACKEND", "legacy")
    changed_profile = _envelope_data(local_client.post(
        "/v1/projects/profile-change/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert changed_profile["cache_hit"] is False
    assert changed_profile["context_id"] != first["context_id"]
    assert changed_profile["snapshot"]["sampling_config"]["analysis_profile"] != (
        first["snapshot"]["sampling_config"]["analysis_profile"])
    assert changed_profile["snapshot"]["asset_coverage"][0][
        "analysis_cache_state"] == "computed"
    assert calls == [media.name, media.name]

    repeated_changed_profile = _envelope_data(local_client.post(
        "/v1/projects/profile-change/film-context:snapshot",
        json={"manifest_revision": 1},
    ))
    assert repeated_changed_profile["cache_hit"] is True
    assert calls == [media.name, media.name]


def test_local_vlm_project_profile_uses_pinned_local_pipeline_and_reuses_cache(
    tmp_path, monkeypatch
):
    from director_brain.analysis_cache import RepositoryAnalysisCache
    from director_brain.models import ProjectAnalysisProfile
    from director_brain.utils import file_sha256
    from observation_service import pipeline
    from observation_service.ollama_vlm_adapter import OllamaVLMAdapter

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "local-vlm-project.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-local-vlm-token")
    monkeypatch.setenv("SHOT_DISCOVERY_BACKEND", "scenedetect")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_MODEL", "qwen3-vl:4b")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_DIGEST", "a" * 64)
    monkeypatch.setenv("PROJECT_LOCAL_VLM_RUNTIME_VERSION", "0.32.14")
    media = tmp_path / "local-vlm-asset.mp4"
    _make_test_video(str(media), duration_sec=2)
    local_client = TestClient(
        app, client=("127.0.0.1", 54131),
        headers={"Authorization": "Bearer p1-local-vlm-token"},
    )
    manifest = _envelope_data(local_client.put(
        "/v1/projects/local-vlm-project/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/local-vlm",
            "boundary_evidence_refs": ["test-fixture://project/local-vlm-manifest"],
            "analysis_profile": ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1.value,
            "assets": [{
                "asset_id": "asset-local-vlm",
                "source_ref": str(media),
                "order": 0,
                "rights": {
                    "state": "local_processing_allowed",
                    "basis": "owner_permission",
                    "evidence_ref": "test-fixture://rights/local-vlm-asset",
                },
            }],
        }))
    assert manifest["manifest"]["analysis_profile"] == "local_vlm_shadow_v1"

    calls: list[str] = []

    def analyze_local_profile(
        path, *, vlm, asr, vlm_adapter, vlm_cache
    ):
        calls.append(Path(path).name)
        assert vlm is True
        assert asr is False
        assert isinstance(vlm_adapter, OllamaVLMAdapter)
        assert vlm_adapter.enforce_loopback is True
        assert vlm_adapter.model_digest == "a" * 64
        assert isinstance(vlm_cache, RepositoryAnalysisCache)
        source_observation = _make_obs(
            "local-vlm-source-obs", "asset-local-vlm", 0, 1_000_000,
        ).model_copy(update={
            "media_hash": file_sha256(path),
            "provider": "ollama_qwen3_vl",
            "model_version": "qwen3-vl:4b@sha256:" + "a" * 64,
            "prompt_version": "vlm_prompt_v4_people",
        })
        technical = source_observation.model_copy(update={
            "observation_id": "local-vlm-tech-obs",
            "observation_type": "deterministic_technical",
            "provider": "deterministic_opencv",
            "model_version": "v1",
            "prompt_version": "n/a",
            "claim_kind": ClaimKind.MEASURED,
        })
        semantic = source_observation.model_copy(update={
            "observation_id": "local-vlm-semantic-obs",
            "media_asset_id": "shot-local-vlm",
            "observation_type": "vlm_semantic",
            "claim": json.dumps({
                "people": ["red coat, short hair"],
                "action_type": "action",
                "scene_description": "A person carries flowers across the room.",
                "temporal_notes": "The person turns toward the doorway.",
            }),
            "claim_kind": ClaimKind.MODEL_OBSERVATION,
            "review_state": "auto_generated",
        })
        return [technical, semantic]

    monkeypatch.setattr(pipeline, "analyze_media_full", analyze_local_profile)
    request = {"manifest_revision": 1}
    first = _envelope_data(local_client.post(
        "/v1/projects/local-vlm-project/film-context:snapshot", json=request,
    ))
    cached = _envelope_data(local_client.post(
        "/v1/projects/local-vlm-project/film-context:snapshot", json=request,
    ))

    assert first["cache_hit"] is False
    assert cached["cache_hit"] is True
    assert calls == [media.name]
    coverage = first["snapshot"]["asset_coverage"][0]
    assert coverage["analysis_state"] == "observed"
    assert coverage["analysis_cache_state"] == "computed"
    profile = first["snapshot"]["sampling_config"]["analysis_profile"]
    assert "model=qwen3-vl:4b@sha256:" + "a" * 64 in profile

    graph = _envelope_data(local_client.post(
        "/v1/projects/local-vlm-project/story-graph:build"))
    asset_graph = graph["story_graph"]["assets"][0]["story_graph"]
    assert asset_graph["schema_version"] == "1.2"
    person_mention = next(
        node for node in asset_graph["nodes"]
        if node["node_type"] == "person_mention")
    event_mention = next(
        node for node in asset_graph["nodes"]
        if node["node_type"] == "event_mention")
    stored_observations = [
        _envelope_data(local_client.get(
            f"/v1/projects/local-vlm-project/observations/{evidence_ref}"
        ))["observation"]
        for evidence_ref in first["snapshot"]["evidence_refs"]
    ]
    stored_semantic = next(
        item for item in stored_observations
        if item["observation_type"] == "vlm_semantic")
    assert person_mention["ref_id"] == stored_semantic["observation_id"]
    assert event_mention["ref_id"] == stored_semantic["observation_id"]
    assert person_mention["attributes"]["evidence_refs"] == [
        stored_semantic["observation_id"]]
    assert person_mention["attributes"]["identity_scope"] == "single_observation"
    assert person_mention["attributes"]["source_anchor"]["project_asset_id"] == (
        "asset-local-vlm")
    assert graph["story_graph"]["cross_asset_relations_state"] == "not_attempted"
    # Persisted graph reads must not require the inference provider currently
    # configured in this process. The graph already binds its exact Context.
    with monkeypatch.context() as no_model_config:
        no_model_config.delenv("PROJECT_LOCAL_VLM_MODEL", raising=False)
        no_model_config.delenv("PROJECT_LOCAL_VLM_DIGEST", raising=False)
        no_model_config.delenv("PROJECT_LOCAL_VLM_RUNTIME_VERSION", raising=False)
        no_model_config.delenv("OLLAMA_BASE_URL", raising=False)
        graph_response = local_client.get(
            "/v1/projects/local-vlm-project/story-graph")
        assert graph_response.status_code == 200
        graph_readback = _envelope_data(graph_response)
        assert graph_readback["story_graph"] == graph["story_graph"]
        exact_context_readback = local_client.get(
            "/v1/projects/local-vlm-project/story-graph",
            params={"context_id": graph["story_graph"]["context_id"]},
        )
        assert exact_context_readback.status_code == 200
        assert _envelope_data(exact_context_readback)["story_graph"] == graph["story_graph"]
        missing_context_readback = local_client.get(
            "/v1/projects/local-vlm-project/story-graph",
            params={"context_id": "ctx_project_not-persisted"},
        )
        assert missing_context_readback.status_code == 404
    assert calls == [media.name]

    preview_path = (
        f"/v1/projects/local-vlm-project/story-graph/"
        f"{graph['story_graph']['graph_id']}/mentions/"
        f"{person_mention['node_id']}/preview"
    )
    anonymous_client = TestClient(app, client=("127.0.0.1", 54132))
    assert anonymous_client.get(preview_path, params={
        "manifest_revision": 1,
        "relation_kind": "person_identity",
    }).status_code == 401

    preview_response = local_client.get(preview_path, params={
        "manifest_revision": 1,
        "relation_kind": "person_identity",
    })
    assert preview_response.status_code == 200
    assert "no-store" in preview_response.headers["cache-control"]
    preview = _envelope_data(preview_response)
    assert preview["source_hash_validation_state"] == "verified_before_and_after"
    assert preview["anchor"]["story_graph_node_id"] == person_mention["node_id"]
    assert [frame["relative_position"] for frame in preview["frames"]] == [
        0.15, 0.5, 0.85]
    import base64
    import hashlib
    for frame in preview["frames"]:
        jpeg = base64.b64decode(frame["jpeg_base64"], validate=True)
        assert jpeg.startswith(b"\xff\xd8\xff")
        assert hashlib.sha256(jpeg).hexdigest() == frame["sha256"]
        assert max(frame["width"], frame["height"]) <= 1280
    assert str(media) not in preview_response.text

    injected = local_client.post(
        "/v1/projects/local-vlm-project/film-context:snapshot",
        json={"manifest_revision": 1, "observations_by_asset": {
            "asset-local-vlm": _observations(),
        }},
    )
    assert injected.status_code == 422
    assert "本地模型分析 profile 必须由绑定的本地管线生成观测" in injected.text

    with media.open("ab") as source:
        source.write(b"source-changed-after-registration")
    stale_preview = local_client.get(preview_path, params={
        "manifest_revision": 1,
        "relation_kind": "person_identity",
    })
    assert stale_preview.status_code == 409


def test_local_asr_project_profile_maps_audio_clock_and_reuses_cache(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import director_brain.context_gateway as context_gateway
    from director_brain.models import ProjectAnalysisProfile, ProjectAssetTimeMap
    from director_brain.utils import file_sha256
    from observation_service import asr, pipeline

    db_path = tmp_path / "local-asr-project.db"
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-local-asr-token")
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    model_path = tmp_path / "asr-model"
    model_path.mkdir()
    (model_path / "model.bin").write_bytes(b"synthetic-model-artifact")
    monkeypatch.setenv("ASR_MODEL_PATH", str(model_path))
    model_digest = asr.local_model_tree_sha256(model_path)
    model_binding = (
        model_path, model_digest, model_path.name,
        "faster-whisper=test;ctranslate2=test;av=test",
    )
    monkeypatch.setattr(
        context_gateway, "project_local_asr_model_binding", lambda: model_binding)
    monkeypatch.setattr(
        api_main, "project_local_asr_model_binding", lambda: model_binding)

    media = tmp_path / "local-asr-offset-asset.mp4"
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i",
        "testsrc=duration=3:size=320x240:rate=25",
        "-itsoffset", "0.5", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=2",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "3",
        str(media), "-y",
    ], capture_output=True, text=True, check=True)

    technical = _make_obs("local-asr-tech", "asset-local-asr", 0, 1_000_000)
    technical = technical.model_copy(update={
        "media_hash": file_sha256(str(media)),
        "media_asset_id": media.name,
    })
    monkeypatch.setattr(
        pipeline,
        "analyze_media",
        lambda path, cache=None: [technical],
    )

    model_calls: list[str] = []

    class _FakeWhisper:
        def __init__(self, model_name, device="cpu", compute_type="int8"):
            assert model_name == str(model_path)

        def transcribe(self, video_path, **kwargs):
            model_calls.append(video_path)
            return [SimpleNamespace(start=0.25, end=0.50, text="spoken sample")], None

    monkeypatch.setattr(asr, "_whisper_cls", lambda: _FakeWhisper)
    client = TestClient(
        app,
        client=("127.0.0.1", 54134),
        headers={"Authorization": "Bearer p1-local-asr-token"},
    )
    project_id = "local-asr-project"
    manifest = _envelope_data(client.put(
        f"/v1/projects/{project_id}/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/local-asr",
            "boundary_evidence_refs": ["test-fixture://manifest/local-asr"],
            "analysis_profile": ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1.value,
            "assets": [{
                "asset_id": "asset-local-asr",
                "source_ref": str(media),
                "order": 0,
                "rights": {
                    "state": "local_processing_allowed",
                    "basis": "owner_permission",
                    "evidence_ref": "test-fixture://rights/local-asr",
                },
            }],
        }))
    asset = manifest["manifest"]["assets"][0]
    assert asset["has_audio"] is True
    time_map = ProjectAssetTimeMap.model_validate(asset["time_map"])
    stream_index, source_offset = time_map.first_audio_stream_clock_from_video()
    assert source_offset > 0

    request = {"manifest_revision": 1}
    first = _envelope_data(client.post(
        f"/v1/projects/{project_id}/film-context:snapshot", json=request))
    cached = _envelope_data(client.post(
        f"/v1/projects/{project_id}/film-context:snapshot", json=request))

    assert first["cache_hit"] is False
    assert cached["cache_hit"] is True
    assert model_calls == [str(media)]
    coverage = first["snapshot"]["asset_coverage"][0]
    assert coverage["analysis_state"] == "observed"
    assert coverage["analysis_cache_state"] == "computed"
    observations = [
        _envelope_data(client.get(
            f"/v1/projects/{project_id}/observations/{ref}"
        ))["observation"]
        for ref in first["snapshot"]["evidence_refs"]
    ]
    transcript = next(
        item for item in observations
        if item["observation_type"] == "speech_transcript")
    offset_us = round(source_offset * 1_000_000)
    assert transcript["start_frame"] == 250_000 + offset_us
    assert transcript["end_frame"] == 500_000 + offset_us
    assert transcript["timebase_unit"] == "microseconds"
    assert transcript["source_stream_index"] == stream_index
    assert transcript["project_asset_id"] == "asset-local-asr"
    assert transcript["media_hash"] == file_sha256(str(media))
    assert "provider=faster_whisper" in first["snapshot"]["model"]


def test_local_asr_api_retains_partial_and_failed_asset_states_without_caching(
    tmp_path, monkeypatch
):
    import sqlite3

    import director_brain.context_gateway as context_gateway
    from director_brain.models import ProjectAnalysisProfile
    from director_brain.utils import file_sha256
    from observation_service import asr, pipeline
    from observation_service.pipeline import ProjectASRAnalysisResult

    db_path = tmp_path / "local-asr-failure-states.db"
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-local-asr-state-token")
    monkeypatch.setenv("ASR_DEVICE", "cpu")

    model_path = tmp_path / "asr-model"
    model_path.mkdir()
    (model_path / "model.bin").write_bytes(b"synthetic-model-artifact")
    model_digest = asr.local_model_tree_sha256(model_path)
    model_binding = (
        model_path, model_digest, model_path.name,
        "faster-whisper=test;ctranslate2=test;av=test",
    )
    monkeypatch.setattr(
        context_gateway, "project_local_asr_model_binding", lambda: model_binding)
    monkeypatch.setattr(
        api_main, "project_local_asr_model_binding", lambda: model_binding)

    assets = []
    for asset_id, frequency in (("asset-partial", 440), ("asset-failed", 660)):
        media = tmp_path / f"{asset_id}.mp4"
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i",
            "testsrc=duration=2:size=320x240:rate=25",
            "-f", "lavfi", "-i",
            f"sine=frequency={frequency}:sample_rate=48000:duration=2",
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(media), "-y",
        ], capture_output=True, text=True, check=True)
        assets.append((asset_id, media))

    def report_asr_outcome(path, **kwargs):
        assert kwargs["source_has_audio"] is True
        asset_id = Path(path).stem
        if asset_id == "asset-partial":
            technical = _make_obs(
                "local-asr-partial-tech", asset_id, 0, 1_000_000,
            ).model_copy(update={"media_hash": file_sha256(path)})
            return ProjectASRAnalysisResult(
                [technical], "speech_analysis_partial")
        return ProjectASRAnalysisResult([], "speech_analysis_failed")

    monkeypatch.setattr(
        pipeline, "analyze_media_with_project_asr", report_asr_outcome)
    client = TestClient(
        app,
        client=("127.0.0.1", 54136),
        headers={"Authorization": "Bearer p1-local-asr-state-token"},
    )
    project_id = "local-asr-failure-states"
    manifest_data = _envelope_data(client.put(
        f"/v1/projects/{project_id}/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/local-asr-states",
            "boundary_evidence_refs": ["test-fixture://manifest/local-asr-states"],
            "analysis_profile": ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1.value,
            "assets": [
                {
                    "asset_id": asset_id,
                    "source_ref": str(media),
                    "order": order,
                    "rights": {
                        "state": "local_processing_allowed",
                        "basis": "owner_permission",
                        "evidence_ref": f"test-fixture://rights/{asset_id}",
                    },
                }
                for order, (asset_id, media) in enumerate(assets)
            ],
        }))
    assert manifest_data["current_revision"] == 1
    assert all(asset["has_audio"] for asset in manifest_data["manifest"]["assets"])

    result = _envelope_data(client.post(
        f"/v1/projects/{project_id}/film-context:snapshot",
        json={"manifest_revision": 1},
    ))

    assert result["coverage"] == "project_multi_asset_partial"
    by_id = {entry["asset_id"]: entry for entry in result["snapshot"]["asset_coverage"]}
    assert by_id["asset-partial"]["analysis_state"] == "partial"
    assert by_id["asset-partial"]["failure_code"] == "speech_analysis_partial"
    assert by_id["asset-partial"]["analysis_cache_state"] == "partial"
    assert by_id["asset-failed"]["analysis_state"] == "failed"
    assert by_id["asset-failed"]["failure_code"] == "speech_analysis_failed"
    assert by_id["asset-failed"]["analysis_cache_state"] == "failed"
    assert len(result["snapshot"]["evidence_refs"]) == 1
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM analysis_result_cache"
        ).fetchone()[0] == 0


def test_local_vlm_partial_context_retries_without_overwriting_failed_evidence(
    tmp_path, monkeypatch
):
    import sqlite3

    from director_brain.models import ProjectAnalysisProfile
    from director_brain.utils import file_sha256
    from observation_service import pipeline

    db_path = tmp_path / "local-vlm-partial-retry.db"
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-partial-retry-token")
    monkeypatch.setenv("SHOT_DISCOVERY_BACKEND", "scenedetect")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_MODEL", "qwen3-vl:4b")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_DIGEST", "b" * 64)
    monkeypatch.setenv("PROJECT_LOCAL_VLM_RUNTIME_VERSION", "0.32.14")

    media = tmp_path / "partial-retry-asset.mp4"
    _make_test_video(str(media), duration_sec=2)
    local_client = TestClient(
        app, client=("127.0.0.1", 54130),
        headers={"Authorization": "Bearer p1-partial-retry-token"},
    )
    project_id = "local-vlm-partial-retry"
    manifest = _envelope_data(local_client.put(
        f"/v1/projects/{project_id}/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/partial-retry",
            "boundary_evidence_refs": ["test-fixture://project/partial-retry-manifest"],
            "analysis_profile": ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1.value,
            "assets": [{
                "asset_id": "asset-partial-retry",
                "source_ref": str(media),
                "order": 0,
                "rights": {
                    "state": "local_processing_allowed",
                    "basis": "owner_permission",
                    "evidence_ref": "test-fixture://rights/partial-retry-asset",
                },
            }],
        }))
    assert manifest["current_revision"] == 1

    calls = 0

    def analyze_local_profile(
        path, *, vlm, asr, vlm_adapter, vlm_cache
    ):
        nonlocal calls
        calls += 1
        assert vlm is True and asr is False
        content_hash = file_sha256(path)
        technical = _make_obs(
            "partial-retry-tech-source", "shot-retry", 0, 1_000_000,
        ).model_copy(update={
            "media_hash": content_hash,
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        })
        semantic_claim_kind = (
            ClaimKind.NOT_DETERMINED if calls == 1
            else ClaimKind.MODEL_OBSERVATION
        )
        semantic = _make_obs(
            "partial-retry-semantic-source", "shot-retry", 0, 1_000_000,
        ).model_copy(update={
            "media_hash": content_hash,
            "media_asset_id": "shot-retry",
            "observation_type": "vlm_semantic",
            "provider": "ollama_qwen3_vl",
            "model_version": "qwen3-vl:4b@sha256:" + "b" * 64,
            "prompt_version": "vlm_prompt_v4_people",
            "claim_kind": semantic_claim_kind,
            "review_state": "auto_generated",
            "claim": "model request failed"
            if calls == 1 else "person walks into the room",
        })
        return [technical, semantic]

    monkeypatch.setattr(pipeline, "analyze_media_full", analyze_local_profile)
    request = {"manifest_revision": 1}
    first = _envelope_data(local_client.post(
        f"/v1/projects/{project_id}/film-context:snapshot", json=request,
    ))
    assert first["cache_hit"] is False
    assert first["coverage"] == "project_single_asset_partial"
    first_asset = first["snapshot"]["asset_coverage"][0]
    assert first_asset["analysis_state"] == "partial"
    assert first_asset["analysis_cache_state"] == "partial"
    assert first_asset["failure_code"] == "semantic_analysis_partial"
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM analysis_result_cache"
        ).fetchone()[0] == 0

    first_observations = [
        _envelope_data(local_client.get(
            f"/v1/projects/{project_id}/observations/{evidence_ref}"
        ))["observation"]
        for evidence_ref in first["snapshot"]["evidence_refs"]
    ]
    first_failure = next(
        item for item in first_observations
        if item["observation_type"] == "vlm_semantic")
    assert first_failure["claim_kind"] == ClaimKind.NOT_DETERMINED.value

    second = _envelope_data(local_client.post(
        f"/v1/projects/{project_id}/film-context:snapshot", json=request,
    ))
    assert second["cache_hit"] is False
    assert second["coverage"] == "project_single_asset_observed"
    second_asset = second["snapshot"]["asset_coverage"][0]
    assert second_asset["analysis_state"] == "observed"
    assert second_asset["analysis_cache_state"] == "computed"
    second_observations = [
        _envelope_data(local_client.get(
            f"/v1/projects/{project_id}/observations/{evidence_ref}"
        ))["observation"]
        for evidence_ref in second["snapshot"]["evidence_refs"]
    ]
    second_semantic = next(
        item for item in second_observations
        if item["observation_type"] == "vlm_semantic")
    assert second_semantic["claim_kind"] == ClaimKind.MODEL_OBSERVATION.value
    assert second_semantic["observation_id"] != first_failure["observation_id"]
    assert _envelope_data(local_client.get(
        f"/v1/projects/{project_id}/observations/"
        f"{first_failure['observation_id']}"
    ))["observation"]["claim_kind"] == ClaimKind.NOT_DETERMINED.value
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM analysis_result_cache"
        ).fetchone()[0] == 1

    third = _envelope_data(local_client.post(
        f"/v1/projects/{project_id}/film-context:snapshot", json=request,
    ))
    assert third["cache_hit"] is True
    assert calls == 2


def test_project_manifest_rejects_duplicate_ids_and_orders_before_media_read(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "invalid_manifest.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-invalid-token")
    local_client = TestClient(
        app, client=("127.0.0.1", 54129),
        headers={"Authorization": "Bearer p1-invalid-token"},
    )
    media_reads: list[str] = []

    def forbidden_media_read(source_ref: str, *_args, **_kwargs):
        media_reads.append(source_ref)
        raise AssertionError("invalid manifest must fail before media access")

    monkeypatch.setattr(
        "api.main._authorized_local_media_path", forbidden_media_read)

    def asset(asset_id: str, order: int) -> dict:
        return {
            "asset_id": asset_id,
            "source_ref": str(tmp_path / "not-opened.mp4"),
            "order": order,
            "rights": {
                "state": "local_processing_allowed",
                "basis": "owner_permission",
                "evidence_ref": f"test-fixture://rights/{asset_id}-{order}",
            },
        }

    invalid_asset_sets = [
        ("duplicate-ids", [asset("same-id", 0), asset("same-id", 1)]),
        ("duplicate-orders", [asset("asset-a", 0), asset("asset-b", 0)]),
    ]
    for project_id, assets in invalid_asset_sets:
        response = local_client.put(
            f"/v1/projects/{project_id}/film-manifest", json={
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": f"test-fixture://project/{project_id}",
                "boundary_evidence_refs": ["test-fixture://project/manifest"],
                "assets": assets,
            })
        assert response.status_code == 422
        assert media_reads == []
        assert not (tmp_path / "invalid_manifest.db").exists()


def test_project_manifest_reports_duplicate_content_hash_asset_ids(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "duplicate_assets.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-duplicate-token")
    media = tmp_path / "same-source.mp4"
    _make_test_video(str(media), duration_sec=2)
    local_client = TestClient(
        app, client=("127.0.0.1", 54128),
        headers={"Authorization": "Bearer p1-duplicate-token"},
    )

    rejected = local_client.put(
        "/v1/projects/duplicate-assets/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/duplicate-assets",
            "boundary_evidence_refs": ["test-fixture://project/manifest"],
            "assets": [{
                "asset_id": asset_id,
                "source_ref": str(media),
                "order": order,
                "rights": {
                    "state": "local_processing_allowed",
                    "basis": "owner_permission",
                    "evidence_ref": f"test-fixture://rights/{asset_id}",
                },
            } for order, asset_id in enumerate(("asset-copy-a", "asset-copy-b"))],
        })

    assert rejected.status_code == 422
    assert "asset-copy-a" in rejected.json()["detail"]
    assert "asset-copy-b" in rejected.json()["detail"]
    assert not (tmp_path / "duplicate_assets.db").exists()
    assert local_client.get(
        "/v1/projects/duplicate-assets/film-manifest").status_code == 404


def test_project_manifest_is_loopback_and_media_root_scoped(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    media = allowed_root / "selected.mp4"
    media.write_bytes(b"not opened because the peer is remote")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "must_not_be_created.db"))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(allowed_root))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-test-token")
    body = {
        "expected_revision": 0,
        "boundary_basis": "user_project_manifest",
        "boundary_source_ref": "test-fixture://project/locality",
        "boundary_evidence_refs": ["test-fixture://project/manifest"],
        "assets": [{
            "asset_id": "asset-a",
            "source_ref": str(media),
            "order": 0,
            "rights": {
                "state": "local_processing_allowed",
                "basis": "owner_permission",
                "evidence_ref": "test-fixture://rights/asset-a",
            },
        }],
    }
    remote_client = TestClient(
        app, client=("10.0.0.8", 54125),
        headers={"Authorization": "Bearer p1-test-token"},
    )
    remote = remote_client.put(
        "/v1/projects/project-remote/film-manifest", json=body)
    assert remote.status_code == 403

    local_client = TestClient(
        app, client=("127.0.0.1", 54126),
        headers={"Authorization": "Bearer p1-test-token"},
    )
    outside_path = tmp_path / "outside.mp4"
    outside_path.write_bytes(b"outside the configured root")
    outside_root = {**body, "assets": [{**body["assets"][0],
                                         "source_ref": str(outside_path)}]}
    escaped = local_client.put(
        "/v1/projects/project-local/film-manifest", json=outside_root)
    assert escaped.status_code == 403
    assert not (tmp_path / "must_not_be_created.db").exists()


def test_changed_project_source_does_not_overwrite_current_context(
    tmp_path, monkeypatch
):
    """A changed source must preserve the last valid, persisted snapshot."""
    import sqlite3

    from observation_service import pipeline

    db_path = tmp_path / "context_immutability.db"
    media = tmp_path / "asset.mp4"
    _make_test_video(str(media), duration_sec=2)
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-immutable-token")
    monkeypatch.setattr(pipeline, "analyze_media", lambda _path, *, cache=None: [])
    local_client = TestClient(
        app, client=("127.0.0.1", 54130),
        headers={"Authorization": "Bearer p1-immutable-token"},
    )

    manifest = local_client.put(
        "/v1/projects/immutable-context/film-manifest", json={
            "expected_revision": 0,
            "boundary_basis": "user_project_manifest",
            "boundary_source_ref": "test-fixture://project/immutable-context",
            "boundary_evidence_refs": ["test-fixture://project/manifest"],
            "assets": [{
                "asset_id": "asset-a",
                "source_ref": str(media),
                "order": 0,
                "rights": {
                    "state": "local_processing_allowed",
                    "basis": "owner_permission",
                    "evidence_ref": "test-fixture://rights/asset-a",
                },
            }],
        },
    )
    assert manifest.status_code == 200

    first_response = local_client.post(
        "/v1/projects/immutable-context/film-context:snapshot",
        json={"manifest_revision": 1},
    )
    assert first_response.status_code == 200
    first = _envelope_data(first_response)
    assert first["cache_hit"] is False
    context_id = first["context_id"]
    with sqlite3.connect(db_path) as connection:
        original_row = connection.execute(
            "SELECT data FROM film_context_snapshots WHERE context_id = ?",
            (context_id,),
        ).fetchone()
    assert original_row is not None

    media.write_bytes(media.read_bytes() + b"changed-after-context")
    changed_response = local_client.post(
        "/v1/projects/immutable-context/film-context:snapshot",
        json={"manifest_revision": 1},
    )

    assert changed_response.status_code == 409
    assert "asset-a" in changed_response.json()["detail"]
    assert "manifest revision" in changed_response.json()["detail"]
    assert "现有上下文已保留" in changed_response.json()["detail"]
    with sqlite3.connect(db_path) as connection:
        preserved_row = connection.execute(
            "SELECT data FROM film_context_snapshots WHERE context_id = ?",
            (context_id,),
        ).fetchone()
        context_count = connection.execute(
            "SELECT COUNT(*) FROM film_context_snapshots WHERE project_id = ?",
            ("immutable-context",),
        ).fetchone()[0]
    assert preserved_row == original_row
    assert context_count == 1
    readback = local_client.get(f"/v1/film-context/{context_id}")
    assert readback.status_code == 200
    assert _envelope_data(readback)["snapshot"] == first["snapshot"]


def test_project_neutral_cache_reuse_keeps_project_evidence_isolated(
    tmp_path, monkeypatch
):
    """Identical authorized media may reuse analysis without aliasing records."""
    import sqlite3
    import shutil

    from director_brain.utils import file_sha256
    from observation_service import pipeline

    db_path = tmp_path / "project_cache_isolation.db"
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("DIRECTOR_BRAIN_PROJECT_API_TOKEN", "p1-cache-isolation")
    monkeypatch.setenv("SHOT_DISCOVERY_BACKEND", "scenedetect")

    media_a = tmp_path / "project-a" / "same-content.mp4"
    media_b = tmp_path / "project-b" / "same-content.mp4"
    media_a.parent.mkdir()
    media_b.parent.mkdir()
    _make_test_video(str(media_a), duration_sec=2)
    shutil.copyfile(media_a, media_b)
    assert file_sha256(media_a) == file_sha256(media_b)

    client = TestClient(
        app, client=("127.0.0.1", 54132),
        headers={"Authorization": "Bearer p1-cache-isolation"},
    )
    projects = [
        ("project-cache-isolation-a", "asset-a", media_a),
        ("project-cache-isolation-b", "asset-b", media_b),
    ]
    calls: list[str] = []

    def analyze_same_content(path: str, *, cache=None):
        calls.append(path)
        return [_make_obs(
            "source-observation-shared", "source-asset", 0, 1_000_000,
        ).model_copy(update={
            "project_id": "analysis-cache-origin-marker",
            "media_hash": file_sha256(path),
            "source_ref": "analysis-cache-origin-path-marker",
        })]

    monkeypatch.setattr(pipeline, "analyze_media", analyze_same_content)
    snapshots: dict[str, dict] = {}
    observations: dict[str, dict] = {}

    for project_id, asset_id, media_path in projects:
        manifest_response = client.put(
            f"/v1/projects/{project_id}/film-manifest", json={
                "expected_revision": 0,
                "boundary_basis": "user_project_manifest",
                "boundary_source_ref": f"test-fixture://project/{project_id}",
                "boundary_evidence_refs": [
                    f"test-fixture://project/{project_id}/manifest",
                ],
                "assets": [{
                    "asset_id": asset_id,
                    "source_ref": str(media_path),
                    "order": 0,
                    "rights": {
                        "state": "local_processing_allowed",
                        "basis": "owner_permission",
                        "evidence_ref": (
                            f"test-fixture://rights/{project_id}/{asset_id}"),
                    },
                }],
            },
        )
        assert manifest_response.status_code == 200, manifest_response.text

        context_response = client.post(
            f"/v1/projects/{project_id}/film-context:snapshot",
            json={"manifest_revision": 1},
        )
        assert context_response.status_code == 200, context_response.text
        snapshot = context_response.json()["data"]
        snapshots[project_id] = snapshot
        assert snapshot["snapshot"]["asset_coverage"][0][
            "source_content_hash"] == file_sha256(media_path)

        evidence_ids = snapshot["snapshot"]["evidence_refs"]
        assert len(evidence_ids) == 1
        observation_response = client.get(
            f"/v1/projects/{project_id}/observations/{evidence_ids[0]}",
        )
        assert observation_response.status_code == 200
        observations[project_id] = observation_response.json()["data"][
            "observation"]

    project_a, project_b = (item[0] for item in projects)
    asset_a, asset_b = (item[1] for item in projects)
    context_a = snapshots[project_a]
    context_b = snapshots[project_b]
    observation_a = observations[project_a]
    observation_b = observations[project_b]

    assert calls == [str(media_a)]
    assert context_a["cache_hit"] is False
    assert context_b["cache_hit"] is False
    assert context_b["snapshot"]["asset_coverage"][0][
        "analysis_cache_state"] == "reused"
    assert context_a["context_id"] != context_b["context_id"]
    assert context_a["snapshot"]["project_id"] == project_a
    assert context_b["snapshot"]["project_id"] == project_b

    scoped_a = client.get(
        f"/v1/projects/{project_a}/film-context/{context_a['context_id']}"
    )
    scoped_b = client.get(
        f"/v1/projects/{project_b}/film-context/{context_b['context_id']}"
    )
    assert scoped_a.status_code == 200, scoped_a.text
    assert scoped_b.status_code == 200, scoped_b.text
    assert scoped_a.json()["data"]["snapshot"]["project_id"] == project_a
    assert scoped_b.json()["data"]["snapshot"]["project_id"] == project_b
    assert scoped_a.json()["data"]["source_hash_validation_state"] == (
        "not_revalidated_by_read")
    assert client.get(
        f"/v1/projects/{project_a}/film-context/{context_b['context_id']}"
    ).status_code == 404
    assert client.get(
        f"/v1/projects/{project_b}/film-context/{context_a['context_id']}"
    ).status_code == 404

    assert observation_a["project_id"] == project_a
    assert observation_b["project_id"] == project_b
    assert observation_a["project_asset_id"] == asset_a
    assert observation_b["project_asset_id"] == asset_b
    assert observation_a["observation_id"] != observation_b["observation_id"]
    assert observation_a["source_ref"] == f"local-media://{asset_a}"
    assert observation_b["source_ref"] == f"local-media://{asset_b}"
    assert "analysis-cache-origin-marker" not in json.dumps(observation_b)
    assert "analysis-cache-origin-path-marker" not in json.dumps(observation_b)

    cross_project_a = client.get(
        f"/v1/projects/{project_b}/observations/"
        f"{observation_a['observation_id']}"
    )
    cross_project_b = client.get(
        f"/v1/projects/{project_a}/observations/"
        f"{observation_b['observation_id']}"
    )
    assert cross_project_a.status_code == 404
    assert cross_project_b.status_code == 404

    with sqlite3.connect(db_path) as connection:
        cache_count = connection.execute(
            "SELECT COUNT(*) FROM analysis_result_cache"
        ).fetchone()[0]
        project_rows = connection.execute(
            "SELECT project_id, COUNT(*) FROM film_observations "
            "GROUP BY project_id ORDER BY project_id"
        ).fetchall()
    assert cache_count == 1
    assert project_rows == [(project_a, 1), (project_b, 1)]
