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
import subprocess

import pytest
from fastapi.testclient import TestClient

from api.main import app
from director_brain.models import ClaimKind, FilmObservation

client = TestClient(app)


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


def test_confirm_strategy_endpoint(video_path):
    plan, edl = _plan_and_edl(video_path)
    data = _envelope_data(client.post(
        f"/v1/director-plans/{plan['plan_id']}:confirm-strategy",
        json={"plan_id": plan["plan_id"], "edl_id": edl["edl_id"],
              "plan_json": plan, "edl_json": edl, "confirmed_by": "owner"}))
    assert data["state"] == "strategy_confirmed"
    assert data["confirmation"]["plan_hash"]


def test_validate_endpoint(video_path):
    plan, edl = _plan_and_edl(video_path)
    data = _envelope_data(client.post(
        f"/v1/director-plans/{plan['plan_id']}:validate",
        json={"plan_json": plan, "edl_json": edl,
              "observations_json": _observations()}))
    assert "valid" in data and "errors" in data


def test_propose_revision_endpoint():
    data = _envelope_data(client.post("/v1/revisions:propose", json={
        "finding_ids": ["fql_001"], "change_summary": "黑帧段剔除"}))
    assert data["status"] == "DRAFT"
    assert data["source_finding_ids"] == ["fql_001"]


def test_decision_ledger_endpoint():
    resp = client.get("/v1/projects/proj_api/decision-ledger")
    body = resp.json()
    assert resp.status_code == 200
    assert "entries" in body["data"]
