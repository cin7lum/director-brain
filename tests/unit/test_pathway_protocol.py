"""T4 信号通路协议单元测试。"""
from __future__ import annotations

import pytest

from director_brain import pathway_protocol as pp


@pytest.fixture()
def restore_statuses():
    """每个用例后恢复登记表到用例前快照（随默认态演进，不写死）。"""
    names = ("vlm_semantic", "relation_inference", "asr_transcript",
             "semantic_reasoner")
    snapshot = {name: pp.get_pathway_status(name) for name in names}
    yield
    for name, status in snapshot.items():
        pp.set_pathway_status(name, status)


def test_defaults_experimental_with_asr_shadow():
    """默认态（2026-09-30）：vlm_semantic 灰度转 ACTIVE；relation 仍 EXPERIMENTAL；
    asr SHADOW。"""
    assert pp.get_pathway_status("vlm_semantic") is pp.PathwayStatus.ACTIVE
    assert pp.get_pathway_status("relation_inference") is pp.PathwayStatus.EXPERIMENTAL
    assert pp.get_pathway_status("asr_transcript") is pp.PathwayStatus.ACTIVE  # D3 灰度转正


def test_describe_lists_all_pathways():
    text = pp.describe()
    for name in ("vlm_semantic", "relation_inference", "asr_transcript"):
        assert name in text
    assert "EXPERIMENTAL" in text


def test_decision_use_blocked_unless_active(restore_statuses):
    pp.set_pathway_status("vlm_semantic", pp.PathwayStatus.EXPERIMENTAL)
    with pytest.raises(pp.PathwayNotActiveError):
        pp.ensure_decision_use_allowed("vlm_semantic")

    pp.set_pathway_status("vlm_semantic", pp.PathwayStatus.SHADOW)
    with pytest.raises(pp.PathwayNotActiveError):
        pp.ensure_decision_use_allowed("vlm_semantic")  # SHADOW 只记录不驱动

    pp.set_pathway_status("vlm_semantic", pp.PathwayStatus.ACTIVE)
    pp.ensure_decision_use_allowed("vlm_semantic")  # 不抛


def test_status_transitions_roundtrip(restore_statuses):
    pp.set_pathway_status("asr_transcript", pp.PathwayStatus.SHADOW)
    assert pp.get_pathway_status("asr_transcript") is pp.PathwayStatus.SHADOW
    pp.set_pathway_status("asr_transcript", "ACTIVE")  # 字符串也可
    assert pp.get_pathway_status("asr_transcript") is pp.PathwayStatus.ACTIVE


def test_unknown_pathway_rejected():
    with pytest.raises(pp.UnknownPathwayError):
        pp.get_pathway_status("nonexistent_pathway")
    with pytest.raises(pp.UnknownPathwayError):
        pp.set_pathway_status("nonexistent_pathway", pp.PathwayStatus.ACTIVE)


def test_reasoner_blocks_vlm_when_experimental(restore_statuses):
    """决策入口闸门：EXPERIMENTAL 下传入 VLM 观测 → PathwayNotActiveError。"""
    import json
    import time

    pp.set_pathway_status("vlm_semantic", pp.PathwayStatus.EXPERIMENTAL)

    from director_brain.director_reasoner import HeuristicDirectorReasoner, _build_candidates
    from director_brain.models.film_observation import ClaimKind, FilmObservation

    def vlm_obs():
        return FilmObservation(
            observation_id="vlm_0", media_asset_id="shot_0",
            media_hash="h0", start_frame=0, end_frame=2_000_000,
            timebase=1_000_000, observation_type="vlm_semantic",
            claim=json.dumps({"shot_function": "ACTION", "proposed_role_v2": "hero"}),
            provider="t", model_version="t", prompt_version="t", confidence=0.7,
            review_state="auto_generated", claim_kind=ClaimKind.MODEL_OBSERVATION,
            schema_version="1.0", project_id="t", created_at=int(time.time()),
            producer="t", source_ref="t.mp4",
        )

    with pytest.raises(pp.PathwayNotActiveError):
        _build_candidates(tech_obs=[], vlm_obs=[vlm_obs()])
    with pytest.raises(pp.PathwayNotActiveError):
        pp.ensure_decision_use_allowed("vlm_semantic")
    # 空列表不触发闸门（无 VLM 信号 = 纯确定性路径）
    assert _build_candidates(tech_obs=[], vlm_obs=[]) == []
