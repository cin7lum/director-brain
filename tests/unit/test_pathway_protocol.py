"""T4 信号通路协议单元测试。"""
from __future__ import annotations

import pytest

from director_brain import pathway_protocol as pp


@pytest.fixture()
def restore_statuses():
    """每个用例后恢复登记表默认态。"""
    yield
    for name in ("vlm_semantic", "relation_inference", "asr_transcript"):
        pp.set_pathway_status(name, pp.PathwayStatus.EXPERIMENTAL)


def test_defaults_experimental_with_asr_shadow():
    """vlm/relation 默认 EXPERIMENTAL（止血态）；asr 已随主链计算+归因上报 → SHADOW。"""
    assert pp.get_pathway_status("vlm_semantic") is pp.PathwayStatus.EXPERIMENTAL
    assert pp.get_pathway_status("relation_inference") is pp.PathwayStatus.EXPERIMENTAL
    assert pp.get_pathway_status("asr_transcript") is pp.PathwayStatus.SHADOW


def test_describe_lists_all_pathways():
    text = pp.describe()
    for name in ("vlm_semantic", "relation_inference", "asr_transcript"):
        assert name in text
    assert "EXPERIMENTAL" in text


def test_decision_use_blocked_unless_active(restore_statuses):
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


def test_reasoner_blocks_vlm_when_experimental():
    """决策入口闸门：EXPERIMENTAL 下传入 VLM 观测 → PathwayNotActiveError。"""
    import json
    import time

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
