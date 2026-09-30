"""阶段 P3-2 语义评分库测试 + 候选①内核融合测试。

评分数学（compute_semantic_score）沿用原 P3-2 用例；独立选片循环
（semantic_aware_select / fused_selection）已删除——选片垄断权在内核，
对应行为改由 HeuristicDirectorReasoner 融合测试覆盖。
"""
import json
import time

from director_brain.acts import ROLE_TO_ACT
from director_brain.brief_compiler import compile_brief
from director_brain.director_reasoner import _build_candidates, get_director_reasoner
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.pathway_protocol import (
    PathwayNotActiveError,
    PathwayStatus,
    set_pathway_status,
)
from director_brain.semantic_scorer import SemanticScore, compute_semantic_score
from director_brain.story_graph_builder import build_story_graph

import pytest


def _brief(arc="upbeat", lang="fast_cut", target_s: int = 10) -> DirectorBrief:
    return DirectorBrief(
        schema_version="1.0", project_id="t", created_at=0, producer="t",
        source_ref="t.mp4", brief_id="b1", version="0.1", source_text="test",
        language="zh", intent="test", audience="not_determined",
        target_duration=target_s * 1_000_000, source_duration_us=20_000_000,
        delivery_profile="short_form", themes=[], relationships=[],
        emotional_arc=arc, visual_language="consistent_exposure",
        editing_language=lang, sound_language="no_speech",
        must_include=[], must_avoid=[], privacy_constraints=[],
        approval_state="draft")


def _cand(shot_id: str, blur: float, dur_s: float) -> dict:
    return {"source_shot_id": shot_id, "blur_score": blur,
            "duration_us": int(dur_s * 1_000_000),
            "exposure_ok": True, "technical_usable": True}


def _sem(role: str, tone: str, action: str, imp: int, desc: str) -> dict:
    return {"narrative_role": role, "emotional_tone": tone,
            "action_type": action, "importance": imp,
            "scene_description": desc}


# ---------------------------------------------------------------------------
# 评分数学（compute_semantic_score）
# ---------------------------------------------------------------------------

def test_narrative_role_drives_act_assignment():
    """climax 镜头 assigned_act=peak，setup→hook（映射来自 acts 单源）。"""
    brief = _brief()
    c = _cand("shot_a", 200.0, 3.0)
    sem = _sem("climax", "tense", "action", 5, "爆炸场景")
    score = compute_semantic_score(c, sem, brief, "peak", [])
    assert score.assigned_act == "peak"
    assert score.narrative_bonus > 0


def test_emotional_tone_match():
    """情绪弧 upbeat 匹配 energetic 镜头加分。"""
    brief = _brief(arc="upbeat")
    c = _cand("shot_a", 200.0, 3.0)
    sem = _sem("climax", "energetic", "action", 5, "激烈战斗")
    score = compute_semantic_score(c, sem, brief, "peak", [])
    assert score.emotion_match > 0


def test_action_type_match():
    """fast_cut 意图下 action_type=action 加分。"""
    brief = _brief(lang="fast_cut")
    c = _cand("shot_a", 200.0, 3.0)
    sem = _sem("climax", "tense", "action", 5, "爆炸")
    score = compute_semantic_score(c, sem, brief, "peak", [])
    assert score.action_match > 0


def test_diversity_penalty_for_similar_descriptions():
    """描述相似的镜头被降权（避免重复感）。"""
    brief = _brief()
    c = _cand("shot_a", 200.0, 3.0)
    sem = _sem("climax", "tense", "action", 5, "雪山中的人物手持木杖站立")
    seen = ["雪山中的人物手持木杖站立在山上"]
    score = compute_semantic_score(c, sem, brief, "peak", seen)
    assert score.diversity_penalty < 0


# ---------------------------------------------------------------------------
# 候选①内核融合（HeuristicDirectorReasoner）
# ---------------------------------------------------------------------------

def _tech_obs(idx: int, start_us: int, end_us: int, blur: float = 150.0) -> FilmObservation:
    claim = json.dumps({"blur_score": blur, "brightness_mean": 120.0,
                        "exposure_ok": True, "shake_score": 5.0})
    return FilmObservation(
        observation_id=f"det_{idx}", media_asset_id=f"shot_{idx:08d}",
        media_hash=f"hash_{idx}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim=claim, provider="deterministic_opencv", model_version="opencv",
        prompt_version="n/a", confidence=1.0, review_state="auto_verified",
        claim_kind=ClaimKind.MEASURED, schema_version="1.0",
        project_id="t", created_at=int(time.time()),
        producer="deterministic_opencv", source_ref="t.mp4")


def _vlm_obs(idx: int, start_us: int, end_us: int, **claim_fields) -> FilmObservation:
    base = {"shot_function": "ACTION", "proposed_role_v2": "support",
            "motion_amount": "subtle", "frame_description": "test",
            "importance": None, "narrative_role": None,
            "emotional_tone": None, "action_type": None,
            "scene_description": ""}
    base.update(claim_fields)
    return FilmObservation(
        observation_id=f"vlm_{idx}", media_asset_id=f"shot_{idx:08d}",
        media_hash=f"hash_{idx}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="vlm_semantic",
        claim=json.dumps(base, ensure_ascii=False),
        provider="ollama_qwen3_vl", model_version="qwen3-vl:latest",
        prompt_version="vlm_prompt_v3_semantic", confidence=0.7,
        review_state="auto_generated", claim_kind=ClaimKind.MODEL_OBSERVATION,
        schema_version="1.0", project_id="t", created_at=int(time.time()),
        producer="ollama_qwen3_vl", source_ref="t.mp4")


@pytest.fixture()
def vlm_active():
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    yield
    set_pathway_status("vlm_semantic", PathwayStatus.EXPERIMENTAL)


def _three_shots():
    return [
        _tech_obs(1, 0, 3_000_000, blur=150.0),
        _tech_obs(2, 3_000_000, 6_000_000, blur=180.0),
        _tech_obs(3, 6_000_000, 9_000_000, blur=120.0),
    ]


def test_kernel_semantic_role_moves_act():
    """源时间在 hook 的 climax 镜头被语义分到 peak 幕（时间比例被接管）。"""
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        obs = _three_shots() + [
            # shot_1 源时间最早（时间图归 hook），但语义是 climax
            _vlm_obs(1, 0, 3_000_000, narrative_role="climax",
                     emotional_tone="tense", action_type="action",
                     importance=5, scene_description="爆炸高潮"),
            _vlm_obs(2, 3_000_000, 6_000_000, narrative_role="setup",
                     emotional_tone="calm", action_type="establishing",
                     importance=4, scene_description="开场全景"),
            _vlm_obs(3, 6_000_000, 9_000_000, narrative_role="resolution",
                     emotional_tone="calm", action_type="sensory",
                     importance=3, scene_description="收尾宁静"),
        ]
        brief = _brief()
        graph = build_story_graph(brief, obs[:3])
        reasoner = get_director_reasoner("heuristic")
        edl, plan = reasoner.generate_plan(brief, graph, obs)
        act_of = {e.source_asset_id: e.act for e in edl.ordered_edits}
        assert act_of.get("shot_00000001") == "peak"
        assert act_of.get("shot_00000002") == "hook"
        assert act_of.get("shot_00000003") == "resolve"
        assert any(q.startswith("semantic_selection:applied")
                   for q in plan.open_questions)
        # 语义幕分配不是降级：degraded 不得因此置位
        assert not plan.degraded or all(
            not ev.startswith("semantic_act_assignment")
            for ev in plan.degradation_events)
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.EXPERIMENTAL)


def test_kernel_semantic_score_drives_selection_under_quota():
    """配额紧张时 importance 高的镜头胜出（修复 S4 死代码后真正生效）。

    语义分决定"谁入选"（贪心）；幕内顺序仍按时间轴（除非叙事序介入）——
    这是内核设计而非缺陷。
    """
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        # 同幕（都语义标为 development）：blur 高的 importance=1（160 分），
        # blur 低的 importance=5（240 分）。develop 配额 = 0.35×6s = 2.1s，
        # 只够一个 2s 镜头——语义分高者入选。
        obs = [
            _tech_obs(1, 0, 2_000_000, blur=200.0),
            _tech_obs(2, 2_000_000, 4_000_000, blur=100.0),
            _vlm_obs(1, 0, 2_000_000, importance=1, narrative_role="development",
                     scene_description="模糊废镜头"),
            _vlm_obs(2, 2_000_000, 4_000_000, importance=5, narrative_role="development",
                     scene_description="关键剧情"),
        ]
        brief = _brief(target_s=6)
        graph = build_story_graph(brief, obs[:2])
        edl, plan = get_director_reasoner("heuristic").generate_plan(
            brief, graph, obs)
        by_shot = {e.source_asset_id: e.rationale
                   for e in edl.ordered_edits}
        # 语义分决定主槽位：imp=5 赢得 develop 主槽（无 topup 标记）；
        # imp=1 只能经 topup 响亮补入（幕结构允许时排在时间顺序位）
        assert "topup" not in by_shot["shot_00000002"]
        assert "topup" in by_shot["shot_00000001"]
        assert any(ev.startswith("semantic_topup:")
                   for ev in plan.degradation_events)
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.EXPERIMENTAL)


def test_kernel_narrative_boundaries_and_order():
    """叙事弧 resolved 字段驱动幕分配与幕内排序（P3-3 进内核）。"""
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        obs = _three_shots()
        brief = _brief()
        graph = build_story_graph(brief, obs)
        narrative = {
            "act_boundaries_resolved": [
                {"act": "peak", "shot_ids": ["shot_00000001"]},
                {"act": "resolve", "shot_ids": ["shot_00000002"]},
            ],
            "suggested_order_resolved": [
                "shot_00000003", "shot_00000002", "shot_00000001"],
        }
        edl, plan = get_director_reasoner("heuristic").generate_plan(
            brief, graph, obs, narrative=narrative)
        act_of = {e.source_asset_id: e.act for e in edl.ordered_edits}
        assert act_of.get("shot_00000001") == "peak"
        assert act_of.get("shot_00000002") == "resolve"
        assert any(q.startswith("narrative_reorder:applied")
                   for q in plan.open_questions)
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.EXPERIMENTAL)


def test_kernel_gate_blocks_vlm_obs_when_not_active():
    """vlm 观测存在而通路非 ACTIVE → PathwayNotActiveError（内核执法）。"""
    obs = _three_shots() + [_vlm_obs(1, 0, 3_000_000)]
    brief = _brief()
    graph = build_story_graph(brief, obs[:3])
    with pytest.raises(PathwayNotActiveError):
        get_director_reasoner("heuristic").generate_plan(brief, graph, obs)


def test_kernel_no_vlm_obs_regression():
    """无 VLM 观测 → 与旧纯技术路径一致（四幕时间分配，无语义事件）。"""
    obs = _three_shots()
    brief = _brief()
    graph = build_story_graph(brief, obs)
    edl, plan = get_director_reasoner("heuristic").generate_plan(brief, graph, obs)
    assert edl.ordered_edits  # 正常出片
    assert not any(q.startswith("semantic_selection")
                   for q in plan.open_questions)
    assert not any(q.startswith("narrative_reorder")
                   for q in plan.open_questions)
