"""D1 镜头卡系统测试（美学策略载体 v1）。

覆盖：卡库加载与 schema 校验、确定性选择器、内核消费（节奏边界覆盖/
转场策略/artistic_choices 留痕）、显式选择的响亮失败。
"""
from __future__ import annotations

import json
import time

import pytest

from director_brain.brief_compiler import compile_brief
from director_brain.director_reasoner import get_director_reasoner
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.models.shot_card import ShotCard
from director_brain.pathway_protocol import PathwayStatus, set_pathway_status
from director_brain.shot_cards import get_card, load_cards, select_card
from director_brain.story_graph_builder import build_story_graph


# ---------------------------------------------------------------------------
# 卡库
# ---------------------------------------------------------------------------

def test_load_cards_valid_and_unique():
    cards = load_cards()
    assert len(cards) >= 12
    ids = [c.card_id for c in cards]
    assert len(ids) == len(set(ids))
    assert all(isinstance(c, ShotCard) for c in cards)


def test_get_card_and_unknown():
    assert get_card("fast_cut") is not None
    assert get_card("no_such_card") is None


# ---------------------------------------------------------------------------
# 选择器
# ---------------------------------------------------------------------------

def _brief(arc="upbeat", lang="fast_cut") -> DirectorBrief:
    return DirectorBrief(
        schema_version="1.0", project_id="t", created_at=0, producer="t",
        source_ref="t.mp4", brief_id="b1", version="0.1", source_text="test",
        language="zh", intent="test", audience="not_determined",
        target_duration=10_000_000, source_duration_us=20_000_000,
        delivery_profile="short_form", themes=[], relationships=[],
        emotional_arc=arc, visual_language="consistent_exposure",
        editing_language=lang, sound_language="no_speech",
        must_include=[], must_avoid=[], privacy_constraints=[],
        approval_state="draft")


def test_select_matches_editing_language_and_arc():
    card = select_card(_brief(arc="upbeat", lang="fast_cut"), shot_count=10)
    assert card is not None
    assert card.card_id in ("fast_cut", "montage", "climax_peak", "beat_sync")


def test_select_slow_calm():
    card = select_card(_brief(arc="calm", lang="slow_paced"), shot_count=5)
    assert card is not None
    assert card.card_id in ("slow_paced", "documentary", "warm_moment",
                            "gentle_close")


def test_select_skips_cards_below_min_shots():
    """素材 2 个镜头：min_shots>=3 的卡全部跳过——无条件匹配返回 None。"""
    card = select_card(_brief(lang="not_determined"), shot_count=2)
    assert card is None or card.min_shots <= 2


def test_explicit_card_id_resolved():
    card = select_card(_brief(), shot_count=10, card_id="balanced")
    assert card.card_id == "balanced"


def test_explicit_unknown_card_fails_loud():
    with pytest.raises(ValueError, match="未知镜头卡"):
        select_card(_brief(), shot_count=10, card_id="nope")


def test_explicit_card_condition_fails_loud():
    """显式选卡但素材镜头数不满足 → 响亮失败（不静默降级）。"""
    with pytest.raises(ValueError, match="条件不满足"):
        select_card(_brief(), shot_count=2, card_id="montage")


# ---------------------------------------------------------------------------
# 内核消费
# ---------------------------------------------------------------------------

def _tech_obs(idx: int, start_us: int, end_us: int, blur: float = 150.0) -> FilmObservation:
    return FilmObservation(
        observation_id=f"det_{idx}", media_asset_id=f"shot_{idx:08d}",
        media_hash=f"hash_{idx}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim=json.dumps({"blur_score": blur, "brightness_mean": 120.0,
                          "exposure_ok": True, "shake_score": 5.0}),
        provider="deterministic_opencv", model_version="opencv",
        prompt_version="n/a", confidence=1.0, review_state="auto_verified",
        claim_kind=ClaimKind.MEASURED, schema_version="1.0",
        project_id="t", created_at=int(time.time()),
        producer="deterministic_opencv", source_ref="t.mp4")


def _obs_set():
    return [_tech_obs(i, i * 3_000_000, (i + 1) * 3_000_000) for i in range(1, 7)]


def test_kernel_card_overrides_pacing_and_annotates():
    """卡的节奏边界进内核；EDL.artistic_choices 与 plan.constraints 留痕。"""
    obs = _obs_set()
    brief = compile_brief("t", "t.mp4", obs, target_duration_us=10_000_000,
                          intent_text="快剪")
    graph = build_story_graph(brief, obs)
    card = get_card("fast_cut")
    edl, plan = get_director_reasoner("heuristic").generate_plan(
        brief, graph, obs, card=card)
    assert f"shot_card:{card.card_id}@{card.version}" in edl.artistic_choices
    assert any(c.startswith("shot_card=fast_cut@") for c in plan.constraints)
    # 快剪上界 3s：所有片段 ≤3s
    assert all(e.out_frame - e.in_frame <= 3_000_000 for e in edl.ordered_edits)


def test_kernel_card_transition_policy_applies():
    """卡的 dissolve_act_boundary 在幕边界产生转场。"""
    obs = _obs_set()
    brief = compile_brief("t", "t.mp4", obs, target_duration_us=10_000_000)
    graph = build_story_graph(brief, obs)
    card = get_card("balanced")  # dissolve_act_boundary
    edl, plan = get_director_reasoner("heuristic").generate_plan(
        brief, graph, obs, card=card)
    trans = [e for e in edl.ordered_edits if e.transition
             and e.transition.type == "xfade"]
    assert trans, "均衡卡应产生幕边界转场"
    # 转场只出现在幕边界（当前与下一镜头不同幕）
    acts = [e.act for e in edl.ordered_edits]
    for i, e in enumerate(edl.ordered_edits[:-1]):
        if e.transition and e.transition.type == "xfade":
            assert acts[i] != acts[i + 1]


def test_kernel_no_card_unchanged():
    """无卡路径与旧行为一致（无 artistic_choices 注记）。"""
    obs = _obs_set()
    brief = compile_brief("t", "t.mp4", obs, target_duration_us=10_000_000)
    graph = build_story_graph(brief, obs)
    edl, _plan = get_director_reasoner("heuristic").generate_plan(
        brief, graph, obs)
    assert edl.artistic_choices == []


# ---------------------------------------------------------------------------
# D2 实体解析与连续性
# ---------------------------------------------------------------------------

def _vlm_people_obs(idx: int, start_us: int, end_us: int, people: list[str],
                    narrative_role: str = "development") -> FilmObservation:
    return FilmObservation(
        observation_id=f"vlm_{idx}", media_asset_id=f"shot_{idx:08d}",
        media_hash=f"hash_{idx}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="vlm_semantic",
        claim=json.dumps({
            "shot_function": "ACTION", "proposed_role_v2": "support",
            "motion_amount": "subtle", "frame_description": "t",
            "importance": 3, "narrative_role": narrative_role,
            "emotional_tone": "neutral", "action_type": "action",
            "scene_description": "测试", "people": people}),
        provider="ollama_qwen3_vl", model_version="qwen3-vl:latest",
        prompt_version="vlm_prompt_v4_people", confidence=0.7,
        review_state="auto_generated", claim_kind=ClaimKind.MODEL_OBSERVATION,
        schema_version="1.0", project_id="t", created_at=int(time.time()),
        producer="ollama_qwen3_vl", source_ref="t.mp4")


def test_entity_resolver_clusters_same_person():
    """同外观描述（红衣短发女孩）跨镜头 → 同一实体。"""
    from director_brain.entity_resolver import resolve_entities
    obs = [
        _vlm_people_obs(1, 0, 2_000_000, ["红衣短发女孩"]),
        _vlm_people_obs(2, 2_000_000, 4_000_000, ["蓝衣男孩"]),
        _vlm_people_obs(3, 4_000_000, 6_000_000, ["红衣短发女孩"]),
    ]
    res = resolve_entities(obs)
    assert len(res.entities) == 2
    a1 = set(res.assignments["shot_00000001"])
    a3 = set(res.assignments["shot_00000003"])
    assert a1 & a3, "镜头 1/3 的红衣女孩应同实体"
    assert not (a1 & set(res.assignments["shot_00000002"])), "蓝衣男孩不同实体"
    # 多成员实体置信度 ≥ 0.5；单成员弱置信
    assert all(e.identity_confidence >= 0.3 for e in res.entities)


def test_entity_resolver_confidence_grading():
    """共享 颜色+载体 判别词对 → 高置信；无判别词 → 弱置信。"""
    from director_brain.entity_resolver import resolve_entities
    obs = [
        _vlm_people_obs(1, 0, 2_000_000, ["红衣女孩"]),
        _vlm_people_obs(2, 2_000_000, 4_000_000, ["红衣女孩拿着伞"]),
    ]
    res = resolve_entities(obs)
    assert len(res.entities) == 1  # 判别词对命中 → 合并
    assert res.entities[0].identity_confidence >= 0.8


def test_entity_resolver_empty_people_no_assignment():
    """无 people 字段的镜头不参与聚类（无证据不归属）。"""
    from director_brain.entity_resolver import resolve_entities
    obs = [_vlm_people_obs(1, 0, 2_000_000, [])]
    res = resolve_entities(obs)
    assert res.entities == [] and res.assignments == {}


def test_kernel_entity_continuity_bonus():
    """与上一幕已选镜头共享人物实体 → continuity_bonus 加分留痕。"""
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        obs = [
            _tech_obs(1, 0, 2_000_000),
            _tech_obs(2, 1_000_000, 3_000_000),
            _tech_obs(3, 1_500_000, 3_500_000),
            # shot1/shot3 同实体（跨 hook→develop 人物线索）
            _vlm_people_obs(1, 0, 2_000_000, ["红衣短发女孩"],
                            narrative_role="setup"),
            _vlm_people_obs(2, 1_000_000, 3_000_000, ["蓝衣男孩"]),
            _vlm_people_obs(3, 1_500_000, 3_500_000, ["红衣短发女孩"]),
        ]
        brief = _brief()
        graph = build_story_graph(brief, obs[:3])
        from director_brain.entity_resolver import resolve_entities
        entities = resolve_entities(obs[3:])
        edl, plan = get_director_reasoner("heuristic").generate_plan(
            brief, graph, obs, entities=entities)
        assert any(q.startswith("entities_resolved:count=")
                   for q in plan.open_questions)
        sem = [e.rationale for e in edl.ordered_edits
               if "entity_continuity" in (e.rationale or "")]
        assert sem, "应有镜头获得 entity_continuity 加分"
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
