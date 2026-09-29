"""阶段 P3-2 语义感知选片测试。"""
import json
import time

import pytest

from director_brain.models.director_brief import DirectorBrief
from director_brain.semantic_scorer import (
    SemanticScore,
    compute_semantic_score,
    fused_selection,
    semantic_aware_select,
)


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


def _cand(shot_id: str, blur: float, dur_s: float) -> dict:
    return {"source_shot_id": shot_id, "blur_score": blur,
            "duration_us": int(dur_s * 1_000_000),
            "exposure_ok": True, "technical_usable": True}


def _sem(role: str, tone: str, action: str, imp: int, desc: str) -> dict:
    return {"narrative_role": role, "emotional_tone": tone,
            "action_type": action, "importance": imp,
            "scene_description": desc}


def test_narrative_role_drives_act_assignment():
    """climax 镜头分到 peak 幕，setup 分到 hook 幕（替代时间比例）。"""
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


def test_semantic_aware_select_fills_by_act():
    """语义选片：narrative_role 决定幕分配，每幕贪心填满配额。"""
    brief = _brief(lang="fast_cut")
    cands = [
        _cand("s_setup", 200.0, 3.0),
        _cand("s_develop", 180.0, 3.0),
        _cand("s_climax", 300.0, 3.0),
        _cand("s_resolve", 150.0, 3.0),
        _cand("s_broll", 250.0, 3.0),
    ]
    sems = {
        "s_setup": _sem("setup", "calm", "establishing", 5, "开场全景"),
        "s_develop": _sem("development", "neutral", "action", 4, "发展剧情"),
        "s_climax": _sem("climax", "tense", "action", 5, "高潮爆发"),
        "s_resolve": _sem("resolution", "calm", "sensory", 4, "收尾"),
        "s_broll": _sem("development", "neutral", "detail", 3, "补充镜头"),
    }
    selected, scores = semantic_aware_select(
        cands, sems, brief, 10_000_000)
    acts = {c["source_shot_id"]: s.assigned_act
            for s, c in zip(scores, selected)}
    # setup→hook, development→develop, climax→peak, resolution→resolve
    assert acts.get("s_setup") == "hook"
    assert acts.get("s_climax") == "peak"
    assert acts.get("s_resolve") == "resolve"


def test_fused_selection_semantic_plus_tech_fallback():
    """融合选片：有语义的走语义分，无语义的技术兜底补位。"""
    brief = _brief()
    cands = [
        _cand("s_sem", 200.0, 3.0),
        _cand("s_tech", 180.0, 3.0),
    ]
    sems = {"s_sem": _sem("climax", "tense", "action", 5, "爆炸场景")}
    selected, scores, used = fused_selection(cands, sems, brief, 5_000_000)
    assert used is True
    assert len(selected) >= 1
    assert any(c["source_shot_id"] == "s_sem" for c in selected)
