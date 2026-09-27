"""P1-a 意图约束解释器单元测试。

覆盖：解释映射（技术词/语义词分流）、规则判定、编码↔解码往返、
reasoner 过滤接线、validator Rule 9 判红（负样本）。
"""
from __future__ import annotations

import json
import time

import pytest

from director_brain.brief_compiler import compile_brief
from director_brain.director_reasoner import (
    EvidenceTooPoorError,
    HeuristicDirectorReasoner,
)
from director_brain.intent_constraints import (
    TechnicalAvoidRule,
    candidate_violated_rules,
    interpret_constraints,
)
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.plan_validator import validate_plan
from director_brain.story_graph_builder import build_story_graph


def _obs(asset_id: str, start: int, end: int, blur: float,
         brightness: float = 120.0, shake: float = 0.05) -> FilmObservation:
    claim = json.dumps({
        "blur_score": blur, "brightness_mean": brightness,
        "exposure_ok": True, "shake_score": shake,
    })
    return FilmObservation(
        observation_id=f"obs_{asset_id}", media_asset_id=asset_id,
        media_hash=f"hash_{asset_id}", start_frame=start, end_frame=end,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim=claim, provider="test", model_version="test", prompt_version="test",
        confidence=1.0, review_state="auto_verified", claim_kind=ClaimKind.MEASURED,
        schema_version="1.0", project_id="p1a", created_at=int(time.time()),
        producer="test", source_ref="test.mp4",
    )


# ---------------------------------------------------------------------------
# 解释映射
# ---------------------------------------------------------------------------

def _brief(must_avoid: list[str], must_include: list[str]) -> DirectorBrief:
    b = compile_brief("p1a", "t.mp4", [])
    return b.model_copy(update={"must_avoid": must_avoid, "must_include": must_include})


def test_interpret_splits_technical_and_semantic():
    d = interpret_constraints(_brief(
        must_avoid=["模糊镜头", "陌生人"], must_include=["日出"],
    ))
    terms = [r.term for _, r in d.avoid_rules]
    assert terms == ["模糊"]  # 命中技术词表
    assert ("must_avoid", "陌生人") in d.unverifiable
    assert ("must_include", "日出") in d.unverifiable


def test_technical_rule_predicates():
    below = TechnicalAvoidRule("模糊", "blur_score", "below", 50.0)
    above = TechnicalAvoidRule("过曝", "brightness_mean", "above", 220.0)
    assert below.violated({"blur_score": 30}) is True
    assert below.violated({"blur_score": 200}) is False
    assert below.violated({}) is False  # 无数据不判违反
    assert above.violated({"brightness_mean": 240}) is True


def test_candidate_violated_rules_uses_claim_metrics():
    rule = TechnicalAvoidRule("过暗", "brightness_mean", "below", 40.0)
    candidate = {"_claim_metrics": {"brightness_mean": 20.0, "blur_score": 200.0}}
    assert candidate_violated_rules(candidate, [rule]) == [rule]


# ---------------------------------------------------------------------------
# reasoner 接线：过滤 + 留痕 + 编码进 plan.constraints
# ---------------------------------------------------------------------------

def test_reasoner_excludes_violating_candidates():
    """用户说"避免模糊镜头" → 低 blur 镜头被排除并留痕，不再进成片。"""
    obs = [
        _obs("shot_blurry", 0, 3_000_000, blur=30.0),
        _obs("shot_clear", 20_000_000, 23_000_000, blur=200.0),
    ]
    brief = compile_brief("p1a", "t.mp4", obs, target_duration_us=6_000_000,
                          intent_text="避免模糊镜头")
    graph = build_story_graph(brief, obs)
    edl, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)

    selected = [e.source_asset_id for e in edl.ordered_edits]
    assert "shot_blurry" not in selected
    assert "shot_clear" in selected
    # 编码进 plan.constraints（validator 判红依据）
    assert any(c.startswith("must_avoid:blur_score<50.0:") for c in plan.constraints)
    # 排除动作留痕
    assert any("must_avoid" in q and "shot_blurry" in q for q in plan.open_questions)


def test_reasoner_all_excluded_raises():
    """全部候选都违反约束 → fail-closed 拒绝导演，不产出无声片子。"""
    obs = [
        _obs("shot_a", 0, 3_000_000, blur=30.0),
        _obs("shot_b", 20_000_000, 23_000_000, blur=20.0),
    ]
    brief = compile_brief("p1a", "t.mp4", obs, target_duration_us=6_000_000,
                          intent_text="避免模糊镜头")
    graph = build_story_graph(brief, obs)
    with pytest.raises(EvidenceTooPoorError):
        HeuristicDirectorReasoner().generate_plan(brief, graph, obs)


def test_reasoner_records_unverifiable_semantic_constraint():
    """语义类约束（"必须有日出"）无法用技术观测验证 → 诚实留痕不假装执行。"""
    obs = [
        _obs("shot_a", 0, 3_000_000, blur=200.0),
        _obs("shot_b", 20_000_000, 23_000_000, blur=180.0),
    ]
    brief = compile_brief("p1a", "t.mp4", obs, target_duration_us=6_000_000,
                          intent_text="必须包含日出镜头")
    graph = build_story_graph(brief, obs)
    _, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)
    assert any("constraint_unverifiable" in q and "日出" in q
               for q in plan.open_questions)


# ---------------------------------------------------------------------------
# validator Rule 9：判红负样本
# ---------------------------------------------------------------------------

def _edl_and_plan_with_constraint():
    obs = [_obs("shot_a", 0, 10_000_000, blur=30.0)]  # blur=30 < 50 → 违反
    edits = [EditItem(
        source_asset_id="shot_a", source_media_hash="hash_shot_a",
        in_frame=1_000_000, out_frame=3_000_000, timebase=1_000_000,
    )]
    edl = EditorialDecisionList(
        schema_version="1.0", project_id="p1a", created_at=int(time.time()),
        producer="test", source_ref="t", edl_id="edl_t", version="0.1",
        brief_version="0.1", context_id="c", timebase=1_000_000,
        ordered_edits=edits, expected_duration=2_000_000, approval_state="draft",
    )
    ids = [e.source_asset_id for e in edl.ordered_edits]
    plan = DirectorDecisionPlan(
        schema_version="1.0", project_id="p1a", created_at=int(time.time()),
        producer="test", source_ref="t", plan_id="plan_t", version="0.1",
        brief_version="0.1", film_state_version="0.1", sequence=ids,
        decisions=[Decision(decision_id="dec_0", purpose="select_shot", shot_refs=ids)],
        constraints=["target_duration_us=2000000",
                     TechnicalAvoidRule("模糊", "blur_score", "below", 50.0).encode()],
        open_questions=[], validation_status="pending", approval_state="draft",
    )
    return obs, edl, plan


def test_validator_rule9_flags_violation():
    """负样本：EDL 含违反 must_avoid 约束的镜头 → 必须判红。"""
    obs, edl, plan = _edl_and_plan_with_constraint()
    ok, errors = validate_plan(edl, plan, obs)
    assert ok is False
    assert any("must_avoid violation" in e and "模糊" in e for e in errors)


def test_validator_rule9_passes_compliant():
    """满足约束的镜头不误伤。"""
    obs, edl, plan = _edl_and_plan_with_constraint()
    obs[0] = _obs("shot_a", 0, 10_000_000, blur=200.0)  # 改为清晰
    ok, errors = validate_plan(edl, plan, obs)
    assert ok is True, f"errors={errors}"


def test_constraint_encoding_roundtrip():
    """编码 → 解码往返保真（含阈值与词条）。"""
    rule = TechnicalAvoidRule("抖动", "shake_score", "above", 0.5)
    from director_brain.plan_validator import _parse_must_avoid_constraints
    parsed = _parse_must_avoid_constraints([rule.encode()])
    assert parsed == [rule]
