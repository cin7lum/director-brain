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
    unresolved_constraint_review_items,
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


def test_unresolved_constraint_review_items_keep_exact_brief_positions():
    brief = _brief(
        must_avoid=["模糊镜头", "陌生人", "陌生人"],
        must_include=["日出", "日出"],
    )

    items = unresolved_constraint_review_items(brief)

    assert [(item["constraint_ref"], item["text"]) for item in items] == [
        ("must_include:0", "日出"),
        ("must_include:1", "日出"),
        ("must_avoid:1", "陌生人"),
        ("must_avoid:2", "陌生人"),
    ]


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
        source_asset_hashes=[edits[0].source_media_hash],
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


# ---------------------------------------------------------------------------
# P1-b：editing_language → 片段时长界 + 结构化字段
# ---------------------------------------------------------------------------

def test_editing_language_bounds_mapping():
    from director_brain.intent_constraints import (
        editing_language_bounds,
        encode_bounds,
    )
    b = compile_brief("p1b", "t.mp4", [], intent_text="快剪风格")
    assert b.editing_language == "fast_cut"
    assert editing_language_bounds(b, (800_000, 6_000_000)) == (400_000, 3_000_000)
    b2 = compile_brief("p1b", "t.mp4", [])  # 未声明 → 回退默认
    assert editing_language_bounds(b2, (800_000, 6_000_000)) == (800_000, 6_000_000)
    assert encode_bounds((400_000, 3_000_000)) == [
        "min_clip_us=400000", "max_clip_us=3000000",
    ]


def test_reasoner_applies_fast_cut_bounds():
    """“快剪”意图 → 单镜头不超过 3s（默认上限 6s），且界编码进 plan.constraints。"""
    obs = [
        _obs("shot_a", 0, 8_000_000, blur=200.0),
        _obs("shot_b", 20_000_000, 28_000_000, blur=180.0),
    ]
    brief = compile_brief("p1b", "t.mp4", obs, target_duration_us=8_000_000,
                          intent_text="快剪风格")
    graph = build_story_graph(brief, obs)
    edl, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)
    assert "max_clip_us=3000000" in plan.constraints
    for e in edl.ordered_edits:
        assert (e.out_frame - e.in_frame) <= 3_000_000, f"{e.source_asset_id} 超快剪上界"


def test_reasoner_sets_structured_act_and_evidence_type():
    """EditItem.act / evidence_type 结构化字段由生成端写入（去字符串协议）。"""
    obs = [
        _obs("shot_a", 0, 2_000_000, blur=200.0),
        _obs("shot_b", 20_000_000, 22_000_000, blur=180.0),
    ]
    brief = compile_brief("p1b", "t.mp4", obs, target_duration_us=8_000_000)
    graph = build_story_graph(brief, obs)
    edl, _ = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)
    for e in edl.ordered_edits:
        assert e.act in ("hook", "develop", "peak", "resolve"), e.act
        assert e.evidence_type == "heuristic"  # 无 VLM 观测


def test_no_data_observation_never_selected():
    """读帧失败（claim 带 error）的镜头：放宽阶梯不得将其注水入选。"""
    error_obs = FilmObservation(
        observation_id="obs_bad", media_asset_id="shot_bad",
        media_hash="h_bad", start_frame=0, end_frame=2_000_000,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim=json.dumps({"error": "no_frame"}),
        provider="test", model_version="test", prompt_version="test",
        confidence=1.0, review_state="auto_verified", claim_kind=ClaimKind.MEASURED,
        schema_version="1.0", project_id="p1b", created_at=int(time.time()),
        producer="test", source_ref="t.mp4",
    )
    obs = [
        error_obs,
        _obs("shot_ok", 20_000_000, 22_000_000, blur=5.0),  # 低 blur → 触发放宽
    ]
    brief = compile_brief("p1b", "t.mp4", obs, target_duration_us=8_000_000)
    graph = build_story_graph(brief, obs)
    edl, _ = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)
    selected = [e.source_asset_id for e in edl.ordered_edits]
    assert "shot_bad" not in selected
    assert "shot_ok" in selected


def test_repair_honors_constraint_bounds():
    """修复器使用 plan.constraints 里的时长界（与生成端同界）。"""
    from director_brain.plan_repair import _parse_clip_bounds
    assert _parse_clip_bounds(["target_duration_us=8000000",
                               "min_clip_us=400000", "max_clip_us=3000000"]) == (
        400_000, 3_000_000,
    )
    assert _parse_clip_bounds([]) == (800_000, 6_000_000)  # 回退默认


def test_repair_backward_extension_and_margin():
    """P2-d：向后延长（in_frame）+ 2% 边界余量——居中裁剪的前部余量不再浪费。

    构造：单镜头源 0-10s，选中片段 8.0-10.0s（贴源尾、向前无余量），目标 3s。
    旧实现只向前延长 → 不可达 ABSTAIN；新实现向后延长到低界+2%。
    """
    from director_brain.plan_repair import repair_plan

    obs = [_obs("shot_a", 0, 10_000_000, 200.0)]
    edits = [EditItem(
        source_asset_id="shot_a", source_media_hash="hash_shot_a",
        in_frame=8_000_000, out_frame=10_000_000, timebase=1_000_000,
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
        constraints=["target_duration_us=3000000"],
        open_questions=[], validation_status="pending", approval_state="draft",
    )
    outcome = repair_plan(edl, plan, obs)
    assert outcome.status == "ok", outcome.reason
    e = outcome.edl.ordered_edits[0]
    total = e.out_frame - e.in_frame
    # 目标延长线 = low(2.7M) + 2%×3M = 2.76M
    assert total >= 2_760_000, f"total={total}"
    assert total <= 3_000_000
    # 任一调整走的是 in_frame（向后）路径
    assert any(":in " in a for a in outcome.adjustments), outcome.adjustments
    ok, errors = validate_plan(outcome.edl, outcome.plan, obs)
    assert ok, f"errors={errors}"


def test_repair_truncation_boundary_margin():
    """P2-d 对称修复：截断目标取上界 -2%（慢节奏实测 10.0% 压线翻车）。

    构造：单镜头 4s（0-4M），慢节奏界 min 1.5s/max 8s，目标 2s
    （high=2.2M）。旧实现截到 high=2.2M 恰好压线；新实现截到 2.16M。
    """
    from director_brain.plan_repair import repair_plan

    obs = [_obs("shot_a", 0, 10_000_000, 200.0)]
    edits = [EditItem(source_asset_id="shot_a", source_media_hash="hash_shot_a",
                      in_frame=0, out_frame=4_000_000, timebase=1_000_000)]
    edl = EditorialDecisionList(
        schema_version="1.0", project_id="p1a", created_at=int(time.time()),
        producer="test", source_ref="t", edl_id="edl_t", version="0.1",
        brief_version="0.1", context_id="c", timebase=1_000_000,
        ordered_edits=edits, expected_duration=4_000_000, approval_state="draft")
    ids = [e.source_asset_id for e in edl.ordered_edits]
    plan = DirectorDecisionPlan(
        schema_version="1.0", project_id="p1a", created_at=int(time.time()),
        producer="test", source_ref="t", plan_id="plan_t", version="0.1",
        brief_version="0.1", film_state_version="0.1", sequence=ids,
        decisions=[Decision(decision_id="dec_0", purpose="select_shot",
                            shot_refs=ids)],
        constraints=["target_duration_us=2000000", "min_clip_us=1500000",
                     "max_clip_us=8000000"],
        open_questions=[], validation_status="pending", approval_state="draft")
    outcome = repair_plan(edl, plan, obs)
    assert outcome.status == "ok", outcome.reason
    total = sum(e.out_frame - e.in_frame for e in outcome.edl.ordered_edits)
    high = int(1.1 * 2_000_000)
    assert total < high, f"total={total} 仍压上界 {high}"
    assert total >= high - int(0.02 * 2_000_000) - 1, f"total={total} 过度截断"
    ok, errors = validate_plan(outcome.edl, outcome.plan, obs)
    assert ok, f"errors={errors}"
