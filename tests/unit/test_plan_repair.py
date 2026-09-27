"""M2.3 Plan Repair 单元测试（T1 收权版契约）。

裁定（"Plan = 导演依据"）：repair 只做**物理修复**（已选镜头内 in/out 调整），
无权增删镜头、无权重排。修复不了就 ABSTAIN（一等返回值 RepairOutcome，
带受控枚举原因码），退回导演层重新出 plan。

覆盖：
- ABSTAIN：空 EDL / unknown asset / 重复选片 / out<=0 / 源过短 /
  大幅重叠 / 目标时长物理不可达
- OK：交换端点 / 扩展到 MIN / 截断到 MAX / 小幅重叠截断 / 时长延长与缩短
- 不变量：镜头集合与顺序不变、plan↔EDL 一致性保持、原对象不被修改、
  物理调整写入 plan.open_questions 留痕、producer 更新
"""
from __future__ import annotations

import time

import pytest

from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.plan_repair import (
    REASON_DUPLICATE_SHOT,
    REASON_DURATION_UNREACHABLE,
    REASON_EMPTY_SELECTION,
    REASON_INVALID_TIME_RANGE,
    REASON_OVERLAP_UNRESOLVABLE,
    REASON_SHOT_TOO_SHORT,
    REASON_UNKNOWN_SOURCE_ASSET,
    RepairOutcome,
    repair_plan,
)
from director_brain.plan_validator import validate_plan
from gen1_adapter.heuristic_baseline import MAX_CLIP_US, MIN_CLIP_US


# ---------------------------------------------------------------------------
# Fixture 辅助
# ---------------------------------------------------------------------------

def _obs(asset_id: str, start: int = 0, end: int = 10_000_000) -> FilmObservation:
    return FilmObservation(
        observation_id=f"obs_{asset_id}",
        media_asset_id=asset_id,
        media_hash=f"hash_{asset_id}",
        start_frame=start,
        end_frame=end,
        timebase=1_000_000,
        observation_type="deterministic_technical",
        claim="{}",
        provider="test",
        model_version="test",
        prompt_version="test",
        confidence=1.0,
        review_state="auto_verified",
        claim_kind=ClaimKind.MEASURED,
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="test",
        source_ref="test.mp4",
    )


def _edit(
    asset_id: str,
    in_frame: int,
    out_frame: int,
    media_hash: str | None = None,
) -> EditItem:
    return EditItem(
        source_asset_id=asset_id,
        source_media_hash=media_hash or f"hash_{asset_id}",
        in_frame=in_frame,
        out_frame=out_frame,
        timebase=1_000_000,
    )


def _edl(edits: list[EditItem]) -> EditorialDecisionList:
    hashes = []
    seen = set()
    for e in edits:
        if e.source_media_hash not in seen:
            seen.add(e.source_media_hash)
            hashes.append(e.source_media_hash)
    return EditorialDecisionList(
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="test",
        source_ref="test",
        edl_id="test_edl",
        version="0.1",
        brief_version="0.1",
        context_id="ctx_test",
        source_asset_hashes=hashes,
        timebase=1_000_000,
        ordered_edits=edits,
        expected_duration=sum(e.out_frame - e.in_frame for e in edits),
        approval_state="draft",
    )


def _plan(
    edl: EditorialDecisionList | None = None,
    constraints: list[str] | None = None,
) -> DirectorDecisionPlan:
    """构造与 edl 逐位一致的 plan（ok 路径要求输入即一致）；edl=None 时为空 plan。"""
    if edl is None:
        sequence: list[str] = []
        decisions: list[Decision] = []
    else:
        sequence = [e.source_asset_id for e in edl.ordered_edits]
        decisions = [
            Decision(
                decision_id=f"dec_{i}",
                purpose="select_shot",
                shot_refs=[e.source_asset_id],
            )
            for i, e in enumerate(edl.ordered_edits)
        ]
    return DirectorDecisionPlan(
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="test",
        source_ref="test",
        plan_id="test_plan",
        version="0.1",
        brief_version="0.1",
        film_state_version="0.1",
        sequence=sequence,
        decisions=decisions,
        constraints=constraints or [],
        open_questions=[],
        validation_status="pending",
        approval_state="draft",
    )


# 三个已知 asset：shot_a/shot_b 源镜头长 10M；shot_c 源镜头仅 0.5M（< MIN）
OBS = [
    _obs("shot_a", start=0, end=10_000_000),
    _obs("shot_b", start=0, end=10_000_000),
    _obs("shot_c", start=0, end=500_000),
]


def _assert_abstain(outcome: RepairOutcome, reason_code: str) -> None:
    assert outcome.status == "abstain"
    assert outcome.requires_director is True
    assert outcome.reason_code == reason_code
    assert outcome.reason, "abstain 必须带人类可读原因"
    assert outcome.edl is None and outcome.plan is None


# ---------------------------------------------------------------------------
# ABSTAIN：修复器无权处理的镜头集合级缺陷
# ---------------------------------------------------------------------------

def test_empty_edl_abstains():
    outcome = repair_plan(_edl([]), _plan(), OBS)
    _assert_abstain(outcome, REASON_EMPTY_SELECTION)


def test_unknown_asset_abstains():
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_ghost", in_frame=4_000_000, out_frame=6_000_000),
    ]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    _assert_abstain(outcome, REASON_UNKNOWN_SOURCE_ASSET)


def test_duplicate_shot_abstains():
    """同一镜头被选中多次 → 保留哪一次属创作决策 → ABSTAIN。"""
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_a", in_frame=5_000_000, out_frame=7_000_000),  # 重复 asset
    ]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    _assert_abstain(outcome, REASON_DUPLICATE_SHOT)


def test_invalid_time_range_out_zero_abstains():
    """in=5M, out=0 → 无法交换 → ABSTAIN（历史行为是静默移除）。"""
    edits = [_edit("shot_a", in_frame=5_000_000, out_frame=0)]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    _assert_abstain(outcome, REASON_INVALID_TIME_RANGE)


def test_too_short_source_abstains():
    """shot_c 源镜头仅 0.5M < MIN_CLIP_US → 无法物理扩展 → ABSTAIN。"""
    edits = [_edit("shot_c", in_frame=100_000, out_frame=300_000)]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    _assert_abstain(outcome, REASON_SHOT_TOO_SHORT)


def test_large_overlap_abstains():
    """大幅重叠（>= 较短片段 50%）→ 留谁删谁是创作决策 → ABSTAIN。"""
    # shot_a: 0-4M (dur=4M), shot_b: 2-3M (dur=1M)；overlap=2M, ratio=2.0
    edits = [
        _edit("shot_a", in_frame=0, out_frame=4_000_000),
        _edit("shot_b", in_frame=2_000_000, out_frame=3_000_000),
    ]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    _assert_abstain(outcome, REASON_OVERLAP_UNRESOLVABLE)


def test_duration_unreachable_abstains():
    """单镜头物理延长到极限（MAX_CLIP_US=6M）仍达不到目标下界 → ABSTAIN。

    历史行为是从未选中镜头追加（shot_function="repair_appended"）——已移除。
    """
    obs = [
        _obs("shot_a", 0, 10_000_000),
        _obs("shot_b", 10_000_000, 20_000_000),
        _obs("shot_c", 20_000_000, 30_000_000),
    ]
    edits = [_edit("shot_a", in_frame=1_000_000, out_frame=3_000_000)]
    edl = _edl(edits)
    plan = _plan(edl, constraints=["target_duration_us=15000000"])
    outcome = repair_plan(edl, plan, obs)
    _assert_abstain(outcome, REASON_DURATION_UNREACHABLE)


def test_abstain_does_not_modify_originals():
    edits = [
        _edit("shot_a", in_frame=5_000_000, out_frame=1_000_000),  # in > out
        _edit("shot_ghost", in_frame=0, out_frame=2_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    original_edit_count = len(edl.ordered_edits)
    original_validation_status = plan.validation_status

    outcome = repair_plan(edl, plan, OBS)

    assert outcome.requires_director is True
    assert len(edl.ordered_edits) == original_edit_count
    assert plan.validation_status == original_validation_status


# ---------------------------------------------------------------------------
# OK：白名单内的物理修复
# ---------------------------------------------------------------------------

def _edl_ids(outcome: RepairOutcome) -> list[str]:
    return [e.source_asset_id for e in outcome.edl.ordered_edits]


def test_swap_fixes_inverted_range():
    # in=5M, out=1M → 交换 → in=1M, out=5M；镜头集合与顺序不变
    edits = [_edit("shot_a", in_frame=5_000_000, out_frame=1_000_000)]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    assert outcome.status == "ok"
    assert _edl_ids(outcome) == ["shot_a"]
    e = outcome.edl.ordered_edits[0]
    assert e.in_frame == 1_000_000
    assert e.out_frame == 5_000_000
    assert any(a.startswith("swap_endpoints:") for a in outcome.adjustments)


def test_extend_to_min_within_source():
    # edit 仅 0.5M (4.0M-4.5M)，shot_a 源镜头长 10M → 物理扩展到 MIN_CLIP_US
    edits = [_edit("shot_a", in_frame=4_000_000, out_frame=4_500_000)]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    assert outcome.status == "ok"
    assert _edl_ids(outcome) == ["shot_a"]
    e = outcome.edl.ordered_edits[0]
    assert (e.out_frame - e.in_frame) >= MIN_CLIP_US


def test_truncate_to_max():
    # edit 10M (0-10M) > MAX_CLIP_US (6M) → 截断到 6M，镜头保留
    edits = [_edit("shot_a", in_frame=0, out_frame=10_000_000)]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    assert outcome.status == "ok"
    assert _edl_ids(outcome) == ["shot_a"]
    e = outcome.edl.ordered_edits[0]
    assert (e.out_frame - e.in_frame) <= MAX_CLIP_US


def test_small_overlap_trimmed_keeps_both_shots():
    """小幅重叠（< 较短片段 50%）→ 截断时间在前的片段，两个镜头都保留。"""
    # shot_a: 0-5M, shot_b: 4.5-8M → overlap=0.5M, smaller=3.5M, ratio≈0.14
    edits = [
        _edit("shot_a", in_frame=0, out_frame=5_000_000),
        _edit("shot_b", in_frame=4_500_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    assert outcome.status == "ok"
    assert _edl_ids(outcome) == ["shot_a", "shot_b"], "小幅重叠不得删镜头"
    a = outcome.edl.ordered_edits[0]
    b = outcome.edl.ordered_edits[1]
    assert a.out_frame <= b.in_frame, "修复后不得再重叠"
    ok, errors = validate_plan(outcome.edl, outcome.plan, OBS)
    assert ok, f"errors={errors}"


def test_duration_extension_within_reach():
    """目标 4M：edit 2M 在源范围内物理延长到下界 3.6M → OK（不追加镜头）。"""
    obs = [_obs("shot_a", 0, 10_000_000)]
    edits = [_edit("shot_a", in_frame=1_000_000, out_frame=3_000_000)]
    edl = _edl(edits)
    plan = _plan(edl, constraints=["target_duration_us=4000000"])
    outcome = repair_plan(edl, plan, obs)
    assert outcome.status == "ok"
    total = sum(e.out_frame - e.in_frame for e in outcome.edl.ordered_edits)
    assert 3_600_000 <= total <= 4_400_000, f"total={total}"
    ok, errors = validate_plan(outcome.edl, outcome.plan, obs)
    assert ok, f"errors={errors}"


def test_duration_truncation_within_reach():
    """总时长 20M > 目标 15M 上界 16.5M → 物理缩短到范围内 → OK（不删镜头）。"""
    obs = [
        _obs("shot_a", 0, 5_000_000),
        _obs("shot_b", 5_000_000, 10_000_000),
        _obs("shot_c", 10_000_000, 15_000_000),
        _obs("shot_d", 15_000_000, 20_000_000),
    ]
    edits = [
        _edit("shot_a", in_frame=0, out_frame=5_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=10_000_000),
        _edit("shot_c", in_frame=10_000_000, out_frame=15_000_000),
        _edit("shot_d", in_frame=15_000_000, out_frame=20_000_000),
    ]
    edl = _edl(edits)
    plan = _plan(edl, constraints=["target_duration_us=15000000"])
    outcome = repair_plan(edl, plan, obs)
    assert outcome.status == "ok"
    assert _edl_ids(outcome) == ["shot_a", "shot_b", "shot_c", "shot_d"], "截短不得删镜头"
    total = sum(e.out_frame - e.in_frame for e in outcome.edl.ordered_edits)
    assert 13_500_000 <= total <= 16_500_000, f"total={total}"


# ---------------------------------------------------------------------------
# 不变量
# ---------------------------------------------------------------------------

def test_ok_repair_preserves_plan_edl_consistency():
    """物理修复不得破坏 plan↔EDL 一致性（T1 新校验必须保持 PASS）。"""
    edits = [
        _edit("shot_a", in_frame=5_000_000, out_frame=1_000_000),
        _edit("shot_b", in_frame=5_500_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan(edl)
    outcome = repair_plan(edl, plan, OBS)
    assert outcome.status == "ok"
    ok, errors = validate_plan(outcome.edl, outcome.plan, OBS)
    assert ok, f"修复后 plan↔EDL 不一致: {errors}"
    assert outcome.plan.sequence == _edl_ids(outcome)


def test_ok_repair_records_adjustments_in_open_questions():
    edits = [_edit("shot_a", in_frame=5_000_000, out_frame=1_000_000)]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    assert outcome.status == "ok"
    assert outcome.adjustments
    assert any("repair: " in q for q in outcome.plan.open_questions)


def test_ok_repair_does_not_modify_originals():
    edits = [_edit("shot_a", in_frame=5_000_000, out_frame=1_000_000)]
    edl = _edl(edits)
    plan = _plan(edl)
    original_edits = [e.model_copy(deep=True) for e in edl.ordered_edits]
    original_status = plan.validation_status

    repair_plan(edl, plan, OBS)

    assert [e.model_copy(deep=True) for e in edl.ordered_edits] == original_edits
    assert plan.validation_status == original_status


def test_expected_duration_recomputed():
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    assert outcome.status == "ok"
    expected = sum(e.out_frame - e.in_frame for e in outcome.edl.ordered_edits)
    assert outcome.edl.expected_duration == expected


def test_producer_updated():
    edits = [_edit("shot_a", in_frame=1_000_000, out_frame=3_000_000)]
    edl = _edl(edits)
    outcome = repair_plan(edl, _plan(edl), OBS)
    assert outcome.status == "ok"
    assert outcome.edl.producer == "plan_repair_v0.2"
    assert outcome.plan.producer == "plan_repair_v0.2"


def test_composite_physical_fix_reduces_errors():
    """可物理修复的坏 EDL：交换端点 + 扩展过短 → 修复后校验通过、集合不变。"""
    edits = [
        _edit("shot_a", in_frame=5_000_000, out_frame=1_000_000),  # in > out
        _edit("shot_b", in_frame=5_500_000, out_frame=6_000_000),  # 过短（0.5M）
    ]
    edl = _edl(edits)
    plan = _plan(edl)
    is_valid_before, errors_before = validate_plan(edl, plan, OBS)
    assert is_valid_before is False
    assert len(errors_before) > 0

    outcome = repair_plan(edl, plan, OBS)
    assert outcome.status == "ok"
    assert _edl_ids(outcome) == ["shot_a", "shot_b"]
    ok, errors = validate_plan(outcome.edl, outcome.plan, OBS)
    assert ok, f"errors={errors}"
    assert len(errors) < len(errors_before)
