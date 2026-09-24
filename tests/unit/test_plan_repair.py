"""M2.3 Plan Repair 单元测试。

验证 repair_plan 的 rule1-9：
- rule1: 移除 unknown source_asset_id
- rule2: 修复 in_frame >= out_frame（交换或移除）
- rule3: 过短片段拉长到 MIN_CLIP_US 或移除
- rule4: 去重同一 source_asset_id
- rule5: 过长片段截断到 MAX_CLIP_US
- rule6: 修复重叠片段
- rule7: 按 in_frame 排序
- rule8: 重新计算 expected_duration
- rule9: 重新验证并更新 plan.validation_status

额外验证：
- 不修改原对象
- 空 EDL 返回空 EDL，不抛异常
- producer == "plan_repair_v0.1"
"""
from __future__ import annotations

import time

import pytest

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.plan_repair import repair_plan
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


def _plan(constraints: list[str] | None = None) -> DirectorDecisionPlan:
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
        sequence=[],
        decisions=[],
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


# ---------------------------------------------------------------------------
# rule1: unknown asset removed
# ---------------------------------------------------------------------------

def test_unknown_asset_removed():
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_ghost", in_frame=4_000_000, out_frame=6_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    asset_ids = [e.source_asset_id for e in new_edl.ordered_edits]
    assert "shot_ghost" not in asset_ids
    assert "shot_a" in asset_ids


# ---------------------------------------------------------------------------
# rule2: in >= out fixed or removed
# ---------------------------------------------------------------------------

def test_in_ge_out_swapped_when_out_positive():
    # in=5M, out=1M → 交换 → in=1M, out=5M
    edits = [_edit("shot_a", in_frame=5_000_000, out_frame=1_000_000)]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    assert len(new_edl.ordered_edits) == 1
    e = new_edl.ordered_edits[0]
    assert e.in_frame < e.out_frame
    assert e.in_frame == 1_000_000
    assert e.out_frame == 5_000_000


def test_in_ge_out_removed_when_out_zero():
    # in=5M, out=0 → 无法交换 → 移除
    edits = [_edit("shot_a", in_frame=5_000_000, out_frame=0)]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    assert len(new_edl.ordered_edits) == 0


# ---------------------------------------------------------------------------
# rule3: too-short clips extended or removed
# ---------------------------------------------------------------------------

def test_too_short_clip_extended_when_source_long():
    # edit 仅 0.5M (4.0M-4.5M)，但 shot_a 源镜头长 10M → 拉长到 MIN_CLIP_US
    edits = [_edit("shot_a", in_frame=4_000_000, out_frame=4_500_000)]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    assert len(new_edl.ordered_edits) == 1
    e = new_edl.ordered_edits[0]
    assert (e.out_frame - e.in_frame) >= MIN_CLIP_US


def test_too_short_clip_removed_when_source_short():
    # shot_c 源镜头仅 0.5M < MIN_CLIP_US → 移除
    edits = [_edit("shot_c", in_frame=100_000, out_frame=300_000)]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    assert len(new_edl.ordered_edits) == 0


# ---------------------------------------------------------------------------
# rule5: too-long clips truncated
# ---------------------------------------------------------------------------

def test_too_long_clip_truncated():
    # edit 10M (0-10M) > MAX_CLIP_US (6M) → 截断到 6M
    edits = [_edit("shot_a", in_frame=0, out_frame=10_000_000)]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    assert len(new_edl.ordered_edits) == 1
    e = new_edl.ordered_edits[0]
    assert (e.out_frame - e.in_frame) <= MAX_CLIP_US


# ---------------------------------------------------------------------------
# rule6: overlap fixed
# ---------------------------------------------------------------------------

def test_overlap_fixed():
    # shot_a: 0-5M, shot_b: 3-8M → 重叠 3-5M
    edits = [
        _edit("shot_a", in_frame=0, out_frame=5_000_000),
        _edit("shot_b", in_frame=3_000_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    # 修复后不应有重叠
    sorted_edits = sorted(new_edl.ordered_edits, key=lambda e: e.in_frame)
    for i in range(len(sorted_edits) - 1):
        assert sorted_edits[i].out_frame <= sorted_edits[i + 1].in_frame


def test_rule6_large_overlap_removed_recorded():
    # shot_a: 0-4M (dur=4M), shot_b: 2-3M (dur=1M)
    # overlap = 4M-2M = 2M, smaller = 1M, ratio = 2.0 >= 0.5 → 大幅重叠移除 shot_b
    edits = [
        _edit("shot_a", in_frame=0, out_frame=4_000_000),
        _edit("shot_b", in_frame=2_000_000, out_frame=3_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    new_edl, new_plan = repair_plan(edl, plan, OBS)
    # shot_b 被移除，只剩 shot_a
    asset_ids = [e.source_asset_id for e in new_edl.ordered_edits]
    assert asset_ids == ["shot_a"]
    # open_questions 记录了移除
    assert any("removed edit" in q for q in new_plan.open_questions)


# ---------------------------------------------------------------------------
# rule4: dedupe same source_asset_id
# ---------------------------------------------------------------------------

def test_dedupe_same_asset():
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_a", in_frame=5_000_000, out_frame=7_000_000),  # 重复 asset
    ]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    asset_ids = [e.source_asset_id for e in new_edl.ordered_edits]
    # 同一 asset 只保留一个
    assert asset_ids.count("shot_a") == 1


# ---------------------------------------------------------------------------
# rule7/8/9: sort, recompute duration, revalidate
# ---------------------------------------------------------------------------

def test_sorted_by_in_frame():
    edits = [
        _edit("shot_b", in_frame=8_000_000, out_frame=10_000_000),
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    in_frames = [e.in_frame for e in new_edl.ordered_edits]
    assert in_frames == sorted(in_frames)


def test_expected_duration_recomputed():
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    new_edl, _ = repair_plan(edl, plan, OBS)
    expected = sum(e.out_frame - e.in_frame for e in new_edl.ordered_edits)
    assert new_edl.expected_duration == expected


def test_plan_validation_status_updated():
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    _, new_plan = repair_plan(edl, plan, OBS)
    assert new_plan.validation_status in ("valid", "repaired_with_errors")


# ---------------------------------------------------------------------------
# Composite: bad EDL repaired, errors reduced
# ---------------------------------------------------------------------------

def test_bad_edl_errors_reduced():
    edits = [
        _edit("shot_a", in_frame=5_000_000, out_frame=1_000_000),  # in > out
        _edit("shot_ghost", in_frame=0, out_frame=2_000_000),  # unknown
        _edit("shot_a", in_frame=3_000_000, out_frame=5_000_000),  # dup + overlap
        _edit("shot_a", in_frame=4_000_000, out_frame=6_000_000),  # overlap
    ]
    edl = _edl(edits)
    plan = _plan()

    is_valid_before, errors_before = validate_plan(edl, plan, OBS)
    assert is_valid_before is False
    assert len(errors_before) > 0

    new_edl, new_plan = repair_plan(edl, plan, OBS)
    is_valid_after, errors_after = validate_plan(new_edl, new_plan, OBS)
    # 修复后 errors 必须减少（或完全消除）
    assert len(errors_after) <= len(errors_before)
    assert len(errors_after) < len(errors_before), (
        f"errors not reduced: before={errors_before}, after={errors_after}"
    )


# ---------------------------------------------------------------------------
# Immutability: original objects not modified
# ---------------------------------------------------------------------------

def test_original_not_modified():
    edits = [
        _edit("shot_a", in_frame=5_000_000, out_frame=1_000_000),  # in > out
        _edit("shot_ghost", in_frame=0, out_frame=2_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    original_edit_count = len(edl.ordered_edits)
    original_validation_status = plan.validation_status

    repair_plan(edl, plan, OBS)

    assert len(edl.ordered_edits) == original_edit_count
    assert plan.validation_status == original_validation_status


# ---------------------------------------------------------------------------
# Empty EDL
# ---------------------------------------------------------------------------

def test_empty_edl_returns_empty():
    edl = _edl([])
    plan = _plan()
    new_edl, new_plan = repair_plan(edl, plan, OBS)
    assert new_edl.ordered_edits == []
    assert new_edl.expected_duration == 0


# ---------------------------------------------------------------------------
# producer
# ---------------------------------------------------------------------------

def test_producer_updated():
    edits = [_edit("shot_a", in_frame=1_000_000, out_frame=3_000_000)]
    edl = _edl(edits)
    plan = _plan()
    new_edl, new_plan = repair_plan(edl, plan, OBS)
    assert new_edl.producer == "plan_repair_v0.1"
    assert new_plan.producer == "plan_repair_v0.1"


# ---------------------------------------------------------------------------
# rule10: duration repair (too-short / too-long)
# ---------------------------------------------------------------------------

def _obs_timeline(asset_id: str, start: int, end: int) -> FilmObservation:
    """构造在连续时间线上的观测（模拟真实镜头分段，互不重叠）。"""
    return _obs(asset_id, start=start, end=end)


def test_rule10_extends_and_appends_when_too_short():
    """总时长 2s < target 15s 的 90% → repair 后进入 [13.5M, 16.5M] 且 valid。"""
    obs = [
        _obs_timeline("shot_a", 0, 10_000_000),
        _obs_timeline("shot_b", 10_000_000, 20_000_000),
        _obs_timeline("shot_c", 20_000_000, 30_000_000),
    ]
    # 初始只有 shot_a 的 1M-3M（2s）
    edits = [_edit("shot_a", in_frame=1_000_000, out_frame=3_000_000)]
    edl = _edl(edits)
    plan = _plan(constraints=["target_duration_us=15000000"])

    new_edl, new_plan = repair_plan(edl, plan, obs)
    total = sum(e.out_frame - e.in_frame for e in new_edl.ordered_edits)
    assert 13_500_000 <= total <= 16_500_000, f"total={total} not in range"
    ok, errors = validate_plan(new_edl, new_plan, obs)
    assert ok, f"repair still invalid: {errors}"


def test_rule10_truncates_when_too_long():
    """总时长 20s > target 15s 的 110% → repair 后缩短到目标范围。"""
    obs = [
        _obs_timeline("shot_a", 0, 5_000_000),
        _obs_timeline("shot_b", 5_000_000, 10_000_000),
        _obs_timeline("shot_c", 10_000_000, 15_000_000),
        _obs_timeline("shot_d", 15_000_000, 20_000_000),
    ]
    edits = [
        _edit("shot_a", in_frame=0, out_frame=5_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=10_000_000),
        _edit("shot_c", in_frame=10_000_000, out_frame=15_000_000),
        _edit("shot_d", in_frame=15_000_000, out_frame=20_000_000),
    ]
    edl = _edl(edits)
    plan = _plan(constraints=["target_duration_us=15000000"])

    new_edl, new_plan = repair_plan(edl, plan, obs)
    total = sum(e.out_frame - e.in_frame for e in new_edl.ordered_edits)
    assert 13_500_000 <= total <= 16_500_000, f"total={total} not in range"
    ok, errors = validate_plan(new_edl, new_plan, obs)
    assert ok, f"repair still invalid: {errors}"
