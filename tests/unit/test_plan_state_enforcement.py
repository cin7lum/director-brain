"""plan_state 执法点测试（架构体检候选③）。

评审指认：validate_transition / is_confirmation_valid 导入后从未被生产
路径调用、"未 STRATEGY_CONFIRMED 不得渲染"红线无处生效。本文件验证
执法点真实存在：转换表拒绝非法推进、渲染闸门查状态、hash 绑定只锁
内容不锁流程元数据。
"""
from __future__ import annotations

import json
import time

import pytest

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.plan_state import (
    InvalidTransition,
    PlanState,
    compute_edl_hash,
    compute_plan_hash,
    confirm_strategy,
    is_confirmation_valid,
    transition_plan,
    validate_transition,
)


def _plan(state: str = "draft") -> DirectorDecisionPlan:
    return DirectorDecisionPlan(
        schema_version="1.0", project_id="t", created_at=0,
        producer="t", source_ref="t.mp4",
        plan_id="plan_t", version="0.1", brief_version="0.1",
        film_state_version="0.1", validation_status="valid",
        approval_state="draft", state=state,
        sequence=["shot_1"],
    )


def _edl(out_last: int = 3_000_000) -> EditorialDecisionList:
    return EditorialDecisionList(
        schema_version="1.0", project_id="t", created_at=0,
        producer="t", source_ref="t.mp4", edl_id="edl_t", version="0.1",
        brief_version="0.1", context_id="ctx", timebase=1_000_000,
        ordered_edits=[EditItem(
            source_asset_id="shot_1", source_media_hash="h1",
            in_frame=0, out_frame=out_last, timebase=1_000_000,
        )],
        approval_state="draft",
    )


def test_enum_has_16_states():
    """枚举 16 态（10 正常 + 6 异常）——文档曾写 15，漂移已修正。"""
    assert len(PlanState) == 16


def test_illegal_transition_rejected():
    """draft 直接跳确认（跳过验证）→ InvalidTransition。"""
    with pytest.raises(InvalidTransition):
        validate_transition(PlanState.DRAFT, PlanState.STRATEGY_CONFIRMED)


def test_transition_plan_writes_state():
    """transition_plan 是唯一执法入口：校验后写回 plan.state。"""
    plan = _plan()
    transition_plan(plan, PlanState.CONTEXT_READY)
    transition_plan(plan, PlanState.VALIDATING)
    transition_plan(plan, PlanState.READY_FOR_STRATEGY_CONFIRMATION)
    assert plan.state == "ready_for_strategy_confirmation"
    with pytest.raises(InvalidTransition):
        transition_plan(plan, PlanState.DISPATCH_ELIGIBLE)


def test_hash_binds_content_not_workflow_metadata():
    """hash 只锁内容：确认后合法推进状态不破坏绑定；改序列则失效。"""
    plan = _plan(state="ready_for_strategy_confirmation")
    edl = _edl()
    confirmation = confirm_strategy(plan, edl, confirmed_by="t")

    # 合法流程推进（状态字段变化）→ 绑定仍有效
    transition_plan(plan, PlanState.STRATEGY_CONFIRMED)
    transition_plan(plan, PlanState.DISPATCH_ELIGIBLE)
    assert is_confirmation_valid(confirmation, plan, edl)

    # 内容变化（EDL 出点改变）→ 绑定失效
    edl_changed = _edl(out_last=3_500_000)
    assert not is_confirmation_valid(confirmation, plan, edl_changed)

    # 内容变化（plan 序列改变）→ 绑定失效
    plan.sequence = ["shot_2"]
    assert not is_confirmation_valid(confirmation, plan, edl)


def test_hash_functions_exclude_workflow_fields():
    """compute_*_hash 排除流程字段（state/validation_status/approval_state）。"""
    p1 = _plan(state="draft")
    p2 = _plan(state="dispatch_eligible")
    assert compute_plan_hash(p1) == compute_plan_hash(p2)

    e1 = _edl()
    e1.approval_state = "draft"
    e2 = _edl()
    e2.approval_state = "approved"
    assert compute_edl_hash(e1) == compute_edl_hash(e2)
