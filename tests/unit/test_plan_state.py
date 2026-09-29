"""阶段 S2 Plan 状态机 + Strategy Confirmation 测试。"""
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
    validate_transition,
)


def _plan() -> DirectorDecisionPlan:
    return DirectorDecisionPlan(
        schema_version="1.0", project_id="t", created_at=int(time.time()),
        producer="t", source_ref="t", plan_id="p1", version="0.1",
        brief_version="0.1", film_state_version="0.1",
        sequence=["a"], decisions=[], constraints=["target_duration_us=2000000"],
        open_questions=[], validation_status="pending", approval_state="draft")


def _edl() -> EditorialDecisionList:
    return EditorialDecisionList(
        schema_version="1.0", project_id="t", created_at=0, producer="t",
        source_ref="t", edl_id="e1", version="0.1", brief_version="0.1",
        context_id="c", timebase=1_000_000,
        ordered_edits=[EditItem(source_asset_id="a", source_media_hash="ha",
                                in_frame=0, out_frame=2000000, timebase=1000000)],
        expected_duration=2000000, approval_state="draft")


def test_valid_transitions():
    validate_transition("draft", "context_ready")
    validate_transition("context_ready", "validating")
    validate_transition("validating", "ready_for_strategy_confirmation")
    validate_transition("ready_for_strategy_confirmation", "strategy_confirmed")
    validate_transition("strategy_confirmed", "dispatch_eligible")
    validate_transition("dispatch_eligible", "execution_verified")
    validate_transition("execution_verified", "fql_l0_recorded")


def test_invalid_transitions_rejected():
    with pytest.raises(InvalidTransition):
        validate_transition("draft", "strategy_confirmed")  # 跳级
    with pytest.raises(InvalidTransition):
        validate_transition("rejected", "draft")  # 终态
    with pytest.raises(InvalidTransition):
        validate_transition("fql_film_quality_reviewed", "draft")  # 终态


def test_confirmation_binds_hash():
    plan, edl = _plan(), _edl()
    conf = confirm_strategy(plan, edl, confirmed_by="user")
    assert is_confirmation_valid(conf, plan, edl)
    # 改动 plan → 确认失效
    plan.constraints.append("new_constraint=1")
    assert not is_confirmation_valid(conf, plan, edl)


def test_hash_deterministic():
    h1 = compute_plan_hash(_plan())
    h2 = compute_plan_hash(_plan())
    assert h1 == h2 and len(h1) == 16
    e1 = compute_edl_hash(_edl())
    e2 = compute_edl_hash(_edl())
    assert e1 == e2 and len(e1) == 16
