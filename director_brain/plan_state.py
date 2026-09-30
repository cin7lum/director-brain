"""Plan 状态机 + Strategy Confirmation（阶段 S2 · 方案 §4.6/§7 落地）。

方案 §7 定义了 9 个正常态（其中 CONTEXT_READY/PARTIAL 为两个变体，合计
10 个）+ 6 个异常态，枚举共 **16 态**。本模块实现：

- **PlanState** 枚举（16 态）与合法转换表（非法转换抛 :class:`InvalidTransition`）
- **transition_plan**：唯一执法入口——校验转换合法性并写回 ``plan.state``
  （候选③：此前 validate_transition 导入后从未被生产路径调用，红线无牙）
- **Strategy Confirmation**：用户批准绑定 ``plan_hash`` + ``edl_hash``；
  任何内容/约束变化使旧确认失效（``is_confirmation_valid`` 返回 False）
- **hash 计算**：plan + EDL 序列化后 SHA-256，确认时锁定版本

红线：未 STRATEGY_CONFIRMED 的 plan 不得路由到渲染/执行。
"""
from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList


class PlanState(str, enum.Enum):
    """方案 §7 定义的 16 个计划状态（10 正常 + 6 异常）。"""

    DRAFT = "draft"
    CONTEXT_READY = "context_ready"
    CONTEXT_PARTIAL = "context_partial"
    VALIDATING = "validating"
    READY_FOR_STRATEGY_CONFIRMATION = "ready_for_strategy_confirmation"
    STRATEGY_CONFIRMED = "strategy_confirmed"
    DISPATCH_ELIGIBLE = "dispatch_eligible"
    EXECUTION_VERIFIED = "execution_verified"
    FQL_L0_RECORDED = "fql_l0_recorded"
    FQL_FILM_QUALITY_REVIEWED = "fql_film_quality_reviewed"
    # 异常态
    NEEDS_INPUT = "needs_input"
    REJECTED = "rejected"
    STALE_CONTEXT = "stale_context"
    FAILED_VALIDATION = "failed_validation"
    SUPERSEDED = "superseded"
    EXECUTION_BLOCKED = "execution_blocked"


#: 合法转换表：{当前态: {允许的下一态集合}}
_TRANSITIONS: dict[PlanState, set[PlanState]] = {
    PlanState.DRAFT: {
        PlanState.CONTEXT_READY, PlanState.CONTEXT_PARTIAL,
        PlanState.NEEDS_INPUT, PlanState.SUPERSEDED,
    },
    PlanState.CONTEXT_READY: {
        PlanState.VALIDATING, PlanState.STALE_CONTEXT,
    },
    PlanState.CONTEXT_PARTIAL: {
        PlanState.VALIDATING, PlanState.NEEDS_INPUT, PlanState.STALE_CONTEXT,
    },
    PlanState.VALIDATING: {
        PlanState.READY_FOR_STRATEGY_CONFIRMATION,
        PlanState.FAILED_VALIDATION,
    },
    PlanState.READY_FOR_STRATEGY_CONFIRMATION: {
        PlanState.STRATEGY_CONFIRMED, PlanState.REJECTED,
        PlanState.STALE_CONTEXT, PlanState.SUPERSEDED,
    },
    PlanState.STRATEGY_CONFIRMED: {
        PlanState.DISPATCH_ELIGIBLE, PlanState.STALE_CONTEXT,
        PlanState.SUPERSEDED,
    },
    PlanState.DISPATCH_ELIGIBLE: {
        PlanState.EXECUTION_VERIFIED, PlanState.EXECUTION_BLOCKED,
        PlanState.SUPERSEDED,
    },
    PlanState.EXECUTION_VERIFIED: {
        PlanState.FQL_L0_RECORDED,
    },
    PlanState.FQL_L0_RECORDED: {
        PlanState.FQL_FILM_QUALITY_REVIEWED,
    },
    PlanState.FQL_FILM_QUALITY_REVIEWED: set(),
    PlanState.NEEDS_INPUT: {
        PlanState.DRAFT, PlanState.SUPERSEDED,
    },
    PlanState.REJECTED: set(),
    PlanState.STALE_CONTEXT: {
        PlanState.DRAFT, PlanState.SUPERSEDED,
    },
    PlanState.FAILED_VALIDATION: {
        PlanState.DRAFT, PlanState.SUPERSEDED,
    },
    PlanState.SUPERSEDED: set(),
    PlanState.EXECUTION_BLOCKED: {
        PlanState.DISPATCH_ELIGIBLE, PlanState.SUPERSEDED,
    },
}


class InvalidTransition(Exception):
    """非法状态转换。"""


def validate_transition(current: PlanState | str, next_: PlanState | str) -> PlanState:
    """校验状态转换是否合法；非法抛 :class:`InvalidTransition`。"""
    cur = PlanState(current)
    nxt = PlanState(next_)
    allowed = _TRANSITIONS.get(cur, set())
    if nxt not in allowed:
        raise InvalidTransition(
            f"非法状态转换 {cur.value} → {nxt.value}；"
            f"允许: {sorted(s.value for s in allowed)}"
        )
    return nxt


def transition_plan(
    plan: DirectorDecisionPlan,
    next_: PlanState | str,
) -> DirectorDecisionPlan:
    """唯一执法入口：校验 ``plan.state → next_`` 合法后写回（候选③）。

    生产路径（roughcut / API）一律经此函数推进状态——直接改
    ``plan.state`` 绕过转换表属违规。非法转换抛 :class:`InvalidTransition`。
    """
    nxt = validate_transition(plan.state, next_)
    plan.state = nxt.value
    return plan


#: hash 绑定排除的工作流易变字段：state/validation_status/approval_state
#: 随管线推进而变（确认后 DISPATCH_ELIGIBLE 是合法推进），若纳入哈希，
#: 确认会破坏自己的绑定。绑定对象是**内容**（序列/决策/约束/降级留痕）。
_PLAN_HASH_EXCLUDE = {"created_at", "state", "validation_status", "approval_state"}


def compute_plan_hash(plan: DirectorDecisionPlan) -> str:
    """plan 内容序列化后 SHA-256 前 16 位（strategy confirmation 绑定用）。"""
    canonical = json.dumps(
        plan.model_dump(exclude=_PLAN_HASH_EXCLUDE, exclude_none=True),
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def compute_edl_hash(edl: EditorialDecisionList) -> str:
    """EDL 内容序列化后 SHA-256 前 16 位（排除审批状态等流程字段）。"""
    canonical = json.dumps(
        edl.model_dump(exclude={"created_at", "approval_state"},
                       exclude_none=True),
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class StrategyConfirmation:
    """一次策略确认的完整记录。"""

    plan_hash: str
    edl_hash: str
    confirmed_by: str
    confirmed_at: int
    brief_version: str
    output_target: str  # "preview" | "delivery" | ...
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "edl_hash": self.edl_hash,
            "confirmed_by": self.confirmed_by,
            "confirmed_at": self.confirmed_at,
            "brief_version": self.brief_version,
            "output_target": self.output_target,
            "notes": self.notes,
        }


def confirm_strategy(
    plan: DirectorDecisionPlan,
    edl: EditorialDecisionList,
    confirmed_by: str = "user",
    output_target: str = "delivery",
    notes: str = "",
) -> StrategyConfirmation:
    """策略确认：绑定 plan_hash + edl_hash，任何变化使旧确认失效。

    调用方须先确保 ``plan.validation_status`` 为 valid 且已过渲染闸门——
    本函数只做 hash 绑定，不重跑验证。
    """
    plan_hash = compute_plan_hash(plan)
    edl_hash = compute_edl_hash(edl)
    return StrategyConfirmation(
        plan_hash=plan_hash,
        edl_hash=edl_hash,
        confirmed_by=confirmed_by,
        confirmed_at=int(time.time()),
        brief_version=plan.brief_version,
        output_target=output_target,
        notes=notes,
    )


def is_confirmation_valid(
    confirmation: StrategyConfirmation,
    plan: DirectorDecisionPlan,
    edl: EditorialDecisionList,
) -> bool:
    """检查现有确认是否仍有效（plan/EDL hash 未变）。"""
    return (confirmation.plan_hash == compute_plan_hash(plan)
            and confirmation.edl_hash == compute_edl_hash(edl))

