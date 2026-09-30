"""M3.2 Revision Engine：基于 FQL findings / 观测生成修订提案。

权限模型（架构体检⑦拆雷，与 T1 裁定分层）：

- **修复器**（plan_repair，管线内自动执行）：物理-only 白名单
  （已选镜头内裁时间/交换），无权增删镜头；
- **修订**（本模块，finding 驱动的显式提案）：允许导演级操作
  （删除低质镜头/重排/延长高潮/调时长），但应用产物是**取代性
  新草案**——``supersedes_plan_id/edl_id`` 指向旧版、``state=draft``、
  验证与确认全部重走；绝不原地修改已确认的 plan/EDL。

修订类型为 :class:`RevisionType` 枚举（白名单内字符串兼容），
取代旧版"类型拼在 change_summary 前缀"的字符串协议。

"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import enum

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_observation import FilmObservation
from director_brain.models.revision import RevisionProposal

# extend_peak 时单个镜头最多延长的微秒数（2 秒）
_PEAK_EXTEND_US = 2_000_000
# remove_low_quality 默认 blur 阈值
_DEFAULT_BLUR_THRESHOLD = 50.0
# adjust_duration 截断时为最后一个镜头保留的最小微秒数
_MIN_TAIL_US = 1
# 合法 revision_type 白名单（字符串兼容旧调用方）
_REVISION_TYPES = ("extend_peak", "remove_low_quality", "adjust_duration", "reorder")


class RevisionType(str, enum.Enum):
    """修订类型（导演级，应用产物为取代性新草案）。"""

    EXTEND_PEAK = "extend_peak"
    REMOVE_LOW_QUALITY = "remove_low_quality"
    ADJUST_DURATION = "adjust_duration"
    REORDER = "reorder"

    @classmethod
    def coerce(cls, value: "RevisionType | str") -> "RevisionType":
        if isinstance(value, cls):
            return value
        if value in _REVISION_TYPES:
            return cls(value)
        raise ValueError(
            f"unknown revision_type: {value!r}; expected one of {_REVISION_TYPES}"
        )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _parse_blur_map(
    observations: list[FilmObservation],
) -> dict[str, float]:
    """从 deterministic_technical 观测中解析 ``asset_id -> blur_score``。

    claim 不是合法 JSON 或缺字段时静默跳过，返回的 map 可能不包含某 asset。
    """
    blur_map: dict[str, float] = {}
    for o in observations:
        if o.observation_type != "deterministic_technical":
            continue
        try:
            claim = json.loads(o.claim) if o.claim else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(claim, dict):
            continue
        blur = claim.get("blur_score")
        if isinstance(blur, (int, float)):
            blur_map[o.media_asset_id] = float(blur)
    return blur_map


def _parse_target_duration_us(constraints: list[str]) -> int | None:
    """从 ``plan.constraints`` 中解析 ``target_duration_us=XXX``。"""
    for c in constraints:
        if "=" not in c:
            continue
        key, _, value = c.partition("=")
        if key.strip() == "target_duration_us":
            try:
                return int(value.strip())
            except ValueError:
                continue
    return None


def _total_duration_us(edl: EditorialDecisionList) -> int:
    return sum(e.out_frame - e.in_frame for e in edl.ordered_edits)


def _new_proposal(
    edl: EditorialDecisionList,
    revision_type: RevisionType,
    reason: str,
    target_ids: list[str],
    expected_effect: str,
    finding_ids: list[str] | None = None,
) -> RevisionProposal:
    """构造一份 RevisionProposal（类型化字段 + 旧 change_summary 兼容）。"""
    return RevisionProposal(
        proposal_id=f"rev_{uuid.uuid4().hex[:8]}",
        source_finding_ids=list(finding_ids or []),
        target_decision_ids=list(target_ids),
        change_summary=f"{revision_type.value}: {reason}",
        revision_type=revision_type.value,
        reason=reason,
        expected_effect=expected_effect,
        regression_risks=[],
        approval_state="pending",
        verification_plan=None,
        result_refs=[],
        schema_version="1.0",
        project_id=edl.project_id,
        created_at=int(time.time()),
        producer="revision_engine_v1.0",
        source_ref=edl.source_ref,
    )


# ---------------------------------------------------------------------------
# propose_revision
# ---------------------------------------------------------------------------

def propose_revision(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    observations: list[FilmObservation],
    revision_type: RevisionType | str,
    reason: str,
    finding_ids: list[str] | None = None,
) -> RevisionProposal:
    """根据修订类型生成一份 RevisionProposal（不修改任何入参）。

    Args:
        edl: 当前编辑决策表。
        plan: 当前导演决策计划。
        observations: 素材观测列表，用于解析 blur_score 等技术指标。
        revision_type: :class:`RevisionType` 或白名单字符串。
        reason: 修订原因。
        finding_ids: FQL findings 来源（spec §6：finding 驱动修订）。

    Raises:
        ValueError: ``revision_type`` 不在白名单内。
    """
    rtype = RevisionType.coerce(revision_type)

    edits = edl.ordered_edits

    if rtype is RevisionType.EXTEND_PEAK:
        # 总时长用 max(out_frame) 作为素材时间轴长度；peak 幕落在 [50%, 80%]
        total = max((e.out_frame for e in edits), default=0)
        low = 0.5 * total
        high = 0.8 * total
        target_ids = [
            e.source_asset_id
            for e in edits
            if low <= e.in_frame <= high
        ]
        expected_effect = "延长高潮幕镜头时长，增强叙事张力"

    elif rtype is RevisionType.REMOVE_LOW_QUALITY:
        blur_map = _parse_blur_map(observations)
        target_ids = [
            e.source_asset_id
            for e in edits
            if blur_map.get(e.source_asset_id, 100.0) < _DEFAULT_BLUR_THRESHOLD
        ]
        expected_effect = "移除低质量镜头，提升整体技术质量"

    elif rtype is RevisionType.ADJUST_DURATION:
        target_us = _parse_target_duration_us(plan.constraints)
        last_id = edits[-1].source_asset_id if edits else ""
        target_ids = [last_id]
        expected_effect = f"调整总时长到目标值 {target_us}us"

    else:  # REORDER
        target_ids = [e.source_asset_id for e in edits]
        expected_effect = "按时间顺序重排镜头，恢复叙事弧"

    return _new_proposal(edl, rtype, reason, target_ids, expected_effect, finding_ids)


# ---------------------------------------------------------------------------
# apply_revision
# ---------------------------------------------------------------------------

def apply_revision(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    proposal: RevisionProposal,
    observations: list[FilmObservation] | None = None,
) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
    """把 proposal 应用为**取代性新草案**（候选⑦拆雷；不修改任何入参）。

    权限模型：修订是导演级重规划，产物是新 plan/EDL——
    ``supersedes_plan_id/supersedes_edl_id`` 指向旧版、``state=draft``、
    ``validation_status="pending"``、``approval_state="draft"``，必须重走
    验证 + 策略确认（渲染闸门会拒绝未确认 plan）。旧版由调用方经
    :func:`supersede_plan` 标记 SUPERSEDED。绝不原地修改入参。

    observations: 源镜头观测（extend_peak 的延长钳制到源镜头出点——
    防止读到下一镜头画面；与修复器的边界余量修复同源教训）。
    """
    revision_type = RevisionType.coerce(
        proposal.revision_type
        or proposal.change_summary.split(":", 1)[0].strip()
    )

    new_edl = edl.model_copy(deep=True)
    new_plan = plan.model_copy(deep=True)
    new_edl.edl_id = f"{edl.edl_id}_rev"
    new_edl.supersedes_edl_id = edl.edl_id
    new_plan.plan_id = f"{plan.plan_id}_rev"
    new_plan.supersedes_plan_id = plan.plan_id

    target_ids = set(proposal.target_decision_ids)

    if revision_type is RevisionType.REMOVE_LOW_QUALITY:
        # 从 EDL 中删除低质量镜头
        new_edl.ordered_edits = [
            e for e in new_edl.ordered_edits
            if e.source_asset_id not in target_ids
        ]
        # 同步删除 plan.decisions 中 shot_refs[0] 命中的 Decision
        new_plan.decisions = [
            d for d in new_plan.decisions
            if not (d.shot_refs and d.shot_refs[0] in target_ids)
        ]
        # 同步重写 sequence
        new_plan.sequence = [
            sid for sid in new_plan.sequence
            if sid not in target_ids
        ]

    elif revision_type is RevisionType.EXTEND_PEAK:
        source_ends = {
            o.media_asset_id: o.end_frame
            for o in (observations or [])
            if o.observation_type == "deterministic_technical"
        }
        for e in new_edl.ordered_edits:
            if e.source_asset_id in target_ids:
                new_out = e.out_frame + _PEAK_EXTEND_US
                # 源边界钳制：延长不得越过源镜头出点（读到下一镜头画面）
                source_end = source_ends.get(e.source_asset_id)
                if source_end is not None:
                    new_out = min(new_out, int(source_end))
                # 保险：保证 out > in（延长操作天然满足，仍防御性处理）
                if new_out <= e.in_frame:
                    new_out = e.in_frame + 1
                e.out_frame = new_out

    elif revision_type is RevisionType.ADJUST_DURATION:
        target_us = _parse_target_duration_us(new_plan.constraints)
        if target_us is not None and new_edl.ordered_edits:
            current = _total_duration_us(new_edl)
            diff = target_us - current
            last = new_edl.ordered_edits[-1]
            if diff > 0:
                last.out_frame = last.out_frame + diff
            elif diff < 0:
                new_out = last.out_frame + diff  # diff 为负 → 截断
                # 至少保留 _MIN_TAIL_US
                if new_out <= last.in_frame:
                    new_out = last.in_frame + _MIN_TAIL_US
                last.out_frame = new_out

    elif revision_type is RevisionType.REORDER:
        # 按 in_frame 升序重排 EDL
        new_edl.ordered_edits.sort(key=lambda e: e.in_frame)
        # 用 source_asset_id 索引 decisions，按重排后的 EDL 顺序重建 decisions
        dec_by_shot: dict[str, Any] = {}
        for d in new_plan.decisions:
            if d.shot_refs:
                dec_by_shot[d.shot_refs[0]] = d
        new_plan.decisions = [
            dec_by_shot[e.source_asset_id]
            for e in new_edl.ordered_edits
            if e.source_asset_id in dec_by_shot
        ]
        new_plan.sequence = [e.source_asset_id for e in new_edl.ordered_edits]

    # 统一更新派生字段：新草案重走管线（state 词汇表见 plan_state）
    new_edl.expected_duration = _total_duration_us(new_edl)
    new_plan.state = "draft"
    new_plan.validation_status = "pending"
    new_plan.approval_state = "draft"

    return new_edl, new_plan


def supersede_plan(plan: DirectorDecisionPlan) -> DirectorDecisionPlan:
    """把被取代的旧 plan 标记为 SUPERSEDED（经状态机合法转换）。

    转换合法性由 :func:`plan_state.transition_plan` 执法——终态
    （EXECUTION_VERIFIED 之后）不可再被取代时抛 InvalidTransition。
    """
    from director_brain.plan_state import PlanState, transition_plan

    return transition_plan(plan, PlanState.SUPERSEDED)


# ---------------------------------------------------------------------------
# list_revisions
# ---------------------------------------------------------------------------

def list_revisions(project_id: str, repository: Any) -> list[RevisionProposal]:
    """从 repository 列出某项目的全部 RevisionProposal。"""
    from director_brain.models.revision import RevisionProposal  # noqa: WPS433

    rows = repository.list(RevisionProposal, project_id=project_id)
    result: list[RevisionProposal] = []
    for r in rows:
        if isinstance(r, RevisionProposal):
            result.append(r)
        else:  # pragma: no cover - repository 可能返回裸 dict
            result.append(RevisionProposal(**r))
    return result
