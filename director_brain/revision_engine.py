"""M3.2 Revision Engine：基于观测/计划生成修订提案并应用到 EDL / Plan。

支持四种修订类型（``revision_type``）：

1. ``extend_peak``：延长落在总时长 50%~80% 区间（高潮幕）的镜头时长；
2. ``remove_low_quality``：移除 ``blur_score < 阈值`` 的镜头；
3. ``adjust_duration``：按 ``plan.constraints`` 中的 ``target_duration_us`` 调整总时长；
4. ``reorder``：按 ``in_frame`` 升序重排全部镜头，恢复叙事弧。

注意：现有 :class:`~director_brain.models.revision.RevisionProposal` 模型没有
``revision_type`` / ``reason`` / ``affected_edit_ids`` / ``expected_outcome`` 字段，
映射关系如下：

- ``revision_id``       → ``proposal_id``（``rev_<8位hex>``）
- ``revision_type``      → 承载在 ``change_summary`` 前缀，格式 ``f"{revision_type}: {reason}"``
- ``reason``             → ``change_summary`` 冒号后的描述
- ``affected_edit_ids``  → ``target_decision_ids``（用 EditItem.source_asset_id 标识）
- ``expected_outcome``   → ``expected_effect``
- ``approval_state``     → 固定 ``"pending"``

解析 ``revision_type`` 时用 ``proposal.change_summary.split(":")[0]``。
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

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
# 合法 revision_type 白名单
_REVISION_TYPES = ("extend_peak", "remove_low_quality", "adjust_duration", "reorder")


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
    revision_type: str,
    reason: str,
    target_ids: list[str],
    expected_effect: str,
) -> RevisionProposal:
    """构造一份 RevisionProposal，统一填充 BaseRecord 公共字段。"""
    return RevisionProposal(
        proposal_id=f"rev_{uuid.uuid4().hex[:8]}",
        source_finding_ids=[],
        target_decision_ids=list(target_ids),
        change_summary=f"{revision_type}: {reason}",
        expected_effect=expected_effect,
        regression_risks=[],
        approval_state="pending",
        verification_plan=None,
        result_refs=[],
        schema_version="1.0",
        project_id=edl.project_id,
        created_at=int(time.time()),
        producer="revision_engine_v0.1",
        source_ref=edl.source_ref,
    )


# ---------------------------------------------------------------------------
# propose_revision
# ---------------------------------------------------------------------------

def propose_revision(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    observations: list[FilmObservation],
    revision_type: str,
    reason: str,
) -> RevisionProposal:
    """根据修订类型生成一份 RevisionProposal（不修改任何入参）。

    Args:
        edl: 当前编辑决策表。
        plan: 当前导演决策计划。
        observations: 素材观测列表，用于解析 blur_score 等技术指标。
        revision_type: 四种之一（``extend_peak`` / ``remove_low_quality`` /
            ``adjust_duration`` / ``reorder``）。
        reason: 修订原因，会拼接到 ``change_summary``。

    Raises:
        ValueError: ``revision_type`` 不在白名单内。
    """
    if revision_type not in _REVISION_TYPES:
        raise ValueError(
            f"unknown revision_type: {revision_type!r}; "
            f"expected one of {_REVISION_TYPES}"
        )

    edits = edl.ordered_edits

    if revision_type == "extend_peak":
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

    elif revision_type == "remove_low_quality":
        blur_map = _parse_blur_map(observations)
        target_ids = [
            e.source_asset_id
            for e in edits
            if blur_map.get(e.source_asset_id, 100.0) < _DEFAULT_BLUR_THRESHOLD
        ]
        expected_effect = "移除低质量镜头，提升整体技术质量"

    elif revision_type == "adjust_duration":
        target_us = _parse_target_duration_us(plan.constraints)
        last_id = edits[-1].source_asset_id if edits else ""
        target_ids = [last_id]
        expected_effect = f"调整总时长到目标值 {target_us}us"

    else:  # reorder
        target_ids = [e.source_asset_id for e in edits]
        expected_effect = "按时间顺序重排镜头，恢复叙事弧"

    return _new_proposal(edl, revision_type, reason, target_ids, expected_effect)


# ---------------------------------------------------------------------------
# apply_revision
# ---------------------------------------------------------------------------

def apply_revision(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    proposal: RevisionProposal,
) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
    """把 proposal 应用到 EDL / Plan，返回新的深拷贝副本（不修改入参）。

    应用后 ``new_plan.validation_status`` 被置为 ``"revised_pending_validation"``，
    由调用方决定是否再调 :func:`~director_brain.plan_validator.validate_plan` 校验。
    """
    revision_type = proposal.change_summary.split(":", 1)[0].strip()

    new_edl = edl.model_copy(deep=True)
    new_plan = plan.model_copy(deep=True)

    target_ids = set(proposal.target_decision_ids)

    if revision_type == "remove_low_quality":
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

    elif revision_type == "extend_peak":
        for e in new_edl.ordered_edits:
            if e.source_asset_id in target_ids:
                new_out = e.out_frame + _PEAK_EXTEND_US
                # 保险：保证 out > in（延长操作天然满足，仍防御性处理）
                if new_out <= e.in_frame:
                    new_out = e.in_frame + 1
                e.out_frame = new_out

    elif revision_type == "adjust_duration":
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

    elif revision_type == "reorder":
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

    # 统一更新派生字段
    new_edl.expected_duration = _total_duration_us(new_edl)
    new_plan.validation_status = "revised_pending_validation"

    return new_edl, new_plan


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
