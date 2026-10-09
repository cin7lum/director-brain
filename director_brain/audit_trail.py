"""M3.3 Decision Ledger + 审计追踪模块。

为导演决策过程提供可回溯的审计能力：

- :func:`log_decision`：每次 EDL / 计划状态变更时向决策账本追加一条记录。
- :func:`get_decision_history`：按 ``decision_id`` 取出该决策的完整时间线。
- :func:`generate_audit_report`：汇总 EDL 摘要、决策时间线与证据完整性校验，
  产出可落盘的审计报告 dict。

``DecisionLedgerEntry`` 复用 :mod:`storage.repository` 中已有的 dataclass，
不重复定义。
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_observation import FilmObservation
from storage.repository import BrainRepository, DecisionLedgerEntry


def log_decision(
    repository: BrainRepository,
    decision_id: str,
    action: str,
    detail: dict[str, Any],
    *,
    project_id: str | None = None,
) -> DecisionLedgerEntry:
    """向决策账本追加一条变更记录并持久化。

    Args:
        repository: 存储后端。
        decision_id: 关联的决策实体 ID（通常是 ``plan.plan_id``）。
        action: 变更动作，取值之一：
            ``"plan_generated" / "plan_selected" / "revision_applied" /
            "plan_validated" / "plan_repaired"``。
        detail: 动作的附加上下文（producer、参数、校验结果等）。

    Returns:
        已构造并保存的 :class:`DecisionLedgerEntry`。
    """
    entry = DecisionLedgerEntry(
        ledger_id=f"ledger_{uuid.uuid4().hex[:8]}",
        decision_id=decision_id,
        action=action,
        timestamp=int(time.time()),
        detail=detail,
        project_id=project_id,
    )
    repository.save(entry)
    return entry


def get_decision_history(
    repository: BrainRepository,
    decision_id: str,
) -> list[DecisionLedgerEntry]:
    """取出某条决策的全部账本记录，按时间戳升序返回。

    ``repository.list`` 只支持按 ``project_id`` 过滤，因此这里全量取出后在
    内存中按 ``decision_id`` 过滤，再按 ``timestamp`` 升序排序。
    """
    all_entries = repository.list(DecisionLedgerEntry)
    matched = [e for e in all_entries if e.decision_id == decision_id]
    return sorted(matched, key=lambda e: e.timestamp)


def generate_audit_report(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    observations: list[FilmObservation],
    repository: BrainRepository,
) -> dict[str, Any]:
    """生成一份审计报告 dict，含 EDL 摘要、决策时间线与证据完整性校验。

    Returns:
        纯 dict，可直接 ``json.dumps`` 落盘。字段：

        - ``edl_summary``：EDL / 计划元信息摘要。
        - ``decision_timeline``：与该 ``plan.plan_id`` 关联的决策时间线。
        - ``evidence_integrity``：EDL 引用素材是否都能在观测中找到。
        - ``edit_traceability``：每条 edit 对应的 observation_id 追溯信息。
    """
    # ---- 1. edl_summary ----
    total_duration_us = sum(
        (e.out_frame - e.in_frame) for e in edl.ordered_edits
    )
    edl_summary: dict[str, Any] = {
        "edl_id": edl.edl_id,
        "shot_count": len(edl.ordered_edits),
        "total_duration_us": total_duration_us,
        "timebase": edl.timebase,
        "approval_state": edl.approval_state,
        "plan_id": plan.plan_id,
        "validation_status": plan.validation_status,
    }

    # ---- 2. decision_timeline ----
    history = get_decision_history(repository, plan.plan_id)
    decision_timeline: list[dict[str, Any]] = [
        {
            "timestamp": h.timestamp,
            "action": h.action,
            "detail": h.detail,
        }
        for h in history
    ]

    # ---- 3. evidence_integrity ----
    known_asset_ids = {o.media_asset_id for o in observations}
    missing_assets: list[str] = []
    verified_count = 0
    for edit in edl.ordered_edits:
        if edit.source_asset_id in known_asset_ids:
            verified_count += 1
        else:
            missing_assets.append(edit.source_asset_id)

    evidence_integrity: dict[str, Any] = {
        "all_assets_found": len(missing_assets) == 0,
        "missing_assets": missing_assets,
        "edit_count": len(edl.ordered_edits),
        "verified_count": verified_count,
    }

    # ---- 4. edit_traceability ----
    asset_to_obs = {o.media_asset_id: o.observation_id for o in observations}
    edit_traceability: list[dict[str, Any]] = [
        {
            "source_asset_id": e.source_asset_id,
            "in_frame": e.in_frame,
            "out_frame": e.out_frame,
            "shot_function": e.shot_function,
            "rationale": e.rationale,
            "observation_id": asset_to_obs.get(e.source_asset_id),
        }
        for e in edl.ordered_edits
    ]

    return {
        "edl_summary": edl_summary,
        "decision_timeline": decision_timeline,
        "evidence_integrity": evidence_integrity,
        "edit_traceability": edit_traceability,
    }
