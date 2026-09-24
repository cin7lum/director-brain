"""V0.1 proposal JSON 导入器：将 GEN-1 V0.1 proposal 转为 Director Brain EDL + Plan。

字段映射规则见任务说明；微秒时基固定为 timebase=1_000_000。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList

#: V0.1 proposal 无 fps 信息，统一使用微秒时基
MICROSECOND_TIMEBASE = 1_000_000

#: 无法映射的字段占位符
NOT_DETERMINED = "NOT_DETERMINED"


def import_v01_proposal(
    json_path: str,
) -> tuple[EditorialDecisionList, DirectorDecisionPlan, list[dict]]:
    """读取 V0.1 proposal JSON 文件，转换为 EDL + Plan + 导入日志。

    Args:
        json_path: V0.1 proposal JSON 文件路径。

    Returns:
        ``(edl, plan, import_log)`` 三元组。``import_log`` 为 list[dict]，
        每条含 ``level``、``message``，可选 ``slot_id``。
    """
    path = Path(json_path)
    import_log: list[dict] = []

    raw = json.loads(path.read_text(encoding="utf-8"))
    proposal_id = raw.get("proposal_id", NOT_DETERMINED)
    project_id = raw.get("project_id", NOT_DETERMINED)
    ai_model = raw.get("ai_model", "unknown")
    slots = raw.get("slots", []) or []

    if not slots:
        import_log.append({"level": "info", "message": "no slots in proposal"})

    # ---- 逐 slot 转换 -------------------------------------------------------
    edit_items: list[EditItem] = []
    decisions: list[Decision] = []
    source_hashes: list[str] = []

    # 按 position 排序；缺失 position 时保持原顺序
    indexed = list(enumerate(slots))
    indexed.sort(key=lambda pair: pair[1].get("position", pair[0]))

    for orig_idx, slot in indexed:
        slot_id = slot.get("slot_id", f"slot_{orig_idx + 1:02d}")
        source_shot_id = slot.get("source_shot_id")
        if not source_shot_id:
            import_log.append({
                "level": "warning",
                "message": "slot missing source_shot_id, skipped",
                "slot_id": slot_id,
            })
            continue

        source_media_hash = slot.get("source_media_hash", NOT_DETERMINED)
        proposed_in = int(slot.get("proposed_in_us", 0))
        proposed_out = int(slot.get("proposed_out_us", 0))
        # reason 兼容两种字段名：reason / proposed_reason
        reason = slot.get("reason") or slot.get("proposed_reason")
        shot_function = slot.get("proposed_role")
        confidence_value = slot.get("confidence_value")

        item = EditItem(
            source_asset_id=source_shot_id,
            source_media_hash=source_media_hash,
            in_frame=proposed_in,
            out_frame=proposed_out,
            timebase=MICROSECOND_TIMEBASE,
            shot_function=shot_function,
            rationale=reason,
        )
        edit_items.append(item)
        source_hashes.append(source_media_hash)

        dec = Decision(
            decision_id=f"dec_{slot_id}",
            purpose=shot_function or "unspecified",
            rationale=reason,
            confidence=confidence_value,
            requires_approval=False,
        )
        decisions.append(dec)

    # ---- 构造 EDL ----------------------------------------------------------
    now = int(time.time())
    edl = EditorialDecisionList(
        edl_id=f"edl_from_{proposal_id}",
        version="1.0",
        brief_version="unknown",
        context_id=NOT_DETERMINED,
        timebase=MICROSECOND_TIMEBASE,
        approval_state="imported",
        ordered_edits=edit_items,
        source_asset_hashes=source_hashes,
        schema_version="1.0",
        project_id=project_id,
        created_at=now,
        producer=ai_model,
        source_ref=str(path),
    )

    # ---- 构造 Plan ---------------------------------------------------------
    plan = DirectorDecisionPlan(
        plan_id=f"plan_from_{proposal_id}",
        version="1.0",
        brief_version="unknown",
        film_state_version="imported",
        validation_status="unverified",
        approval_state="imported",
        decisions=decisions,
        schema_version="1.0",
        project_id=project_id,
        created_at=now,
        producer=ai_model,
        source_ref=str(path),
    )

    return edl, plan, import_log
