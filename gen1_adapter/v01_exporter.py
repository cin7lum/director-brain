"""V0.1 proposal 导出器：将 Director Brain EDL + Plan 转回 V0.1 proposal dict。

用于影子评估（shadow evaluation）：把 EDL/Plan 序列化为 GEN-1 可消费的
V0.1 JSON 结构，与真实 heuristic 输出对比。
"""
from __future__ import annotations

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList

#: edl_id / plan_id 导入时添加的前缀，导出时去除
_EDL_PREFIX = "edl_from_"
_PLAN_PREFIX = "plan_from_"


def _strip_prefix(value: str, prefix: str) -> str:
    """去除导入时添加的前缀；无前缀时原样返回。"""
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


def export_to_v01(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
) -> dict:
    """将 EDL + Plan 反向序列化为 V0.1 proposal dict。

    Args:
        edl: 导入得到的 EDL。
        plan: 导入得到的 Plan，其 decisions 与 edl.ordered_edits 一一对应。

    Returns:
        可直接 ``json.dumps`` 的 V0.1 proposal dict。
    """
    proposal_id = _strip_prefix(edl.edl_id, _EDL_PREFIX)
    project_id = edl.project_id
    ai_model = edl.producer

    # decisions 与 ordered_edits 在导入时同序构造，可直接 zip
    decisions_by_id = {dec.decision_id: dec for dec in plan.decisions}

    slots: list[dict] = []
    for pos, item in enumerate(edl.ordered_edits):
        slot_id = f"slot_{pos + 1:02d}"
        # 尝试按导入时的 decision_id 规则找到对应 decision
        dec = decisions_by_id.get(f"dec_{slot_id}")
        confidence_value = dec.confidence if dec else None
        # confidence_type 无法从 EDL/Plan 恢复，导出 HEURISTIC 作为占位
        confidence_type = "HEURISTIC"

        slots.append({
            "slot_id": slot_id,
            "position": pos,
            "source_shot_id": item.source_asset_id,
            "source_media_hash": item.source_media_hash,
            "proposed_in_us": item.in_frame,
            "proposed_out_us": item.out_frame,
            "proposed_role": item.shot_function,
            "reason": item.rationale,
            "confidence_type": confidence_type,
            "confidence_value": confidence_value,
        })

    return {
        "proposal_id": proposal_id,
        "rough_cut_id": "",
        "project_id": project_id,
        "ai_model": ai_model,
        "prompt_version": ai_model,
        "slots": slots,
    }
