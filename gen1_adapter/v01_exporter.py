"""V0.1 proposal 导出器：将 Director Brain EDL + Plan 转回 V0.1 proposal dict。

用于影子评估（shadow evaluation）：把 EDL/Plan 序列化为 GEN-1 可消费的
V0.1 JSON 结构，与真实 heuristic 输出对比。
"""
from __future__ import annotations

import logging

from director_brain._utils import short_hash
from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList

logger = logging.getLogger(__name__)

#: edl_id / plan_id 导入时添加的前缀，导出时去除
_EDL_PREFIX = "edl_from_"
_PLAN_PREFIX = "plan_from_"


def _strip_prefix(value: str, prefix: str) -> str:
    """去除导入时添加的前缀；无前缀时原样返回。"""
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


def _infer_confidence_type(rationale) -> str:
    """从 Decision.rationale 推断 confidence_type。

    - rationale 含 ``"heuristic:"`` → ``"HEURISTIC"``
    - rationale 含 ``"vlm:"`` → ``"SELF_REPORTED"``
    - 无法推断（含 dec 缺失导致 rationale 为 None）→ warning 并默认 ``"HEURISTIC"``
    """
    if rationale and "heuristic:" in rationale:
        return "HEURISTIC"
    if rationale and "vlm:" in rationale:
        return "SELF_REPORTED"
    logger.warning("无法从 rationale 推断 confidence_type, 默认 HEURISTIC: %s", rationale)
    return "HEURISTIC"


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
        confidence_type = _infer_confidence_type(dec.rationale if dec else None)

        # EditItem 无 clip_instance_id 字段，从 source_asset_id + in/out 派生稳定 ID
        clip_instance_id = (
            f"clip_{short_hash(f'{item.source_asset_id}|{item.in_frame}|{item.out_frame}')}"
        )

        slots.append({
            "slot_id": slot_id,
            "position": pos,
            "source_shot_id": item.source_asset_id,
            "source_media_hash": item.source_media_hash,
            "proposed_in_us": item.in_frame,
            "proposed_out_us": item.out_frame,
            "proposed_role": item.shot_function,
            "reason": item.rationale,
            "clip_instance_id": clip_instance_id,
            "confidence_type": confidence_type,
            "confidence_value": confidence_value,
        })

    return {
        "proposal_id": proposal_id,
        "rough_cut_id": f"rc_{short_hash(edl.edl_id)}",
        "project_id": project_id,
        "ai_model": ai_model,
        "prompt_version": ai_model,
        "slots": slots,
    }
