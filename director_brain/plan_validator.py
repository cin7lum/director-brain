"""M2.3 Plan Validator：校验 EditorialDecisionList 与 DirectorDecisionPlan 的一致性。

迁移自 GEN-1 ``proposal_validator.py``，适配 EDL 模型。检查规则：

1. 总时长 > 0（空 EDL 报错）
2. 每个片段 ``in_frame < out_frame``
3. 无重叠片段（按 ``in_frame`` 排序后相邻检查）
4. 间隙可接受（不同镜头的源引用之间天然有间隙）
5. ``source_asset_id`` 必须存在于 observations 的 media_asset_id 集合
6. 总时长在目标范围 ±10%（仅当 ``plan.constraints`` 含 ``target_duration_us=XXX``）

Validator 如实报告所有错误，不做修复——修复由 :mod:`plan_repair` 或人工决定。
"""
from __future__ import annotations

import re

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_observation import FilmObservation

_TARGET_DUR_RE = re.compile(r"target_duration_us=(\d+)")


def validate_plan(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    observations: list[FilmObservation],
) -> tuple[bool, list[str]]:
    """校验 EDL 与 plan 的一致性。

    Returns:
        ``(is_valid, errors)``；``is_valid = len(errors) == 0``。
    """
    errors: list[str] = []
    edits = edl.ordered_edits

    # ---- Rule 1: 总时长 > 0（空 EDL 报错）----
    if len(edits) == 0:
        errors.append("empty EDL: no edits")
        return (False, errors)

    total_duration = sum(e.out_frame - e.in_frame for e in edits)

    # ---- Rule 5: source_asset_id 存在性（先收集集合，避免重复报错）----
    known_asset_ids = {o.media_asset_id for o in observations}
    seen_unknown: set[str] = set()
    for i, e in enumerate(edits):
        if e.source_asset_id not in known_asset_ids and e.source_asset_id not in seen_unknown:
            seen_unknown.add(e.source_asset_id)
            errors.append(f"unknown source_asset_id: {e.source_asset_id}")

    # ---- Rule 2: in_frame < out_frame ----
    for i, e in enumerate(edits):
        if e.in_frame >= e.out_frame:
            errors.append(
                f"edit {i}: in_frame ({e.in_frame}) >= out_frame ({e.out_frame})"
            )

    # ---- Rule 3: 无重叠（按 in_frame 排序后相邻检查，保留原始索引）----
    indexed = sorted(enumerate(edits), key=lambda pair: pair[1].in_frame)
    for k in range(len(indexed) - 1):
        prev_idx, prev = indexed[k]
        curr_idx, curr = indexed[k + 1]
        if prev.out_frame > curr.in_frame:
            errors.append(
                f"overlap: edit {prev_idx} ({prev.in_frame}-{prev.out_frame}) "
                f"overlaps edit {curr_idx} ({curr.in_frame}-{curr.out_frame})"
            )

    # ---- Rule 6: 时长在目标范围内（±10%）----
    target_us = _parse_target_duration(plan.constraints)
    if target_us is not None:
        low = int(0.9 * target_us)
        high = int(1.1 * target_us)
        if total_duration < low or total_duration > high:
            errors.append(
                f"duration {total_duration}us outside target range "
                f"[{low}, {high}]us"
            )

    return (len(errors) == 0, errors)


def _parse_target_duration(constraints: list[str]) -> int | None:
    """从 plan.constraints 中解析 ``target_duration_us=XXX``。找不到返回 None。"""
    for c in constraints:
        m = _TARGET_DUR_RE.search(c)
        if m:
            return int(m.group(1))
    return None
