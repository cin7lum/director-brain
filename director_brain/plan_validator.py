"""M2.3 Plan Validator：校验 EditorialDecisionList 与 DirectorDecisionPlan 的一致性。

迁移自 GEN-1 ``proposal_validator.py``，适配 EDL 模型。检查规则：

1. 总时长 > 0（空 EDL 报错）
2. 每个片段 ``in_frame < out_frame``
3. 无重叠片段（按 ``in_frame`` 排序后相邻检查）
4. 间隙可接受（不同镜头的源引用之间天然有间隙）
5. ``source_asset_id`` 必须存在于 observations 的 media_asset_id 集合
6. 总时长在目标范围 ±10%（仅当 ``plan.constraints`` 含 ``target_duration_us=XXX``）
7. plan↔EDL 一致性：``plan.sequence`` 必须与 EDL 的 source_asset_id 序列
   逐位相同（T1 修复新增——历史版本只校验 EDL，repair 删镜头后 plan 与
   EDL 静默分裂仍报 PASS）
8. plan↔EDL 一致性：所有 ``decision.shot_refs`` 摊平后与 EDL 的
   source_asset_id 集合一致（多重集比较，T1 修复新增）
9. must_avoid 技术约束：``plan.constraints`` 中由意图约束解释器编码的
   ``must_avoid:<metric><op>:<threshold>:<term>`` 条目，对每条 edit 的
   观测指标判红（P1-a 新增——用户"避免模糊"等约束从安慰剂变为硬校验）

Validator 如实报告所有错误，不做修复——修复由 :mod:`plan_repair` 或人工决定。
"""
from __future__ import annotations

import json
import re

from director_brain.intent_constraints import TechnicalAvoidRule
from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_observation import FilmObservation

_TARGET_DUR_RE = re.compile(r"target_duration_us=(\d+)")
_MUST_AVOID_RE = re.compile(r"must_avoid:([a-z_]+)([<>])([0-9.]+):(.+)")


def _parse_must_avoid_constraints(constraints: list[str]) -> list[TechnicalAvoidRule]:
    """从 plan.constraints 解码 must_avoid 技术约束（P1-a 编码的逆操作）。"""
    rules: list[TechnicalAvoidRule] = []
    for c in constraints:
        m = _MUST_AVOID_RE.match(c.strip())
        if not m:
            continue
        metric, op, threshold, term = m.group(1), m.group(2), m.group(3), m.group(4)
        rules.append(TechnicalAvoidRule(
            term=term,
            metric=metric,
            predicate="below" if op == "<" else "above",
            threshold=float(threshold),
        ))
    return rules


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
        if plan.sequence:
            errors.append(
                "plan/edl split: plan.sequence non-empty while EDL is empty"
            )
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

    # ---- Rule 7: plan.sequence 与 EDL 镜头序列逐位一致（T1 修复新增）----
    edl_ids = [e.source_asset_id for e in edits]
    if plan.sequence != edl_ids:
        errors.append(
            f"plan/edl split: plan.sequence {plan.sequence} != "
            f"EDL source_asset_id sequence {edl_ids}"
        )

    # ---- Rule 8: decision.shot_refs 摊平后与 EDL 镜头集合一致（T1 修复新增）----
    shot_refs = [ref for d in plan.decisions for ref in d.shot_refs]
    if sorted(shot_refs) != sorted(edl_ids):
        errors.append(
            f"plan/edl split: flattened decision.shot_refs {sorted(shot_refs)} != "
            f"EDL source_asset_id set {sorted(edl_ids)}"
        )

    # ---- Rule 9: must_avoid 技术约束判红（P1-a 新增）----
    avoid_rules = _parse_must_avoid_constraints(plan.constraints)
    if avoid_rules:
        obs_metrics: dict[str, dict] = {}
        for o in observations:
            try:
                data = json.loads(o.claim)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict):
                obs_metrics[o.media_asset_id] = data
        for e in edits:
            metrics = obs_metrics.get(e.source_asset_id, {})
            for rule in avoid_rules:
                if rule.violated(metrics):
                    errors.append(
                        f"must_avoid violation: 「{rule.term}」 edit "
                        f"{e.source_asset_id} {rule.metric} 违反 "
                        f"{rule.predicate} {rule.threshold}"
                    )

    return (len(errors) == 0, errors)


def _parse_target_duration(constraints: list[str]) -> int | None:
    """从 plan.constraints 中解析 ``target_duration_us=XXX``。找不到返回 None。"""
    for c in constraints:
        m = _TARGET_DUR_RE.search(c)
        if m:
            return int(m.group(1))
    return None
