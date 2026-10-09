"""M2.3 Plan Repair：对校验失败的 EDL 做确定性**物理**修复（T1 收权版）。

语义裁定（2026-09-27，"Plan = 导演依据"）：:class:`DirectorDecisionPlan` 是
导演创作意图的真实来源（source of truth），修复器**无权改动镜头集合与顺序**
——它只能在已选镜头内部做"不改变任何创作决策"的物理调整（in/out 裁剪）。

**允许的物理操作（白名单，均不增删镜头、不重排顺序）：**

- P1：``in_frame >= out_frame`` 且 ``out_frame > 0`` 时交换两端点（数据缺陷修正）
- P2：过短片段（< ``MIN_CLIP_US``）在源镜头范围内扩展到 ``MIN_CLIP_US``
- P3：过长片段（> ``MAX_CLIP_US``）中点截断到 ``MAX_CLIP_US``
- P4：小幅重叠（重叠量 < 较短片段时长的 50%）截断时间在前片段的尾部
- P5：总时长偏离目标 ±10% 时，在已选镜头的源范围内延长/缩短

**以下情况一律 ABSTAIN（fail-closed，退回导演层重新出 plan）：**

- 镜头引用了不存在的观测（unknown source_asset_id）
- 同一镜头被选中多次（选片重复属导演层缺陷）
- ``out_frame <= 0`` 无法交换修复的时间范围
- 源镜头过短，无法扩展到 ``MIN_CLIP_US``
- 大幅重叠（>= 较短片段时长的 50%）——"留谁删谁"是创作决策
- 物理调整到极限后仍无法满足目标时长
- 空 EDL（选片为空属导演层问题，修复器无权追加镜头）

**历史版本（<= c72f0d1）的越权行为已移除**：rule4 去重删镜头、rule6 大幅
重叠删镜头、rule10 从未选中镜头追加（``shot_function="repair_appended"``）、
rule7 按 ``in_frame`` 重排（顺序是导演决策，修复器不得重排）。以上行为正是
P0-1「plan 与 EDL 分裂而校验器仍 PASS」的成因。

每次 ABSTAIN 都带受控枚举原因码（``REASON_*``），由调用方（导演层/CLI）
决定下一步；所有已执行的物理调整写入 ``plan.open_questions`` 留痕。
不修改原 edl/plan 对象。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import (
    EditItem,
    EditorialDecisionList,
    ordered_unique_source_hashes,
)
from director_brain.models.film_observation import FilmObservation
from director_brain.plan_validator import validate_plan, _parse_target_duration
from director_brain.providers.heuristic import MAX_CLIP_US, MIN_CLIP_US

PRODUCER = "plan_repair_v0.2"

_CLIP_MIN_RE = re.compile(r"min_clip_us=(\d+)")
_CLIP_MAX_RE = re.compile(r"max_clip_us=(\d+)")


def _parse_clip_bounds(constraints: list[str]) -> tuple[int, int]:
    """从 plan.constraints 解析片段时长界（P1-b）；未声明回退模块默认。"""
    min_us = max_us = None
    for c in constraints:
        m = _CLIP_MIN_RE.search(c)
        if m:
            min_us = int(m.group(1))
        m = _CLIP_MAX_RE.search(c)
        if m:
            max_us = int(m.group(1))
    return (min_us or MIN_CLIP_US, max_us or MAX_CLIP_US)


def _is_heuristic_evidence(e) -> bool:
    """片段证据是否为纯启发式（P1-b 结构化字段优先，旧数据回退字符串嗅探）。"""
    if e.evidence_type:
        return e.evidence_type == "heuristic"
    return "heuristic:" in (e.rationale or "")

# ---- ABSTAIN 原因码（受控枚举，禁止自由文本）----
REASON_EMPTY_SELECTION = "empty_shot_selection"
REASON_UNKNOWN_SOURCE_ASSET = "unknown_source_asset"
REASON_DUPLICATE_SHOT = "duplicate_shot_selection"
REASON_INVALID_TIME_RANGE = "invalid_time_range_unfixable"
REASON_SHOT_TOO_SHORT = "shot_too_short_unfixable"
REASON_OVERLAP_UNRESOLVABLE = "shot_overlap_unresolvable"
REASON_DURATION_UNREACHABLE = "target_duration_unreachable"


@dataclass
class RepairOutcome:
    """repair_plan 的一等返回值：OK（已物理修复）或 ABSTAIN（需导演层重决策）。"""

    status: str  # "ok" | "abstain"
    edl: EditorialDecisionList | None = None
    plan: DirectorDecisionPlan | None = None
    #: abstain 时的受控枚举原因码（REASON_* 常量之一）
    reason_code: str | None = None
    #: 人类可读说明
    reason: str | None = None
    #: 已执行的物理调整留痕（ok 时非空；亦已写入 plan.open_questions）
    adjustments: list[str] = field(default_factory=list)

    @property
    def requires_director(self) -> bool:
        """True 表示修复器放弃（repair_requires_director），须导演层重新决策。"""
        return self.status == "abstain"


def _claim_of(o: FilmObservation | None) -> dict:
    if o is None:
        return {}
    try:
        data = json.loads(o.claim)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def repair_plan(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    observations: list[FilmObservation],
) -> RepairOutcome:
    """对 EDL 做确定性物理修复；无法物理修复时 ABSTAIN。不修改原对象。

    Args:
        edl: 待修复的 EDL。
        plan: 对应的导演决策计划（只读依据，本函数绝不改写其镜头集合）。
        observations: 素材观测列表，用于查 source_asset_id 的源镜头范围。

    Returns:
        :class:`RepairOutcome`：``status="ok"`` 携带修复后的 (edl, plan)；
        ``status="abstain"`` 携带 ``reason_code``/``reason``，edl/plan 为 None。
    """
    new_edl = edl.model_copy(deep=True)
    new_plan = plan.model_copy(deep=True)

    edits: list[EditItem] = list(new_edl.ordered_edits)
    obs_by_asset = {o.media_asset_id: o for o in observations}
    known_ids = set(obs_by_asset.keys())
    adjustments: list[str] = []

    def _abstain(code: str, reason: str) -> RepairOutcome:
        logging.info("repair abstain (%s): %s", code, reason)
        return RepairOutcome(status="abstain", reason_code=code, reason=reason)

    # P1-b：与生成端同一组片段时长界（plan.constraints 未声明时回退默认）
    min_clip_us, max_clip_us = _parse_clip_bounds(new_plan.constraints)

    # ---- 前置检查：镜头集合级缺陷（修复器无权处理，ABSTAIN）----
    if not edits:
        return _abstain(
            REASON_EMPTY_SELECTION,
            "EDL 为空：选片为空属导演层决策问题，修复器无权追加镜头",
        )

    unknown = sorted({e.source_asset_id for e in edits if e.source_asset_id not in known_ids})
    if unknown:
        return _abstain(
            REASON_UNKNOWN_SOURCE_ASSET,
            f"EDL 引用了不存在的观测: {unknown}；观测与决策不一致，须导演层重新出 plan",
        )

    ids = [e.source_asset_id for e in edits]
    dupes = sorted({a for a in ids if ids.count(a) > 1})
    if dupes:
        return _abstain(
            REASON_DUPLICATE_SHOT,
            f"同一镜头被选中多次: {dupes}；保留哪一次属创作决策，须导演层重新出 plan",
        )

    for e in edits:
        if e.in_frame >= e.out_frame and e.out_frame <= 0:
            return _abstain(
                REASON_INVALID_TIME_RANGE,
                f"edit {e.source_asset_id} in={e.in_frame} out={e.out_frame} "
                f"无法交换修复，须导演层重新出 plan",
            )

    # ---- P1: 交换反置的 in/out（数据缺陷修正，不改动镜头集合）----
    fixed: list[EditItem] = []
    for e in edits:
        if e.in_frame >= e.out_frame:
            fixed.append(e.model_copy(update={
                "in_frame": e.out_frame,
                "out_frame": e.in_frame,
            }))
            adjustments.append(f"swap_endpoints:{e.source_asset_id}")
        else:
            fixed.append(e)
    edits = fixed

    # ---- P2: 过短片段在源镜头范围内扩展到 MIN_CLIP_US ----
    fixed = []
    for e in edits:
        dur = e.out_frame - e.in_frame
        if dur >= min_clip_us:
            fixed.append(e)
            continue
        obs = obs_by_asset.get(e.source_asset_id)
        src_dur = (obs.end_frame - obs.start_frame) if obs else 0
        if src_dur < min_clip_us:
            return _abstain(
                REASON_SHOT_TOO_SHORT,
                f"edit {e.source_asset_id} 时长 {dur}us < min_clip_us 且源镜头"
                f"（{src_dur}us）无法扩展，须导演层重新出 plan",
            )
        # 以中点为中心扩展到 MIN_CLIP_US，钳制到源镜头范围
        mid = (e.in_frame + e.out_frame) // 2
        half = min_clip_us // 2
        new_in = mid - half
        new_out = mid + half
        if new_in < obs.start_frame:
            new_in = obs.start_frame
            new_out = new_in + min_clip_us
        if new_out > obs.end_frame:
            new_out = obs.end_frame
            new_in = new_out - min_clip_us
        fixed.append(e.model_copy(update={
            "in_frame": new_in,
            "out_frame": new_out,
        }))
        adjustments.append(
            f"extend_to_min:{e.source_asset_id}:{e.in_frame}-{e.out_frame}->{new_in}-{new_out}"
        )
    edits = fixed

    # ---- P3: 过长片段截断到 MAX_CLIP_US（中点居中）----
    fixed = []
    for e in edits:
        dur = e.out_frame - e.in_frame
        if dur <= max_clip_us:
            fixed.append(e)
            continue
        mid = (e.in_frame + e.out_frame) // 2
        half = max_clip_us // 2
        new_in = mid - half
        new_out = mid + half
        fixed.append(e.model_copy(update={
            "in_frame": new_in,
            "out_frame": new_out,
        }))
        adjustments.append(
            f"truncate_to_max:{e.source_asset_id}:{e.in_frame}-{e.out_frame}->{new_in}-{new_out}"
        )
    edits = fixed

    # ---- P4: 重叠修复 —— 小幅截断时间在前的片段；大幅 ABSTAIN ----
    # 只做物理截断：列表顺序（= plan.sequence 顺序）保持不变。
    indexed = sorted(enumerate(edits), key=lambda pair: pair[1].in_frame)
    abstain_reason: str | None = None
    for k in range(len(indexed) - 1):
        i_prev, prev = indexed[k]
        _i_curr, curr = indexed[k + 1]
        if prev.out_frame <= curr.in_frame:
            continue
        overlap = prev.out_frame - curr.in_frame
        prev_dur = prev.out_frame - prev.in_frame
        curr_dur = curr.out_frame - curr.in_frame
        smaller = min(prev_dur, curr_dur)
        ratio = (overlap / smaller) if smaller > 0 else 1.0
        if ratio >= 0.5:
            abstain_reason = (
                f"edit {prev.source_asset_id}（{prev.in_frame}-{prev.out_frame}）与 "
                f"edit {curr.source_asset_id}（{curr.in_frame}-{curr.out_frame}）重叠 "
                f"{overlap}us（>= 较短片段的 50%）；保留哪一个是创作决策，"
                f"须导演层重新出 plan"
            )
            break
        new_out = curr.in_frame
        edits[i_prev] = edits[i_prev].model_copy(update={"out_frame": new_out})
        adjustments.append(
            f"trim_overlap:{prev.source_asset_id}:out->{new_out}"
        )
    if abstain_reason is not None:
        return _abstain(REASON_OVERLAP_UNRESOLVABLE, abstain_reason)

    # ---- P5: 时长修复（在已选镜头内延长/缩短，不增删镜头）----
    target = _parse_target_duration(new_plan.constraints)
    if target is not None:
        low = int(0.9 * target)
        high = int(1.1 * target)
        # 8b：转场感知——修复目标是**成片总时长**（Σd−ΣD）落在 ±10% 内
        current = sum(e.out_frame - e.in_frame for e in edits)
        current -= sum(
            e.transition.duration_us for i, e in enumerate(edits[:-1])
            if e.transition is not None and e.transition.type == "xfade"
        )

        if current < low:
            # P2-d 边界余量：延长目标取下界 +2%，避免渲染帧取整后偏差
            # 恰好压线 ±10% 被 L1 判 FAIL（实测 10.0004% 翻车案例）
            extend_target = low + int(0.02 * target)
            gap = extend_target - current

            # 优先延长纯启发式证据的片段（确定性技术优先级；P1-b 结构化判据）
            def _ext_priority(i: int) -> tuple:
                e = edits[i]
                is_heuristic = 0 if _is_heuristic_evidence(e) else 1
                return (is_heuristic, -(e.out_frame - e.in_frame))

            for idx in sorted(range(len(edits)), key=_ext_priority):
                if gap <= 0:
                    break
                e = edits[idx]
                obs = obs_by_asset.get(e.source_asset_id)
                if obs is None:
                    continue
                dur = e.out_frame - e.in_frame
                next_in = edits[idx + 1].in_frame if idx + 1 < len(edits) else 10**18
                max_ext = min(
                    obs.end_frame - e.out_frame,
                    max_clip_us - dur,
                    next_in - e.out_frame,
                )
                if max_ext <= 0:
                    continue
                ext = min(max_ext, gap)
                old_out = e.out_frame
                new_out = old_out + ext
                edits[idx] = e.model_copy(update={"out_frame": new_out})
                adjustments.append(
                    f"extend:{e.source_asset_id}:out {old_out}->{new_out}"
                )
                gap -= ext

            # P2-d 第二轮：向后延长（in_frame）——居中裁剪留下的源前部余量
            # 同属"已选镜头内 in/out 调整"白名单；先前只向前延长浪费了一半余量
            if gap > 0:
                for idx in sorted(range(len(edits)), key=_ext_priority):
                    if gap <= 0:
                        break
                    e = edits[idx]
                    obs = obs_by_asset.get(e.source_asset_id)
                    if obs is None:
                        continue
                    dur = e.out_frame - e.in_frame
                    prev_out = edits[idx - 1].out_frame if idx > 0 else 0
                    max_back = min(
                        e.in_frame - obs.start_frame,
                        max_clip_us - dur,
                        e.in_frame - prev_out,
                    )
                    if max_back <= 0:
                        continue
                    ext = min(max_back, gap)
                    old_in = e.in_frame
                    new_in = old_in - ext
                    edits[idx] = e.model_copy(update={"in_frame": new_in})
                    adjustments.append(
                        f"extend:{e.source_asset_id}:in {old_in}->{new_in}"
                    )
                    gap -= ext

            if gap > 0:
                return _abstain(
                    REASON_DURATION_UNREACHABLE,
                    f"已选镜头物理延长到极限后仍距目标下界差 {gap}us"
                    f"（target={target}us, current={current}us）；"
                    f"增删镜头属创作决策，须导演层重新出 plan",
                )

        elif current > high:
            # P2-d 边界余量（与延长路径对称）：截断目标取上界 -2%，
            # 避免渲染取整后偏差恰好压线 +10% 被 L1 判 FAIL（慢节奏实测）
            trunc_target = high - int(0.02 * target)
            gap = current - trunc_target

            # 优先缩短非启发式证据的最长片段（确定性技术优先级；P1-b 结构化判据）
            def _trunc_priority(i: int) -> tuple:
                e = edits[i]
                non_heuristic = 0 if not _is_heuristic_evidence(e) else 1
                return (non_heuristic, -(e.out_frame - e.in_frame))

            for idx in sorted(range(len(edits)), key=_trunc_priority):
                if gap <= 0:
                    break
                e = edits[idx]
                dur = e.out_frame - e.in_frame
                can_cut = dur - min_clip_us
                if can_cut <= 0:
                    continue
                cut = min(can_cut, gap)
                old_out = e.out_frame
                new_out = old_out - cut
                edits[idx] = e.model_copy(update={"out_frame": new_out})
                adjustments.append(
                    f"truncate:{e.source_asset_id}:out {old_out}->{new_out}"
                )
                gap -= cut

            if gap > 0:
                return _abstain(
                    REASON_DURATION_UNREACHABLE,
                    f"已选镜头物理缩短到极限后仍超目标上界 {gap}us"
                    f"（target={target}us, current={current}us）；"
                    f"删镜头属创作决策，须导演层重新出 plan",
                )

    # ---- 收尾：重算 expected_duration + source_asset_hashes ----
    expected_duration = sum(e.out_frame - e.in_frame for e in edits)
    source_hashes = ordered_unique_source_hashes(edits)

    new_edl.ordered_edits = edits
    new_edl.expected_duration = expected_duration
    new_edl.source_asset_hashes = source_hashes
    new_edl.producer = PRODUCER
    if not new_edl.edl_id.endswith("_repaired"):
        new_edl.edl_id = new_edl.edl_id + "_repaired"

    # 留痕：物理调整写入 plan.open_questions（append-only，可追溯）
    if adjustments:
        new_plan.open_questions = list(new_plan.open_questions) + [
            f"repair: {a}" for a in adjustments
        ]
    new_plan.producer = PRODUCER
    if not new_plan.plan_id.endswith("_repaired"):
        new_plan.plan_id = new_plan.plan_id + "_repaired"

    # ---- 重新验证（含 plan↔EDL 交叉校验；物理修复不改集合与顺序，应保持一致）----
    is_valid, errors = validate_plan(new_edl, new_plan, observations)
    if is_valid:
        new_plan.validation_status = "valid"
    else:
        new_plan.validation_status = "repaired_with_errors"
        new_plan.open_questions = list(new_plan.open_questions) + [
            f"repair: {err}" for err in errors
        ]

    return RepairOutcome(
        status="ok",
        edl=new_edl,
        plan=new_plan,
        adjustments=adjustments,
    )
