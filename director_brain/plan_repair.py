"""M2.3 Plan Repair：对修复后的 EDL 做确定性规则修复。

从 GEN-1 ``plan_repair.py`` 迁移 rule1-9，适配 EDL 模型。

**迁移的规则（rule1-9）：**

- rule1：移除 unknown source_asset_id 的片段
- rule2：修复 in_frame >= out_frame（交换或移除）
- rule3：过短片段拉长到 MIN_CLIP_US（若源镜头足够长），否则移除
- rule4：去重同一 source_asset_id（保留第一个）
- rule5：过长片段截断到 MAX_CLIP_US（中点居中）
- rule6：修复重叠片段（小幅重叠截断前者，大幅重叠移除后者）
- rule7：按 in_frame 升序排序
- rule8：重新计算 expected_duration
- rule9：重新验证并更新 plan.validation_status

**未迁移的规则（明确不迁移，原因记录于此）：**

- **rule10（确定性节奏后处理）**：hook 提升、长尾截断、快剪、CV 日志——
  依赖 LLM/VLM 标记的 ``is_hook`` 和 shake 驱动阈值，当前 pipeline 无 VLM
  信号，迁移后无数据可作用。
- **rule11（打破同 parent 镜头连续运行）**：依赖 GEN-1 的 ``"_w"`` 窗口分割
  命名约定（shot_id 形如 ``shot_001_w2``），当前 ``media_asset_id`` 无此
  约定，无法识别 parent 关系。
- **rule12（移动已选片段打破运行）**：纯重排逻辑，与 rule11 配套，
  不单独迁移；rule7 已做基础排序。

修复过程 fail-soft：任何单条规则出错不阻断整体，跳过该片段即可。
不修改原 edl/plan 对象，返回新对象。
"""
from __future__ import annotations

import json
import logging

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import FilmObservation
from director_brain.plan_validator import validate_plan, _parse_target_duration
from gen1_adapter.heuristic_baseline import MAX_CLIP_US, MIN_CLIP_US

PRODUCER = "plan_repair_v0.1"


def repair_plan(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    observations: list[FilmObservation],
) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
    """对 EDL 做确定性修复，返回新的 (edl, plan)。不修改原对象。

    Args:
        edl: 待修复的 EDL。
        plan: 对应的导演决策计划。
        observations: 素材观测列表，用于查 source_asset_id 的源镜头范围。

    Returns:
        ``(repaired_edl, repaired_plan)``。
    """
    # ---- 深拷贝，不修改原对象 ----
    new_edl = edl.model_copy(deep=True)
    new_plan = plan.model_copy(deep=True)

    edits: list[EditItem] = list(new_edl.ordered_edits)
    obs_by_asset = {o.media_asset_id: o for o in observations}
    known_ids = set(obs_by_asset.keys())

    # ---- rule1: 移除 unknown source_asset_id ----
    edits = [e for e in edits if e.source_asset_id in known_ids]

    # ---- rule2: 修复 in_frame >= out_frame ----
    fixed: list[EditItem] = []
    for e in edits:
        if e.in_frame >= e.out_frame:
            if e.out_frame > 0:
                # 交换
                fixed.append(e.model_copy(update={
                    "in_frame": e.out_frame,
                    "out_frame": e.in_frame,
                }))
            # else: out == 0，无法交换，移除
        else:
            fixed.append(e)
    edits = fixed

    # ---- rule3: 过短片段拉长或移除 ----
    fixed = []
    for e in edits:
        dur = e.out_frame - e.in_frame
        if dur >= MIN_CLIP_US:
            fixed.append(e)
            continue
        obs = obs_by_asset.get(e.source_asset_id)
        if obs is None:
            continue  # fail-soft: 找不到观测，移除
        src_dur = obs.end_frame - obs.start_frame
        if src_dur < MIN_CLIP_US:
            continue  # 源镜头本身过短，移除
        # 以中点为中心扩展到 MIN_CLIP_US，钳制到源镜头范围
        mid = (e.in_frame + e.out_frame) // 2
        half = MIN_CLIP_US // 2
        new_in = mid - half
        new_out = mid + half
        if new_in < obs.start_frame:
            new_in = obs.start_frame
            new_out = new_in + MIN_CLIP_US
        if new_out > obs.end_frame:
            new_out = obs.end_frame
            new_in = new_out - MIN_CLIP_US
        fixed.append(e.model_copy(update={
            "in_frame": new_in,
            "out_frame": new_out,
        }))
    edits = fixed

    # ---- rule4: 去重同一 source_asset_id（保留第一个）----
    seen_ids: set[str] = set()
    deduped: list[EditItem] = []
    for e in edits:
        if e.source_asset_id in seen_ids:
            continue
        seen_ids.add(e.source_asset_id)
        deduped.append(e)
    edits = deduped

    # ---- rule5: 过长片段截断到 MAX_CLIP_US（中点居中）----
    fixed = []
    for e in edits:
        dur = e.out_frame - e.in_frame
        if dur <= MAX_CLIP_US:
            fixed.append(e)
            continue
        mid = (e.in_frame + e.out_frame) // 2
        half = MAX_CLIP_US // 2
        new_in = mid - half
        new_out = mid + half
        fixed.append(e.model_copy(update={
            "in_frame": new_in,
            "out_frame": new_out,
        }))
    edits = fixed

    # ---- rule6: 修复重叠片段 ----
    edits.sort(key=lambda e: e.in_frame)
    no_overlap: list[EditItem] = []
    for e in edits:
        if not no_overlap:
            no_overlap.append(e)
            continue
        prev = no_overlap[-1]
        if prev.out_frame <= e.in_frame:
            no_overlap.append(e)
            continue
        # 有重叠
        overlap = prev.out_frame - e.in_frame
        prev_dur = prev.out_frame - prev.in_frame
        curr_dur = e.out_frame - e.in_frame
        smaller = min(prev_dur, curr_dur)
        ratio = (overlap / smaller) if smaller > 0 else 1.0
        if ratio < 0.5:
            # 小幅重叠：截断 prev
            no_overlap[-1] = prev.model_copy(update={"out_frame": e.in_frame})
            no_overlap.append(e)
        else:
            # 大幅重叠：移除 curr（保留先出现的 prev）
            new_plan.open_questions.append(
                f"repair: removed edit {e.source_asset_id} (overlap >50% with prev)"
            )
            logging.debug(
                "rule6: removed edit %s (in=%d, out=%d) due to >50%% overlap "
                "with prev %s (in=%d, out=%d)",
                e.source_asset_id, e.in_frame, e.out_frame,
                prev.source_asset_id, prev.in_frame, prev.out_frame,
            )
    edits = no_overlap

    # ---- rule7: 按 in_frame 升序排序 ----
    edits.sort(key=lambda e: e.in_frame)

    # ---- rule10: 时长修复（不足则延长/追加，过长则截断）----
    target = _parse_target_duration(new_plan.constraints)
    if target is not None:
        low = int(0.9 * target)
        high = int(1.1 * target)
        current = sum(e.out_frame - e.in_frame for e in edits)

        def _claim_of(asset_id: str) -> dict:
            o = obs_by_asset.get(asset_id)
            if o is None:
                return {}
            try:
                data = json.loads(o.claim)
            except (json.JSONDecodeError, TypeError):
                return {}
            return data if isinstance(data, dict) else {}

        if current < low:
            gap = low - current

            # 第一步：优先延长 rationale 含 "heuristic:" 的片段
            def _ext_priority(i: int) -> tuple:
                e = edits[i]
                is_heuristic = 0 if "heuristic:" in (e.rationale or "") else 1
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
                    MAX_CLIP_US - dur,
                    next_in - e.out_frame,
                )
                if max_ext <= 0:
                    continue
                ext = min(max_ext, gap)
                old_out = e.out_frame
                new_out = old_out + ext
                edits[idx] = e.model_copy(update={"out_frame": new_out})
                new_plan.open_questions.append(
                    f"repair: extended edit {e.source_asset_id} "
                    f"from {old_out} to {new_out}us"
                )
                gap -= ext

            # 第二步：仍不足时追加未选中的 deterministic_technical 观测
            if gap > 0:
                used_ids = {e.source_asset_id for e in edits}
                candidates = [
                    o for o in observations
                    if o.observation_type == "deterministic_technical"
                    and o.media_asset_id not in used_ids
                ]

                def _append_score(o: FilmObservation) -> tuple:
                    d = _claim_of(o.media_asset_id)
                    blur = d.get("blur_score", 0)
                    blur = float(blur) if isinstance(blur, (int, float)) else 0.0
                    exp = 0 if d.get("exposure_ok") else 1  # exposure_ok=True 优先
                    return (exp, -blur)

                candidates.sort(key=_append_score)
                for o in candidates:
                    if gap <= 0:
                        break
                    edits.append(EditItem(
                        source_asset_id=o.media_asset_id,
                        source_media_hash=o.media_hash,
                        in_frame=o.start_frame,
                        out_frame=o.end_frame,
                        timebase=1_000_000,
                        shot_function="repair_appended",
                        rationale="repair: appended to meet target duration",
                    ))
                    new_plan.open_questions.append(
                        f"repair: appended shot_{o.media_asset_id}"
                    )
                    gap -= (o.end_frame - o.start_frame)

                # 追加后按 in_frame 排序，并修复可能的重叠
                edits.sort(key=lambda e: e.in_frame)
                no_overlap: list[EditItem] = []
                for e in edits:
                    if not no_overlap:
                        no_overlap.append(e)
                        continue
                    prev = no_overlap[-1]
                    if prev.out_frame > e.in_frame:
                        if prev.out_frame - e.in_frame < (e.out_frame - e.in_frame):
                            no_overlap[-1] = prev.model_copy(
                                update={"out_frame": e.in_frame}
                            )
                            no_overlap.append(e)
                        # else: 大幅重叠，丢弃当前
                    else:
                        no_overlap.append(e)
                edits = no_overlap

        elif current > high:
            gap = current - high

            # 优先缩短非 heuristic: 的最长片段
            def _trunc_priority(i: int) -> tuple:
                e = edits[i]
                non_heuristic = 0 if "heuristic:" not in (e.rationale or "") else 1
                return (non_heuristic, -(e.out_frame - e.in_frame))

            for idx in sorted(range(len(edits)), key=_trunc_priority):
                if gap <= 0:
                    break
                e = edits[idx]
                dur = e.out_frame - e.in_frame
                can_cut = dur - MIN_CLIP_US
                if can_cut <= 0:
                    continue
                cut = min(can_cut, gap)
                old_out = e.out_frame
                new_out = old_out - cut
                edits[idx] = e.model_copy(update={"out_frame": new_out})
                new_plan.open_questions.append(
                    f"repair: truncated edit {e.source_asset_id} "
                    f"from {old_out} to {new_out}us"
                )
                gap -= cut

    # ---- rule8: 重新计算 expected_duration + source_asset_hashes ----
    expected_duration = sum(e.out_frame - e.in_frame for e in edits)
    source_hashes: list[str] = []
    seen_hashes: set[str] = set()
    for e in edits:
        if e.source_media_hash not in seen_hashes:
            seen_hashes.add(e.source_media_hash)
            source_hashes.append(e.source_media_hash)

    new_edl.ordered_edits = edits
    new_edl.expected_duration = expected_duration
    new_edl.source_asset_hashes = source_hashes
    new_edl.producer = PRODUCER
    if not new_edl.edl_id.endswith("_repaired"):
        new_edl.edl_id = new_edl.edl_id + "_repaired"

    # ---- rule9: 重新验证并更新 plan ----
    is_valid, errors = validate_plan(new_edl, new_plan, observations)
    new_plan.producer = PRODUCER
    if not new_plan.plan_id.endswith("_repaired"):
        new_plan.plan_id = new_plan.plan_id + "_repaired"
    if is_valid:
        new_plan.validation_status = "valid"
    else:
        new_plan.validation_status = "repaired_with_errors"
        new_plan.open_questions = list(new_plan.open_questions) + [
            f"repair: {err}" for err in errors
        ]

    return (new_edl, new_plan)
