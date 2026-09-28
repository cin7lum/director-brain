"""启发式选片 Provider（T6 内化版）。

自遗留 GEN-1 的启发式基线模块内化而来：02 的链 A 不再依赖 GEN-1 遗留
包，与遗留适配器之间的双向 import 环就此断开。算法与常量与原实现逐位一致：

- 按 ``blur_score × VLM 权重`` 降序贪心填充至 ``target_duration_us``；
- 过短镜头跳过、单镜头截断到 ``MAX_CLIP_US``、片段窗口中点居中。

与旧实现的差异（行为等价性说明）：
- 旧 ``generate_edl`` 先产出 V0.1 proposal dict、写临时 JSON、再经遗留
  V0.1 导入器绕道导入；本版**直接由 slots 构造 EDL**，去掉临时文件与
  JSON 往返。EditItem 字段映射与旧导入器一致
  （shot_function=None、rationale=slot reason、timebase=1_000_000），
  EDL 包装字段保持旧值（``edl_from_<proposal_id>`` 等）；链 A 主消费方
  :mod:`director_brain.director_reasoner` 只消费 ``ordered_edits``。
"""
from __future__ import annotations

import time
import uuid

from director_brain.models.edl import EditItem, EditorialDecisionList

PRODUCER = "heuristic_baseline_v1"
TIMEBASE_US = 1_000_000

# ---------------------------------------------------------------------------
# 常量（与 GEN-1 heuristic_proposal.py 保持一致）
# ---------------------------------------------------------------------------

MIN_CLIP_US = 800_000    # 0.8s: 不生成短于此值的片段
MAX_CLIP_US = 6_000_000  # 6s: 单镜头片段最长不超过此值
LONG_SHOT_US = 5_000_000  # 静态镜头长于此值时降权
SHORT_TRANSITION_US = 600_000  # 转场镜头短于此值时惩罚
TARGET_LONG_US = 15_000_000  # atmospheric 加成仅对 target > 15s 生效

W_ATMOSPHERIC = 1.15
W_TRANSITION_SHORT = 0.5
W_STATIC_LONG = 0.85
W_HERO = 1.1
W_DISCARD = 0.2


# ---------------------------------------------------------------------------
# clip window（内联自 GEN-1 clip_window.py）
# ---------------------------------------------------------------------------

def compute_clip_window(
    shot_in_us: int,
    shot_out_us: int,
    clip_dur_us: int,
    align: str = "center",
) -> tuple[int, int]:
    """按对齐策略取片段窗口，边界钳制到 ``[shot_in_us, shot_out_us]``。

    align:
    - ``"center"``（历史缺省）：镜头中点居中裁剪；
    - ``"head"``（route-9 切点吸附）：窗口起点 = 源镜头起点——镜头起点即
      场景边界（discover_shots 分段依据），**切在场景边界上**是专业剪辑
      原则，切点自然度（scripts/cut_naturalness.py）由此提升。

    Guarantees: ``shot_in_us <= proposed_in_us < proposed_out_us <= shot_out_us``
    （调用方保证 ``clip_dur_us <= shot_out_us - shot_in_us``）。
    """
    if align == "head":
        proposed_in = shot_in_us
        proposed_out = shot_in_us + clip_dur_us
        if proposed_out > shot_out_us:
            proposed_out = shot_out_us
            proposed_in = proposed_out - clip_dur_us
        return proposed_in, proposed_out

    shot_mid = (shot_in_us + shot_out_us) // 2
    half = clip_dur_us // 2
    proposed_in = shot_mid - half
    proposed_out = shot_mid + half
    if proposed_in < shot_in_us:
        proposed_in = shot_in_us
        proposed_out = proposed_in + clip_dur_us
    if proposed_out > shot_out_us:
        proposed_out = shot_out_us
        proposed_in = proposed_out - clip_dur_us
    return proposed_in, proposed_out


# ---------------------------------------------------------------------------
# VLM 权重
# ---------------------------------------------------------------------------

def _vlm_multiplier(
    candidate: dict, target_duration_us: int
) -> tuple[float, list[str]]:
    """返回 ``(multiplier, 命中规则名列表)``；无 VLM 标签时返回 ``(1.0, [])``。"""
    fn = candidate.get("vlm_shot_function")
    motion = candidate.get("vlm_motion")
    role = candidate.get("vlm_role")
    if fn is None and motion is None and role is None:
        return 1.0, []

    shot_dur = candidate.get("duration_us", 0)
    mult = 1.0
    hits: list[str] = []

    if fn == "ATMOSPHERIC_EVIDENCE" and target_duration_us > TARGET_LONG_US:
        mult *= W_ATMOSPHERIC
        hits.append(f"atmospheric(x{W_ATMOSPHERIC})")
    if fn == "TRANSITION" and shot_dur < SHORT_TRANSITION_US:
        mult *= W_TRANSITION_SHORT
        hits.append(f"transition_short(x{W_TRANSITION_SHORT})")
    if motion == "static" and shot_dur > LONG_SHOT_US:
        mult *= W_STATIC_LONG
        hits.append(f"static_long(x{W_STATIC_LONG})")
    if role == "hero":
        mult *= W_HERO
        hits.append(f"hero(x{W_HERO})")
    if role == "discard":
        mult *= W_DISCARD
        hits.append(f"discard(x{W_DISCARD})")
    return mult, hits


# ---------------------------------------------------------------------------
# 生成器
# ---------------------------------------------------------------------------

class HeuristicBaseline:
    """按 blur 质量 × VLM 权重选镜头，贪心填充至目标时长。无 LLM。"""

    @staticmethod
    def generate(
        project_id: str,
        candidates: list[dict],
        target_duration_us: int,
        *,
        min_clip_us: int = MIN_CLIP_US,
        max_clip_us: int = MAX_CLIP_US,
        align: str = "center",
    ) -> dict:
        """生成 V0.1 格式的 proposal dict。

        Args:
            project_id: 项目 ID。
            candidates: 候选镜头 list[dict]，预期字段见模块 docstring。
            target_duration_us: 目标总时长（微秒）。
            min_clip_us/max_clip_us: 片段时长上下界（P1-b：由用户
                ``editing_language`` 意图映射注入，默认沿用模块常量）。

        Returns:
            V0.1 proposal dict，含 ``proposal_id``、``rough_cut_id``、
            ``project_id``、``slots``、``ai_model``、``prompt_version``。
        """
        usable = [c for c in candidates if c.get("technical_usable")]

        def _score(c: dict) -> float:
            blur = c.get("blur_score") or 0
            mult, _ = _vlm_multiplier(c, target_duration_us)
            return blur * mult

        usable.sort(key=lambda c: -_score(c))

        slots: list[dict] = []
        total = 0
        pos = 0
        for c in usable:
            if total >= target_duration_us:
                break
            shot_in = c["source_in_us"]
            shot_out = c["source_out_us"]
            shot_dur = shot_out - shot_in
            if shot_dur < min_clip_us:
                continue  # 跳过过短镜头

            remaining = target_duration_us - total
            if remaining < min_clip_us:
                break  # 剩余时间不足最小片段，停止

            clip_dur = min(shot_dur, max_clip_us, remaining)
            proposed_in, proposed_out = compute_clip_window(
                shot_in, shot_out, clip_dur, align=align
            )

            mult, hits = _vlm_multiplier(c, target_duration_us)
            score = (c.get("blur_score") or 0) * mult
            if hits:
                reason = (
                    f"vlm:fn={c.get('vlm_shot_function')},"
                    f"role={c.get('vlm_role')},"
                    f"blur={c.get('blur_score')},"
                    f"score={score:.1f},"
                    + ",".join(hits)
                )
                ctype = "SELF_REPORTED"
            else:
                reason = f"heuristic:blur={c.get('blur_score')}"
                ctype = "HEURISTIC"

            slots.append({
                "source_shot_id": c["source_shot_id"],
                "source_media_hash": c["source_media_hash"],
                "clip_instance_id": f"clip_{uuid.uuid4().hex[:12]}",
                "slot_id": f"slot_{pos + 1:02d}",
                "position": pos,
                "proposed_in_us": proposed_in,
                "proposed_out_us": proposed_out,
                "reason": reason,
                "confidence_type": ctype,
                "confidence_value": None,
            })
            total += clip_dur
            pos += 1

        return {
            "proposal_id": f"prop_{uuid.uuid4().hex[:12]}",
            "rough_cut_id": f"rc_{uuid.uuid4().hex[:12]}",
            "project_id": project_id,
            "slots": slots,
            "ai_model": "heuristic_v0.2",
            "prompt_version": "heuristic_v0.2",
        }


# ---------------------------------------------------------------------------
# EDL 直出（不再经临时 JSON + v01_importer 绕道）
# ---------------------------------------------------------------------------

def generate_edl(
    project_id: str,
    candidates: list[dict],
    target_duration_us: int,
    *,
    min_clip_us: int = MIN_CLIP_US,
    max_clip_us: int = MAX_CLIP_US,
    align: str = "center",
) -> EditorialDecisionList:
    """生成 EDL：产出 V0.1 proposal slots 后直接构造 EDL。

    注意：heuristic 的 slot 不含 ``proposed_role``，因此
    :attr:`EditItem.shot_function` 为 None，由调用方（director_reasoner）
    按幕覆盖——与旧 v01_importer 行为一致。``evidence_type`` 由 slot 的
    ``confidence_type`` 映射（SELF_REPORTED→vlm，HEURISTIC→heuristic），
    取代下游对 rationale 字符串的嗅探（P1-b）。
    """
    proposal = HeuristicBaseline.generate(
        project_id=project_id,
        candidates=candidates,
        target_duration_us=target_duration_us,
        min_clip_us=min_clip_us,
        max_clip_us=max_clip_us,
        align=align,
    )

    edits = [
        EditItem(
            source_asset_id=slot["source_shot_id"],
            source_media_hash=slot["source_media_hash"],
            in_frame=int(slot["proposed_in_us"]),
            out_frame=int(slot["proposed_out_us"]),
            timebase=TIMEBASE_US,
            shot_function=None,
            rationale=slot["reason"],
            evidence_type=("vlm" if slot["confidence_type"] == "SELF_REPORTED"
                           else "heuristic"),
        )
        for slot in proposal["slots"]
    ]

    now = int(time.time())
    return EditorialDecisionList(
        schema_version="1.0",
        project_id=project_id,
        created_at=now,
        producer=proposal["ai_model"],
        source_ref="internal:heuristic_baseline",
        edl_id=f"edl_from_{proposal['proposal_id']}",
        version="1.0",
        brief_version="unknown",
        context_id="NOT_DETERMINED",
        source_asset_hashes=[e.source_media_hash for e in edits],
        timebase=TIMEBASE_US,
        ordered_edits=edits,
        expected_duration=sum(e.out_frame - e.in_frame for e in edits),
        approval_state="draft",
    )
