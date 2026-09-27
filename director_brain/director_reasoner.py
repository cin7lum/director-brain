"""M2.2 Director Reasoner：从 Brief + 故事图 + 观测产出 EDL 与决策计划。

当前生产推理器为 :class:`HeuristicDirectorReasoner`（确定性算法：基于
blur_score × vlm_multiplier 排序 + 四幕选片）。
:class:`LLMDirectorReasoner` 为预留骨架，未实现，``generate_plan`` 抛
:class:`NotImplementedError`；通过 ``get_director_reasoner(strategy="llm")`
获取时会提前发出 warning。

时间统一微秒，timebase=1_000_000。

T2（静默降级显式化）：任何判据放宽/兜底/借用都写入
``plan.degradation_events`` 并置 ``plan.degraded=True``；候选池为空或
没有任何镜头通过最低技术判据（曝光合格）时，抛
:class:`EvidenceTooPoorError` 拒绝导演（fail-closed）；有锚点但不足时
响亮降级继续工作，confidence 按原始可用率计算并强制 requires_approval。
"""
from __future__ import annotations

import copy
import json
import time
import warnings
from abc import ABC, abstractmethod

from director_brain._utils import short_hash
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.models.story_graph import StoryGraph
from gen1_adapter.heuristic_baseline import generate_edl, MIN_CLIP_US

PRODUCER = "heuristic_director_reasoner_v0.1"
TIMEBASE_US = 1_000_000
#: primary technical_usable 阈值：exposure_ok 且 blur_score 高于此值。
_BLUR_USABLE_THRESHOLD = 10.0


class EvidenceTooPoorError(RuntimeError):
    """素材技术证据不足以支撑导演决策（T2 fail-closed）。

    触发条件（结构性证据缺失，拒绝导演）：
    - 素材未检出任何镜头（候选池为空）；
    - 没有任何镜头通过最低技术判据（曝光合格）——连一个证据锚点都没有。

    有锚点但不足 2 个时**不拒绝**：走响亮降级（degraded + 事件留痕 +
    原始可用率拉低 confidence + requires_approval），保证真实世界的不完美
    素材仍可工作，且用户看得见每一处放宽。
    """

#: 四幕顺序（用于 EDL 排序）与 shot_function 映射。
_ACT_ORDER: dict[str, int] = {"hook": 0, "develop": 1, "peak": 2, "resolve": 3}
_ACT_FUNCTION: dict[str, str] = {
    "hook": "opening",
    "develop": "pacing",
    "peak": "peak",
    "resolve": "closing",
}
#: 四幕时长比例（与 story_graph_builder._ACTS 一致）。
_ACT_RATIO: dict[str, float] = {
    "hook": 0.15,
    "develop": 0.35,
    "peak": 0.30,
    "resolve": 0.20,
}


def _parse_claim(claim: str) -> dict:
    try:
        data = json.loads(claim)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _num(data: dict, key: str) -> float | None:
    v = data.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _build_candidates(
    tech_obs: list[FilmObservation],
    vlm_obs: list[FilmObservation] | None = None,
    threshold: float = _BLUR_USABLE_THRESHOLD,
) -> list[dict]:
    """从 deterministic_technical 观测构建 HeuristicBaseline 候选 dict。

    technical_usable 主条件为 ``exposure_ok and blur_score > threshold``；过滤后可用
    候选 <2 时逐级放宽（去掉 blur 阈值 → 全部可用），保证至少有候选。

    若传入 ``vlm_obs``，从中筛选 ``claim_kind == MODEL_OBSERVATION`` 的 VLM
    语义观测，按 ``media_asset_id`` 建立 claim 映射，把
    ``shot_function / proposed_role_v2 / motion_amount`` 作为
    ``vlm_shot_function / vlm_role / vlm_motion`` 写入 candidate；无对应
    VLM 观测时这三个字段为 ``None``（``_vlm_multiplier`` 返回 1.0，行为不变）。
    """
    vlm_by_shot: dict[str, dict] = {}
    if vlm_obs:
        for o in vlm_obs:
            if getattr(o, "claim_kind", None) is not ClaimKind.MODEL_OBSERVATION:
                continue
            vlm_by_shot[o.media_asset_id] = _parse_claim(o.claim)

    candidates: list[dict] = []
    for o in tech_obs:
        data = _parse_claim(o.claim)
        blur = _num(data, "blur_score")
        exposure_ok = bool(data.get("exposure_ok", False))
        vlm_claim = vlm_by_shot.get(o.media_asset_id, {})
        candidates.append({
            "source_shot_id": o.media_asset_id,
            "source_media_hash": o.media_hash,
            "source_in_us": o.start_frame,
            "source_out_us": o.end_frame,
            "duration_us": o.end_frame - o.start_frame,
            "blur_score": blur if blur is not None else 0.0,
            "exposure_ok": exposure_ok,
            "_obs_id": o.observation_id,
            "vlm_shot_function": vlm_claim.get("shot_function"),
            "vlm_role": vlm_claim.get("proposed_role_v2"),
            "vlm_motion": vlm_claim.get("motion_amount"),
        })

    # T2：原始（未放宽）判据结果单独留档——confidence 用它计算，
    # 防止"注水后可用率变高、置信度反而上升"的历史假象。
    for c in candidates:
        c["technical_usable"] = c["exposure_ok"] and c["blur_score"] > threshold
        c["_primary_usable"] = c["technical_usable"]

    # T2：判据放宽必须可追溯。放宽级别写入每个候选的 _usability_relaxed
    # （0=未放宽，1=仅曝光，2=强制可用），由 generate_plan 汇总为降级事件。
    relaxation_level = 0
    if sum(1 for c in candidates if c["technical_usable"]) < 2:
        for c in candidates:
            c["technical_usable"] = c["exposure_ok"]
        relaxation_level = 1
    if sum(1 for c in candidates if c["technical_usable"]) < 2:
        for c in candidates:
            c["technical_usable"] = True
        relaxation_level = 2
    for c in candidates:
        c["_usability_relaxed"] = relaxation_level
    return candidates


class DirectorReasoner(ABC):
    """导演推理器抽象基类。"""

    @abstractmethod
    def generate_plan(
        self,
        brief: DirectorBrief,
        graph: StoryGraph,
        observations: list[FilmObservation],
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        """从简报/故事图/观测产出 (EDL, 决策计划)。"""
        raise NotImplementedError


class HeuristicDirectorReasoner(DirectorReasoner):
    """基于 HeuristicBaseline 的确定性导演推理器（无 LLM）。"""

    def __init__(self, blur_threshold: float = 10.0) -> None:
        self.blur_threshold = float(blur_threshold)

    def generate_plan(
        self,
        brief: DirectorBrief,
        graph: StoryGraph,
        observations: list[FilmObservation],
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        tech_obs = [
            o for o in observations if o.observation_type == "deterministic_technical"
        ]
        vlm_obs = [
            o for o in observations if o.observation_type == "vlm_semantic"
        ]
        candidates = _build_candidates(tech_obs, vlm_obs, self.blur_threshold)
        shot_to_obs = {o.media_asset_id: o.observation_id for o in tech_obs}

        # ---- T2 fail-closed：结构性证据缺失时拒绝导演 ----
        # 候选池为空，或没有任何镜头通过最低技术判据（曝光合格）——
        # 连一个证据锚点都没有，成片无据可依（历史行为是静默注水）。
        # 有锚点但不足 2 个时不拒绝：响亮降级（见 degradation_events）。
        if not candidates or not any(c["exposure_ok"] for c in candidates):
            exposure_ok_count = sum(1 for c in candidates if c["exposure_ok"])
            raise EvidenceTooPoorError(
                f"候选镜头 {len(candidates)} 个中仅 {exposure_ok_count} 个曝光合格"
                f"——没有任何证据锚点，技术证据不足以支撑选片，"
                f"拒绝导演（fail-closed）；请更换素材、放宽目标或人工介入"
            )

        # ---- T2：降级事件汇总（判据放宽/借用/兜底全部显式化）----
        degradation_events: list[str] = []
        global_relax = max(
            (c.get("_usability_relaxed", 0) for c in candidates), default=0
        )
        if global_relax > 0:
            degradation_events.append(
                f"relax_technical_usable:scope=global:level={global_relax}"
            )

        # ---- 从 graph.nodes 获取四幕 shot_ids，每幕单独选片 ----
        act_nodes = [
            n for n in graph.nodes
            if n.attributes.get("act") in _ACT_ORDER
        ]
        act_nodes.sort(key=lambda n: _ACT_ORDER[n.attributes["act"]])

        paired: list[tuple[str, EditItem, Decision]] = []
        borrowed_any = False
        fallback_any = False
        #: 已被选入 plan 的镜头 id（T1 修复：一份 plan 内同一镜头只选一次）
        selected_ids: set[str] = set()

        for act_node in act_nodes:
            act_name = act_node.attributes["act"]
            shot_ids = set(act_node.attributes.get("shot_ids", []))
            # T1 修复：同一镜头在一份 plan 中只选入一次。重复选入会让 EDL 出现
            # 同 asset 的嵌套区间（validator 判 overlap），历史上由 repair 越权
            # 删镜头"兜底"——按"Plan=导演依据"裁定，缺陷必须在导演层消除。
            # 注意：有意的镜头复用（reprise）未来需以"不重叠子区间"显式表达，
            # 当前 schema 不支持，先按保守规则排除。
            act_cands = [
                c for c in candidates
                if c["source_shot_id"] in shot_ids
                and c["source_shot_id"] not in selected_ids
            ]

            # 空幕兜底：本幕时间范围内无镜头时，从**未被选用**的全局候选借用。
            # VLM 判为 discard 的镜头视为导演层已否决，不参与借用（宁可空幕
            # 也不启用被否决的素材）；resolve 幕优先取时间最靠后的镜头，
            # hook 幕取最靠前的。
            borrowed = False
            if not act_cands and candidates:
                pool = [
                    c for c in candidates
                    if c["source_shot_id"] not in selected_ids
                    and c.get("vlm_role") != "discard"
                ]
                if pool:
                    sorted_cands = sorted(pool, key=lambda c: c["source_in_us"])
                    if act_name == "resolve":
                        act_cands = [copy.deepcopy(sorted_cands[-1])]
                    elif act_name == "hook":
                        act_cands = [copy.deepcopy(sorted_cands[0])]
                    else:
                        mid = len(sorted_cands) // 2
                        act_cands = [copy.deepcopy(sorted_cands[mid])]
                    borrowed = True
                    borrowed_any = True
                    degradation_events.append(
                        f"borrowed_shot:act={act_name}:"
                        f"shot={act_cands[0]['source_shot_id']}"
                    )
                # pool 为空：可用镜头已全被前幕选用（或仅剩 discard），本幕保持空

            per_act_target = max(
                int(_ACT_RATIO[act_name] * brief.target_duration),
                MIN_CLIP_US,
            )

            edl = generate_edl(
                project_id=brief.project_id,
                candidates=act_cands,
                target_duration_us=per_act_target,
            )
            act_edits: list[EditItem] = list(edl.ordered_edits)
            used_fallback = False

            # 时长不足兜底：本幕选中总时长 < 目标 50% 时，放宽 technical_usable
            # 重新选片（仅对本幕候选深拷贝，不影响其他幕）
            if act_edits:
                act_dur = sum(e.out_frame - e.in_frame for e in act_edits)
                if act_dur < per_act_target * 0.5 and not borrowed:
                    relaxed = [copy.deepcopy(c) for c in act_cands]
                    for c in relaxed:
                        c["technical_usable"] = True
                    edl_relaxed = generate_edl(
                        project_id=brief.project_id,
                        candidates=relaxed,
                        target_duration_us=per_act_target,
                    )
                    relaxed_edits = list(edl_relaxed.ordered_edits)
                    if relaxed_edits:
                        relaxed_dur = sum(e.out_frame - e.in_frame for e in relaxed_edits)
                        if relaxed_dur > act_dur:
                            act_edits = relaxed_edits
                            degradation_events.append(
                                "relax_technical_usable:scope=act:"
                                f"{act_name}:reason=act_duration_below_50pct"
                            )

            # 兜底：本幕选不出镜头时，直接取 blur_score 最高的完整镜头
            if not act_edits and act_cands:
                best = max(act_cands, key=lambda c: c["blur_score"])
                act_edits = [EditItem(
                    source_asset_id=best["source_shot_id"],
                    source_media_hash=best["source_media_hash"],
                    in_frame=int(best["source_in_us"]),
                    out_frame=int(best["source_out_us"]),
                    timebase=TIMEBASE_US,
                    shot_function=_ACT_FUNCTION.get(act_name),
                    rationale=f"heuristic:blur={best['blur_score']}",
                )]
                used_fallback = True
                fallback_any = True
                degradation_events.append(
                    f"fallback_selection:act={act_name}:shot={best['source_shot_id']}"
                )

            act_total = len(act_cands)
            # T2：confidence 按原始（未放宽）可用率计算——放宽入选的镜头
            # 会拉低置信度并触发 requires_approval，而不是注水抬高它。
            act_usable = sum(
                1 for c in act_cands
                if c.get("_primary_usable", c.get("technical_usable", False))
            )
            usable_ratio = act_usable / act_total if act_total else 0.0
            max_blur = max((c["blur_score"] for c in act_cands), default=1.0) or 1.0
            for idx, edit in enumerate(act_edits):
                edit.shot_function = _ACT_FUNCTION.get(act_name)
                reason = edit.rationale or ""
                edit.rationale = f"act={act_name}, {reason}"
                slot_label = "fallback" if used_fallback else f"slot_{idx + 1:02d}"

                selected_blur = next(
                    (c["blur_score"] for c in act_cands
                     if c["source_shot_id"] == edit.source_asset_id), 0.0)
                blur_norm = min(selected_blur / max_blur, 1.0) if max_blur > 0 else 0.0
                confidence = round(
                    usable_ratio * 0.4 + blur_norm * 0.4
                    + (0.0 if (used_fallback or borrowed) else 0.2), 2)
                confidence = max(0.1, min(confidence, 1.0))

                other_cands = [c for c in act_cands
                               if c["source_shot_id"] != edit.source_asset_id]
                other_cands.sort(key=lambda c: c["blur_score"], reverse=True)
                alternatives = [c["source_shot_id"] for c in other_cands[:2]]

                requires_approval = (confidence < 0.6 or used_fallback
                                     or borrowed or act_total < 2)

                decision = Decision(
                    decision_id=f"dec_{act_name}_{slot_label}",
                    purpose="select_shot",
                    shot_refs=[edit.source_asset_id],
                    evidence_refs=[shot_to_obs.get(edit.source_asset_id, "")],
                    rationale=edit.rationale,
                    alternatives=alternatives,
                    confidence=confidence,
                    requires_approval=requires_approval,
                )
                paired.append((act_name, edit, decision))
                selected_ids.add(edit.source_asset_id)

        # ---- 按幕顺序（hook→develop→peak→resolve），同幕内按 in_frame 升序 ----
        paired.sort(key=lambda p: (_ACT_ORDER.get(p[0], 99), p[1].in_frame))

        edits: list[EditItem] = [p[1] for p in paired]
        decisions: list[Decision] = [p[2] for p in paired]

        open_questions: list[str] = []
        if any(d.requires_approval for d in decisions):
            open_questions.append("some_decisions_require_approval")
        if borrowed_any:
            open_questions.append("borrowed_shots_from_other_acts")
        if fallback_any:
            open_questions.append("fallback_selection_used")
        if degradation_events:
            open_questions.append("degraded_selection_used")

        source_hashes: list[str] = []
        seen_hashes: set[str] = set()
        for e in edits:
            if e.source_media_hash not in seen_hashes:
                seen_hashes.add(e.source_media_hash)
                source_hashes.append(e.source_media_hash)

        expected_duration = sum(e.out_frame - e.in_frame for e in edits)

        edl = EditorialDecisionList(
            schema_version="1.0",
            project_id=brief.project_id,
            created_at=int(time.time()),
            producer=PRODUCER,
            source_ref=brief.source_ref,
            edl_id=f"edl_{brief.project_id}_{short_hash(brief.brief_id)}",
            version="0.1",
            brief_version=brief.version,
            context_id=graph.graph_id,
            source_asset_hashes=source_hashes,
            timebase=TIMEBASE_US,
            ordered_edits=edits,
            expected_duration=expected_duration,
            approval_state="draft",
        )

        plan = DirectorDecisionPlan(
            schema_version="1.0",
            project_id=brief.project_id,
            created_at=int(time.time()),
            producer=PRODUCER,
            source_ref=brief.source_ref,
            plan_id=f"plan_{brief.project_id}_{short_hash(brief.brief_id)}",
            version="0.1",
            brief_version=brief.version,
            film_state_version="0.1",
            sequence=[e.source_asset_id for e in edits],
            decisions=decisions,
            constraints=[f"target_duration_us={brief.target_duration}"],
            open_questions=open_questions,
            degraded=bool(degradation_events),
            degradation_events=degradation_events,
            validation_status="pending",
            approval_state="draft",
        )

        return edl, plan


class LLMDirectorReasoner(DirectorReasoner):
    """LLM 导演推理器未实现，当前环境使用 HeuristicDirectorReasoner 作为确定性推理器。

    本类为预留骨架，``generate_plan`` 抛 :class:`NotImplementedError`。
    """

    def __init__(self, provider: str = "ollama", config: dict | None = None):
        self.provider = provider
        self.config = config or {}

    def generate_plan(
        self,
        brief: DirectorBrief,
        graph: StoryGraph,
        observations: list[FilmObservation],
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        raise NotImplementedError(
            "LLM director reasoner requires ollama endpoint; "
            "current environment uses heuristic fallback"
        )


def get_director_reasoner(strategy: str = "heuristic", **kwargs) -> DirectorReasoner:
    """按策略名构造导演推理器。"""
    if strategy == "heuristic":
        return HeuristicDirectorReasoner()
    if strategy == "llm":
        warnings.warn(
            "LLM director reasoner is not implemented; returning a placeholder "
            "instance whose generate_plan() raises NotImplementedError. "
            "Use strategy='heuristic' for the deterministic reasoner.",
            stacklevel=2,
        )
        return LLMDirectorReasoner(**kwargs)
    raise ValueError(f"unknown director reasoner strategy: {strategy!r}")
