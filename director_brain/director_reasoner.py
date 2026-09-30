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
import logging
import time
import warnings
from abc import ABC, abstractmethod

from director_brain.acts import ACT_FUNCTION, ACT_ORDER, ACT_RATIO
from director_brain.utils import short_hash
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.models.story_graph import StoryGraph
from director_brain.pathway_protocol import ensure_decision_use_allowed
from director_brain.providers.heuristic import MAX_CLIP_US, MIN_CLIP_US, generate_edl
from director_brain.intent_constraints import (
    TechnicalAvoidRule,
    candidate_violated_rules,
    editing_language_bounds,
    encode_bounds,
    interpret_constraints,
)

logger = logging.getLogger(__name__)

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

#: 四幕顺序 / shot_function / 时长配额：单一事实源见 :mod:`director_brain.acts`
#: （架构体检候选⑥收编——此前本模块自带一份，靠注释与 story_graph 对齐）。


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

    T4 通路闸门：``vlm_obs`` 非空即意味着 VLM 信号将影响选片排序（决策），
    必须 ``vlm_semantic`` 通路处于 ACTIVE，否则抛
    :class:`PathwayNotActiveError`（fail-closed；影子期的 VLM 信号只记录不驱动）。

    若传入 ``vlm_obs``，从中筛选 ``claim_kind == MODEL_OBSERVATION`` 的 VLM
    语义观测，按 ``media_asset_id`` 建立 claim 映射，把
    ``shot_function / proposed_role_v2 / motion_amount`` 作为
    ``vlm_shot_function / vlm_role / vlm_motion`` 写入 candidate；无对应
    VLM 观测时这三个字段为 ``None``（``_vlm_multiplier`` 返回 1.0，行为不变）。
    """
    if vlm_obs:
        ensure_decision_use_allowed("vlm_semantic")

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
            # P1-a：完整 claim 指标透传（意图约束解释器按需读取
            # brightness_mean / shake_score 等）
            "_claim_metrics": data,
            "vlm_shot_function": vlm_claim.get("shot_function"),
            "vlm_role": vlm_claim.get("proposed_role_v2"),
            "vlm_motion": vlm_claim.get("motion_amount"),
            # S4：TVSum 同构 importance（1-5；None=未标注）
            "vlm_importance": vlm_claim.get("importance"),
        })

    # T2：原始（未放宽）判据结果单独留档——confidence 用它计算，
    # 防止"注水后可用率变高、置信度反而上升"的历史假象。
    # P1-b：无数据观测（claim 带 error，如读帧失败）永不可用——
    # "没有数据"不等于"质量最差"，放宽阶梯不得将其注水入选。
    for c in candidates:
        metrics = c.get("_claim_metrics") or {}
        c["_no_data"] = "error" in metrics
        # P0：暗镜头（多点采样黑占比 >= 50%）永久排除——碎内容不属低质，
        # 与 no_data 同等处理，放宽阶梯也不得注水入选
        c["_dark_shot"] = (
            isinstance(metrics.get("dark_sample_ratio"), (int, float))
            and metrics["dark_sample_ratio"] >= 0.5
        )
        c["technical_usable"] = (
            c["exposure_ok"] and c["blur_score"] > threshold
            and not c["_no_data"] and not c["_dark_shot"]
        )
        c["_primary_usable"] = c["technical_usable"]

    # T2：判据放宽必须可追溯。放宽级别写入每个候选的 _usability_relaxed
    # （0=未放宽，1=仅曝光，2=强制可用），由 generate_plan 汇总为降级事件。
    relaxation_level = 0
    if sum(1 for c in candidates if c["technical_usable"]) < 2:
        for c in candidates:
            if not c["_no_data"] and not c["_dark_shot"]:
                c["technical_usable"] = c["exposure_ok"]
        relaxation_level = 1
    if sum(1 for c in candidates if c["technical_usable"]) < 2:
        for c in candidates:
            if not c["_no_data"] and not c["_dark_shot"]:
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

        # ---- P1-a：意图约束解释与过滤（must_avoid 技术子集）----
        # 用户约束不再是一张单程票：技术词表内的 must_avoid 生成排除规则
        # （选片过滤 + 编码进 plan.constraints 供 validator 判红）；语义类
        # 约束诚实标注"当前证据无法验证"，不静默假装执行。
        directives = interpret_constraints(brief)
        constraint_questions: list[str] = []
        applied_rules: list[TechnicalAvoidRule] = []
        if directives.avoid_rules:
            applied_rules = [rule for _, rule in directives.avoid_rules]
            kept: list[dict] = []
            for c in candidates:
                violated = candidate_violated_rules(c, applied_rules)
                if violated:
                    for rule in violated:
                        constraint_questions.append(
                            f"constraint: must_avoid「{rule.term}」排除镜头 "
                            f"{c['source_shot_id']}"
                            f"（{rule.metric} {rule.predicate} {rule.threshold}）"
                        )
                else:
                    kept.append(c)
            if not kept:
                raise EvidenceTooPoorError(
                    f"全部 {len(candidates)} 个候选镜头均违反 must_avoid 约束"
                    f"（{[r.term for r in applied_rules]}）——素材无法满足用户"
                    f"约束，拒绝导演（fail-closed）；请更换素材或调整意图"
                )
            candidates = kept
        for kind, text in directives.unverifiable:
            constraint_questions.append(
                f"constraint_unverifiable: {kind}「{text}」"
                f"（当前技术观测无法验证语义内容，待语义证据通路）"
            )

        # ---- P1-b：剪辑语言意图 → 片段时长上下界（"快剪/慢剪"不再一个味）----
        clip_bounds = editing_language_bounds(brief, (MIN_CLIP_US, MAX_CLIP_US))
        min_clip_us, max_clip_us = clip_bounds

        # ---- 从 graph.nodes 获取四幕 shot_ids，每幕单独选片 ----
        act_nodes = [
            n for n in graph.nodes
            if n.attributes.get("act") in ACT_ORDER
        ]
        act_nodes.sort(key=lambda n: ACT_ORDER[n.attributes["act"]])

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
            # VLM 判为 discard 的镜头视为导演层已否决，不参与借用；无数据
            # 观测（读帧失败）同样不得借用于兜底（宁可空幕）。resolve 幕
            # 优先取时间最靠后的镜头，hook 幕取最靠前的。
            borrowed = False
            if not act_cands and candidates:
                pool = [
                    c for c in candidates
                    if c["source_shot_id"] not in selected_ids
                    and c.get("vlm_role") != "discard"
                    and not c.get("_no_data")
                    and not c.get("_dark_shot")
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
                # pool 为空：可用镜头已全被前幕选用（或仅剩 discard/无数据），本幕保持空

            per_act_target = max(
                int(ACT_RATIO[act_name] * brief.target_duration),
                MIN_CLIP_US,
            )

            # route-9 导演层 policy：切点吸附场景边界（head 对齐）——
            # 片段起点 = 源镜头起点（discover_shots 分段依据 = 场景边界）
            edl = generate_edl(
                project_id=brief.project_id,
                candidates=act_cands,
                target_duration_us=per_act_target,
                min_clip_us=min_clip_us,
                max_clip_us=max_clip_us,
                align="head",
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
                        min_clip_us=min_clip_us,
                        max_clip_us=max_clip_us,
                        align="head",
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
                # 兜底选片同样排除无数据观测（读帧失败 ≠ 质量最差）
                usable_pool = [c for c in act_cands if not c.get("_no_data")]
                if usable_pool:
                    best = max(usable_pool, key=lambda c: c["blur_score"])
                    act_edits = [EditItem(
                        source_asset_id=best["source_shot_id"],
                        source_media_hash=best["source_media_hash"],
                        in_frame=int(best["source_in_us"]),
                        out_frame=int(best["source_out_us"]),
                        timebase=TIMEBASE_US,
                        shot_function=ACT_FUNCTION.get(act_name),
                        rationale=f"heuristic:blur={best['blur_score']}",
                        evidence_type="heuristic",
                    )]
                    used_fallback = True
                    fallback_any = True
                    degradation_events.append(
                        f"fallback_selection:act={act_name}:shot={best['source_shot_id']}"
                    )
                # usable_pool 为空：本幕只剩无数据观测，保持空幕

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
                edit.shot_function = ACT_FUNCTION.get(act_name)
                edit.act = act_name
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
        paired.sort(key=lambda p: (ACT_ORDER.get(p[0], 99), p[1].in_frame))

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
        open_questions.extend(constraint_questions)

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
            constraints=[f"target_duration_us={brief.target_duration}"]
            + [rule.encode() for rule in applied_rules]
            + encode_bounds(clip_bounds),
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


#: 第三方推理器注册表（P2-c 插拔点）：strategy 名 → 工厂 callable。
#: 内置 heuristic/llm 走硬编码分支以保持向后兼容；外部实现经
#: :func:`register_reasoner` 插入，无需改动本模块。
_REASONER_REGISTRY: dict[str, callable] = {}


def register_reasoner(strategy: str, factory) -> None:
    """注册自定义导演推理器工厂（插拔点）。

    factory 签名与内置一致：``factory(**kwargs) -> DirectorReasoner``。
    重复注册同名校盖旧实现；注册不校验基类（鸭子类型，调用方负责）。
    """
    _REASONER_REGISTRY[strategy] = factory
    logger.info("reasoner registered: %s", strategy)


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
    factory = _REASONER_REGISTRY.get(strategy)
    if factory is None:
        raise ValueError(f"unknown director reasoner strategy: {strategy!r}")
    return factory(**kwargs)
