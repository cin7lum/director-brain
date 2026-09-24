"""M2.2 Director Reasoner：从 Brief + 故事图 + 观测产出 EDL 与决策计划。

- :class:`HeuristicDirectorReasoner`：复用
  :class:`~gen1_adapter.heuristic_baseline.HeuristicBaseline` 选镜头，转成
  :class:`EditorialDecisionList` 与 :class:`DirectorDecisionPlan`。
- :class:`LLMDirectorReasoner`：预留 LLM（ollama）接口骨架，当前环境不可用，
  ``generate_plan`` 直接抛 :class:`NotImplementedError`。

时间统一微秒，timebase=1_000_000。
"""
from __future__ import annotations

import json
import time
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
) -> list[dict]:
    """从 deterministic_technical 观测构建 HeuristicBaseline 候选 dict。

    technical_usable 主条件为 ``exposure_ok and blur_score > 10``；过滤后可用
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

    for c in candidates:
        c["technical_usable"] = c["exposure_ok"] and c["blur_score"] > _BLUR_USABLE_THRESHOLD

    if sum(1 for c in candidates if c["technical_usable"]) < 2:
        for c in candidates:
            c["technical_usable"] = c["exposure_ok"]
    if sum(1 for c in candidates if c["technical_usable"]) < 2:
        for c in candidates:
            c["technical_usable"] = True
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
        candidates = _build_candidates(tech_obs, vlm_obs)
        shot_to_obs = {o.media_asset_id: o.observation_id for o in tech_obs}

        # ---- 从 graph.nodes 获取四幕 shot_ids，每幕单独选片 ----
        act_nodes = [
            n for n in graph.nodes
            if n.attributes.get("act") in _ACT_ORDER
        ]
        act_nodes.sort(key=lambda n: _ACT_ORDER[n.attributes["act"]])

        paired: list[tuple[str, EditItem, Decision]] = []

        for act_node in act_nodes:
            act_name = act_node.attributes["act"]
            shot_ids = set(act_node.attributes.get("shot_ids", []))
            act_cands = [c for c in candidates if c["source_shot_id"] in shot_ids]

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

            for idx, edit in enumerate(act_edits):
                edit.shot_function = _ACT_FUNCTION.get(act_name)
                reason = edit.rationale or ""
                edit.rationale = f"act={act_name}, {reason}"
                slot_label = "fallback" if used_fallback else f"slot_{idx + 1:02d}"
                decision = Decision(
                    decision_id=f"dec_{act_name}_{slot_label}",
                    purpose="select_shot",
                    shot_refs=[edit.source_asset_id],
                    evidence_refs=[shot_to_obs.get(edit.source_asset_id, "")],
                    rationale=edit.rationale,
                    alternatives=[],
                    confidence=0.5,
                    requires_approval=False,
                )
                paired.append((act_name, edit, decision))

        # ---- 按幕顺序（hook→develop→peak→resolve），同幕内按 in_frame 升序 ----
        paired.sort(key=lambda p: (_ACT_ORDER.get(p[0], 99), p[1].in_frame))

        edits: list[EditItem] = [p[1] for p in paired]
        decisions: list[Decision] = [p[2] for p in paired]

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
            open_questions=[],
            validation_status="pending",
            approval_state="draft",
        )

        return edl, plan


class LLMDirectorReasoner(DirectorReasoner):
    """LLM（ollama）导演推理器预留骨架；当前环境不可用。"""

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
        return LLMDirectorReasoner(**kwargs)
    raise ValueError(f"unknown director reasoner strategy: {strategy!r}")
