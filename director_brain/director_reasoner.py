"""M2.2 Director Reasoner：从 Brief + 故事图 + 观测产出 EDL 与决策计划。

当前默认推理器为 :class:`HeuristicDirectorReasoner`（确定性算法：基于
blur_score × vlm_multiplier 排序 + 四幕选片）。:class:`LLMDirectorReasoner` 可在
本地 Ollama 上生成叙事候选，但独立通路 ``director_strategy_reasoning`` 仍为
SHADOW；正式入口在通路准入前 fail-closed，多素材比较入口输出不可确认的
Plan/EDL。当前项目级 Reasoner 已支持多素材分层推理，但真实项目质量、跨项目
泛化与产品质量仍 NOT_PROVEN。

时间统一微秒，timebase=1_000_000。

T2（静默降级显式化）：任何判据放宽/兜底/借用都写入
``plan.degradation_events`` 并置 ``plan.degraded=True``；候选池为空或
没有任何镜头通过最低技术判据（曝光合格）时，抛
:class:`EvidenceTooPoorError` 拒绝导演（fail-closed）；有锚点但不足时
响亮降级继续工作，confidence 按原始可用率计算并强制 requires_approval。
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import time
import warnings
from abc import ABC, abstractmethod
from urllib.parse import urlparse

from director_brain.acts import ACT_FUNCTION, ACT_ORDER, ACT_RATIO, ROLE_TO_ACT
from director_brain.utils import short_hash
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import (
    Decision,
    DirectorDecisionPlan,
    ProjectNarrativeCallProvenance,
    ProjectDirectorSourceSelectionAuditEntry,
    ProjectDirectorShadowComparison,
    ProjectDirectorStrategyCandidate,
    derive_candidate_edl_source_asset_alignment,
    ProjectNarrativeActBoundary,
    ProjectNarrativeConstraintAssessment,
    ProjectNarrativeEvidenceClaim,
    ProjectNarrativeEvidenceRef,
    ProjectNarrativeReasoningTrace,
    ProjectNarrativeShotHypothesis,
    ProjectNarrativeSourceRationale,
    ProjectNarrativeStrategyHypothesis,
)
from director_brain.models.edl import (
    EditItem,
    EditorialDecisionList,
    TransitionSpec,
    ordered_unique_source_hashes,
)
from director_brain.models.film_observation import (
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)
from director_brain.models.film_context import AssetAnalysisState, FilmContextSnapshot
from director_brain.models.project import FilmProjectManifest, RightsState
from director_brain.models.project_story_graph import ProjectStoryGraph
from director_brain.models.project_story_link_review import ProjectStoryLinkReview
from director_brain.models.project_story_mention_review import ProjectStoryMentionReview
from director_brain.models.story_graph import StoryGraph, StoryNode, StoryNodeType
from director_brain.pathway_protocol import ensure_decision_use_allowed
from director_brain.providers.heuristic import (
    MAX_CLIP_US,
    MIN_CLIP_US,
    generate_edl,
    vlm_multiplier,
)
from director_brain.semantic_scorer import SemanticScore, compute_semantic_score
from director_brain.models.shot_card import ShotCard
from director_brain.llm_adapter import LLMStructuredOutputError
from director_brain.intent_constraints import (
    EDITING_LANGUAGE_PROFILE_IDS,
    TechnicalAvoidRule,
    candidate_violated_rules,
    editing_language_bounds,
    encode_bounds,
    interpret_constraints,
    unresolved_constraint_review_items,
)
from director_brain.input_sanitizer import sanitize_untrusted

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


def _candidate_identity(
    project_asset_id: str | None, source_media_hash: str, source_asset_id: str,
) -> tuple[str | None, str, str]:
    """Stable selection identity; a shot ID alone is not project-unique."""
    return project_asset_id, source_media_hash, source_asset_id


def _project_candidate_ref(
    project_asset_id: str, source_media_hash: str, source_asset_id: str,
) -> str:
    """Lossless narrative key for a shot whose ID is only asset-local."""
    return json.dumps(
        [project_asset_id, source_media_hash.lower(), source_asset_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _candidate_narrative_ref(candidate: dict) -> str:
    if candidate.get("_project_asset_id") is None:
        return candidate["source_shot_id"]
    return _project_candidate_ref(
        candidate["_project_asset_id"],
        candidate["source_media_hash"],
        candidate["source_shot_id"],
    )


def _edit_identity(edit: EditItem) -> tuple[str | None, str, str]:
    return _candidate_identity(
        edit.project_asset_id, edit.source_media_hash, edit.source_asset_id)


def _edl_execution_signature(edl: EditorialDecisionList) -> tuple:
    """Describe the materialized EDL behavior, excluding lineage-only metadata.

    Source identity order is deliberately kept as a separate comparison: this
    signature also captures trims, transitions, audio offsets, effects and
    output tracks so equal shot order does not hide a different edit.
    """
    def enum_value(value):
        return value.value if hasattr(value, "value") else value

    edit_signatures = []
    for edit in edl.ordered_edits:
        transition = edit.transition
        transition_signature = None if transition is None else (
            transition.type,
            transition.name,
            transition.duration_us,
            transition.audio_duration_us,
        )
        source_identity = (
            edit.project_asset_id,
            edit.source_media_hash.lower(),
            edit.source_asset_id,
        )
        edit_signatures.append((
            source_identity,
            edit.in_frame,
            edit.out_frame,
            edit.timebase,
            enum_value(edit.timebase_unit),
            transition_signature,
            edit.audio_lead_us,
            edit.audio_tail_us,
            tuple(edit.effect_refs),
        ))

    return (
        edl.timebase,
        enum_value(edl.timebase_unit),
        tuple(value.lower() for value in edl.source_asset_hashes),
        edl.expected_duration,
        tuple(edit_signatures),
        tuple(edl.audio_refs),
        tuple(edl.overlay_refs),
        tuple(edl.subtitle_refs),
        tuple(edl.artistic_choices),
    )


# Bump this value whenever persisted-candidate acceptance semantics change.
_PROJECT_STRATEGY_CANDIDATE_BINDING_VERSION = (
    "project-director-candidate-binding-v11"
)


def compute_project_strategy_candidate_binding_digest(
    brief: DirectorBrief,
    manifest: FilmProjectManifest,
    context: FilmContextSnapshot,
    project_graph: ProjectStoryGraph,
    hypothesis: ProjectNarrativeStrategyHypothesis,
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    *,
    project_story_link_review: ProjectStoryLinkReview | None = None,
    project_story_mention_review: ProjectStoryMentionReview | None = None,
) -> str:
    """Bind an unranked candidate to its complete, current review scope.

    Generated artifact IDs and timestamps are intentionally excluded. The
    digest covers exact project/review lineage, stable Brief/Plan/EDL state,
    narrative evidence and claims, plan decisions, reasoner provenance, and
    the materialized EDL behavior. Generated artifact IDs and timestamps are
    excluded so equivalent re-generation can be matched.
    """
    trace = plan.project_narrative_reasoning
    if trace is None:
        raise ValueError("project strategy candidate lacks its narrative trace")

    def review_scope(review):
        return None if review is None else review.model_dump(mode="json")

    payload = {
        "binding_version": _PROJECT_STRATEGY_CANDIDATE_BINDING_VERSION,
        "scope": {
            "project_id": manifest.project_id,
            "manifest_id": manifest.manifest_id,
            "manifest_revision": manifest.revision,
            "context_id": context.context_id,
            "story_graph_id": project_graph.graph_id,
            "brief_id": brief.brief_id,
            # created_at changes when API requests recompile the same brief;
            # bind every stable brief field so same-ID content mutations stale
            # the candidate without making normal regeneration impossible.
            "brief_snapshot": brief.model_dump(
                mode="json", exclude={"created_at"}),
            "manifest_snapshot": manifest.model_dump(mode="json"),
            "context_snapshot": context.model_dump(mode="json"),
            "story_graph_snapshot": project_graph.model_dump(mode="json"),
            "story_link_review": review_scope(project_story_link_review),
            "story_mention_review": review_scope(project_story_mention_review),
        },
        "hypothesis": hypothesis.model_dump(mode="json"),
        "reasoning_trace": trace.model_dump(mode="json"),
        "plan_snapshot": plan.model_dump(
            mode="json",
            exclude={
                "created_at", "plan_id", "edl_id", "decisions",
                "project_narrative_reasoning",
            },
        ),
        "plan_decisions": [
            item.model_dump(mode="json", exclude={"decision_id"})
            for item in plan.decisions
        ],
        "edl_snapshot": edl.model_dump(
            mode="json", exclude={"created_at", "edl_id"}),
        "reasoner_provenance": {
            "producer": plan.producer,
            "constraints": sorted(
                item for item in plan.constraints
                if item.startswith("director_reasoner_")
            ),
        },
        "edl_execution_signature": _edl_execution_signature(edl),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_project_strategy_candidate_binding_digest = (
    compute_project_strategy_candidate_binding_digest
)


def _apply_asr_audio_bridge(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    observations: list[FilmObservation],
    audio_style: str,
    *,
    evidence_prefix: str = "audio_bridge",
    question_prefix: str = "audio_bridge",
    validate_plan_result: bool = False,
) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
    """Apply J/L offsets only from same-source ASR spans crossing picture cuts.

    Source coordinates must be canonical microseconds. Project asset identity,
    media hash, project ID, and claim provenance must match. Missing or
    out-of-scope evidence produces zero offset; the historical fixed 400 ms
    convention is never used as a fallback.
    """
    if audio_style not in {"j_cut", "l_cut"}:
        return edl, plan

    speech_observations = [
        item for item in observations
        if item.observation_type == "speech_transcript"
    ]

    adjacent_indexes = (
        range(1, len(edl.ordered_edits))
        if audio_style == "j_cut"
        else range(0, max(0, len(edl.ordered_edits) - 1))
    )
    applied = 0
    unavailable = 0
    decision_use_checked = False
    for index in adjacent_indexes:
        edit = edl.ordered_edits[index]
        boundary = edit.in_frame if audio_style == "j_cut" else edit.out_frame
        matches: list[tuple[int, FilmObservation]] = []
        for observation in speech_observations:
            if (
                observation.project_id != plan.project_id
                or observation.project_asset_id != edit.project_asset_id
                or observation.media_hash.lower() != edit.source_media_hash.lower()
                or observation.timebase != TIMEBASE_US
                or observation.timebase_unit != TimebaseUnit.MICROSECONDS
                or observation.claim_kind not in {
                    ClaimKind.MODEL_OBSERVATION,
                    ClaimKind.HUMAN_CONFIRMED,
                }
                or not observation.start_frame < boundary < observation.end_frame
            ):
                continue
            offset = (
                boundary - observation.start_frame
                if audio_style == "j_cut"
                else observation.end_frame - boundary
            )
            if offset > 0:
                matches.append((offset, observation))

        if not matches:
            unavailable += 1
            continue

        if not decision_use_checked:
            ensure_decision_use_allowed("asr_transcript")
            decision_use_checked = True

        offset, evidence = min(
            matches, key=lambda item: (item[0], item[1].observation_id))
        if audio_style == "j_cut":
            edit.audio_lead_us = offset
        else:
            edit.audio_tail_us = offset
        edit.audio_evidence_refs = list(dict.fromkeys([
            *edit.audio_evidence_refs, evidence.observation_id,
        ]))
        edl.audio_evidence_refs = list(dict.fromkeys([
            *edl.audio_evidence_refs, evidence.observation_id,
        ]))

        matching_decisions = [
            item for item in plan.decisions
            if item.project_asset_id == edit.project_asset_id
            and item.shot_refs == [edit.source_asset_id]
        ]
        if len(matching_decisions) != 1:
            raise ValueError(
                "ASR audio bridge cannot bind to one exact Plan decision")
        decision = matching_decisions[0]
        decision.evidence_refs = list(dict.fromkeys([
            *decision.evidence_refs, evidence.observation_id,
        ]))
        decision.requires_approval = True
        policy_rationale = (
            f"{evidence_prefix}={audio_style};offset_us={offset};"
            "source=ASR_span_crossing_picture_boundary"
        )
        edit.rationale = "; ".join(
            value for value in (edit.rationale, policy_rationale) if value)
        decision.rationale = edit.rationale
        applied += 1

    plan.constraints.extend([
        f"{evidence_prefix}_policy=asr_crossing_span_v1",
        f"{evidence_prefix}_style={audio_style}",
        f"{evidence_prefix}_fallback=fixed_offset_disabled",
    ])
    timing_question = (
        "shadow_project_audio_bridge_asr_timing_unverified"
        if question_prefix == "project_shadow_audio_bridge"
        else f"{question_prefix}_asr_timing_unverified"
    )
    plan.open_questions.append(timing_question)
    if unavailable:
        plan.open_questions.append(
            f"{question_prefix}_no_supported_boundary={audio_style}:"
            f"count={unavailable}"
        )
    if applied:
        edl.artistic_choices = list(dict.fromkeys([
            *edl.artistic_choices,
            f"{evidence_prefix}={audio_style}:asr_crossing_span",
        ]))
    plan.constraints = list(dict.fromkeys(plan.constraints))
    plan.open_questions = list(dict.fromkeys(plan.open_questions))

    edl = EditorialDecisionList.model_validate(edl.model_dump(mode="python"))
    plan = DirectorDecisionPlan.model_validate(plan.model_dump(mode="python"))
    if validate_plan_result:
        from director_brain.plan_validator import validate_plan

        is_valid, errors = validate_plan(edl, plan, observations)
        if not is_valid:
            raise ValueError(
                "project shadow audio bridge invalidated its Plan: "
                + "; ".join(errors)
            )
    return edl, plan


def _apply_project_shadow_audio_bridge(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    observations: list[FilmObservation],
    audio_style: str,
) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
    """Apply the source-bound bridge policy to a non-confirmable SHADOW Plan."""
    return _apply_asr_audio_bridge(
        edl,
        plan,
        observations,
        audio_style,
        evidence_prefix="project_shadow_audio_bridge",
        question_prefix="shadow_project_audio_bridge",
        validate_plan_result=True,
    )


def _build_project_narrative_reasoning_trace(
    narrative: dict,
    *,
    brief_id: str,
    model: str,
    temperature: float,
    prompt_version: str,
    semantic_constraints: list[dict] | None = None,
    candidate_edl_source_assets: set[tuple[str, str, str]] | None = None,
) -> ProjectNarrativeReasoningTrace:
    candidate_refs = narrative.get("project_candidate_refs")
    source_map = narrative.get("_source_evidence_by_candidate_ref")
    raw_call_provenance = narrative.get("provider_call_provenance")
    if (
        not isinstance(candidate_refs, list)
        or not candidate_refs
        or any(not isinstance(ref, str) or not ref for ref in candidate_refs)
        or len(candidate_refs) != len(set(candidate_refs))
        or not isinstance(source_map, dict)
        or set(source_map) != set(candidate_refs)
        or not isinstance(raw_call_provenance, list)
        or not raw_call_provenance
    ):
        raise ValueError(
            "project narrative trace lacks exact source and provider-call provenance")
    call_provenance = [
        ProjectNarrativeCallProvenance.model_validate(item)
        for item in raw_call_provenance
    ]

    def source_ref(candidate_ref: str) -> ProjectNarrativeEvidenceRef:
        raw = source_map.get(candidate_ref)
        if not isinstance(raw, dict):
            raise ValueError("project narrative statement cites an unknown source")
        return ProjectNarrativeEvidenceRef.model_validate(raw)

    def safe_text(value: object, field: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"project narrative {field} must be text")
        return sanitize_untrusted(value) or "[removed by output sanitizer]"

    def evidence_claims(values: object, field: str) -> list[ProjectNarrativeEvidenceClaim]:
        if not isinstance(values, list) or not values:
            raise ValueError(f"project strategy {field} must contain cited claims")
        claims: list[ProjectNarrativeEvidenceClaim] = []
        for claim in values:
            if (not isinstance(claim, dict)
                    or set(claim) != {"statement", "source_refs"}):
                raise ValueError(f"project strategy {field} claim is invalid")
            refs = claim["source_refs"]
            if (not isinstance(refs, list) or not refs
                    or any(not isinstance(ref, str) or ref not in source_map
                           for ref in refs)
                    or len(refs) != len(set(refs))):
                raise ValueError(f"project strategy {field} evidence refs are invalid")
            claims.append(ProjectNarrativeEvidenceClaim(
                statement=safe_text(claim["statement"], field),
                source_refs=[source_ref(ref) for ref in refs],
            ))
        return claims

    def source_rationales(
        values: object,
        ordered_source_refs: list[str],
    ) -> list[ProjectNarrativeSourceRationale]:
        if not isinstance(values, list) or len(values) != len(ordered_source_refs):
            raise ValueError(
                "project strategy requires one source rationale per ordered source")
        rationales: list[ProjectNarrativeSourceRationale] = []
        for item in values:
            if (not isinstance(item, dict)
                    or set(item) != {
                        "focus_source_ref", "disposition", "statement", "source_refs",
                    }):
                raise ValueError("project strategy source rationale is invalid")
            focus_ref = item["focus_source_ref"]
            disposition = item["disposition"]
            refs = item["source_refs"]
            if (not isinstance(focus_ref, str)
                    or focus_ref not in source_map
                    or focus_ref not in ordered_source_refs
                    or disposition not in {"include", "exclude"}
                    or not isinstance(refs, list)
                    or not refs
                    or any(not isinstance(ref, str) or ref not in source_map
                           for ref in refs)
                    or len(refs) != len(set(refs))
                    or focus_ref not in refs):
                raise ValueError(
                    "project strategy source rationale evidence refs are invalid")
            rationales.append(ProjectNarrativeSourceRationale(
                focus_source=source_ref(focus_ref),
                disposition=disposition,
                statement=safe_text(item["statement"], "source rationale"),
                source_refs=[source_ref(ref) for ref in refs],
            ))
        return rationales

    emotions = narrative.get("emotional_trajectory_resolved")
    if not isinstance(emotions, list) or len(emotions) != len(candidate_refs):
        raise ValueError("project narrative trace lacks mapped emotional hypotheses")
    emotional_hypotheses = []
    for item in emotions:
        if (not isinstance(item, dict)
                or not isinstance(item.get("shot_ref"), str)
                or item["shot_ref"] not in source_map):
            raise ValueError("project narrative emotion has no exact source reference")
        emotional_hypotheses.append(ProjectNarrativeShotHypothesis(
            source=source_ref(item["shot_ref"]),
            label=safe_text(item.get("label"), "emotional label"),
        ))

    raw_key_moments = narrative.get("key_moments_resolved")
    if not isinstance(raw_key_moments, list):
        raise ValueError("project narrative key moments must be a list")
    key_moments = []
    for item in raw_key_moments:
        if (not isinstance(item, dict)
                or not isinstance(item.get("shot_ref"), str)
                or item["shot_ref"] not in source_map):
            raise ValueError("project narrative key moment has no exact source reference")
        key_moments.append(ProjectNarrativeShotHypothesis(
            source=source_ref(item["shot_ref"]),
            label="key_moment",
            rationale=safe_text(item.get("why"), "key moment rationale"),
        ))

    raw_boundaries = narrative.get("act_boundaries_resolved")
    if not isinstance(raw_boundaries, list):
        raise ValueError("project narrative act boundaries must be a list")
    act_boundaries = []
    for item in raw_boundaries:
        if (not isinstance(item, dict)
                or not isinstance(item.get("shot_ids"), list)
                or not item["shot_ids"]
                or any(not isinstance(ref, str) or ref not in source_map
                       for ref in item["shot_ids"])):
            raise ValueError("project narrative act boundary lacks exact source references")
        act_boundaries.append(ProjectNarrativeActBoundary(
            act=item["act"],
            source_refs=[source_ref(ref) for ref in item["shot_ids"]],
        ))

    suggested_order = narrative.get("suggested_order_resolved")
    if (not isinstance(suggested_order, list)
            or len(suggested_order) != len(candidate_refs)
            or any(not isinstance(ref, str) for ref in suggested_order)
            or set(suggested_order) != set(candidate_refs)):
        raise ValueError("project narrative order lacks a complete exact source mapping")
    raw_strategy_hypotheses = narrative.get("strategy_hypotheses_resolved")
    if (not isinstance(raw_strategy_hypotheses, list)
            or len(raw_strategy_hypotheses) < 2):
        raise ValueError("project narrative trace requires multiple strategy hypotheses")
    strategy_hypotheses: list[ProjectNarrativeStrategyHypothesis] = []
    strategy_ids: set[str] = set()
    for item in raw_strategy_hypotheses:
        if not isinstance(item, dict):
            raise ValueError("project strategy hypothesis must be an object")
        hypothesis_id = item.get("hypothesis_id")
        if (not isinstance(hypothesis_id, str) or not hypothesis_id
                or hypothesis_id in strategy_ids):
            raise ValueError("project strategy hypothesis ID is missing or duplicated")
        strategy_ids.add(hypothesis_id)
        option_order = item.get("suggested_order_resolved")
        if (not isinstance(option_order, list)
                or len(option_order) != len(candidate_refs)
                or any(not isinstance(ref, str) for ref in option_order)
                or set(option_order) != set(candidate_refs)):
            raise ValueError("project strategy order lacks a complete source mapping")
        option_boundaries = item.get("act_boundaries_resolved")
        if not isinstance(option_boundaries, list):
            raise ValueError("project strategy act boundaries must be a list")
        typed_boundaries = []
        for boundary in option_boundaries:
            if (not isinstance(boundary, dict)
                    or not isinstance(boundary.get("shot_ids"), list)
                    or not boundary["shot_ids"]
                    or any(not isinstance(ref, str) or ref not in source_map
                           for ref in boundary["shot_ids"])):
                raise ValueError("project strategy act boundary lacks exact source refs")
            typed_boundaries.append(ProjectNarrativeActBoundary(
                act=boundary.get("act"),
                source_refs=[source_ref(ref) for ref in boundary["shot_ids"]],
            ))
        tradeoffs = evidence_claims(item.get("tradeoffs"), "tradeoffs")
        uncertainties = evidence_claims(item.get("uncertainties"), "uncertainties")
        emotional_arc = evidence_claims(
            [item.get("emotional_arc")], "emotional_arc")[0]
        raw_constraint_assessments = item.get("constraint_assessments", [])
        if not isinstance(raw_constraint_assessments, list):
            raise ValueError("project strategy constraint assessments must be a list")
        if semantic_constraints is not None:
            expected_by_ref = {
                constraint["constraint_ref"]: constraint
                for constraint in semantic_constraints
            }
            if len(raw_constraint_assessments) != len(expected_by_ref):
                raise ValueError(
                    "project strategy does not review every open semantic constraint")
        else:
            expected_by_ref = {}
            if raw_constraint_assessments:
                raise ValueError(
                    "project strategy reports an unrequested semantic constraint")
        typed_constraint_assessments = []
        for assessment in raw_constraint_assessments:
            if (
                not isinstance(assessment, dict)
                or set(assessment) != {
                    "constraint_ref", "constraint_kind", "brief_index",
                    "constraint_text", "assessment", "statement", "source_refs",
                }
                or not isinstance(assessment.get("source_refs"), list)
                or any(
                    not isinstance(ref, str) or ref not in source_map
                    for ref in assessment["source_refs"]
                )
            ):
                raise ValueError(
                    "project strategy constraint assessment lacks exact source refs")
            constraint_ref = assessment["constraint_ref"]
            if not isinstance(constraint_ref, str):
                raise ValueError(
                    "project strategy constraint assessment reference is invalid")
            expected = expected_by_ref.get(constraint_ref)
            if (
                expected is None
                or assessment["constraint_kind"] != expected["kind"]
                or assessment["brief_index"] != expected["brief_index"]
                or assessment["constraint_text"] != expected["text"]
            ):
                raise ValueError(
                    "project strategy constraint assessment differs from the Brief")
            typed_constraint_assessments.append(
                ProjectNarrativeConstraintAssessment(
                    constraint_kind=assessment["constraint_kind"],
                    brief_index=assessment["brief_index"],
                    constraint_text=assessment["constraint_text"],
                    assessment=assessment["assessment"],
                    statement=safe_text(
                        assessment.get("statement"),
                        "constraint assessment statement"),
                    source_refs=[
                        source_ref(ref) for ref in assessment["source_refs"]
                    ],
                )
            )
        editing_language_choice = item.get("editing_language_choice")
        editing_language_rationale = None
        if editing_language_choice is not None:
            editing_language_rationale = evidence_claims(
                [item.get("editing_language_rationale")],
                "editing_language_rationale")[0]
        audio_style_choice = item.get("audio_style_choice")
        audio_style_rationale = None
        if audio_style_choice is not None:
            audio_style_rationale = evidence_claims(
                [item.get("audio_style_rationale")],
                "audio_style_rationale")[0]
        transition_policy_choice = item.get("transition_policy_choice")
        transition_policy_rationale = None
        if transition_policy_choice is not None:
            transition_policy_rationale = evidence_claims(
                [item.get("transition_policy_rationale")],
                "transition_policy_rationale")[0]
        typed_source_rationales = source_rationales(
            item.get("source_rationales_resolved"), option_order)
        strategy_hypotheses.append(ProjectNarrativeStrategyHypothesis(
            hypothesis_id=hypothesis_id,
            label=safe_text(item.get("label"), "strategy label"),
            editorial_intent=safe_text(
                item.get("editorial_intent"), "strategy editorial intent"),
            emotional_arc=emotional_arc,
            editing_language_choice=editing_language_choice,
            editing_language_rationale=editing_language_rationale,
            audio_style_choice=audio_style_choice,
            audio_style_rationale=audio_style_rationale,
            transition_policy_choice=transition_policy_choice,
            transition_policy_rationale=transition_policy_rationale,
            transition_duration_us=(
                item.get("transition_duration_us") or None),
            ordered_sources=[source_ref(ref) for ref in option_order],
            source_rationales=typed_source_rationales,
            act_boundaries=typed_boundaries,
            tradeoffs=tradeoffs,
            uncertainties=uncertainties,
            constraint_assessments=typed_constraint_assessments,
        ))
    active_hypothesis_id = narrative.get("_active_strategy_hypothesis_id")
    if (active_hypothesis_id is not None
            and active_hypothesis_id not in strategy_ids):
        raise ValueError("active project strategy hypothesis is not in the trace")
    if candidate_edl_source_assets is not None:
        if active_hypothesis_id is None:
            raise ValueError(
                "candidate EDL source alignment requires an active strategy hypothesis")
        strategy_hypotheses = [
            strategy.model_copy(update={
                "constraint_assessments": [
                    assessment.model_copy(update={
                        "candidate_edl_source_asset_alignment": (
                            derive_candidate_edl_source_asset_alignment(
                                assessment.source_refs,
                                candidate_edl_source_assets,
                            )
                        ),
                    })
                    for assessment in strategy.constraint_assessments
                ],
            })
            if strategy.hypothesis_id == active_hypothesis_id
            else strategy
            for strategy in strategy_hypotheses
        ]
    limitations = narrative.get("limitations")
    if not isinstance(limitations, list):
        raise ValueError("project narrative limitations must be a list")
    limitations = [safe_text(item, "limitation") for item in limitations]
    hierarchy = narrative.get("analysis_hierarchy")
    if hierarchy is not None:
        if (not isinstance(hierarchy, dict)
                or hierarchy.get("mode") != "bounded_segment_then_project_synthesis"
                or type(hierarchy.get("provider_call_count")) is not int
                or hierarchy["provider_call_count"] < 1):
            raise ValueError("project narrative hierarchy trace is invalid")
        limitations.append(
            "Bounded hierarchical project reasoning used "
            f"{hierarchy['provider_call_count']} local provider call(s); "
            "intermediate summaries remain model-derived and unverified."
        )
    hint_count = narrative.get(
        "caller_asserted_link_hints_supplied",
        narrative.get("caller_asserted_link_hints_used", 0),
    )
    if type(hint_count) is not int or hint_count < 0:
        raise ValueError("project narrative caller link hint count is invalid")
    supplied_hint_ids = narrative.get(
        "caller_asserted_link_hint_ids_supplied", [])
    omitted_hint_ids = narrative.get(
        "caller_asserted_link_hint_ids_omitted", [])
    review_link_count = narrative.get(
        "caller_asserted_link_review_link_count", 0)
    link_provenance_state = narrative.get(
        "caller_asserted_link_provenance_state", "legacy_not_captured")
    limitations.extend([
        "Source references establish input lineage only; they do not verify narrative claims.",
        "Cross-asset identity and causal relationships were not established.",
    ])
    if hint_count:
        limitations.append(
            "Caller-asserted identity hints were supplied as unverified context; "
            "model use or acceptance was not measured."
        )

    return ProjectNarrativeReasoningTrace(
        brief_id=brief_id,
        model=model,
        prompt_version=prompt_version,
        temperature=temperature,
        provider_call_provenance=call_provenance,
        provider_call_provenance_state="captured",
        story_arc=safe_text(narrative.get("story_arc"), "story arc"),
        input_evidence_refs=[source_ref(ref) for ref in candidate_refs],
        emotional_hypotheses=emotional_hypotheses,
        key_moments=key_moments,
        act_boundaries=act_boundaries,
        suggested_order=[source_ref(ref) for ref in suggested_order],
        strategy_hypotheses=strategy_hypotheses,
        active_strategy_hypothesis_id=active_hypothesis_id,
        limitations=list(dict.fromkeys(limitations)),
        caller_asserted_link_hints_supplied=hint_count,
        caller_asserted_link_hint_ids_supplied=supplied_hint_ids,
        caller_asserted_link_hint_ids_omitted=omitted_hint_ids,
        caller_asserted_link_review_link_count=review_link_count,
        caller_asserted_link_provenance_state=link_provenance_state,
    )


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
    ``shot_function / shot_scale / proposed_role_v2 / motion_amount / importance`` 与
    P3-1 深度语义（``narrative_role / emotional_tone / action_type /
    scene_description``）写入 candidate；无对应 VLM 观测时这些字段为
    ``None``（按无语义处理，行为与技术路径一致）。
    """
    if vlm_obs:
        ensure_decision_use_allowed("vlm_semantic")

    vlm_by_shot: dict[tuple[str | None, str, str], dict] = {}
    if vlm_obs:
        for o in vlm_obs:
            if getattr(o, "claim_kind", None) is not ClaimKind.MODEL_OBSERVATION:
                continue
            vlm_by_shot[_candidate_identity(
                o.project_asset_id, o.media_hash, o.media_asset_id
            )] = _parse_claim(o.claim)

    candidates: list[dict] = []
    for o in tech_obs:
        data = _parse_claim(o.claim)
        blur = _num(data, "blur_score")
        exposure_ok = bool(data.get("exposure_ok", False))
        identity = _candidate_identity(
            o.project_asset_id, o.media_hash, o.media_asset_id)
        vlm_claim = vlm_by_shot.get(identity, {})
        candidates.append({
            "source_shot_id": o.media_asset_id,
            "source_media_hash": o.media_hash,
            "source_in_us": o.start_frame,
            "source_out_us": o.end_frame,
            "duration_us": o.end_frame - o.start_frame,
            "blur_score": blur if blur is not None else 0.0,
            "exposure_ok": exposure_ok,
            "_obs_id": o.observation_id,
            "_project_asset_id": o.project_asset_id,
            "_candidate_identity": identity,
            "_source_timebase": o.timebase,
            "_source_timebase_unit": o.timebase_unit,
            # P1-a：完整 claim 指标透传（意图约束解释器按需读取
            # brightness_mean / shake_score 等）
            "_claim_metrics": data,
            "vlm_shot_function": vlm_claim.get("shot_function"),
            "vlm_role": vlm_claim.get("proposed_role_v2"),
            "vlm_motion": vlm_claim.get("motion_amount"),
            # S4：TVSum 同构 importance（1-5；None=未标注）——来源是 VLM
            # 语义观测 claim（修正：此前 generate._score 误读技术观测 claim，
            # 该处永无 importance，S4 权重在生产内核实为死代码）
            "vlm_importance": vlm_claim.get("importance"),
            # P3-1 深度语义（内核语义融合消费，须过通路闸门）
            "vlm_narrative_role": vlm_claim.get("narrative_role"),
            "vlm_emotional_tone": vlm_claim.get("emotional_tone"),
            "vlm_action_type": vlm_claim.get("action_type"),
            "vlm_scene_description": vlm_claim.get("scene_description", ""),
            # D2：实体归属（entities 传入时由内核注入 _entity_ids）
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


def _bind_edit_provenance(edit: EditItem, candidates: list[dict]) -> EditItem:
    """Attach one exact observation and its project/source clock to a selected edit."""
    matches = [
        candidate for candidate in candidates
        if candidate["source_shot_id"] == edit.source_asset_id
        and candidate["source_media_hash"] == edit.source_media_hash
        and candidate["source_in_us"] <= edit.in_frame
        and candidate["source_out_us"] >= edit.out_frame
    ]
    if len(matches) != 1:
        raise ValueError(
            "selected edit must map to exactly one source observation; "
            f"found {len(matches)} for {edit.source_asset_id}"
        )
    candidate = matches[0]
    provenance = {
        "timebase_unit": TimebaseUnit.MICROSECONDS,
        "project_asset_id": candidate["_project_asset_id"],
        "source_observation_refs": [candidate["_obs_id"]],
    }
    if candidate["_project_asset_id"] is not None:
        provenance.update({
            "source_observation_start": candidate["source_in_us"],
            "source_observation_end": candidate["source_out_us"],
            "source_timebase": candidate["_source_timebase"],
            "source_timebase_unit": candidate["_source_timebase_unit"],
        })
    return edit.model_copy(update=provenance)


def _project_aggregate_story_graph(
    manifest: FilmProjectManifest,
    project_graph: ProjectStoryGraph,
    observations: list[FilmObservation],
) -> tuple[StoryGraph, dict[str, int]]:
    """Create a planner-only ACT view; source graphs and clocks stay separate."""
    if project_graph.project_id != manifest.project_id:
        raise ValueError("project graph and manifest project_id values must match")
    if (project_graph.project_manifest_id != manifest.manifest_id
            or project_graph.project_revision != manifest.revision):
        raise ValueError("project StoryGraph does not match the supplied manifest revision")
    if project_graph.boundary_state != "declared_unverified":
        raise ValueError("project boundary must remain declared_unverified")
    if project_graph.timeline_scope != "project_per_asset":
        raise ValueError("project StoryGraph must preserve per-asset timelines")
    if project_graph.cross_asset_relations_state != "not_attempted":
        raise ValueError("cross-asset relations must remain not_attempted")

    ordered_assets = sorted(manifest.assets, key=lambda item: item.order)
    if [item.asset_id for item in project_graph.assets] != [
        item.asset_id for item in ordered_assets
    ]:
        raise ValueError("project StoryGraph assets do not match manifest order")
    asset_order = {asset.asset_id: asset.order for asset in ordered_assets}
    observations_by_id: dict[str, FilmObservation] = {}
    for observation in observations:
        if observation.observation_id in observations_by_id:
            raise ValueError("duplicate observation record supplied")
        observations_by_id[observation.observation_id] = observation
    if set(observations_by_id) != set(project_graph.evidence_refs):
        raise ValueError("observations must exactly match project StoryGraph evidence refs")

    # Temporal claims used by the existing reasoner must already declare
    # canonical microseconds per asset. No conversion or shared clock is inferred.
    temporal_types = {
        "deterministic_technical", "vlm_semantic", "speech_transcript",
    }
    for observation in observations:
        if observation.project_id != manifest.project_id:
            raise ValueError("project observation belongs to another project")
        asset = next(
            (item for item in ordered_assets
             if item.asset_id == observation.project_asset_id),
            None,
        )
        if asset is None:
            raise ValueError("project observation asset is absent from manifest")
        if (asset.source_content_hash is None
                or observation.media_hash.lower() != asset.source_content_hash.lower()):
            raise ValueError("project observation hash does not match manifest asset")
        if observation.observation_type in temporal_types and (
            observation.timebase != TIMEBASE_US
            or observation.timebase_unit != TimebaseUnit.MICROSECONDS
        ):
            raise ValueError(
                "project DirectorReasoner requires canonical microsecond evidence "
                "for each source asset"
            )

    observations_by_asset: dict[str, list[FilmObservation]] = {
        asset.asset_id: [] for asset in ordered_assets
    }
    for observation in observations:
        assert observation.project_asset_id is not None
        observations_by_asset[observation.project_asset_id].append(observation)

    candidate_refs_by_act: dict[str, list[dict[str, str]]] = {
        act: [] for act in ACT_ORDER
    }
    for asset, asset_graph in zip(ordered_assets, project_graph.assets, strict=True):
        if asset_graph.order != asset.order:
            raise ValueError("project StoryGraph asset order differs from manifest")
        if asset_graph.source_content_hash != (
            asset.source_content_hash.lower()
            if asset.source_content_hash else None
        ):
            raise ValueError("project StoryGraph asset hash differs from manifest")
        if set(asset_graph.evidence_refs) != {
            item.observation_id for item in observations_by_asset[asset.asset_id]
        }:
            raise ValueError("asset StoryGraph evidence refs differ from context observations")
        if asset_graph.story_graph_state != "constructed":
            if asset_graph.story_graph is not None:
                raise ValueError("non-constructed asset graph cannot supply candidates")
            continue
        if asset.rights.state is not RightsState.LOCAL_PROCESSING_ALLOWED:
            raise ValueError("candidate graph includes an asset without local processing authority")
        source_graph = asset_graph.story_graph
        if source_graph is None or source_graph.project_asset_id != asset.asset_id:
            raise ValueError("constructed asset graph has invalid project asset scope")
        if (source_graph.timebase != TIMEBASE_US
                or source_graph.timebase_unit != TimebaseUnit.MICROSECONDS):
            raise ValueError("asset StoryGraph must use its canonical microsecond source clock")

        technical = [
            item for item in observations_by_asset[asset.asset_id]
            if item.observation_type == "deterministic_technical"
        ]
        from director_brain.story_graph_builder import build_asset_story_graph
        expected_source_graph = build_asset_story_graph(
            manifest.project_id,
            asset.asset_id,
            observations_by_asset[asset.asset_id],
        )
        if (source_graph.model_dump(exclude={"created_at"})
                != expected_source_graph.model_dump(exclude={"created_at"})):
            raise ValueError(
                "asset StoryGraph differs from the structure derived from exact observations")
        technical_by_shot: dict[str, FilmObservation] = {}
        for observation in technical:
            if observation.media_asset_id in technical_by_shot:
                raise ValueError(
                    "one asset cannot bind a shot id to multiple technical observations")
            technical_by_shot[observation.media_asset_id] = observation

        seen_shots: set[str] = set()
        for node in source_graph.nodes:
            act_name = node.attributes.get("act")
            if act_name not in ACT_ORDER:
                continue
            for shot_id in node.attributes.get("shot_ids", []):
                if shot_id in seen_shots:
                    raise ValueError("asset StoryGraph assigns one shot to multiple acts")
                observation = technical_by_shot.get(shot_id)
                if observation is None:
                    raise ValueError("asset StoryGraph shot has no exact technical observation")
                seen_shots.add(shot_id)
                candidate_refs_by_act[act_name].append({
                    "project_asset_id": asset.asset_id,
                    "source_media_hash": observation.media_hash.lower(),
                    "source_asset_id": shot_id,
                })
        if seen_shots != set(technical_by_shot):
            raise ValueError("asset StoryGraph does not assign every technical observation")

    nodes = [
        StoryNode(
            node_id=f"project_act_{act_name}",
            node_type=StoryNodeType.ACT,
            ref_id=f"act_{act_name}",
            attributes={
                "act": act_name,
                # The project view contains identity tuples, never a joined
                # timeline or cross-asset temporal edge.
                "shot_ids": [],
                "candidate_refs": candidate_refs_by_act[act_name],
            },
        )
        for act_name in ACT_ORDER
    ]
    aggregate = StoryGraph(
        schema_version="1.1",
        project_id=manifest.project_id,
        created_at=project_graph.created_at,
        producer="project_director_reasoner_aggregate",
        source_ref=manifest.manifest_id,
        graph_id=project_graph.graph_id,
        version="0.1",
        timeline_scope="project_aggregate",
        timebase=None,
        timebase_unit=None,
        nodes=nodes,
        edges=[],
    )
    return aggregate, asset_order


def _project_target_topup_pool(
    candidates: list[dict],
    selected_ids: set[tuple[str | None, str, str]],
    *,
    min_clip_us: int,
    narrative_order_priority: bool,
) -> list[dict]:
    """Return viable, ordered project candidates for duration top-up.

    Filter sub-minimum source intervals before ranking; otherwise the highest
    ranked short interval can stop top-up while a later usable shot remains.
    """
    pool = [
        candidate for candidate in candidates
        if candidate["_candidate_identity"] not in selected_ids
        and not candidate.get("_no_data")
        and not candidate.get("_dark_shot")
        and candidate.get("vlm_role") != "discard"
        and type(candidate.get("duration_us")) is int
        and candidate["duration_us"] >= min_clip_us
    ]
    if narrative_order_priority:
        pool.sort(key=lambda candidate: (
            candidate.get("_director_strategy_rank", 10 ** 9),
            -candidate.get("selection_score", candidate.get("blur_score", 0)),
        ))
    else:
        pool.sort(
            key=lambda candidate: candidate.get(
                "selection_score", candidate.get("blur_score", 0)),
            reverse=True,
        )
    return pool


class DirectorReasoner(ABC):
    """导演推理器抽象基类。"""

    @abstractmethod
    def generate_plan(
        self,
        brief: DirectorBrief,
        graph: StoryGraph,
        observations: list[FilmObservation],
        *,
        narrative: dict | None = None,
        transition_policy: str = "none",
        transition_duration_us: int | None = None,
        card: ShotCard | None = None,
        entities=None,
        voice_led: bool = False,
        beat_grid=None,
        audio_style: str = "none",
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        """从简报/故事图/观测产出 (EDL, 决策计划)。

        narrative: 跨镜头叙事理解（narrative_analyzer 产物，P3-3）。传入时
        act_boundaries_resolved 驱动幕分配、suggested_order_resolved 驱动
        幕内排序——属于 vlm_semantic 通路的决策消费，调用方须已确保通路
        ACTIVE（消费侧闸门在 _build_candidates 统一执法）。
        """
        raise NotImplementedError


class HeuristicDirectorReasoner(DirectorReasoner):
    """基于 HeuristicBaseline 的确定性导演推理器（无 LLM）。

    候选①内核融合：VLM 语义观测（vlm_semantic 通路，闸门内）驱动
    三层语义选片——narrative_role/叙事幕边界决定幕分配（替代纯时间
    比例）、emotional_tone/action_type/importance/内容多样性参与评分、
    叙事推荐顺序决定幕内排序。无语义观测时逐位回退纯技术路径。
    """

    def __init__(self, blur_threshold: float = 10.0) -> None:
        self.blur_threshold = float(blur_threshold)

    def generate_project_plan(
        self,
        brief: DirectorBrief,
        manifest: FilmProjectManifest,
        context: FilmContextSnapshot,
        project_graph: ProjectStoryGraph,
        observations: list[FilmObservation],
        *,
        transition_policy: str = "none",
        transition_duration_us: int | None = None,
        card: ShotCard | None = None,
        voice_led: bool = False,
        audio_style: str = "none",
        pacing_style: str = "brief",
        project_story_link_review: ProjectStoryLinkReview | None = None,
        project_story_mention_review: ProjectStoryMentionReview | None = None,
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        """Generate the heuristic project plan used by the normal product path."""
        if audio_style not in {"none", "j_cut", "l_cut"}:
            raise ValueError(
                "strategy-selected audio requires the LLM strategy comparison path")
        if pacing_style != "brief":
            raise ValueError(
                "strategy-selected pacing requires the LLM strategy comparison path")
        return self._generate_project_plan(
            brief,
            manifest,
            context,
            project_graph,
            observations,
            transition_policy=transition_policy,
            transition_duration_us=transition_duration_us,
            card=card,
            voice_led=voice_led,
            audio_style=audio_style,
            project_story_link_review=project_story_link_review,
            project_story_mention_review=project_story_mention_review,
        )

    def _generate_project_plan(
        self,
        brief: DirectorBrief,
        manifest: FilmProjectManifest,
        context: FilmContextSnapshot,
        project_graph: ProjectStoryGraph,
        observations: list[FilmObservation],
        *,
        transition_policy: str = "none",
        transition_duration_us: int | None = None,
        card: ShotCard | None = None,
        voice_led: bool = False,
        audio_style: str = "none",
        project_story_link_review: ProjectStoryLinkReview | None = None,
        project_story_mention_review: ProjectStoryMentionReview | None = None,
        narrative: dict | None = None,
        narrative_order_priority: bool = False,
        artifact_variant: str | None = None,
        selection_reason_events: dict | None = None,
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        """Plan from exact per-asset evidence without inventing one project clock."""
        if brief.project_id != manifest.project_id:
            raise ValueError("DirectorBrief project_id must match project manifest")
        if (context.project_id != manifest.project_id
                or context.project_manifest_id != manifest.manifest_id
                or context.project_revision != manifest.revision
                or context.invalidated_at is not None):
            raise ValueError("FilmContext must be current for the supplied manifest revision")
        if (project_graph.context_id != context.context_id
                or project_graph.analysis_fingerprint != context.analysis_fingerprint):
            raise ValueError("ProjectStoryGraph does not match the supplied context snapshot")
        from director_brain.project_story_graph import project_story_graph_id
        if project_graph.graph_id != project_story_graph_id(context):
            raise ValueError("ProjectStoryGraph identity is not bound to this context")
        if context.timeline_scope != "project_per_asset":
            raise ValueError("project DirectorReasoner requires per-asset source clocks")
        if context.asset_refs != [
            asset.asset_id for asset in sorted(manifest.assets, key=lambda item: item.order)
        ]:
            raise ValueError("FilmContext asset membership/order differs from manifest")
        ordered_assets = sorted(manifest.assets, key=lambda item: item.order)
        expected_hashes = [
            asset.source_content_hash.lower()
            if asset.source_content_hash is not None else None
            for asset in ordered_assets
        ]
        if context.source_content_hashes != expected_hashes:
            raise ValueError("FilmContext source hashes differ from manifest assets")
        if len(context.asset_coverage) != len(ordered_assets):
            raise ValueError("FilmContext coverage must include every manifest asset")
        if [item.asset_id for item in context.asset_coverage] != context.asset_refs:
            raise ValueError("FilmContext coverage order differs from manifest assets")
        if [item.evidence_refs for item in context.asset_coverage] != [
            item.evidence_refs for item in project_graph.assets
        ]:
            raise ValueError("FilmContext and ProjectStoryGraph per-asset evidence differ")
        if [
            ref for item in context.asset_coverage for ref in item.evidence_refs
        ] != context.evidence_refs:
            raise ValueError("FilmContext evidence order differs from per-asset coverage")
        for asset, coverage, asset_graph in zip(
            ordered_assets, context.asset_coverage, project_graph.assets, strict=True
        ):
            expected_hash = (
                asset.source_content_hash.lower()
                if asset.source_content_hash is not None else None
            )
            if (coverage.source_content_hash != expected_hash
                    or coverage.rights_state != asset.rights.state.value
                    or coverage.rights_evidence_state != asset.rights.evidence_state
                    or coverage.analysis_state != asset_graph.analysis_state):
                raise ValueError("FilmContext asset identity/rights differ from manifest")
        if context.evidence_refs != project_graph.evidence_refs:
            raise ValueError("FilmContext and ProjectStoryGraph evidence refs differ")
        if len({item.observation_id for item in observations}) != len(observations):
            raise ValueError("duplicate observation record supplied")
        if {item.observation_id for item in observations} != set(context.evidence_refs):
            raise ValueError("observations must exactly match the current context evidence refs")

        aggregate, asset_order = _project_aggregate_story_graph(
            manifest, project_graph, observations)
        edl, plan = self.generate_plan(
            brief,
            aggregate,
            observations,
            transition_policy=transition_policy,
            transition_duration_us=transition_duration_us,
            card=card,
            voice_led=voice_led,
            audio_style=audio_style,
            narrative=narrative,
            narrative_order_priority=narrative_order_priority,
            artifact_variant=artifact_variant,
            _project_asset_order=asset_order,
            _selection_reason_events=selection_reason_events,
        )

        # Keep the persisted project lineage on the executable artifacts, not
        # only in the API envelope or free-form constraints. The aggregate
        # graph is an ephemeral planner view; the EDL points to the persisted
        # project StoryGraph that supplied its candidate evidence.
        edl = EditorialDecisionList.model_validate({
            **edl.model_dump(mode="python"),
            "context_id": project_graph.graph_id,
        })
        plan = DirectorDecisionPlan.model_validate({
            **plan.model_dump(mode="python"),
            "brief_id": brief.brief_id,
            "edl_id": edl.edl_id,
            "project_manifest_id": manifest.manifest_id,
            "project_revision": manifest.revision,
            "project_context_id": context.context_id,
            "project_story_graph_id": project_graph.graph_id,
            "film_state_version": context.context_id,
        })

        if any(edit.project_asset_id is None for edit in edl.ordered_edits):
            raise ValueError("project plan edits must retain project_asset_id")
        plan.sequence_project_asset_ids = [
            edit.project_asset_id for edit in edl.ordered_edits
        ]
        plan.film_state_version = context.context_id
        plan.constraints.extend([
            f"project_manifest={manifest.manifest_id}@{manifest.revision}",
            f"project_story_graph={project_graph.graph_id}",
            "project_timeline_scope=per_asset",
            "project_story_graph_cross_asset_relations_state=not_attempted",
            "automatic_cross_asset_inference=not_attempted",
        ])
        plan.open_questions.extend([
            "project_source_duration_not_aggregated_across_asset_clocks",
            "project_boundary_not_independently_verified",
            "project_processing_rights_evidence_declared_unverified",
            "automatic_cross_asset_person_event_inference_not_attempted",
            "director_quality_not_established_by_planner_execution",
        ])
        for asset_graph in project_graph.assets:
            if asset_graph.story_graph_state != "constructed":
                plan.open_questions.append(
                    f"project_asset_not_available_for_selection:{asset_graph.asset_id}:"
                    f"{asset_graph.story_graph_state}"
                )
        plan.open_questions = list(dict.fromkeys(plan.open_questions))
        from director_brain.project_story_link_review import (
            bind_project_story_link_review_to_plan,
        )

        edl, plan = bind_project_story_link_review_to_plan(
            edl,
            plan,
            project_story_link_review,
            manifest,
            context,
            project_graph,
            observations,
            project_story_mention_review,
        )
        return edl, plan

    def generate_plan(
        self,
        brief: DirectorBrief,
        graph: StoryGraph,
        observations: list[FilmObservation],
        *,
        narrative: dict | None = None,
        transition_policy: str = "none",
        transition_duration_us: int | None = None,
        card: ShotCard | None = None,
        entities=None,
        voice_led: bool = False,
        beat_grid=None,
        audio_style: str = "none",
        narrative_order_priority: bool = False,
        artifact_variant: str | None = None,
        _project_asset_order: dict[str, int] | None = None,
        _selection_reason_events: dict | None = None,
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        if graph.project_id != brief.project_id:
            raise ValueError("StoryGraph project_id must match DirectorBrief")
        if (transition_duration_us is not None
                and transition_policy != "dissolve_act_boundary"):
            raise ValueError(
                "transition duration requires dissolve_act_boundary policy")
        project_aggregate = graph.timeline_scope == "project_aggregate"
        if project_aggregate:
            if _project_asset_order is None or graph.project_asset_id is not None:
                raise ValueError("project aggregate graphs require manifest asset order")
            if any(
                observation.project_id != brief.project_id
                or observation.project_asset_id not in _project_asset_order
                for observation in observations
            ):
                raise ValueError("project aggregate observations must match manifest scope")
            if any(
                observation.observation_type in {
                    "deterministic_technical", "vlm_semantic", "speech_transcript",
                }
                and (observation.timebase != TIMEBASE_US
                     or observation.timebase_unit != TimebaseUnit.MICROSECONDS)
                for observation in observations
            ):
                raise ValueError("project aggregate evidence must use canonical microseconds")
            if entities is not None or beat_grid is not None:
                raise ValueError(
                    "project aggregate planning does not accept unscoped entity "
                    "or beat-grid evidence"
                )
        elif _project_asset_order is not None:
            raise ValueError("manifest asset order is only valid for a project aggregate graph")
        elif graph.project_asset_id is not None:
            if graph.timeline_scope != "project_asset":
                raise ValueError("project-bound StoryGraph must declare project_asset scope")
            if graph.timebase_unit != TimebaseUnit.MICROSECONDS:
                raise ValueError(
                    "project-bound DirectorReasoner currently requires an explicit "
                    "microsecond source timebase"
                )
            if graph.timebase is None:
                raise ValueError("project-bound StoryGraph requires a source timebase")
            if graph.timebase != TIMEBASE_US:
                raise ValueError(
                    "project-bound DirectorReasoner requires the canonical "
                    "microsecond timebase (1000000 ticks per second)"
                )
            if any(
                observation.project_id != brief.project_id
                or observation.project_asset_id != graph.project_asset_id
                for observation in observations
            ):
                raise ValueError(
                    "DirectorReasoner observations must match the StoryGraph project asset"
                )
            if any(
                observation.timebase_unit != TimebaseUnit.MICROSECONDS
                or observation.timebase != graph.timebase
                for observation in observations
            ):
                raise ValueError(
                    "DirectorReasoner observations must match the StoryGraph "
                    "microsecond timebase"
                )
        elif any(observation.project_asset_id is not None for observation in observations):
            raise ValueError(
                "unbound StoryGraph cannot consume project-bound observations"
            )

        tech_obs = [
            o for o in observations if o.observation_type == "deterministic_technical"
        ]
        vlm_obs = [
            o for o in observations if o.observation_type == "vlm_semantic"
        ]
        candidates = _build_candidates(tech_obs, vlm_obs, self.blur_threshold)
        if project_aggregate:
            graph_candidate_ids = {
                _candidate_identity(
                    ref["project_asset_id"],
                    ref["source_media_hash"],
                    ref["source_asset_id"],
                )
                for node in graph.nodes
                for ref in node.attributes.get("candidate_refs", [])
            }
            # Asset observations can be present in the project context while
            # that asset's graph is unavailable (for example, rights or
            # analysis coverage is incomplete). Only graph-bound candidates
            # may enter a project Plan or the local narrative prompt.
            candidates = [
                candidate for candidate in candidates
                if candidate["_candidate_identity"] in graph_candidate_ids
            ]

        def record_selection_reason(candidate: dict, reason: str) -> None:
            identity = candidate.get("_candidate_identity")
            if (_selection_reason_events is None
                    or not isinstance(identity, tuple)):
                return
            _selection_reason_events.setdefault(identity, set()).add(reason)

        if _selection_reason_events is not None:
            for candidate in candidates:
                if candidate.get("_no_data"):
                    record_selection_reason(
                        candidate, "source_has_no_technical_data")
                if candidate.get("_dark_shot"):
                    record_selection_reason(candidate, "source_is_dark_shot")
                if candidate.get("vlm_role") == "discard":
                    record_selection_reason(candidate, "source_role_discarded")
        narrative_candidate_map: dict[str, dict] = {}
        strategy_dispositions: dict[str, str] | None = None
        for candidate in candidates:
            ref = _candidate_narrative_ref(candidate)
            if ref in narrative_candidate_map:
                raise ValueError("narrative input contains an ambiguous source shot identity")
            narrative_candidate_map[ref] = candidate
        if project_aggregate and narrative is not None:
            refs = narrative.get("project_candidate_refs")
            order = narrative.get("suggested_order_resolved")
            if (not isinstance(refs, list)
                    or any(not isinstance(ref, str) for ref in refs)
                    or len(refs) != len(set(refs))
                    or not isinstance(order, list)
                    or len(order) != len(refs)
                    or set(order) != set(refs)
                    or any(ref not in narrative_candidate_map for ref in refs)):
                raise ValueError(
                    "project narrative must resolve to exact, unique project shot identities"
                )
            if narrative_order_priority and narrative.get(
                    "_active_strategy_hypothesis_id") is not None:
                raw_dispositions = narrative.get("_active_source_dispositions")
                if (not isinstance(raw_dispositions, dict)
                        or set(raw_dispositions) != set(refs)
                        or any(value not in {"include", "exclude"}
                               for value in raw_dispositions.values())
                        or sum(value == "include"
                               for value in raw_dispositions.values()) < 2):
                    raise ValueError(
                        "active SHADOW strategy must bind an include/exclude "
                        "decision for every project source and include at least two")
                strategy_dispositions = raw_dispositions
        if _project_asset_order is not None:
            candidates.sort(key=lambda item: (
                _project_asset_order[item["_project_asset_id"]],
                item["source_in_us"],
                item["source_media_hash"],
                item["source_shot_id"],
            ))

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
                    record_selection_reason(c, "excluded_by_must_avoid")
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
        if strategy_dispositions is not None:
            kept = []
            for candidate in candidates:
                disposition = strategy_dispositions[
                    _candidate_narrative_ref(candidate)]
                if disposition == "exclude":
                    record_selection_reason(
                        candidate, "excluded_by_director_strategy")
                else:
                    kept.append(candidate)
            candidates = kept
            if len(candidates) < 2 or not any(
                    candidate["exposure_ok"] for candidate in candidates):
                raise EvidenceTooPoorError(
                    "strategy include decisions leave fewer than two usable "
                    "source shots; no heuristic fallback may restore excluded sources")
        for kind, text in directives.unverifiable:
            constraint_questions.append(
                f"constraint_unverifiable: {kind}「{text}」"
                f"（当前技术观测无法验证语义内容，待语义证据通路）"
            )

        # ---- P1-b：剪辑语言意图 → 片段时长上下界（"快剪/慢剪"不再一个味）----
        clip_bounds = editing_language_bounds(brief, (MIN_CLIP_US, MAX_CLIP_US))
        min_clip_us, max_clip_us = clip_bounds
        if card is not None:
            # D1 镜头卡：节奏边界覆盖（卡是导演创作选择包）
            po = card.pacing_override or {}
            if "min_clip_us" in po:
                min_clip_us = int(po["min_clip_us"])
            if "max_clip_us" in po:
                max_clip_us = int(po["max_clip_us"])
            if min_clip_us >= max_clip_us:
                raise ValueError(
                    f"镜头卡 {card.card_id} 节奏边界非法: "
                    f"min {min_clip_us} >= max {max_clip_us}")
            clip_bounds = (min_clip_us, max_clip_us)
            # 卡声明转场策略且调用方未显式指定时，卡生效（显式参数优先）
            if transition_policy == "none":
                transition_policy = card.transition_policy

        # ---- D3：语音价值（voice_led 显式开启；asr_transcript 通路闸门）----
        speech_obs = [
            o for o in observations if o.observation_type == "speech_transcript"
        ]
        speech_shots = 0
        if voice_led and speech_obs:
            ensure_decision_use_allowed("asr_transcript")
            utterances_by_asset: dict[str | None, list[tuple[int, int]]] = {}
            for observation in speech_obs:
                utterances_by_asset.setdefault(
                    observation.project_asset_id, []).append(
                        (observation.start_frame, observation.end_frame))
            for c in candidates:
                s_in, s_out = c["source_in_us"], c["source_out_us"]
                utterances = utterances_by_asset.get(
                    c["_project_asset_id"], [])
                overlap = sum(
                    max(0, min(s_out, u_end) - max(s_in, u_start))
                    for u_start, u_end in utterances)
                coverage = overlap / max(s_out - s_in, 1)
                c["_speech_coverage"] = round(coverage, 2)
                c["_speech_bonus"] = 0.1 if coverage >= 0.2 else 0.0
                speech_shots += 1 if coverage >= 0.2 else 0
        if not (voice_led and speech_obs) and voice_led:
            # voice_led 但素材无语音观测——响亮归因（对齐 ASR 归因先例）
            speech_shots = -1
        # ---- D2：实体归属注入（entities: EntityResolution）----
        entity_assignments: dict[str, list[str]] = (
            dict(entities.assignments) if entities is not None else {})
        for c in candidates:
            c["_entity_ids"] = entity_assignments.get(c["source_shot_id"], [])

        # ---- 候选①：语义幕分配表（vlm_semantic 闸门已在上游执法）----
        # 优先级：叙事弧幕边界（P3-3 序列级理解）> 逐镜头 narrative_role
        # （P3-2）> 时间比例（story_graph 默认）。无任何语义时表为空，
        # 逐位走旧路径（回归安全）。
        semantic_act: dict[tuple[str | None, str, str], str] = {}
        if narrative:
            for b in narrative.get("act_boundaries_resolved", []):
                act_name = b.get("act")
                if act_name in ACT_ORDER:
                    for sid in b.get("shot_ids", []):
                        candidate = narrative_candidate_map.get(sid)
                        if candidate is not None:
                            semantic_act[candidate["_candidate_identity"]] = act_name
        for c in candidates:
            identity = c["_candidate_identity"]
            if identity in semantic_act:
                continue
            role = c.get("vlm_narrative_role")
            if role in ROLE_TO_ACT:
                semantic_act[identity] = ROLE_TO_ACT[role]
        # 注意：语义幕分配是能力升级而非降级——不写 degradation_events
        # （否则会误置 plan.degraded=True），消费留痕走 open_questions。
        semantic_moves = bool(semantic_act)

        # ---- 候选①：叙事推荐顺序 → 幕内排序位次（P3-3）----
        order_rank: dict[tuple[str | None, str, str], int] = {}
        if narrative:
            resolved_order = narrative.get("suggested_order_resolved") or []
            for rank, sid in enumerate(resolved_order):
                candidate = narrative_candidate_map.get(sid)
                if candidate is not None:
                    order_rank[candidate["_candidate_identity"]] = rank
        if narrative_order_priority:
            for candidate in candidates:
                rank = order_rank.get(candidate["_candidate_identity"])
                if rank is not None:
                    candidate["_director_strategy_rank"] = rank

        # ---- 从 graph.nodes 获取四幕 shot_ids，每幕单独选片 ----
        act_nodes = [
            n for n in graph.nodes
            if n.attributes.get("act") in ACT_ORDER
        ]
        act_nodes.sort(key=lambda n: ACT_ORDER[n.attributes["act"]])

        paired: list[tuple[str, EditItem, Decision]] = []
        borrowed_any = False
        fallback_any = False
        #: 候选①：本 plan 内的语义评分明细（rationale 留痕用）
        sem_scores: dict[tuple[str | None, str, str], SemanticScore] = {}
        #: 候选①：内容多样性——已评分镜头的归一化描述（跨幕累积）
        seen_descs: list[str] = []
        #: 已被选入 plan 的镜头 id（T1 修复：一份 plan 内同一镜头只选一次）
        selected_ids: set[tuple[str | None, str, str]] = set()

        prev_act_entity_ids: set[str] = set()
        for act_node in act_nodes:
            act_name = act_node.attributes["act"]
            shot_ids = set(act_node.attributes.get("shot_ids", []))
            candidate_refs = act_node.attributes.get("candidate_refs")
            if candidate_refs is not None:
                if any(
                    not isinstance(item, dict)
                    or not item.get("source_media_hash")
                    or not item.get("source_asset_id")
                    or not item.get("project_asset_id")
                    for item in candidate_refs
                ):
                    raise ValueError("project ACT candidate refs are malformed")
                shot_identities = {
                    _candidate_identity(
                        item["project_asset_id"],
                        item["source_media_hash"],
                        item["source_asset_id"],
                    )
                    for item in candidate_refs
                }
            else:
                shot_identities = {
                    candidate["_candidate_identity"] for candidate in candidates
                    if candidate["source_shot_id"] in shot_ids
                }
            # T1 修复：同一镜头在一份 plan 中只选入一次。重复选入会让 EDL 出现
            # 同 asset 的嵌套区间（validator 判 overlap），历史上由 repair 越权
            # 删镜头"兜底"——按"Plan=导演依据"裁定，缺陷必须在导演层消除。
            # 注意：有意的镜头复用（reprise）未来需以"不重叠子区间"显式表达，
            # 当前 schema 不支持，先按保守规则排除。
            if semantic_moves:
                # 候选①：语义幕分配接管（叙事边界 > narrative_role）；
                # 无语义归属的候选仍按时间图分幕，两种来源并存。
                def _belongs(c: dict) -> bool:
                    identity = c["_candidate_identity"]
                    if identity in selected_ids:
                        return False
                    if identity in semantic_act:
                        return semantic_act[identity] == act_name
                    return identity in shot_identities

                act_cands = [c for c in candidates if _belongs(c)]
            else:
                act_cands = [
                    c for c in candidates
                    if c["_candidate_identity"] in shot_identities
                    and c["_candidate_identity"] not in selected_ids
                ]

            # 空幕兜底：本幕时间范围内无镜头时，从**未被选用**的全局候选借用。
            # VLM 判为 discard 的镜头视为导演层已否决，不参与借用；无数据
            # 观测（读帧失败）同样不得借用于兜底（宁可空幕）。resolve 幕
            # 优先取时间最靠后的镜头，hook 幕取最靠前的。
            borrowed = False
            if not act_cands and candidates:
                pool = [
                    c for c in candidates
                    if c["_candidate_identity"] not in selected_ids
                    and c.get("vlm_role") != "discard"
                    and not c.get("_no_data")
                    and not c.get("_dark_shot")
                    # 候选①：语义归属其他幕的镜头不得借入本幕
                    # （无语义归属者保持时间兜底行为）
                    and semantic_act.get(c["_candidate_identity"], act_name)
                    == act_name
                ]
                if pool:
                    if narrative_order_priority:
                        selected = min(pool, key=lambda c: (
                            c.get("_director_strategy_rank", 10 ** 9),
                            c["source_in_us"],
                        ))
                        act_cands = [copy.deepcopy(selected)]
                    else:
                        sorted_cands = sorted(
                            pool, key=lambda c: c["source_in_us"])
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

            # ---- 候选①：语义评分（P3-2 数学收编进内核）----
            # importance 基础加权 + 情绪弧/剪辑语言匹配 + 内容多样性降权；
            # 再乘结构性 VLM 权重（hero/discard/长静态等，保留旧权重语义）。
            # 有任一语义分 → generate_edl 改按 selection_score 贪心排序。
            score_key: str | None = None
            for c in act_cands:
                sem = {
                    "importance": c.get("vlm_importance"),
                    "emotional_tone": c.get("vlm_emotional_tone"),
                    "action_type": c.get("vlm_action_type"),
                    "narrative_role": c.get("vlm_narrative_role"),
                    "scene_description": c.get("vlm_scene_description", ""),
                }
                if not any(
                    v is not None
                    for k, v in sem.items() if k != "scene_description"
                ):
                    continue
                c["_prev_entity_ids"] = prev_act_entity_ids
                s = compute_semantic_score(c, sem, brief, act_name, seen_descs)
                desc = (sem.get("scene_description") or "")[:30].lower().strip()
                if desc:
                    seen_descs.append(desc)
                structural_mult, _hits = vlm_multiplier(c, brief.target_duration)
                c["selection_score"] = round(s.total * structural_mult, 2)
                sem_scores[c["_candidate_identity"]] = s
                score_key = "selection_score"

            # route-9 导演层 policy：切点吸附场景边界（head 对齐）——
            # 片段起点 = 源镜头起点（discover_shots 分段依据 = 场景边界）
            edl = generate_edl(
                project_id=brief.project_id,
                candidates=act_cands,
                target_duration_us=per_act_target,
                min_clip_us=min_clip_us,
                max_clip_us=max_clip_us,
                align="head",
                score_key=score_key,
                priority_key=(
                    "_director_strategy_rank"
                    if narrative_order_priority else None
                ),
                selection_reason_events=_selection_reason_events,
            )
            act_edits: list[EditItem] = list(edl.ordered_edits)
            used_fallback = False

            # 时长不足兜底：本幕选中总时长 < 目标 50% 时，放宽 technical_usable
            # 重新选片（仅对本幕候选深拷贝，不影响其他幕）
            if act_edits:
                act_dur = sum(e.out_frame - e.in_frame for e in act_edits)
                if act_dur < per_act_target * 0.5 and not borrowed:
                    # P0 不变量：放宽阶梯不得注水 no_data/暗镜头——
                    # 此前本通道漏查 _dark_shot（语义 pilot 黑帧 4s 根因）
                    relaxed = [
                        copy.deepcopy(c) for c in act_cands
                        if not c.get("_no_data") and not c.get("_dark_shot")
                    ]
                    for c in relaxed:
                        c["technical_usable"] = True
                    edl_relaxed = generate_edl(
                        project_id=brief.project_id,
                        candidates=relaxed,
                        target_duration_us=per_act_target,
                        min_clip_us=min_clip_us,
                        max_clip_us=max_clip_us,
                        align="head",
                        score_key=score_key,
                        priority_key=(
                            "_director_strategy_rank"
                            if narrative_order_priority else None
                        ),
                        selection_reason_events=_selection_reason_events,
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
                # 兜底选片同样排除无数据观测与暗镜头（P0 不变量：
                # "读帧失败/黑屏"≠"质量最差"，放宽阶梯不得注水入选）
                usable_pool = [
                    c for c in act_cands
                    if not c.get("_no_data") and not c.get("_dark_shot")
                ]
                if usable_pool:
                    if narrative_order_priority:
                        best = min(usable_pool, key=lambda c: (
                            c.get("_director_strategy_rank", 10 ** 9),
                            -c["blur_score"],
                        ))
                    else:
                        best = max(usable_pool, key=lambda c: c["blur_score"])
                    act_edits = [EditItem(
                        source_asset_id=best["source_shot_id"],
                        source_media_hash=best["source_media_hash"],
                        in_frame=int(best["source_in_us"]),
                        out_frame=int(best["source_out_us"]),
                        timebase=TIMEBASE_US,
                        timebase_unit=TimebaseUnit.MICROSECONDS,
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

            # This selection path accepts canonical microseconds only. The
            # source binding records the original observation interval and
            # rejects ambiguous duplicate matches instead of overwriting refs.
            act_edits = [
                _bind_edit_provenance(edit, act_cands) for edit in act_edits
            ]

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
                edit_identity = _edit_identity(edit)
                sem = sem_scores.get(edit_identity)
                if sem is not None:
                    edit.rationale = (
                        f"act={act_name}, sem[{sem.reason}], {reason}"
                    )
                else:
                    edit.rationale = f"act={act_name}, {reason}"
                slot_label = "fallback" if used_fallback else f"slot_{idx + 1:02d}"

                selected_blur = next(
                    (c["blur_score"] for c in act_cands
                     if c["_candidate_identity"] == edit_identity), 0.0)
                blur_norm = min(selected_blur / max_blur, 1.0) if max_blur > 0 else 0.0
                confidence = round(
                    usable_ratio * 0.4 + blur_norm * 0.4
                    + (0.0 if (used_fallback or borrowed) else 0.2), 2)
                confidence = max(0.1, min(confidence, 1.0))

                other_cands = [c for c in act_cands
                               if c["_candidate_identity"] != edit_identity]
                other_cands.sort(key=lambda c: c["blur_score"], reverse=True)
                alternatives = [c["source_shot_id"] for c in other_cands[:2]]

                requires_approval = (confidence < 0.6 or used_fallback
                                     or borrowed or act_total < 2)

                decision = Decision(
                    decision_id=f"dec_{act_name}_{slot_label}",
                    purpose="select_shot",
                    shot_refs=[edit.source_asset_id],
                    project_asset_id=edit.project_asset_id,
                    evidence_refs=list(edit.source_observation_refs),
                    rationale=edit.rationale,
                    alternatives=alternatives,
                    alternative_project_asset_ids=[
                        c["_project_asset_id"] for c in other_cands[:2]
                    ] if any(c["_project_asset_id"] is not None
                             for c in other_cands[:2]) else [],
                    confidence=confidence,
                    requires_approval=requires_approval,
                )
                paired.append((act_name, edit, decision))
                selected_ids.add(edit_identity)
            prev_act_entity_ids = {
                eid
                for c in act_cands
                if c["_candidate_identity"] in selected_ids
                for eid in (c.get("_entity_ids") or [])
            }

        # ---- 目标时长补齐（路径无关的可达性保底；逐条响亮留痕）----
        # 幕配额无法填满时（素材没有某类内容/暗镜头被 P0 正确排除/scenedetect
        # 后端更准地切出黑字幕段），若总时长低于目标下界（validator ±10%），
        # 按剩余配额从全局合格候选补齐（语义分/技术分排序）。不补则
        # validator 必 FAIL → fail-closed 拒绝出片：诚实但可用性倒退。
        # 补齐镜头 requires_approval；事件名区分路径（semantic/target topup）。
        # ---- 转场策略（导演层 artistic choice）：幕边界 dissolve。
        # 分配在补齐之前——补齐目标加 ΣD（生成端预补偿；validator/repair
        # 的转场感知预算链继续兜底，8b 起既有能力）----
        n_transitions = 0
        total_overlap_us = 0
        if transition_policy == "dissolve_act_boundary":
            duration_us = (
                400_000 if transition_duration_us is None
                else transition_duration_us)
            if type(duration_us) is not int or duration_us <= 0:
                raise ValueError("dissolve transition duration must be positive")
            for i in range(len(paired) - 1):
                if paired[i][0] != paired[i + 1][0]:
                    left_duration_us = (
                        paired[i][1].out_frame - paired[i][1].in_frame)
                    right_duration_us = (
                        paired[i + 1][1].out_frame
                        - paired[i + 1][1].in_frame)
                    if duration_us >= min(left_duration_us, right_duration_us):
                        raise ValueError(
                            "dissolve duration must be shorter than both "
                            "adjacent selected clips")
                    paired[i][1].transition = TransitionSpec(
                        type="xfade", name="dissolve", duration_us=duration_us)
                    n_transitions += 1
                    total_overlap_us += duration_us
            if transition_duration_us is not None and n_transitions == 0:
                raise ValueError(
                    "dissolve strategy has no generated act boundary")

        topup_event = "semantic_topup" if semantic_moves else "target_topup"
        from director_brain.providers.heuristic import compute_clip_window
        target_low = int(brief.target_duration * 0.9) + total_overlap_us
        act_filled: dict[str, int] = {}
        for a_name, a_edit, _d in paired:
            act_filled[a_name] = act_filled.get(a_name, 0) + (
                a_edit.out_frame - a_edit.in_frame)
        topup_idx = 0
        while sum(act_filled.values()) < target_low:
            placed = False
            for act_name in ACT_ORDER:
                if sum(act_filled.values()) >= target_low:
                    break
                remaining_q = max(
                    int(ACT_RATIO[act_name] * brief.target_duration),
                    MIN_CLIP_US,
                ) - act_filled.get(act_name, 0)
                if remaining_q < MIN_CLIP_US:
                    continue
                pool = _project_target_topup_pool(
                    candidates,
                    selected_ids,
                    min_clip_us=min_clip_us,
                    narrative_order_priority=narrative_order_priority,
                )
                if not pool:
                    break
                take = pool[0]
                clip_dur = min(
                    take["duration_us"], max_clip_us, remaining_q)
                if clip_dur < min_clip_us:
                    continue
                in_us, out_us = compute_clip_window(
                    take["source_in_us"], take["source_out_us"],
                    clip_dur, align="head")
                topup_idx += 1
                edit = EditItem(
                    source_asset_id=take["source_shot_id"],
                    source_media_hash=take["source_media_hash"],
                    in_frame=int(in_us),
                    out_frame=int(out_us),
                    timebase=TIMEBASE_US,
                    timebase_unit=TimebaseUnit.MICROSECONDS,
                    shot_function=ACT_FUNCTION.get(act_name),
                    rationale=(
                        f"act={act_name}, topup, "
                        f"heuristic:blur={take['blur_score']}"),
                    evidence_type=(
                        "vlm" if take.get("vlm_shot_function")
                        else "heuristic"),
                )
                edit = _bind_edit_provenance(edit, [take])
                decision = Decision(
                    decision_id=f"dec_{act_name}_topup_{topup_idx:02d}",
                    purpose="select_shot",
                    shot_refs=[take["source_shot_id"]],
                    project_asset_id=edit.project_asset_id,
                    evidence_refs=list(edit.source_observation_refs),
                    rationale=edit.rationale,
                    alternatives=[],
                    confidence=0.3,
                    requires_approval=True,
                )
                paired.append((act_name, edit, decision))
                selected_ids.add(take["_candidate_identity"])
                act_filled[act_name] = (
                    act_filled.get(act_name, 0) + out_us - in_us)
                degradation_events.append(
                    f"{topup_event}:act={act_name}"
                    f":shot={take['source_shot_id']}")
                placed = True
            if not placed:
                break  # 无幕可放且无进展：物理上限，交 validator 判定

        # ---- D5：切点吸附节拍（beat_grid 通路 ACTIVE 时生效）----
        # 输出时间轴的每个剪切点吸附最近节拍（容差半拍）；调整量钳制在
        # 源镜头边界内。EXPERIMENTAL 下闸门拒绝吸附——只留注记（禁止半消费）。
        beat_aligned = 0
        snap_enabled = False
        if beat_grid is not None and beat_grid.beat_times_us:
            from director_brain.pathway_protocol import (
                PathwayStatus as _PS,
                ensure_decision_use_allowed as _gate,
                get_pathway_status as _get,
            )
            snap_enabled = _get("beat_grid") is _PS.ACTIVE
            if snap_enabled:
                _gate("beat_grid")
            cand_by_id = {c["_candidate_identity"]: c for c in candidates}
            cum = 0
            half_beat = int(500_000 * 60.0 / max(beat_grid.bpm, 30.0) / 2) + 1
            for _act, edit, _dec in paired:
                overlap = (edit.transition.duration_us
                           if (edit.transition and edit.transition.type == "xfade")
                           else 0)
                cum += edit.out_frame - edit.in_frame
                cut_pos = cum - overlap  # 该切点在输出时间轴的位置
                nearest = min(beat_grid.beat_times_us,
                              key=lambda b: abs(b - cut_pos))
                delta = nearest - cut_pos
                if not snap_enabled or delta == 0 or abs(delta) > half_beat:
                    continue
                cand = cand_by_id.get(_edit_identity(edit))
                if cand is None:
                    continue
                new_out = edit.out_frame + delta
                if new_out < edit.in_frame + MIN_CLIP_US:
                    continue
                if new_out > cand.get("source_out_us", new_out):
                    continue
                shift = new_out - edit.out_frame
                edit.out_frame = int(new_out)
                cum += shift
                beat_aligned += 1
                degradation_events.append(
                    f"beat_aligned:shot={edit.source_asset_id}:shift={shift}")

        # ---- 按幕顺序（hook→develop→peak→resolve）----
        # 幕内排序：有叙事推荐顺序（P3-3 suggested_order）时按其位次，
        # 未覆盖的镜头排在其后；否则按 in_frame 升序（旧行为）。
        sequence_before_final_order = [
            _edit_identity(item[1]) for item in paired
        ]
        if narrative_order_priority:
            if not narrative or not order_rank:
                raise ValueError(
                    "narrative-order planning requires a complete project order")
            paired.sort(key=lambda p: (
                order_rank.get(_edit_identity(p[1]), 10 ** 9),
                ACT_ORDER.get(p[0], 99),
                _project_asset_order.get(
                    p[1].project_asset_id, -1
                ) if _project_asset_order is not None else -1,
                p[1].in_frame,
            ))
        else:
            paired.sort(key=lambda p: (
                ACT_ORDER.get(p[0], 99),
                order_rank.get(_edit_identity(p[1]), 10 ** 9),
                _project_asset_order.get(
                    p[1].project_asset_id, -1
                ) if _project_asset_order is not None else -1,
                p[1].in_frame,
            ))
        narrative_order_changed = sequence_before_final_order != [
            _edit_identity(item[1]) for item in paired
        ]

        if narrative_order_priority:
            for _act_name, edit, decision in paired:
                rank = order_rank.get(_edit_identity(edit))
                if rank is None:
                    raise ValueError(
                        "selected project edit is absent from the strategy order")
                edit.rationale = (
                    f"{edit.rationale or ''}; shadow_strategy_order_rank={rank}"
                ).strip("; ")
                decision.rationale = edit.rationale

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
        if sem_scores:
            open_questions.append(
                f"semantic_selection:applied:shots={len(sem_scores)}")
        if order_rank:
            open_questions.append(
                "narrative_reorder:applied:"
                f"planner_sort_changed={str(narrative_order_changed).lower()}:"
                f"shots={len(order_rank)}")
        if narrative_order_priority:
            ranked_candidate_count = sum(
                candidate["_candidate_identity"] in order_rank
                for candidate in candidates
            )
            open_questions.append(
                "shadow_strategy_model_order_influences_selection_empty_act_fallback_and_output_order;"
                "technical_eligibility_and_clip_window_remain_rule_driven")
            open_questions.append(
                "shadow_strategy_ranked_candidate_coverage="
                f"{ranked_candidate_count}/{len(candidates)};"
                "all_retained_candidates_have_model_rank")
        if strategy_dispositions is not None:
            open_questions.append(
                "shadow_strategy_source_disposition:"
                f"included={sum(value == 'include' for value in strategy_dispositions.values())};"
                f"excluded={sum(value == 'exclude' for value in strategy_dispositions.values())};"
                "planner_exclusions_remain_additional_hard_filters")
        if entities is not None and entities.entities:
            open_questions.append(
                f"entities_resolved:count={len(entities.entities)}")
        if voice_led:
            open_questions.append(
                f"voice_led:applied:speech_shots={speech_shots}")
        if beat_grid is not None:
            open_questions.append(
                f"beat_grid:bpm={beat_grid.bpm}:aligned={beat_aligned}")
        open_questions.extend(constraint_questions)

        source_hashes = ordered_unique_source_hashes(edits)

        expected_duration = sum(e.out_frame - e.in_frame for e in edits)

        artifact_suffix = (
            f"_{short_hash(artifact_variant)}" if artifact_variant else ""
        )
        edl = EditorialDecisionList(
            schema_version="1.0",
            project_id=brief.project_id,
            created_at=int(time.time()),
            producer=PRODUCER,
            source_ref=brief.source_ref,
            edl_id=(
                f"edl_{brief.project_id}_{short_hash(brief.brief_id)}"
                f"{artifact_suffix}"
            ),
            version="0.1",
            brief_version=brief.version,
            context_id=graph.graph_id,
            source_asset_hashes=source_hashes,
            timebase=TIMEBASE_US,
            timebase_unit=TimebaseUnit.MICROSECONDS,
            ordered_edits=edits,
            expected_duration=expected_duration,
            approval_state="draft",
            artistic_choices=(
                [f"shot_card:{card.card_id}@{card.version}"] if card else []
            ),
        )

        plan = DirectorDecisionPlan(
            schema_version="1.0",
            project_id=brief.project_id,
            created_at=int(time.time()),
            producer=PRODUCER,
            source_ref=brief.source_ref,
            plan_id=(
                f"plan_{brief.project_id}_{short_hash(brief.brief_id)}"
                f"{artifact_suffix}"
            ),
            version="0.1",
            brief_version=brief.version,
            film_state_version="0.1",
            sequence=[e.source_asset_id for e in edits],
            sequence_project_asset_ids=(
                [e.project_asset_id for e in edits]
                if edits and all(e.project_asset_id is not None for e in edits)
                else []
            ),
            decisions=decisions,
            constraints=[f"target_duration_us={brief.target_duration}"]
            + [rule.encode() for rule in applied_rules]
            + encode_bounds(clip_bounds)
            + ([f"transition_policy=dissolve_act_boundary:applied={n_transitions}"]
               if n_transitions else [])
            + ([f"transition_duration_us={duration_us}"]
               if n_transitions else [])
            + ([f"shot_card={card.card_id}@{card.version}"] if card else [])
            + ([f"voice_led=on:speech_shots={speech_shots}"] if voice_led else [])
            + ([f"audio_style={audio_style}"] if audio_style != "none" else []),
            open_questions=open_questions,
            degraded=bool(degradation_events),
            degradation_events=degradation_events,
            validation_status="pending",
            approval_state="draft",
        )

        return _apply_asr_audio_bridge(
            edl,
            plan,
            observations,
            audio_style,
        )


class LLMDirectorReasoner(DirectorReasoner):
    """Evidence-bounded narrative ordering on top of the deterministic planner.

    The local LLM may influence the shot order and act boundaries, while the
    existing planner still owns source bounds, technical constraints and EDL
    construction. The normal entry is gated by the separate
    ``director_strategy_reasoning`` pathway;
    ``generate_shadow_plan`` is an explicitly non-confirmable comparison path.
    """

    def __init__(self, provider: str = "ollama", config: dict | None = None):
        self.provider = provider
        self.config = dict(config or {})

        from director_brain.llm_adapter import LLMTransportError

        def configuration_error(message: str) -> LLMTransportError:
            return LLMTransportError(
                message, failure_code="provider_configuration_error")

        if provider != "ollama":
            raise configuration_error(
                "LLMDirectorReasoner currently supports local Ollama only")
        from director_brain.config import load_settings

        settings = load_settings()
        self.model = str(self.config.get("model", "qwen2.5:7b"))
        configured_base_url = self.config.get("base_url")
        if configured_base_url is None:
            configured_base_url = settings.ollama_base_url.rstrip("/")
            if not configured_base_url.endswith("/v1"):
                configured_base_url += "/v1"
        self.base_url = str(configured_base_url)
        try:
            parsed = urlparse(self.base_url)
        except ValueError:
            raise configuration_error(
                "private-media reasoning requires a loopback Ollama endpoint"
            ) from None
        if (parsed.scheme not in {"http", "https"}
                or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username or parsed.password):
            raise configuration_error(
                "private-media reasoning requires a loopback Ollama endpoint",
            )
        try:
            self.timeout = int(self.config.get("timeout", 180))
            self.temperature = float(self.config.get("temperature", 0.0))
        except (OverflowError, TypeError, ValueError):
            raise configuration_error(
                "invalid local LLM timeout or temperature") from None
        if self.timeout <= 0 or not 0.0 <= self.temperature <= 2.0:
            raise configuration_error("invalid local LLM timeout or temperature")
        model_digest = self.config.get(
            "model_digest", settings.project_local_reasoner_model_digest)
        runtime_version = self.config.get(
            "runtime_version", settings.project_local_reasoner_runtime_version)
        model_digest = model_digest.strip() if isinstance(model_digest, str) else model_digest
        runtime_version = (
            runtime_version.strip()
            if isinstance(runtime_version, str) else runtime_version)
        if (model_digest is not None and not isinstance(model_digest, str)) or (
            runtime_version is not None and not isinstance(runtime_version, str)
        ):
            raise configuration_error(
                "local Reasoner model digest and runtime version pins must be text")
        model_digest = model_digest or None
        runtime_version = runtime_version or None
        if (model_digest is None) != (runtime_version is None):
            raise configuration_error(
                "local Reasoner model digest and runtime version must be configured together")
        self.model_digest = None
        self.runtime_version = None
        self._runtime_binding = None
        if model_digest is not None:
            normalized_digest = model_digest.removeprefix("sha256:").lower()
            if not re.fullmatch(r"[0-9a-f]{64}", normalized_digest):
                raise configuration_error(
                    "local Reasoner model digest must be 64 hex characters")
            if (not isinstance(runtime_version, str)
                    or len(runtime_version) > 128
                    or not re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", runtime_version)):
                raise configuration_error(
                    "local Reasoner runtime version pin is invalid")
            if (parsed.path not in {"", "/", "/v1", "/v1/"}
                    or parsed.query or parsed.fragment):
                raise configuration_error(
                    "pinned local Reasoner endpoint must use the Ollama /v1 route")
            from observation_service.ollama_vlm_adapter import OllamaVLMAdapter

            runtime_base_url = parsed._replace(
                path="", params="", query="", fragment="").geturl().rstrip("/")
            self.model_digest = normalized_digest
            self.runtime_version = runtime_version
            self._runtime_binding = OllamaVLMAdapter(
                model=self.model,
                base_url=runtime_base_url,
                model_digest=self.model_digest,
                runtime_version=self.runtime_version,
                enforce_loopback=True,
            )

    def _verify_model_binding(self, *, required: bool = False) -> dict[str, str] | None:
        """Check the pinned local Ollama runtime and model before one provider call."""
        from director_brain.llm_adapter import LLMTransportError

        if self._runtime_binding is None:
            if required:
                raise LLMTransportError(
                    "formal project Reasoner requires a pinned local model and runtime",
                    failure_code="provider_configuration_error",
                )
            return None
        from observation_service.ollama_vlm_adapter import LocalVLMRuntimeBindingError

        try:
            self._runtime_binding.verify_runtime_binding()
        except LocalVLMRuntimeBindingError:
            raise LLMTransportError(
                "local project Reasoner model/runtime binding verification failed",
                failure_code="provider_model_binding_error",
            ) from None
        return {
            "model_digest": self.model_digest,
            "runtime_version": self.runtime_version,
        }

    def _analyze(
        self,
        brief: DirectorBrief,
        observations: list[FilmObservation],
    ) -> dict:
        from director_brain.narrative_analyzer import analyze_narrative

        semantic = [
            item for item in observations
            if item.observation_type == "vlm_semantic"
            and item.claim_kind == ClaimKind.MODEL_OBSERVATION
        ]
        if len(semantic) < 2:
            raise EvidenceTooPoorError(
                "LLM narrative reasoning requires at least two model-observed shot summaries"
            )
        identities = {
            (item.project_asset_id, item.media_hash.lower()) for item in semantic
        }
        if len(identities) != 1:
            raise ValueError(
                "single-source LLMDirectorReasoner cannot combine independent asset clocks"
            )
        if len({item.media_asset_id for item in semantic}) != len(semantic):
            raise ValueError("LLM narrative reasoning requires unique source shot IDs")
        technical_by_shot: dict[str, list[FilmObservation]] = {}
        for item in observations:
            if item.observation_type == "deterministic_technical":
                technical_by_shot.setdefault(item.media_asset_id, []).append(item)
        for item in semantic:
            matches = technical_by_shot.get(item.media_asset_id, [])
            if len(matches) != 1:
                raise ValueError(
                    "each semantic shot must bind to one exact technical source observation"
                )
            technical = matches[0]
            if (technical.media_hash.lower() != item.media_hash.lower()
                    or technical.project_asset_id != item.project_asset_id
                    or technical.start_frame != item.start_frame
                    or technical.end_frame != item.end_frame
                    or technical.timebase != item.timebase
                    or technical.timebase_unit != item.timebase_unit):
                raise ValueError(
                    "semantic and technical observations do not share an exact source interval"
                )
        for item in semantic:
            if item.timebase_unit != TimebaseUnit.MICROSECONDS or item.timebase != TIMEBASE_US:
                raise ValueError("LLM narrative reasoning requires canonical microsecond observations")
        semantic.sort(key=lambda item: (
            item.start_frame, item.end_frame, item.media_asset_id,
        ))
        claim_data = [_parse_claim(item.claim) for item in semantic]
        semantics = [{
            "scene_description": claim.get("scene_description")
                or claim.get("frame_description"),
            "action_type": claim.get("action_type"),
            "emotional_tone": claim.get("emotional_tone"),
            "narrative_role": claim.get("narrative_role")
                or claim.get("proposed_role_v2"),
            "importance": claim.get("importance"),
            "motion_amount": claim.get("motion_amount"),
        } for claim in claim_data]
        if not any(any(value not in (None, "") for value in item.values())
                   for item in semantics):
            raise EvidenceTooPoorError(
                "LLM narrative reasoning has no semantic fields to ground a sequence"
            )
        brief_context = json.dumps({
            "intent": brief.intent,
            "creator_direction": brief.creator_direction,
            "audience": brief.audience,
            "themes": brief.themes,
            "relationships": brief.relationships,
            "emotional_arc": brief.emotional_arc,
            "visual_language": brief.visual_language,
            "editing_language": brief.editing_language,
            "sound_language": brief.sound_language,
            "must_include": brief.must_include,
            "must_avoid": brief.must_avoid,
            "privacy_constraints": brief.privacy_constraints,
            "target_duration_us": brief.target_duration,
        }, ensure_ascii=False)
        return analyze_narrative(
            semantics,
            shot_ids=[item.media_asset_id for item in semantic],
            base_url=self.base_url,
            model=self.model,
            api_key="ollama-local",
            timeout=self.timeout,
            temperature=self.temperature,
            director_brief=brief_context,
        )

    def _analyze_project(
        self,
        brief: DirectorBrief,
        manifest: FilmProjectManifest,
        context: FilmContextSnapshot,
        project_graph: ProjectStoryGraph,
        observations: list[FilmObservation],
        *,
        project_story_link_review: ProjectStoryLinkReview | None = None,
        project_story_mention_review: ProjectStoryMentionReview | None = None,
        include_audio_style_choice: bool = False,
        include_editing_language_choice: bool = False,
        include_transition_policy_choice: bool = False,
        require_model_binding: bool = False,
    ) -> dict:
        from director_brain.narrative_analyzer import analyze_project_narrative

        if project_story_link_review is not None:
            from director_brain.project_story_link_review import (
                validate_project_story_link_review_for_plan,
            )

            validate_project_story_link_review_for_plan(
                project_story_link_review,
                manifest,
                context,
                project_graph,
                observations,
                project_story_mention_review,
            )

        aggregate, _asset_order = _project_aggregate_story_graph(
            manifest, project_graph, observations)
        eligible_refs = {
            (
                ref["project_asset_id"],
                ref["source_media_hash"].lower(),
                ref["source_asset_id"],
            )
            for node in aggregate.nodes
            for ref in node.attributes.get("candidate_refs", [])
        }
        technical_by_identity: dict[tuple[str, str, str], list[FilmObservation]] = {}
        for item in observations:
            if item.observation_type != "deterministic_technical":
                continue
            assert item.project_asset_id is not None
            identity = (
                item.project_asset_id, item.media_hash.lower(), item.media_asset_id,
            )
            technical_by_identity.setdefault(identity, []).append(item)

        ordered_assets = sorted(manifest.assets, key=lambda item: item.order)
        asset_order = {asset.asset_id: asset.order for asset in ordered_assets}
        semantic_observations = [
            item for item in observations
            if item.observation_type == "vlm_semantic"
            and item.claim_kind == ClaimKind.MODEL_OBSERVATION
            and item.project_asset_id is not None
            and (
                item.project_asset_id, item.media_hash.lower(), item.media_asset_id,
            ) in eligible_refs
        ]
        semantic_observations.sort(key=lambda item: (
            asset_order[item.project_asset_id],
            item.start_frame,
            item.end_frame,
            item.media_asset_id,
        ))

        semantics: list[dict] = []
        asset_ids: list[str] = []
        candidate_refs: list[str] = []
        source_evidence_by_candidate_ref: dict[str, dict] = {}
        candidate_index_by_identity: dict[tuple[str, str, str], int] = {}
        semantic_by_identity: dict[tuple[str, str, str], FilmObservation] = {}
        seen_source_refs: set[tuple[str, str, str]] = set()
        for item in semantic_observations:
            assert item.project_asset_id is not None
            identity = (
                item.project_asset_id, item.media_hash.lower(), item.media_asset_id,
            )
            if identity in seen_source_refs:
                raise ValueError(
                    "project narrative requires one semantic observation per exact source shot"
                )
            seen_source_refs.add(identity)
            technical_matches = technical_by_identity.get(identity, [])
            if len(technical_matches) != 1:
                raise ValueError(
                    "each project semantic shot must bind to one exact technical observation"
                )
            technical = technical_matches[0]
            if (
                technical.start_frame != item.start_frame
                or technical.end_frame != item.end_frame
                or technical.timebase != item.timebase
                or technical.timebase_unit != item.timebase_unit
            ):
                raise ValueError(
                    "project semantic and technical observations must share an exact source interval"
                )
            if (item.timebase != TIMEBASE_US
                    or item.timebase_unit != TimebaseUnit.MICROSECONDS):
                raise ValueError(
                    "project narrative reasoning requires canonical per-asset microseconds"
                )

            claim = _parse_claim(item.claim)
            summary = {
                "scene_description": claim.get("scene_description")
                    or claim.get("frame_description"),
                "shot_function": claim.get("shot_function"),
                "shot_scale": claim.get("shot_scale"),
                "action_type": claim.get("action_type"),
                "emotional_tone": claim.get("emotional_tone"),
                "narrative_role": claim.get("narrative_role")
                    or claim.get("proposed_role_v2"),
                "importance": claim.get("importance"),
                "motion_amount": claim.get("motion_amount"),
            }
            if not any(value not in (None, "") for value in summary.values()):
                continue
            candidate_index_by_identity[identity] = len(semantics)
            semantic_by_identity[identity] = item
            semantics.append(summary)
            asset_ids.append(item.project_asset_id)
            candidate_ref = _project_candidate_ref(*identity)
            candidate_refs.append(candidate_ref)
            source_evidence_by_candidate_ref[candidate_ref] = {
                "project_asset_id": item.project_asset_id,
                "source_media_hash": item.media_hash.lower(),
                "source_asset_id": item.media_asset_id,
                "observation_id": item.observation_id,
            }

        if len(set(asset_ids)) < 2:
            raise EvidenceTooPoorError(
                "project narrative reasoning requires usable semantic evidence "
                "from at least two locally processable source assets"
            )
        if len(candidate_refs) < 2:
            raise EvidenceTooPoorError(
                "project narrative reasoning requires at least two exact source shots"
            )

        caller_asserted_link_hints: list[dict] = []
        caller_asserted_link_hint_ids_supplied: list[str] = []
        review_link_count = (
            len(project_story_link_review.links)
            if project_story_link_review is not None else 0
        )
        if project_story_link_review is not None:
            observations_by_id = {item.observation_id: item for item in observations}
            for link in project_story_link_review.links:
                mapped_indices: set[int] = set()
                complete_mapping = True
                for anchor in link.anchors:
                    source_observation = observations_by_id.get(anchor.observation_id)
                    if source_observation is None:
                        complete_mapping = False
                        break
                    identity = (
                        anchor.project_asset_id,
                        anchor.source_content_hash.lower(),
                        source_observation.media_asset_id,
                    )
                    candidate_index = candidate_index_by_identity.get(identity)
                    semantic_observation = semantic_by_identity.get(identity)
                    if (
                        candidate_index is None
                        or semantic_observation is None
                        or anchor.source_start >= semantic_observation.end_frame
                        or semantic_observation.start_frame >= anchor.source_end
                    ):
                        complete_mapping = False
                        break
                    mapped_indices.add(candidate_index)
                if (
                    complete_mapping
                    and len(mapped_indices) >= 2
                    and len({asset_ids[index] for index in mapped_indices}) >= 2
                ):
                    caller_asserted_link_hints.append({
                        "entity_kind": link.entity_kind,
                        "display_label": link.display_label,
                        "shot_indices": sorted(mapped_indices),
                    })
                    caller_asserted_link_hint_ids_supplied.append(link.link_id)

        brief_context = json.dumps({
            # Asset-derived summaries are passed only through the evidence-bound
            # semantics list below, not duplicated here.
            "source_text": brief.source_text,
            "creator_direction": brief.creator_direction,
            "intent": brief.intent,
            "audience": brief.audience,
            "emotional_arc": brief.emotional_arc,
            "editing_language": brief.editing_language,
            "sound_language": brief.sound_language,
            "must_include": brief.must_include,
            "must_avoid": brief.must_avoid,
            "privacy_constraints": brief.privacy_constraints,
            "target_duration_us": brief.target_duration,
        }, ensure_ascii=False)
        narrative = analyze_project_narrative(
            semantics,
            asset_ids=asset_ids,
            shot_ids=candidate_refs,
            base_url=self.base_url,
            model=self.model,
            api_key="ollama-local",
            timeout=self.timeout,
            temperature=self.temperature,
            director_brief=brief_context,
            caller_asserted_links=caller_asserted_link_hints,
            include_audio_style_choice=include_audio_style_choice,
            include_editing_language_choice=include_editing_language_choice,
            include_transition_policy_choice=include_transition_policy_choice,
            semantic_constraints=(
                unresolved_constraint_review_items(brief) or None),
            runtime_binding_verifier=lambda: self._verify_model_binding(
                required=require_model_binding),
        )
        narrative["caller_asserted_link_review_link_count"] = review_link_count
        narrative["caller_asserted_link_hints_supplied"] = len(
            caller_asserted_link_hint_ids_supplied)
        narrative["caller_asserted_link_hint_ids_supplied"] = (
            caller_asserted_link_hint_ids_supplied)
        supplied_link_ids = set(caller_asserted_link_hint_ids_supplied)
        narrative["caller_asserted_link_hint_ids_omitted"] = [
            link.link_id for link in (
                project_story_link_review.links
                if project_story_link_review is not None else [])
            if link.link_id not in supplied_link_ids
        ]
        narrative["caller_asserted_link_provenance_state"] = "captured"
        narrative["_source_evidence_by_candidate_ref"] = (
            source_evidence_by_candidate_ref
        )
        return narrative

    def generate_project_shadow_plan(
        self,
        brief: DirectorBrief,
        manifest: FilmProjectManifest,
        context: FilmContextSnapshot,
        project_graph: ProjectStoryGraph,
        observations: list[FilmObservation],
        *,
        transition_policy: str = "none",
        card: ShotCard | None = None,
        voice_led: bool = False,
        audio_style: str = "none",
        pacing_style: str = "brief",
        project_story_link_review: ProjectStoryLinkReview | None = None,
        project_story_mention_review: ProjectStoryMentionReview | None = None,
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        """Generate a non-confirmable, local multi-asset reasoning comparison."""
        if audio_style not in {"none", "j_cut", "l_cut"}:
            raise ValueError(
                "single-plan shadow generation requires a fixed audio style")
        if pacing_style != "brief":
            raise ValueError(
                "single-plan shadow generation requires the Brief pacing profile")
        baseline_edl, _baseline_plan = HeuristicDirectorReasoner().generate_project_plan(
            brief,
            manifest,
            context,
            project_graph,
            observations,
            transition_policy=transition_policy,
            card=card,
            voice_led=voice_led,
            audio_style=audio_style,
            project_story_link_review=project_story_link_review,
            project_story_mention_review=project_story_mention_review,
        )
        narrative = self._analyze_project(
            brief,
            manifest,
            context,
            project_graph,
            observations,
            project_story_link_review=project_story_link_review,
            project_story_mention_review=project_story_mention_review,
        )
        edl, plan = HeuristicDirectorReasoner()._generate_project_plan(
            brief,
            manifest,
            context,
            project_graph,
            observations,
            transition_policy=transition_policy,
            card=card,
            voice_led=voice_led,
            # Compare the historical fixed-offset baseline with a shadow
            # candidate driven only by exact same-source ASR spans.
            audio_style=(
                "none" if audio_style in {"j_cut", "l_cut"} else audio_style
            ),
            project_story_link_review=project_story_link_review,
            project_story_mention_review=project_story_mention_review,
            narrative=narrative,
        )
        edl, plan = _apply_project_shadow_audio_bridge(
            edl, plan, observations, audio_style)
        from director_brain.narrative_analyzer import PROJECT_NARRATIVE_PROMPT_VERSION

        plan.project_narrative_reasoning = _build_project_narrative_reasoning_trace(
            narrative,
            brief_id=brief.brief_id,
            model=self.model,
            temperature=self.temperature,
            prompt_version=PROJECT_NARRATIVE_PROMPT_VERSION,
            semantic_constraints=(
                unresolved_constraint_review_items(brief) or None),
        )

        sequence_changed = [
            _edit_identity(item) for item in baseline_edl.ordered_edits
        ] != [_edit_identity(item) for item in edl.ordered_edits]
        mode = (
            "llm_project_director_reasoner_shadow_v0.5"
            if audio_style in {"j_cut", "l_cut"}
            else "llm_project_director_reasoner_shadow_v0.3"
        )
        edl.producer = mode
        plan.producer = mode
        plan.constraints.extend([
            "director_reasoner_pathway=director_strategy_reasoning:SHADOW",
            f"director_reasoner_provider=ollama:{self.model}",
            f"director_reasoner_prompt_version={PROJECT_NARRATIVE_PROMPT_VERSION}",
            f"director_reasoner_temperature={self.temperature:g}",
            "director_reasoner_scope=project_editorial_order_per_asset_clocks",
            "cross_asset_identity_and_relation_inference=not_attempted",
            "director_reasoner_shadow_candidate=not_confirmable",
            f"director_reasoner_heuristic_sequence_changed={str(sequence_changed).lower()}",
            "director_reasoner_model_relationship_claims_dropped="
            f"{narrative.get('project_relationship_claims_dropped', 0)}",
            "director_reasoner_caller_asserted_link_hints_supplied="
            f"{narrative.get('caller_asserted_link_hints_supplied', 0)}",
            "director_reasoner_caller_asserted_link_hints_omitted="
            f"{len(narrative.get('caller_asserted_link_hint_ids_omitted', []))}",
        ])
        plan.open_questions.extend([
            "director_reasoner_quality_not_proven",
            "director_reasoner_independent_project_validation_not_run",
            "director_reasoner_cross_asset_identity_not_inferred",
            "semantic_reasoner_shadow_candidate_not_for_confirmation",
        ])
        plan.open_questions = list(dict.fromkeys(plan.open_questions))
        return edl, plan

    def generate_project_shadow_strategy_options(
        self,
        brief: DirectorBrief,
        manifest: FilmProjectManifest,
        context: FilmContextSnapshot,
        project_graph: ProjectStoryGraph,
        observations: list[FilmObservation],
        *,
        transition_policy: str = "none",
        card: ShotCard | None = None,
        voice_led: bool = False,
        audio_style: str = "none",
        pacing_style: str = "brief",
        project_story_link_review: ProjectStoryLinkReview | None = None,
        project_story_mention_review: ProjectStoryMentionReview | None = None,
        _require_model_binding: bool = False,
    ) -> ProjectDirectorShadowComparison:
        """Return unranked story-strategy Plan/EDL candidates for local review.

        The heuristic baseline remains available for comparison. The model never
        chooses a winner; duplicate final EDL sequences stay visible but are
        marked as structurally indistinct so they cannot be described as separate
        editing strategies.
        """
        if audio_style not in {"none", "j_cut", "l_cut", "strategy"}:
            raise ValueError("project strategy audio style is invalid")
        if pacing_style not in {"brief", "strategy"}:
            raise ValueError("project strategy pacing style is invalid")
        if transition_policy not in {
            "none", "dissolve_act_boundary", "strategy",
        }:
            raise ValueError("project strategy transition policy is invalid")
        if (transition_policy == "strategy" and card is not None
                and card.transition_policy != "none"):
            raise ValueError(
                "strategy transition policy cannot override a declared ShotCard policy")
        if (pacing_style == "strategy"
                and brief.editing_language in EDITING_LANGUAGE_PROFILE_IDS):
            raise ValueError(
                "strategy pacing cannot override a declared Brief editing language")
        strategy_audio_choice = audio_style == "strategy"
        strategy_editing_language_choice = pacing_style == "strategy"
        strategy_transition_policy_choice = transition_policy == "strategy"
        baseline_audio_style = "none" if strategy_audio_choice else audio_style
        baseline_transition_policy = (
            "none" if strategy_transition_policy_choice else transition_policy)
        heuristic = HeuristicDirectorReasoner()
        baseline_edl, baseline_plan = heuristic.generate_project_plan(
            brief,
            manifest,
            context,
            project_graph,
            observations,
            transition_policy=baseline_transition_policy,
            card=card,
            voice_led=voice_led,
            audio_style=baseline_audio_style,
            project_story_link_review=project_story_link_review,
            project_story_mention_review=project_story_mention_review,
        )
        narrative = self._analyze_project(
            brief,
            manifest,
            context,
            project_graph,
            observations,
            project_story_link_review=project_story_link_review,
            project_story_mention_review=project_story_mention_review,
            include_audio_style_choice=strategy_audio_choice,
            include_editing_language_choice=strategy_editing_language_choice,
            include_transition_policy_choice=strategy_transition_policy_choice,
            require_model_binding=_require_model_binding,
        )
        options = narrative.get("strategy_hypotheses_resolved")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("project reasoner did not return multiple strategy hypotheses")
        from director_brain.narrative_analyzer import PROJECT_NARRATIVE_PROMPT_VERSION

        candidates: list[dict] = []
        for option in options:
            option_id = option.get("hypothesis_id")
            if not isinstance(option_id, str) or not option_id:
                raise ValueError("project strategy hypothesis ID is invalid")
            candidate_audio_style = audio_style
            if strategy_audio_choice:
                candidate_audio_style = option.get("audio_style_choice")
                if candidate_audio_style not in {"none", "j_cut", "l_cut"}:
                    raise ValueError(
                        "strategy hypothesis has no valid audio style choice")
            candidate_transition_policy = transition_policy
            candidate_transition_duration_us = None
            if strategy_transition_policy_choice:
                candidate_transition_policy = option.get(
                    "transition_policy_choice")
                if candidate_transition_policy not in {
                    "none", "dissolve_act_boundary",
                }:
                    raise ValueError(
                        "strategy hypothesis has no valid transition policy choice")
                candidate_transition_duration_us = (
                    option.get("transition_duration_us") or None)
            candidate_editing_language = None
            candidate_brief = brief
            if strategy_editing_language_choice:
                candidate_editing_language = option.get(
                    "editing_language_choice")
                if candidate_editing_language not in EDITING_LANGUAGE_PROFILE_IDS:
                    raise ValueError(
                        "strategy hypothesis has no valid editing language choice")
                candidate_brief = brief.model_copy(update={
                    "editing_language": candidate_editing_language,
                })
            option_narrative = copy.deepcopy(narrative)
            option_narrative["suggested_order_resolved"] = list(
                option["suggested_order_resolved"])
            option_narrative["act_boundaries_resolved"] = copy.deepcopy(
                option["act_boundaries_resolved"])
            option_narrative["_active_strategy_hypothesis_id"] = option_id
            option_narrative["_active_source_dispositions"] = {
                item["focus_source_ref"]: item["disposition"]
                for item in option["source_rationales_resolved"]
            }
            selection_reason_events: dict = {}
            edl, plan = heuristic._generate_project_plan(
                candidate_brief,
                manifest,
                context,
                project_graph,
                observations,
                transition_policy=candidate_transition_policy,
                transition_duration_us=candidate_transition_duration_us,
                card=card,
                voice_led=voice_led,
                # Keep the baseline's historical offset. The SHADOW Plan
                # receives source-bound offsets only after materialization.
                audio_style=(
                    "none" if candidate_audio_style in {"j_cut", "l_cut"}
                    else candidate_audio_style
                ),
                project_story_link_review=project_story_link_review,
                project_story_mention_review=project_story_mention_review,
                narrative=option_narrative,
                narrative_order_priority=True,
                artifact_variant=f"project-shadow-strategy:{option_id}",
                selection_reason_events=selection_reason_events,
            )
            edl, plan = _apply_project_shadow_audio_bridge(
                edl, plan, observations, candidate_audio_style)
            plan.project_narrative_reasoning = (
                _build_project_narrative_reasoning_trace(
                    option_narrative,
                    brief_id=brief.brief_id,
                    model=self.model,
                    temperature=self.temperature,
                    prompt_version=PROJECT_NARRATIVE_PROMPT_VERSION,
                    semantic_constraints=(
                        unresolved_constraint_review_items(brief) or None),
                    candidate_edl_source_assets={
                        _edit_identity(edit) for edit in edl.ordered_edits
                    },
                )
            )
            sequence_changed = (
                [_edit_identity(item) for item in baseline_edl.ordered_edits]
                != [_edit_identity(item) for item in edl.ordered_edits]
            )
            mode = "llm_project_director_reasoner_shadow_v0.6"
            if strategy_audio_choice:
                mode = "llm_project_director_reasoner_shadow_v0.7"
            if strategy_editing_language_choice:
                mode = "llm_project_director_reasoner_shadow_v0.8"
            if strategy_transition_policy_choice:
                mode = "llm_project_director_reasoner_shadow_v1.0"
            edl.producer = mode
            plan.producer = mode
            plan.constraints.extend([
                "director_reasoner_pathway=director_strategy_reasoning:SHADOW",
                f"director_reasoner_provider=ollama:{self.model}",
                f"director_reasoner_prompt_version={PROJECT_NARRATIVE_PROMPT_VERSION}",
                f"director_reasoner_strategy_hypothesis={option_id}",
                *([f"director_reasoner_strategy_audio_style={candidate_audio_style}"]
                  if strategy_audio_choice else []),
                *([f"director_reasoner_strategy_editing_language="
                   f"{candidate_editing_language}"]
                  if strategy_editing_language_choice else []),
                *([f"director_reasoner_strategy_transition_policy="
                   f"{candidate_transition_policy}"]
                  if strategy_transition_policy_choice else []),
                *([f"director_reasoner_strategy_transition_duration_us="
                   f"{candidate_transition_duration_us}"]
                  if (strategy_transition_policy_choice
                      and candidate_transition_duration_us is not None) else []),
                "director_reasoner_strategy_options=unranked",
                "director_reasoner_selection=validated_model_order_rank_within_eligible_candidates",
                "director_reasoner_shadow_candidate=not_confirmable",
                f"director_reasoner_heuristic_sequence_changed={str(sequence_changed).lower()}",
                "cross_asset_identity_and_relation_inference=not_attempted",
            ])
            plan.open_questions.extend([
                "director_reasoner_quality_not_proven",
                "director_reasoner_independent_project_validation_not_run",
                "director_reasoner_cross_asset_identity_not_inferred",
                "semantic_reasoner_shadow_candidate_not_for_confirmation",
            ])
            plan.open_questions = list(dict.fromkeys(plan.open_questions))
            source_sequence = tuple(
                _edit_identity(edit) for edit in edl.ordered_edits
            )
            candidates.append({
                "hypothesis": option,
                "edl": edl,
                "plan": plan,
                "selection_reason_events": selection_reason_events,
                "source_sequence": source_sequence,
                "edl_execution_signature": _edl_execution_signature(edl),
                "sequence_changed": source_sequence != tuple(
                    _edit_identity(edit)
                    for edit in baseline_edl.ordered_edits
                ),
            })

        execution_signatures = [
            item["edl_execution_signature"] for item in candidates
        ]
        if len(execution_signatures) != len(set(execution_signatures)):
            failure = LLMStructuredOutputError(
                "project strategy candidates did not produce distinct EDL execution sequences",
                failure_code="project_edl_candidates_identical",
            )
            failure.failure_stage = "candidate_materialization"
            failure.provider_call_count = len(
                narrative.get("provider_call_provenance", []))
            raise failure
        execution_signature_counts = {
            signature: execution_signatures.count(signature)
            for signature in set(execution_signatures)
        }
        baseline_execution_signature = _edl_execution_signature(baseline_edl)
        typed_candidates = []
        for item in candidates:
            hypothesis = item["hypothesis"]
            trace = item["plan"].project_narrative_reasoning
            if trace is None:
                raise ValueError("project strategy Plan lost its narrative trace")
            typed_hypothesis = next(
                (candidate for candidate in trace.strategy_hypotheses
                 if candidate.hypothesis_id == hypothesis["hypothesis_id"]),
                None,
            )
            if typed_hypothesis is None:
                raise ValueError("project strategy Plan lost its exact hypothesis")
            selected_edl_sources = {
                _edit_identity(edit) for edit in item["edl"].ordered_edits
            }
            disposition_by_source = {
                (
                    rationale.focus_source.project_asset_id,
                    rationale.focus_source.source_media_hash.lower(),
                    rationale.focus_source.source_asset_id,
                ): rationale.disposition
                for rationale in typed_hypothesis.source_rationales
            }
            selection_audit = []
            for rank, source in enumerate(typed_hypothesis.ordered_sources):
                source_identity = (
                    source.project_asset_id,
                    source.source_media_hash.lower(),
                    source.source_asset_id,
                )
                selected = source_identity in selected_edl_sources
                reason_codes = (
                    ["present_in_candidate_edl"]
                    if selected
                    else sorted(item["selection_reason_events"].get(
                        source_identity, set()))
                )
                disposition = disposition_by_source.get(source_identity)
                if disposition == "exclude":
                    reason_codes = sorted(set(reason_codes) | {
                        "excluded_by_director_strategy",
                    })
                if not reason_codes:
                    raise ValueError(
                        "non-selected strategy source lacks a retained planner reason")
                selection_audit.append(ProjectDirectorSourceSelectionAuditEntry(
                    strategy_rank=rank,
                    strategy_disposition=disposition,
                    selected_in_edl=selected,
                    reason_codes=reason_codes,
                ))
            typed_candidates.append(ProjectDirectorStrategyCandidate(
                hypothesis=typed_hypothesis,
                candidate_binding_digest=(
                    compute_project_strategy_candidate_binding_digest(
                        brief,
                        manifest,
                        context,
                        project_graph,
                        typed_hypothesis,
                        item["edl"],
                        item["plan"],
                        project_story_link_review=project_story_link_review,
                        project_story_mention_review=project_story_mention_review,
                    )
                ),
                edl=item["edl"],
                plan=item["plan"],
                selection_audit=selection_audit,
                sequence_changed_vs_heuristic=item["sequence_changed"],
                distinct_edl_sequence_from_baseline=(
                    item["edl_execution_signature"]
                    != baseline_execution_signature),
                distinct_edl_sequence_from_other_candidates=(
                    execution_signature_counts[
                        item["edl_execution_signature"]
                    ] == 1),
            ))

        return ProjectDirectorShadowComparison(
            baseline_edl=baseline_edl,
            baseline_plan=baseline_plan,
            strategy_candidates=typed_candidates,
        )

    def generate_project_plan(
        self,
        brief: DirectorBrief,
        manifest: FilmProjectManifest,
        context: FilmContextSnapshot,
        project_graph: ProjectStoryGraph,
        observations: list[FilmObservation],
        *,
        selected_strategy_hypothesis_id: str,
        selected_candidate_binding_digest: str,
        transition_policy: str = "none",
        card: ShotCard | None = None,
        voice_led: bool = False,
        audio_style: str = "none",
        pacing_style: str = "brief",
        project_story_link_review: ProjectStoryLinkReview | None = None,
        project_story_mention_review: ProjectStoryMentionReview | None = None,
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        """Materialize only an explicitly selected, freshly bound strategy.

        The pathway gate runs before provider work. Even after admission, this
        method does not select a winner: the caller must present the hypothesis
        ID and digest from a reviewed comparison, and the regenerated candidate
        must match both exactly.
        """
        ensure_decision_use_allowed("director_strategy_reasoning")
        if (not selected_strategy_hypothesis_id
                or len(selected_candidate_binding_digest) != 64
                or any(ch not in "0123456789abcdef"
                       for ch in selected_candidate_binding_digest)):
            raise ValueError("explicit project strategy selection is invalid")

        comparison = self.generate_project_shadow_strategy_options(
            brief,
            manifest,
            context,
            project_graph,
            observations,
            transition_policy=transition_policy,
            card=card,
            voice_led=voice_led,
            audio_style=audio_style,
            pacing_style=pacing_style,
            project_story_link_review=project_story_link_review,
            project_story_mention_review=project_story_mention_review,
            _require_model_binding=True,
        )
        selected = [
            candidate for candidate in comparison.strategy_candidates
            if candidate.hypothesis.hypothesis_id
            == selected_strategy_hypothesis_id
        ]
        if (len(selected) != 1
                or selected[0].candidate_binding_digest
                != selected_candidate_binding_digest):
            raise ValueError(
                "selected project strategy candidate is stale or unavailable")

        return self.materialize_project_plan_from_candidate(
            brief,
            manifest,
            context,
            project_graph,
            selected[0],
            selected_strategy_hypothesis_id=selected_strategy_hypothesis_id,
            selected_candidate_binding_digest=selected_candidate_binding_digest,
            project_story_link_review=project_story_link_review,
            project_story_mention_review=project_story_mention_review,
        )

    def materialize_project_plan_from_candidate(
        self,
        brief: DirectorBrief,
        manifest: FilmProjectManifest,
        context: FilmContextSnapshot,
        project_graph: ProjectStoryGraph,
        candidate: ProjectDirectorStrategyCandidate,
        *,
        selected_strategy_hypothesis_id: str,
        selected_candidate_binding_digest: str,
        project_story_link_review: ProjectStoryLinkReview | None = None,
        project_story_mention_review: ProjectStoryMentionReview | None = None,
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        """Materialize one exact saved candidate after the pathway gate.

        This path performs no provider call. The candidate is re-bound to the
        current project, review, Brief, Plan and EDL snapshots, then checked
        against the current verified local model/runtime pin before use.
        """
        ensure_decision_use_allowed("director_strategy_reasoning")
        if (candidate.hypothesis.hypothesis_id != selected_strategy_hypothesis_id
                or candidate.candidate_binding_digest
                != selected_candidate_binding_digest):
            raise ValueError("selected project strategy candidate is stale or unavailable")
        current_digest = compute_project_strategy_candidate_binding_digest(
            brief,
            manifest,
            context,
            project_graph,
            candidate.hypothesis,
            candidate.edl,
            candidate.plan,
            project_story_link_review=project_story_link_review,
            project_story_mention_review=project_story_mention_review,
        )
        if current_digest != selected_candidate_binding_digest:
            raise ValueError("selected project strategy candidate is stale or unavailable")

        edl = candidate.edl.model_copy(deep=True)
        plan = candidate.plan.model_copy(deep=True)
        suffix = short_hash(
            f"{brief.brief_id}:{selected_strategy_hypothesis_id}:"
            f"{selected_candidate_binding_digest}"
        )
        edl.edl_id = f"edl_{brief.project_id}_{suffix}"
        edl.producer = "llm_project_director_reasoner_active_v0.2"
        plan.plan_id = f"plan_{brief.project_id}_{suffix}"
        plan.edl_id = edl.edl_id
        plan.producer = edl.producer
        if plan.project_narrative_reasoning is None:
            raise ValueError("selected project strategy lost its narrative trace")
        call_provenance = (
            plan.project_narrative_reasoning.provider_call_provenance)
        if (
            not call_provenance
            or any(
                item.runtime_binding_state != "verified"
                for item in call_provenance
            )
        ):
            raise ValueError(
                "selected project strategy candidate lacks a verified Ollama model binding")
        current_runtime_binding = self._verify_model_binding(required=True)
        if any(
            item.verified_model_digest != current_runtime_binding["model_digest"]
            or item.verified_runtime_version
            != current_runtime_binding["runtime_version"]
            for item in call_provenance
        ):
            raise ValueError(
                "selected project strategy candidate is stale or unavailable")
        plan.project_narrative_reasoning = plan.project_narrative_reasoning.model_copy(
            update={"state": "ACTIVE_PATHWAY_UNVERIFIED"}
        )
        plan.constraints = [
            item for item in plan.constraints
            if item not in {
                "director_reasoner_pathway=director_strategy_reasoning:SHADOW",
                "director_reasoner_shadow_candidate=not_confirmable",
            }
        ]
        plan.constraints.extend([
            "director_reasoner_pathway=director_strategy_reasoning:ACTIVE",
            "director_reasoner_selection=caller_selected_bound_hypothesis",
            "director_reasoner_candidate_binding_sha256="
            f"{selected_candidate_binding_digest}",
        ])
        plan.open_questions = [
            item for item in plan.open_questions
            if item != "semantic_reasoner_shadow_candidate_not_for_confirmation"
        ]
        plan.open_questions.extend([
            "director_reasoner_quality_not_proven",
            "director_reasoner_independent_project_validation_not_run",
        ])
        plan.open_questions = list(dict.fromkeys(plan.open_questions))
        return edl, plan

    def _generate(
        self,
        brief: DirectorBrief,
        graph: StoryGraph,
        observations: list[FilmObservation],
        *,
        shadow: bool,
        transition_policy: str,
        transition_duration_us: int | None,
        card: ShotCard | None,
        entities,
        voice_led: bool,
        beat_grid,
        audio_style: str,
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        if graph.timeline_scope == "project_aggregate":
            raise ValueError(
                "single-source LLM reasoning cannot consume a project aggregate graph"
            )
        narrative = self._analyze(brief, observations)
        edl, plan = HeuristicDirectorReasoner().generate_plan(
            brief,
            graph,
            observations,
            narrative=narrative,
            transition_policy=transition_policy,
            transition_duration_us=transition_duration_us,
            card=card,
            entities=entities,
            voice_led=voice_led,
            beat_grid=beat_grid,
            audio_style=audio_style,
        )
        mode = "SHADOW" if shadow else "ACTIVE"
        producer = f"llm_director_reasoner_{mode.lower()}_v0.1"
        from director_brain.narrative_analyzer import NARRATIVE_PROMPT_VERSION

        edl.producer = producer
        plan.producer = producer
        plan.constraints.extend([
            f"director_reasoner_pathway=director_strategy_reasoning:{mode}",
            f"director_reasoner_provider=ollama:{self.model}",
            f"director_reasoner_prompt_version={NARRATIVE_PROMPT_VERSION}",
            f"director_reasoner_temperature={self.temperature:g}",
            "director_reasoner_scope=single_source_narrative_ordering",
        ])
        plan.open_questions.extend([
            "director_reasoner_quality_not_proven",
            "director_reasoner_independent_project_validation_not_run",
        ])
        if shadow:
            plan.constraints.append("director_reasoner_shadow_candidate=not_confirmable")
            plan.open_questions.append(
                "semantic_reasoner_shadow_candidate_not_for_confirmation"
            )
        plan.open_questions = list(dict.fromkeys(plan.open_questions))
        return edl, plan

    def generate_plan(
        self,
        brief: DirectorBrief,
        graph: StoryGraph,
        observations: list[FilmObservation],
        *,
        narrative: dict | None = None,
        transition_policy: str = "none",
        transition_duration_us: int | None = None,
        card: ShotCard | None = None,
        entities=None,
        voice_led: bool = False,
        beat_grid=None,
        audio_style: str = "none",
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        ensure_decision_use_allowed("director_strategy_reasoning")
        if narrative is not None:
            raise ValueError(
                "LLMDirectorReasoner derives narrative only from supplied observations"
            )
        return self._generate(
            brief, graph, observations,
            shadow=False,
            transition_policy=transition_policy,
            transition_duration_us=transition_duration_us,
            card=card,
            entities=entities,
            voice_led=voice_led,
            beat_grid=beat_grid,
            audio_style=audio_style,
        )

    def generate_shadow_plan(
        self,
        brief: DirectorBrief,
        graph: StoryGraph,
        observations: list[FilmObservation],
        *,
        transition_policy: str = "none",
        transition_duration_us: int | None = None,
        card: ShotCard | None = None,
        entities=None,
        voice_led: bool = False,
        beat_grid=None,
        audio_style: str = "none",
    ) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
        """Return a local comparison artifact that cannot be strategy-confirmed."""
        return self._generate(
            brief, graph, observations,
            shadow=True,
            transition_policy=transition_policy,
            transition_duration_us=transition_duration_us,
            card=card,
            entities=entities,
            voice_led=voice_led,
            beat_grid=beat_grid,
            audio_style=audio_style,
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
            "LLM director reasoning is gated by director_strategy_reasoning admission; "
            "use generate_shadow_plan() only for non-confirmable local comparison.",
            stacklevel=2,
        )
        return LLMDirectorReasoner(**kwargs)
    factory = _REASONER_REGISTRY.get(strategy)
    if factory is None:
        raise ValueError(f"unknown director reasoner strategy: {strategy!r}")
    return factory(**kwargs)
