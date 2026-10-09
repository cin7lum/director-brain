"""02 导演脑 REST API（方案 §6 · 8 端点全量对齐）。

FastAPI 实现；每个请求携带 correlation_id + schema_version 信封；
有副作用端点接受 idempotency_key。

启动：uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import time
import uuid
import ipaddress
import hmac
import hashlib
import re
import subprocess
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, StrictInt, ValidationError, model_validator

from director_brain import (
    compile_brief,
    compile_project_brief,
    generate_plan,
    generate_project_plan,
    generate_project_shadow_strategy_options,
    materialize_project_plan_from_candidate,
    propose_revision,
    repair_plan,
    validate_plan,
)
from director_brain.context_gateway import (
    ContextGateway,
    build_asset_context,
    build_evidence_context,
    build_project_context,
    build_scene_context,
    project_local_asr_model_binding,
    validate_window_us,
)
from director_brain.plan_state import (
    PlanState,
    confirm_strategy,
    compute_edl_hash,
    compute_plan_hash,
    is_confirmation_valid,
    transition_plan,
    validate_transition,
)
from director_brain.llm_adapter import (
    LLMInputCapacityError,
    LLMStructuredOutputError,
    LLMTransportError,
)
from director_brain.director_reasoner import (
    EvidenceTooPoorError,
    PRODUCER as DIRECTOR_REASONER_PRODUCER,
    compute_project_strategy_candidate_binding_digest,
)
from director_brain.pathway_protocol import PathwayNotActiveError
from director_brain.narrative_analyzer import PROJECT_NARRATIVE_FAILURE_CODES
from director_brain.intent_constraints import interpret_constraints
from storage.project_director_shadow_runs import (
    ProjectDirectorShadowRunConflictError,
    ProjectDirectorShadowRunInProgressError,
    ProjectDirectorShadowRunLeaseLostError,
    ProjectDirectorShadowRunOutcomeUnknownError,
    ProjectDirectorShadowRunClaim,
    SqliteProjectDirectorShadowRunStore,
)
from director_brain.models.project import (
    FilmProjectManifest,
    ProcessingRights,
    ProjectAnalysisProfile,
    ProjectAsset,
    ProjectBoundaryBasis,
    RightsState,
)
from director_brain.models.project_director_shadow_comparison import (
    ProjectDirectorShadowComparisonRecord,
)
from director_brain.models.project_story_link_candidates import (
    ProjectStoryLinkComparisonRequestData,
)
from director_brain.models.project_story_mention_review import (
    ProjectStoryMentionDecision,
    ProjectStoryMentionReview,
)
from director_brain.models.project_story_mention_preview import (
    ProjectStoryMentionPreview,
    ProjectStoryMentionPreviewFrame,
)
from director_brain.models.project_story_graph import ProjectStoryGraph
from director_brain.models.project_story_link_candidate_preview import (
    ProjectStoryLinkCandidatePreview,
)
from director_brain.models.project_story_link_review import ProjectStoryLinkAnchor
from director_brain.project_story_link_ranking import (
    ProjectStoryLinkRankingError,
    RANKING_METRIC,
    RANKING_MODEL,
    RANKING_PROFILE_ID,
    RANKING_STATE,
    apply_project_story_link_ranking_snapshot,
    project_story_link_ranking_snapshot_entries,
    rank_project_story_link_candidates,
    ranked_candidate_set_id,
    resolve_project_story_link_ranking_binding,
)


_SAFE_PROJECT_REASONER_FAILURE_CODES = PROJECT_NARRATIVE_FAILURE_CODES
_UNCLASSIFIED_REASONER_VALUE_ERROR_CODE = (
    "reasoner_value_error_unclassified"
)
_SAFE_PROJECT_REASONER_API_FAILURE_CODES = (
    _SAFE_PROJECT_REASONER_FAILURE_CODES
    | frozenset({_UNCLASSIFIED_REASONER_VALUE_ERROR_CODE})
)
_SAFE_PROJECT_REASONER_TRANSPORT_CODES = frozenset({
    "provider_connection_error",
    "provider_configuration_error",
    "provider_empty_response",
    "provider_exchange_error",
    "provider_http_error",
    "provider_model_binding_error",
    "provider_transport_error",
})
_PROJECT_SHADOW_RUN_STATES = (
    "running",
    "recoverable",
    "outcome_unknown",
    "idempotency_conflict",
    "ownership_lost",
)


def _safe_project_reasoner_failure_headers(
    exc: LLMStructuredOutputError,
) -> dict[str, str] | None:
    """Expose only a fixed, non-semantic reasoner failure code to API clients."""
    code = exc.failure_code
    if not isinstance(code, str) or code not in _SAFE_PROJECT_REASONER_FAILURE_CODES:
        return None
    return {"X-Director-Brain-Failure-Code": code}


def _safe_project_reasoner_transport_headers(
    exc: LLMTransportError,
) -> dict[str, str] | None:
    """Expose only a fixed, non-semantic provider failure code."""
    code = exc.failure_code
    if not isinstance(code, str) or code not in _SAFE_PROJECT_REASONER_TRANSPORT_CODES:
        return None
    return {"X-Director-Brain-Failure-Code": code}


def _safe_project_reasoner_failure_diagnostics(exc: Exception) -> dict[str, Any]:
    """Project only bounded, allowlisted provider envelope fields into local receipts."""
    diagnostics: dict[str, Any] = {}
    failure_stage = getattr(exc, "failure_stage", None)
    if failure_stage in {
        "flat_project", "segment", "project_synthesis", "preflight",
        "input_capacity", "project_validation", "candidate_materialization",
    }:
        diagnostics["provider_failure_stage"] = failure_stage
    call_count = getattr(exc, "provider_call_count", None)
    if type(call_count) is int and 1 <= call_count <= 10_000:
        diagnostics["provider_call_count"] = call_count

    metadata = getattr(exc, "provider_response_metadata", None)
    if isinstance(metadata, dict):
        safe_metadata: dict[str, str | int] = {}
        for key in ("model", "system_fingerprint"):
            value = metadata.get(key)
            if (isinstance(value, str)
                    and re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}", value)):
                safe_metadata[key] = value
        finish_reason = metadata.get("finish_reason")
        if (isinstance(finish_reason, str)
                and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", finish_reason)):
            safe_metadata["finish_reason"] = finish_reason
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = metadata.get(key)
            if type(value) is int and 0 <= value <= 1_000_000_000:
                safe_metadata[key] = value
        if safe_metadata:
            diagnostics["provider_response_metadata"] = safe_metadata
    return diagnostics


_PROJECT_REASONER_FAILURE_RESPONSES = {
    502: {
        "description": "Project Director Reasoner structured output was invalid.",
        "headers": {
            "X-Director-Brain-Failure-Code": {
                "description": (
                    "Optional allowlisted non-semantic validation code; "
                    "raw provider output is never returned."
                ),
                "schema": {
                    "type": "string",
                    "enum": sorted(_SAFE_PROJECT_REASONER_FAILURE_CODES),
                },
            },
        },
    },
    503: {
        "description": "Local Project Director Reasoner provider is unavailable.",
        "headers": {
            "X-Director-Brain-Failure-Code": {
                "description": (
                    "Optional allowlisted provider transport code; raw provider "
                    "details and response content are never returned."
                ),
                "schema": {
                    "type": "string",
                    "enum": sorted(_SAFE_PROJECT_REASONER_TRANSPORT_CODES),
                },
            },
        },
    },
}
_PROJECT_SHADOW_COMPARISON_RESPONSES = {
    **_PROJECT_REASONER_FAILURE_RESPONSES,
    500: {
        "description": (
            "The local project comparison failed with an unclassified error."
        ),
        "headers": {
            "X-Director-Brain-Failure-Code": {
                "description": "Fixed code for an unclassified ValueError.",
                "schema": {
                    "type": "string",
                    "enum": [_UNCLASSIFIED_REASONER_VALUE_ERROR_CODE],
                },
            },
        },
    },
    409: {
        "description": (
            "The idempotency key conflicts, is still running, or has an "
            "uncertain prior provider outcome."
        ),
        "headers": {
            "Retry-After": {
                "description": (
                    "Present while the same-key comparison remains in progress."
                ),
                "schema": {"type": "integer"},
            },
            "X-Director-Brain-Run-State": {
                "description": (
                    "Fixed lifecycle state for a local SHADOW comparison request."
                ),
                "schema": {
                    "type": "string",
                    "enum": [
                        "running", "recoverable", "outcome_unknown",
                        "idempotency_conflict", "ownership_lost",
                    ],
                },
            },
        },
    },
}


app = FastAPI(
    title="Director Brain API",
    version="1.0.0",
    description="02 导演脑 · 可解释可追溯可修订的导演决策 REST 接口",
)

# log_level 配置接线（成品级扫荡：此前 LOG_LEVEL 配置零消费）
import logging as _logging  # noqa: E402

try:
    from director_brain.config import load_settings as _load_settings
    _logging.basicConfig(level=_load_settings().log_level)
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# 信封
# ---------------------------------------------------------------------------

def _envelope(correlation_id: str, data: Any) -> dict:
    return {
        "correlation_id": correlation_id,
        "schema_version": "1.0",
        "timestamp": int(time.time()),
        "data": data,
    }


def _new_correlation_id() -> str:
    return f"corr_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class CompileBriefRequest(BaseModel):
    project_id: str
    video_path: str
    intent_text: str | None = None
    target_duration_us: int = 15_000_000
    observations_json: list[dict] | None = None  # 可选：外部观测注入


class GeneratePlanRequest(BaseModel):
    project_id: str
    video_path: str
    brief_id: str | None = None
    intent_text: str | None = None
    target_duration_us: int = 15_000_000
    observations_json: list[dict] | None = None


class ConfirmStrategyRequest(BaseModel):
    plan_id: str
    edl_id: str
    plan_json: dict
    edl_json: dict
    confirmed_by: str = "user"
    output_target: str = "delivery"
    notes: str = ""


class ProjectStrategyConfirmationRequest(BaseModel):
    expected_plan_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    expected_edl_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    idempotency_key: str = Field(min_length=1, max_length=128)
    confirmed_by: str = Field(min_length=1, max_length=200)
    output_target: Literal["preview", "delivery"] = "delivery"
    notes: str = Field(default="", max_length=2000)


class ProjectStrategyRejectionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    expected_plan_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    expected_edl_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    idempotency_key: str = Field(min_length=1, max_length=128)
    rejected_by: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_nonblank_rejection_fields(self):
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be blank")
        if not self.rejected_by.strip():
            raise ValueError("rejected_by must not be blank")
        if not self.reason.strip():
            raise ValueError("reason must not be blank")
        return self


class ProjectConstraintAssessmentRejectionRequest(BaseModel):
    """Caller disposition of one exact, unverified SHADOW constraint suggestion."""

    model_config = {"extra": "forbid"}

    candidate_binding_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypothesis_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}$")
    constraint_kind: Literal["must_include", "must_avoid"]
    brief_index: StrictInt = Field(ge=0)
    expected_assessment: Literal["candidate_supported", "candidate_conflicted"]
    idempotency_key: str = Field(min_length=1, max_length=128)
    rejected_by: str = Field(min_length=1, max_length=200)
    reason_code: Literal[
        "source_evidence_insufficient", "misread_constraint", "other",
    ]

    @model_validator(mode="after")
    def validate_nonblank_rejection_identity(self):
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be blank")
        if not self.rejected_by.strip():
            raise ValueError("rejected_by must not be blank")
        return self


class ValidateRequest(BaseModel):
    plan_json: dict
    edl_json: dict
    observations_json: list[dict]


class RevisionRequest(BaseModel):
    """spec §6：POST /v1/revisions:propose——输入 findings，输出新的
    EDL/策略候选，**不直接执行**（候选⑦：端点此前返回硬编码 DRAFT）。"""

    finding_ids: list[str] = Field(default_factory=list)
    target_decision_ids: list[str] = Field(default_factory=list)
    change_summary: str = ""
    revision_type: str = "adjust_duration"
    plan_json: dict | None = None
    edl_json: dict | None = None
    observations_json: list[dict] | None = None


class FilmContextSnapshotRequest(BaseModel):
    """方案 §6 第一条：POST /v1/film-context:snapshot 的请求体。"""

    project_id: str
    video_path: str
    layer: str = "asset"  # project | asset | scene | evidence
    intent_text: str | None = None  # project/scene 层编译 Brief 用
    target_duration_us: int = 15_000_000
    observations_json: list[dict] | None = None  # 缺省时走 analyze_media
    shot_ids: list[str] = Field(default_factory=list)  # evidence 层显式镜头
    window_us: list[int] | None = None  # scene/evidence 层时间窗 [start, end]
    reason: str = ""  # evidence 层必须（审计留痕）


class ProjectManifestAssetRequest(BaseModel):
    """One source asset and its explicit processing-route rights declaration."""

    model_config = {"extra": "forbid"}

    asset_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    order: int = Field(ge=0)
    rights: ProcessingRights = Field(default_factory=ProcessingRights)


class ProjectManifestWriteRequest(BaseModel):
    """Append an immutable project asset manifest revision."""

    model_config = {"extra": "forbid"}

    expected_revision: int = Field(ge=0)
    boundary_basis: ProjectBoundaryBasis
    boundary_source_ref: str = Field(min_length=1)
    boundary_evidence_refs: list[str] = Field(min_length=1)
    analysis_profile: ProjectAnalysisProfile = ProjectAnalysisProfile.DETERMINISTIC_V1
    assets: list[ProjectManifestAssetRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_asset_identifiers_before_ingest(self):
        asset_ids = [asset.asset_id for asset in self.assets]
        orders = [asset.order for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("project manifest asset_id values must be unique")
        if len(orders) != len(set(orders)):
            raise ValueError("project manifest asset order values must be unique")
        return self


class ProjectContextSnapshotRequest(BaseModel):
    """Build or reuse a project context for one exact manifest revision."""

    model_config = {"extra": "forbid"}

    manifest_revision: int = Field(ge=1)
    observations_by_asset: dict[str, list[dict]] | None = None


class ProjectContextJobRequest(BaseModel):
    """Submit one durable analysis for an exact current manifest revision."""

    model_config = {"extra": "forbid"}

    manifest_revision: int = Field(ge=1)


class ProjectPlanInputs(BaseModel):
    """Inputs shared by project-plan generation and SHADOW comparison."""

    model_config = {"extra": "forbid"}

    manifest_revision: int = Field(ge=1)
    target_duration_us: int = Field(default=15_000_000, gt=0)
    intent_text: str = Field(min_length=1)
    voice_led: bool = False
    audio_style: Literal["none", "j_cut", "l_cut", "strategy"] = "none"
    pacing_style: Literal["brief", "strategy"] = "brief"
    transition_policy: Literal[
        "none", "dissolve_act_boundary", "strategy",
    ] = "none"


class ProjectPlanRequest(ProjectPlanInputs):
    """Generate a Plan candidate from one exact persisted evidence revision."""

    reasoner_strategy: Literal["heuristic", "llm"] = "heuristic"
    selected_strategy_hypothesis_id: str | None = Field(default=None, min_length=1)
    selected_comparison_id: str | None = Field(default=None, min_length=1)
    selected_candidate_binding_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$")
    supersedes_plan_id: str | None = Field(default=None, min_length=1)
    expected_superseded_plan_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{16}$")
    expected_superseded_edl_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{16}$")
    clarified_by: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_explicit_reasoner_selection(self):
        if self.audio_style == "strategy" and self.reasoner_strategy != "llm":
            raise ValueError(
                "strategy-selected audio is available only for LLM strategy Plans")
        if self.pacing_style == "strategy" and self.reasoner_strategy != "llm":
            raise ValueError(
                "strategy-selected pacing is available only for LLM strategy Plans")
        if (self.transition_policy == "strategy"
                and self.reasoner_strategy != "llm"):
            raise ValueError(
                "strategy-selected transitions are available only for LLM strategy Plans")
        selection_supplied = (
            self.selected_strategy_hypothesis_id is not None
            or self.selected_candidate_binding_digest is not None
            or self.selected_comparison_id is not None
        )
        if self.reasoner_strategy == "llm":
            if (self.selected_strategy_hypothesis_id is None
                    or self.selected_candidate_binding_digest is None
                    or self.selected_comparison_id is None):
                raise ValueError(
                    "LLM project planning requires a saved comparison ID, "
                    "hypothesis ID and candidate binding digest")
        elif selection_supplied:
            raise ValueError(
                "hypothesis selection is only valid for the LLM project reasoner")
        supersession_supplied = (
            self.supersedes_plan_id is not None
            or self.expected_superseded_plan_hash is not None
            or self.expected_superseded_edl_hash is not None
            or self.clarified_by is not None
        )
        if supersession_supplied and (
            self.supersedes_plan_id is None
            or self.expected_superseded_plan_hash is None
            or self.expected_superseded_edl_hash is None
            or self.clarified_by is None
        ):
            raise ValueError(
                "Plan supersession requires the prior Plan ID, both reviewed hashes, and caller")
        if self.supersedes_plan_id is not None and not self.supersedes_plan_id.strip():
            raise ValueError("supersedes_plan_id must not be blank")
        if self.clarified_by is not None and not self.clarified_by.strip():
            raise ValueError("clarified_by must not be blank")
        return self


class ProjectShadowStrategyComparisonRequest(ProjectPlanInputs):
    """Request for an unranked, locally persisted, non-confirmable comparison."""

    idempotency_key: str = Field(min_length=1, max_length=128)


class ProjectStoryLinkAnchorSelection(BaseModel):
    """A source interval selected within one already stored observation."""

    model_config = {"extra": "forbid"}

    observation_id: str = Field(min_length=1)
    story_graph_node_id: str | None = Field(default=None, min_length=1)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.source_end <= self.source_start:
            raise ValueError("source_end must be greater than source_start")
        return self


class ProjectStoryEntityLinkSelection(BaseModel):
    """A caller-entered person/event identity grouping across assets."""

    model_config = {"extra": "forbid"}

    link_id: str = Field(min_length=1, max_length=128)
    entity_kind: Literal[
        "person_identity", "event_identity", "place_identity"]
    display_label: str = Field(min_length=1, max_length=200)
    anchors: list[ProjectStoryLinkAnchorSelection] = Field(
        min_length=2, max_length=500)
    source_comparison_ids: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def validate_unique_anchors(self):
        anchors = [
            (item.observation_id, item.story_graph_node_id,
             item.source_start, item.source_end)
            for item in self.anchors
        ]
        if len(anchors) != len(set(anchors)):
            raise ValueError("a link cannot repeat an exact source anchor")
        if len(self.source_comparison_ids) != len(set(self.source_comparison_ids)):
            raise ValueError("a link cannot repeat a source comparison id")
        return self


class ProjectStoryRelationEndpointSelection(ProjectStoryLinkAnchorSelection):
    """Exact mention-node endpoint for a caller-authored project relation."""

    story_graph_node_id: str = Field(min_length=1)


class ProjectStoryRelationSelection(BaseModel):
    """Caller-authored, cross-asset semantic edge; never model-inferred."""

    model_config = {"extra": "forbid"}

    relation_id: str = Field(min_length=1, max_length=128)
    relation_type: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    from_anchor: ProjectStoryRelationEndpointSelection
    to_anchor: ProjectStoryRelationEndpointSelection
    privacy_class: Literal["public", "internal", "private", "sensitive"] = "private"


class ProjectStoryLinkComparisonDispositionRequest(BaseModel):
    """Caller-entered decision over one stored, unreviewed comparison."""

    model_config = {"extra": "forbid"}

    comparison_id: str = Field(min_length=1, max_length=160)
    decision: Literal[
        "accepted_as_caller_asserted",
        "rejected",
        "needs_more_evidence",
    ]
    link_id: str | None = Field(default=None, min_length=1, max_length=128)
    rationale: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_link_binding(self):
        if self.decision == "accepted_as_caller_asserted" and self.link_id is None:
            raise ValueError("accepted comparison disposition requires a link_id")
        if self.decision != "accepted_as_caller_asserted" and self.link_id is not None:
            raise ValueError("only an accepted disposition can bind a link_id")
        return self


class ProjectStoryLinkReviewRequest(BaseModel):
    """Replace caller-asserted links, relations, and dispositions for one graph."""

    model_config = {"extra": "forbid"}

    manifest_revision: int = Field(ge=1)
    story_graph_id: str = Field(min_length=1)
    expected_review_revision: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1, max_length=128)
    links: list[ProjectStoryEntityLinkSelection] = Field(
        default_factory=list, max_length=500)
    relations: list[ProjectStoryRelationSelection] = Field(
        default_factory=list, max_length=500)
    comparison_dispositions: list[ProjectStoryLinkComparisonDispositionRequest] = Field(
        default_factory=list, max_length=500)

    @model_validator(mode="after")
    def validate_unique_link_ids(self):
        link_ids = [item.link_id for item in self.links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("link ids must be unique within one review request")
        relation_ids = [item.relation_id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation ids must be unique within one review request")
        if set(link_ids) & set(relation_ids):
            raise ValueError("identity link and semantic relation ids must be distinct")
        comparison_ids = [item.comparison_id for item in self.comparison_dispositions]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("comparison ids must be unique within one review request")
        links_by_id = {item.link_id: item for item in self.links}
        for disposition in self.comparison_dispositions:
            if disposition.decision == "accepted_as_caller_asserted":
                link = links_by_id.get(disposition.link_id or "")
                if link is None or disposition.comparison_id not in link.source_comparison_ids:
                    raise ValueError(
                        "accepted comparison disposition must bind its exact link")
            elif any(
                disposition.comparison_id in link.source_comparison_ids
                for link in self.links
            ):
                raise ValueError(
                    "non-accepted comparison disposition cannot create a link")
        accepted_by_link = {
            (item.link_id, item.comparison_id)
            for item in self.comparison_dispositions
            if item.decision == "accepted_as_caller_asserted"
        }
        for link in self.links:
            if any(
                (link.link_id, comparison_id) not in accepted_by_link
                for comparison_id in link.source_comparison_ids
            ):
                raise ValueError(
                    "source comparisons require caller-accepted dispositions")
        return self


class ProjectStoryMentionReviewRequest(BaseModel):
    """Replace caller dispositions for exact source-local mentions."""

    model_config = {"extra": "forbid"}

    manifest_revision: int = Field(ge=1)
    story_graph_id: str = Field(min_length=1)
    expected_review_revision: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1, max_length=128)
    decisions: list[ProjectStoryMentionDecision] = Field(max_length=10_000)

    @model_validator(mode="after")
    def validate_unique_targets(self):
        keys = [
            (item.relation_kind, item.anchor.project_asset_id,
             item.anchor.story_graph_node_id)
            for item in self.decisions
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("a review request cannot repeat an exact source mention")
        return self


class ProjectStoryLinkComparisonRequest(ProjectStoryLinkComparisonRequestData):
    """Ask the pinned local VLM to compare one exact cross-asset pair."""


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------

@app.post("/v1/briefs:compile")
def compile_brief_endpoint(req: CompileBriefRequest):
    corr = _new_correlation_id()
    try:
        observations = []
        if req.observations_json:
            from director_brain.models.film_observation import FilmObservation
            observations = [FilmObservation(**o) for o in req.observations_json]
        brief = compile_brief(
            req.project_id, req.video_path, observations,
            target_duration_us=req.target_duration_us,
            intent_text=req.intent_text,
        )
        return _envelope(corr, {
            "brief": brief.model_dump(),
            "brief_id": brief.brief_id,
            "version": brief.version,
        })
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


@app.post("/v1/director-plans:generate")
def generate_plan_endpoint(req: GeneratePlanRequest):
    corr = _new_correlation_id()
    try:
        observations = []
        if req.observations_json:
            from director_brain.models.film_observation import FilmObservation
            observations = [FilmObservation(**o) for o in req.observations_json]
        else:
            from observation_service.pipeline import analyze_media
            observations = analyze_media(req.video_path)

        brief = compile_brief(
            req.project_id, req.video_path, observations,
            target_duration_us=req.target_duration_us,
            intent_text=req.intent_text,
        )
        graph = build_story_graph(brief, observations)
        edl, plan = generate_plan(brief, graph, observations)

        ok, errors = validate_plan(edl, plan, observations)
        plan.validation_status = "valid" if ok else "invalid"

        return _envelope(corr, {
            "plan": plan.model_dump(),
            "edl": edl.model_dump(),
            "plan_hash": compute_plan_hash(plan),
            "edl_hash": compute_edl_hash(edl),
            "validation": {"valid": ok, "errors": errors},
        })
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


@app.post("/v1/director-plans/{plan_id}:confirm-strategy")
def confirm_strategy_endpoint(plan_id: str, req: ConfirmStrategyRequest):
    corr = _new_correlation_id()
    repo = None
    try:
        from director_brain.models.director_plan import DirectorDecisionPlan
        from director_brain.models.edl import EditorialDecisionList
        from director_brain.plan_state import transition_plan
        plan = DirectorDecisionPlan(**req.plan_json)
        edl = EditorialDecisionList(**req.edl_json)
        if plan.project_manifest_id is not None:
            raise HTTPException(
                409,
                detail=(
                    "project-bound Plans must use the persisted, project-scoped "
                    "strategy confirmation endpoint"
                ),
            )
        # 候选③执法：确认前 plan 必须已到 READY_FOR_STRATEGY_CONFIRMATION
        # （非法转换在此抛 InvalidTransition），确认绑定真实 hash 并落账本。
        if plan.state != PlanState.READY_FOR_STRATEGY_CONFIRMATION.value:
            raise HTTPException(
                409,
                detail=f"plan 状态为 {plan.state}，须先验证通过到达 "
                       f"ready_for_strategy_confirmation 才能确认",
            )
        confirmation = confirm_strategy(
            plan, edl, confirmed_by=req.confirmed_by,
            output_target=req.output_target, notes=req.notes,
        )
        transition_plan(plan, PlanState.STRATEGY_CONFIRMED)
        transition_plan(plan, PlanState.DISPATCH_ELIGIBLE)
        # 持久化：决策账本（此前端点返回硬编码状态、不落任何记录）
        from director_brain.audit_trail import log_decision
        try:
            repo = _project_manifest_repository()
            log_decision(repo, plan.plan_id, "strategy_confirmed", {
                "plan_hash": confirmation.plan_hash,
                "edl_hash": confirmation.edl_hash,
                "confirmed_by": confirmation.confirmed_by,
                "output_target": confirmation.output_target,
                "state": plan.state,
                "correlation_id": corr,
            }, project_id=plan.project_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, detail=f"确认记录落账本失败: {exc}") from exc
        return _envelope(corr, {
            "plan_id": plan_id,
            "confirmation": confirmation.to_dict(),
            "state": plan.state,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])
    finally:
        if repo is not None:
            repo.close()


@app.post("/v1/director-plans/{plan_id}:validate")
def validate_endpoint(plan_id: str, req: ValidateRequest):
    corr = _new_correlation_id()
    try:
        from director_brain.models.director_plan import DirectorDecisionPlan
        from director_brain.models.edl import EditorialDecisionList
        from director_brain.models.film_observation import FilmObservation
        plan = DirectorDecisionPlan(**req.plan_json)
        edl = EditorialDecisionList(**req.edl_json)
        observations = [FilmObservation(**o) for o in req.observations_json]
        ok, errors = validate_plan(edl, plan, observations)
        return _envelope(corr, {"valid": ok, "errors": errors})
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


@app.post("/v1/revisions:propose")
def propose_revision_endpoint(req: RevisionRequest):
    corr = _new_correlation_id()
    try:
        if not req.plan_json or not req.edl_json:
            raise HTTPException(
                400, detail="revisions:propose 需要 plan_json 与 edl_json"
                            "（端点生成真实修订候选，不直接执行）")
        from director_brain.models.director_plan import DirectorDecisionPlan
        from director_brain.models.edl import EditorialDecisionList
        from director_brain.models.film_observation import FilmObservation
        from director_brain.revision_engine import propose_revision

        plan = DirectorDecisionPlan(**req.plan_json)
        edl = EditorialDecisionList(**req.edl_json)
        observations = [
            FilmObservation(**o) for o in (req.observations_json or [])
        ]
        proposal = propose_revision(
            edl, plan, observations,
            revision_type=req.revision_type,
            reason=req.change_summary or "api revision request",
            finding_ids=req.finding_ids,
        )
        return _envelope(corr, {
            "proposal": proposal.model_dump(mode="json"),
            "proposal_id": proposal.proposal_id,
            "revision_type": proposal.revision_type,
            "source_finding_ids": proposal.source_finding_ids,
            "target_decision_ids": proposal.target_decision_ids,
            "status": proposal.approval_state,
            "note": "提案未执行；应用产物为取代性新草案，须重走验证+策略确认",
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


# ---------------------------------------------------------------------------
# Film Context 端点（方案 §6 前两条：快照 + 按层查询）
# ---------------------------------------------------------------------------

# 进程内快照存储：POST 写入、GET 读取；键为 context_id（输入确定性哈希，
# 同输入同 id → 天然去重复用）。服务重启即失（快照可由 POST 重建）。
_CONTEXT_STORE: dict[str, dict] = {}


def _load_observations(req: FilmContextSnapshotRequest) -> list:
    if req.observations_json:
        from director_brain.models.film_observation import FilmObservation
        return [FilmObservation(**o) for o in req.observations_json]
    from observation_service.pipeline import analyze_media
    return analyze_media(req.video_path)


@app.post("/v1/film-context:snapshot")
def film_context_snapshot_endpoint(req: FilmContextSnapshotRequest):
    corr = _new_correlation_id()
    try:
        layer = (req.layer or "").lower()
        if layer not in ("project", "asset", "scene", "evidence"):
            raise HTTPException(400, detail="layer 须为 project|asset|scene|evidence")
        if req.window_us is not None:
            if layer not in ("scene", "evidence"):
                raise HTTPException(
                    422, detail="window_us is supported only for scene/evidence layers"
                )
            if len(req.window_us) != 2:
                raise HTTPException(
                    422, detail="window_us must contain exactly [start_us, end_us]"
                )

        observations = _load_observations(req)
        if req.window_us is not None:
            try:
                validate_window_us(observations, tuple(req.window_us))
            except ValueError as exc:
                raise HTTPException(422, detail=str(exc)) from exc

        brief = None
        if layer in ("project", "scene"):
            brief = compile_brief(
                req.project_id, req.video_path, observations,
                target_duration_us=req.target_duration_us,
                intent_text=req.intent_text,
            )
        window = tuple(req.window_us) if req.window_us is not None else None

        # 候选⑤：ContextGateway 类是渐进披露的会话入口（此前休眠）——
        # 端点经其取层快照，status() 汇报已展开层
        if layer == "scene":
            graph = build_story_graph(brief, observations)
            edl, _plan = generate_plan(brief, graph, observations)
            gw = ContextGateway(req.video_path, observations, brief=brief,
                                graph=graph, edl=edl)
            snap = gw.scene(window)
        else:
            gw = ContextGateway(req.video_path, observations, brief=brief)
            if layer == "project":
                snap = gw.project()
            elif layer == "asset":
                snap = gw.asset()
            else:  # evidence
                if not req.reason.strip():
                    raise HTTPException(
                        400,
                        detail="EVIDENCE 层展开必须提供 reason（方案 §4.2 审计留痕）")
                targets = list(req.shot_ids)
                if not targets and window:
                    # 时间窗 → 相交镜头（与 build_scene_context 同一口径：
                    # 窗口对 start_frame/end_frame 区间比较）
                    targets = sorted({
                        o.media_asset_id for o in observations
                        if o.end_frame > window[0] and o.start_frame < window[1]
                    })
                if not targets:
                    raise HTTPException(
                        400, detail="EVIDENCE 层需要 shot_ids 或 window_us 定位目标镜头")
                snap = gw.evidence(targets, req.reason)

        cache_hit = snap.context_id in _CONTEXT_STORE
        payload = snap.model_dump(mode="json")
        _CONTEXT_STORE[snap.context_id] = payload
        return _envelope(corr, {
            "snapshot": payload,
            "context_id": snap.context_id,
            "analysis_fingerprint": snap.analysis_fingerprint,
            "coverage": snap.coverage,
            "evidence_refs": snap.evidence_refs,
            "cache_hit": cache_hit,
            "gateway_status": gw.status(),
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


@app.get("/v1/film-context/{context_id}")
def film_context_get_endpoint(
    context_id: str,
    request: Request,
    layer: str | None = Query(
        default=None, description="按需请求的层：project|asset|scene|evidence"),
    reason: str = Query(
        default="", description="EVIDENCE 层用途理由（方案 §6：必须）"),
):
    corr = _new_correlation_id()
    snap = _CONTEXT_STORE.get(context_id)
    if snap is None:
        try:
            from director_brain.models.film_context import FilmContextSnapshot
            repo = _project_manifest_repository()
            try:
                stored = repo.get(FilmContextSnapshot, context_id)
            finally:
                repo.close()
            if stored is not None:
                snap = stored.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, detail=f"持久化上下文读取失败: {exc}") from exc
    if snap is None:
        raise HTTPException(
            404, detail="context 不存在")
    if snap.get("project_manifest_id"):
        _require_project_api_access(request)
    if snap.get("invalidated_at") is not None:
        raise HTTPException(410, detail="context 已因项目 manifest 更新而失效")
    if layer:
        want = layer.lower()
        have = [str(v) for v in snap.get("layers", [])]
        if want not in have:
            raise HTTPException(
                400, detail=f"快照层 {have} 不包含请求层 {want}")
        if want == "evidence" and not reason.strip():
            raise HTTPException(
                400, detail="EVIDENCE 层按需展开必须提供 reason（方案 §6）")
    return _envelope(corr, {
        "snapshot": snap,
        "requested_layer": layer,
        "expansion_reason": reason or None,
    })


def _project_manifest_repository():
    """Create a request-scoped SQLite repository using the configured DB path."""
    from director_brain.config import load_settings
    from storage.sqlite_repository import SqliteRepository

    db_path = Path(load_settings().sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteRepository(str(db_path))


def _latest_project_context(repo, project_id: str):
    """Resolve the latest manifest and its exact, non-invalidated context."""
    from director_brain.context_gateway import project_context_id
    from director_brain.models.film_context import FilmContextSnapshot

    manifest = repo.latest_project_manifest(project_id)
    if manifest is None:
        raise HTTPException(404, detail="project manifest 不存在")
    context = repo.get(FilmContextSnapshot, project_context_id(manifest))
    if context is None:
        raise HTTPException(
            409,
            detail=(
                "当前 manifest 尚无匹配当前 schema/profile 的持久化 "
                "project context"),
        )
    if (
        context.invalidated_at is not None
        or context.project_manifest_id != manifest.manifest_id
        or context.project_revision != manifest.revision
    ):
        raise HTTPException(409, detail="project context 与当前 manifest 不匹配")
    return manifest, context


def _validate_project_story_mention_review_snapshot(
    repo,
    review: ProjectStoryMentionReview | None,
    manifest: FilmProjectManifest,
    context,
    graph,
) -> None:
    """Revalidate persisted mention-review anchors at each consumer boundary."""
    if review is None:
        return
    from director_brain.models.film_observation import FilmObservation
    from director_brain.project_story_mention_review import (
        validate_project_story_mention_review,
    )

    observations = {}
    for observation_id in {
        item.anchor.observation_id for item in review.decisions
    }:
        observation = repo.get(FilmObservation, observation_id)
        if observation is None:
            raise ValueError("mention review cites a missing stored observation")
        observations[observation_id] = observation
    validate_project_story_mention_review(
        review, manifest, context, graph, observations)


def _require_current_project_sources(manifest: FilmProjectManifest) -> None:
    """Refuse project planning when authorized local files drifted after registration.

    The manifest and persisted observations are content-addressed, but the
    filesystem can change after a context snapshot or StoryGraph is built. A
    fresh hash check at the point where a new Plan is created keeps that Plan
    bound to the bytes represented by its evidence.
    """
    from director_brain.utils import file_sha256

    stale_asset_ids: list[str] = []
    for asset in manifest.assets:
        if asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED:
            continue
        if asset.source_content_hash is None:
            stale_asset_ids.append(asset.asset_id)
            continue
        media_path = _authorized_local_media_path(asset.source_ref)
        try:
            current_hash = file_sha256(str(media_path)).lower()
        except OSError:
            stale_asset_ids.append(asset.asset_id)
            continue
        if current_hash != asset.source_content_hash.lower():
            stale_asset_ids.append(asset.asset_id)

    if stale_asset_ids:
        raise HTTPException(
            409,
            detail=(
                "project source files no longer match the registered manifest; "
                "refresh the manifest and context for assets: "
                + ", ".join(stale_asset_ids)
            ),
        )


def _project_director_candidate_request_identity(
    project_id,
    manifest,
    context,
    graph,
    active_link_review,
    active_mention_review,
    req,
    brief,
) -> dict:
    """Canonical shared identity for previewing and later selecting a candidate."""
    return {
        "identity_version": "project-director-candidate-review-v1",
        "project_id": project_id,
        "manifest_id": manifest.manifest_id,
        "manifest_revision": manifest.revision,
        "context_id": context.context_id,
        "story_graph_id": graph.graph_id,
        "story_link_review_id": (
            active_link_review.review_id if active_link_review else None),
        "story_link_review_revision": (
            active_link_review.review_revision if active_link_review else None),
        "story_mention_review_id": (
            active_mention_review.review_id if active_mention_review else None),
        "story_mention_review_revision": (
            active_mention_review.review_revision if active_mention_review else None),
        "target_duration_us": req.target_duration_us,
        "intent_text": req.intent_text,
        "voice_led": req.voice_led,
        "audio_style": req.audio_style,
        "pacing_style": req.pacing_style,
        **({"transition_policy": req.transition_policy}
           if req.transition_policy != "none" else {}),
        "brief_producer": brief.producer,
        "reasoning_mode": "director_strategy_reasoning:SHADOW",
    }


def _fingerprint_project_director_candidate_request(identity: dict) -> str:
    canonical = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _project_director_shadow_comparison_id(project_id: str, idempotency_key: str) -> str:
    digest = hashlib.sha256(
        f"{project_id}\0{idempotency_key}".encode("utf-8")
    ).hexdigest()
    return f"pds_{digest[:40]}"


class _ProcessKeyedLockRegistry:
    """Serialize same-key work within this API process without retaining keys."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, tuple[Any, int]] = {}

    @contextmanager
    def hold(self, key: str):
        with self._guard:
            entry = self._entries.get(key)
            lock = entry[0] if entry is not None else threading.Lock()
            users = entry[1] if entry is not None else 0
            self._entries[key] = (lock, users + 1)

        acquired = False
        try:
            lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                lock.release()
            with self._guard:
                current_lock, users = self._entries[key]
                if users == 1:
                    del self._entries[key]
                else:
                    self._entries[key] = (current_lock, users - 1)


_project_director_shadow_locks = _ProcessKeyedLockRegistry()
_project_director_shadow_current_claim: ContextVar[
    ProjectDirectorShadowRunClaim | None
] = ContextVar("project_director_shadow_current_claim", default=None)
_project_director_shadow_unknown_outcome: ContextVar[bool] = ContextVar(
    "project_director_shadow_unknown_outcome", default=False,
)


def _project_director_shadow_run_store() -> SqliteProjectDirectorShadowRunStore:
    from director_brain.config import load_settings

    return SqliteProjectDirectorShadowRunStore(load_settings().sqlite_path)


def _project_director_shadow_request_fingerprint(
    project_id: str,
    req: ProjectShadowStrategyComparisonRequest,
) -> str:
    canonical = json.dumps(
        {
            "project_id": project_id,
            "request": req.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _complete_project_director_shadow_run() -> None:
    claim = _project_director_shadow_current_claim.get()
    if claim is not None:
        claim.complete()


def _authorize_project_director_shadow_provider_call(
    comparison_id: str,
) -> None:
    if _project_director_shadow_unknown_outcome.get():
        raise HTTPException(
            status_code=409,
            detail=(
                f"comparison {comparison_id} has an uncertain prior outcome; "
                "use a new idempotency key before requesting another Reasoner call"
            ),
            headers={"X-Director-Brain-Run-State": "outcome_unknown"},
        )
    claim = _project_director_shadow_current_claim.get()
    if claim is not None:
        claim.mark_provider_started()


def _serialize_project_director_shadow_request(
    endpoint: Callable[..., Any],
) -> Callable[..., Any]:
    """Serialize same-key work within and across local API processes."""
    @wraps(endpoint)
    def wrapped(
        project_id: str,
        request: Request,
        req: ProjectShadowStrategyComparisonRequest,
    ):
        comparison_id = _project_director_shadow_comparison_id(
            project_id, req.idempotency_key,
        )
        with _project_director_shadow_locks.hold(comparison_id):
            _require_project_api_access(request)
            store = _project_director_shadow_run_store()
            try:
                claim = store.acquire(
                    comparison_id=comparison_id,
                    project_id=project_id,
                    request_fingerprint=(
                        _project_director_shadow_request_fingerprint(
                            project_id, req)),
                    wait_timeout_seconds=30,
                )
            except ProjectDirectorShadowRunConflictError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"comparison {comparison_id} is bound to another request"
                    ),
                    headers={
                        "X-Director-Brain-Run-State": "idempotency_conflict",
                    },
                ) from exc
            except ProjectDirectorShadowRunInProgressError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"comparison {comparison_id} is still in progress; "
                        "retry the same request and idempotency key"
                    ),
                    headers={
                        "Retry-After": "2",
                        "X-Director-Brain-Run-State": "running",
                    },
                ) from exc
            except ProjectDirectorShadowRunOutcomeUnknownError:
                # Let the endpoint replay a terminal record if the owner saved
                # one before interruption. It blocks a new provider call below.
                token = _project_director_shadow_unknown_outcome.set(True)
                try:
                    return endpoint(project_id, request, req)
                finally:
                    _project_director_shadow_unknown_outcome.reset(token)

            with claim:
                token = _project_director_shadow_current_claim.set(claim)
                try:
                    return endpoint(project_id, request, req)
                finally:
                    _project_director_shadow_current_claim.reset(token)

    return wrapped


def _record_project_director_shadow_failure(
    repo,
    *,
    project_id: str,
    comparison_id: str,
    request_fingerprint: str,
    idempotency_key_sha256: str,
    manifest,
    context,
    graph,
    exc: Exception,
) -> None:
    """Append a content-free local receipt for one failed Reasoner attempt."""
    from director_brain.audit_trail import log_decision

    claim = _project_director_shadow_current_claim.get()
    if claim is not None:
        claim.assert_owner()

    if isinstance(exc, EvidenceTooPoorError):
        failure_code = "evidence_too_poor"
    elif isinstance(exc, LLMInputCapacityError):
        failure_code = "input_capacity_exceeded"
    elif isinstance(exc, LLMTransportError):
        headers = _safe_project_reasoner_transport_headers(exc) or {}
        failure_code = headers.get(
            "X-Director-Brain-Failure-Code", "provider_transport_error")
    elif isinstance(exc, LLMStructuredOutputError):
        headers = _safe_project_reasoner_failure_headers(exc) or {}
        failure_code = headers.get(
            "X-Director-Brain-Failure-Code", "project_schema_invalid")
    elif isinstance(exc, ValueError):
        failure_code = _UNCLASSIFIED_REASONER_VALUE_ERROR_CODE
    else:
        failure_code = "internal_error"

    detail = {
        "stage": "reasoner_generation",
        "failure_code": failure_code,
        "request_fingerprint": request_fingerprint,
        "idempotency_key_sha256": idempotency_key_sha256,
        "manifest_id": manifest.manifest_id,
        "manifest_revision": manifest.revision,
        "context_id": context.context_id,
        "story_graph_id": graph.graph_id,
    }
    if failure_code == _UNCLASSIFIED_REASONER_VALUE_ERROR_CODE:
        detail["attribution_state"] = "unclassified"
        exception_type = type(exc).__name__
        if (isinstance(exception_type, str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", exception_type)):
            detail["exception_type"] = exception_type
    detail.update(_safe_project_reasoner_failure_diagnostics(exc))

    try:
        log_decision(
            repo,
            comparison_id,
            "director_strategy_comparison_failed",
            detail,
            project_id=project_id,
        )
        _complete_project_director_shadow_run()
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Director comparison failed and its local receipt could not be saved",
        ) from None


def _replay_project_director_shadow_failure(
    repo,
    *,
    project_id: str,
    comparison_id: str,
    request_fingerprint: str,
    idempotency_key_sha256: str,
) -> None:
    """Replay a terminal failure receipt without invoking the Reasoner again."""
    from storage.repository import DecisionLedgerEntry

    failures = [
        entry for entry in repo.list(DecisionLedgerEntry, project_id=project_id)
        if entry.decision_id == comparison_id
        and entry.action == "director_strategy_comparison_failed"
    ]
    if not failures:
        return

    entry = max(failures, key=lambda item: (item.timestamp, item.ledger_id))
    detail = entry.detail if isinstance(entry.detail, dict) else {}
    if (
        detail.get("request_fingerprint") != request_fingerprint
        or detail.get("idempotency_key_sha256") != idempotency_key_sha256
    ):
        raise HTTPException(
            status_code=409,
            detail="idempotency key is already bound to a failed comparison",
        )

    failure_code = detail.get("failure_code")
    if not isinstance(failure_code, str):
        raise HTTPException(
            status_code=409,
            detail="prior comparison failure receipt is invalid; use a new idempotency key",
        )
    if failure_code == "evidence_too_poor":
        status_code = 422
    elif failure_code == "input_capacity_exceeded":
        status_code = 413
    elif failure_code in _SAFE_PROJECT_REASONER_TRANSPORT_CODES:
        status_code = 503
    elif failure_code in _SAFE_PROJECT_REASONER_FAILURE_CODES:
        status_code = 502
    elif failure_code == _UNCLASSIFIED_REASONER_VALUE_ERROR_CODE:
        status_code = 500
    elif failure_code == "project_evidence_inconsistent":
        status_code = 409
    elif failure_code == "internal_error":
        status_code = 500
    else:
        raise HTTPException(
            status_code=409,
            detail="prior comparison failure receipt is invalid; use a new idempotency key",
        )

    headers = {}
    if (failure_code in _SAFE_PROJECT_REASONER_API_FAILURE_CODES
            or failure_code in _SAFE_PROJECT_REASONER_TRANSPORT_CODES):
        headers["X-Director-Brain-Failure-Code"] = failure_code
    raise HTTPException(
        status_code=status_code,
        detail="prior comparison attempt failed; use a new idempotency key to retry",
        headers=headers,
    )


def _project_director_strategy_asset_coverage(candidate) -> dict:
    """Summarize strategy source selection by asset for human review."""
    ordered_sources = candidate.hypothesis.ordered_sources
    selection_audit = candidate.selection_audit
    if len(ordered_sources) != len(selection_audit):
        raise ValueError("strategy selection audit does not cover ordered sources")

    by_asset = {}
    for rank, (source, audit) in enumerate(
        zip(ordered_sources, selection_audit, strict=True)
    ):
        if audit.strategy_rank != rank:
            raise ValueError("strategy selection audit order is inconsistent")
        row = by_asset.setdefault(source.project_asset_id, {
            "project_asset_id": source.project_asset_id,
            "source_count": 0,
            "strategy_include_count": 0,
            "strategy_exclude_count": 0,
            "selected_in_edl_count": 0,
            "included_but_not_selected_count": 0,
        })
        row["source_count"] += 1
        if audit.strategy_disposition == "include":
            row["strategy_include_count"] += 1
            if audit.selected_in_edl:
                row["selected_in_edl_count"] += 1
            else:
                row["included_but_not_selected_count"] += 1
        else:
            row["strategy_exclude_count"] += 1

    return {
        "hypothesis_id": candidate.hypothesis.hypothesis_id,
        "assets": [by_asset[asset_id] for asset_id in sorted(by_asset)],
    }


def _project_director_shadow_comparison_payload(
    record: ProjectDirectorShadowComparisonRecord,
    observations,
    *,
    active_link_review,
    active_mention_review,
    idempotent_replay: bool = False,
) -> dict:
    comparison = record.comparison
    baseline_valid, baseline_errors = validate_plan(
        comparison.baseline_edl, comparison.baseline_plan, observations)
    strategy_validations = []
    for candidate in comparison.strategy_candidates:
        valid, errors = validate_plan(candidate.edl, candidate.plan, observations)
        strategy_validations.append({
            "hypothesis_id": candidate.hypothesis.hypothesis_id,
            "valid": valid,
            "errors": errors,
        })
    return {
        "comparison_id": record.comparison_id,
        "comparison": comparison.model_dump(mode="json"),
        "baseline_validation": {
            "valid": baseline_valid,
            "errors": baseline_errors,
        },
        "strategy_validations": strategy_validations,
        "strategy_asset_coverage": [
            _project_director_strategy_asset_coverage(candidate)
            for candidate in comparison.strategy_candidates
        ],
        "persisted": True,
        "idempotent_replay": idempotent_replay,
        "confirmable": False,
        "quality_acceptance": "NOT_PROVEN",
        "provider_scope": "local_ollama_loopback_only",
        "evidence_scope": {
            "manifest_id": record.project_manifest_id,
            "manifest_revision": record.project_revision,
            "context_id": record.context_id,
            "story_graph_id": record.story_graph_id,
            "timeline_scope": "project_per_asset",
            "cross_asset_relations_state": "not_attempted",
            "project_story_mention_review_id": (
                active_mention_review.review_id if active_mention_review else None),
            "project_story_mention_review_revision": (
                active_mention_review.review_revision if active_mention_review else 0),
            "project_story_mention_review_state": (
                "caller_asserted" if active_mention_review else "not_provided"),
            "project_story_link_review_id": (
                active_link_review.review_id if active_link_review else None),
            "project_story_link_review_revision": (
                active_link_review.review_revision if active_link_review else None),
            "project_story_link_review_state": (
                "caller_asserted" if active_link_review else "not_provided"),
            "quality_acceptance": "not_proven",
        },
    }


def _project_director_shadow_currentness_state(repo, project_id: str) -> dict:
    """Resolve shared live state once for shadow-comparison list/readback."""
    current_manifest = repo.latest_project_manifest(project_id)
    try:
        manifest, context = _latest_project_context(repo, project_id)
        from director_brain.models import ProjectStoryGraph
        from director_brain.project_story_graph import project_story_graph_id

        graph = repo.get(ProjectStoryGraph, project_story_graph_id(context))
    except HTTPException:
        return {
            "current_manifest": current_manifest,
            "context": None,
            "graph": None,
            "source_changed": False,
        }
    try:
        _require_current_project_sources(manifest)
        source_changed = False
    except HTTPException:
        source_changed = True
    return {
        "current_manifest": current_manifest,
        "context": context,
        "graph": graph,
        "source_changed": source_changed,
    }


def _project_director_shadow_comparison_currentness(
    repo,
    record: ProjectDirectorShadowComparisonRecord,
    state: dict,
) -> dict:
    """Report staleness against current project, graph, reviews, and sources."""
    reasons: list[str] = []
    current_manifest = state["current_manifest"]
    if (current_manifest is None
            or current_manifest.manifest_id != record.project_manifest_id
            or current_manifest.revision != record.project_revision):
        reasons.append("project_manifest_changed")

    context = state["context"]
    if context is None:
        reasons.append("current_project_context_unavailable")
    else:
        if context.context_id != record.context_id:
            reasons.append("project_context_changed")
        graph = state["graph"]
        if graph is None or graph.graph_id != record.story_graph_id:
            reasons.append("project_story_graph_changed")
        try:
            mention_history, link_history = repo.list_project_story_review_histories(
                record.project_id, record.story_graph_id)
        except HTTPException:
            reasons.append("current_project_context_unavailable")
            unique_reasons = sorted(set(reasons))
            return {
                "source_currentness": "stale",
                "stale_reasons": unique_reasons,
            }
        current_mention = mention_history[-1] if mention_history else None
        current_link = link_history[-1] if link_history else None
        if ((current_mention.review_id if current_mention else None)
                != record.project_story_mention_review_id
                or (current_mention.review_revision if current_mention else 0)
                != record.project_story_mention_review_revision):
            reasons.append("project_mention_review_changed")
        if ((current_link.review_id if current_link else None)
                != record.project_story_link_review_id
                or (current_link.review_revision if current_link else None)
                != record.project_story_link_review_revision):
            reasons.append("project_link_review_changed")
        if state["source_changed"]:
            reasons.append("project_source_changed")

    unique_reasons = sorted(set(reasons))
    return {
        "source_currentness": "stale" if unique_reasons else "current",
        "stale_reasons": unique_reasons,
    }


def _require_loopback_request(request: Request) -> None:
    """Keep new private-media endpoints off LAN interfaces by default."""
    peer = request.client.host if request.client is not None else ""
    try:
        is_loopback = ipaddress.ip_address(peer).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(403, detail="项目素材接口仅允许本机 loopback 客户端")


def _require_project_api_access(request: Request) -> None:
    """Require local origin plus an explicit secret for project-media operations."""
    _require_loopback_request(request)
    from director_brain.config import read_env_key

    expected = (read_env_key("DIRECTOR_BRAIN_PROJECT_API_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(
            503,
            detail="DIRECTOR_BRAIN_PROJECT_API_TOKEN 未配置；项目素材接口保持关闭",
        )
    authorization = request.headers.get("authorization", "")
    scheme, separator, provided = authorization.partition(" ")
    if (
        scheme.casefold() != "bearer"
        or not separator
        or not hmac.compare_digest(
            provided.encode("utf-8"), expected.encode("utf-8")
        )
    ):
        raise HTTPException(
            401,
            detail="项目素材接口凭证无效",
            headers={"WWW-Authenticate": "Bearer"},
        )

def _authorized_local_media_path(
    source_ref: str,
    *,
    require_file: bool = True,
) -> Path:
    """Resolve an asset only beneath an explicitly configured local media root."""
    from director_brain.config import read_env_key

    configured_root = (
        read_env_key("DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT") or ""
    ).strip()
    if not configured_root:
        raise HTTPException(
            503,
            detail="DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT 未配置；项目素材处理默认关闭",
        )
    try:
        root = Path(configured_root).expanduser().resolve(strict=True)
        path = Path(source_ref).expanduser().resolve(strict=require_file)
    except OSError as exc:
        raise HTTPException(503, detail="本地媒体根目录或素材不可用") from exc
    if not root.is_dir():
        raise HTTPException(503, detail="DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT 必须是目录")
    if require_file and not path.is_file():
        raise HTTPException(422, detail="source_ref 必须是可读本地媒体文件")
    if not path.is_relative_to(root):
        raise HTTPException(403, detail="source_ref 超出获准本地媒体根目录")
    return path


def _public_manifest_payload(manifest: FilmProjectManifest) -> dict:
    """Omit host-local filesystem paths from REST responses."""
    payload = manifest.model_dump(mode="json")
    for asset in payload["assets"]:
        scheme = (
            "local-media"
            if asset["source_identity_state"] == "locally_verified"
            else "source-asset"
        )
        asset["source_ref"] = f"{scheme}://{asset['asset_id']}"
    return payload


def _public_observation_payload(observation) -> dict:
    payload = observation.model_dump(mode="json")
    payload["source_ref"] = (
        f"local-media://{observation.project_asset_id or 'unbound'}"
    )
    return payload


@app.put("/v1/projects/{project_id}/film-manifest")
def project_film_manifest_put_endpoint(
    project_id: str,
    request: Request,
    req: ProjectManifestWriteRequest,
):
    """Record an ordered multi-asset boundary without claiming verification."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        if not project_id.strip():
            raise HTTPException(400, detail="project_id 不能为空")
        project_assets: list[ProjectAsset] = []
        for item in req.assets:
            if item.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED:
                project_assets.append(ProjectAsset(
                    asset_id=item.asset_id,
                    source_ref=item.source_ref,
                    source_identity_state="declared_unverified",
                    source_content_hash=None,
                    size_bytes=None,
                    order=item.order,
                    duration_us=None,
                    fps=None,
                    r_frame_rate=None,
                    avg_frame_rate=None,
                    stream_time_base=None,
                    has_audio=None,
                    time_map=None,
                    probe_ok=None,
                    rights=item.rights,
                ))
                continue
            try:
                media_path = _authorized_local_media_path(item.source_ref)
                from director_brain.utils import file_sha256
                from director_brain.context_gateway import _probe_video_meta

                content_hash = file_sha256(str(media_path))
                size_bytes = media_path.stat().st_size
                metadata = _probe_video_meta(str(media_path))
            except (OSError, ValueError) as exc:
                raise HTTPException(
                    422, detail=f"素材 {item.asset_id} 无法读取或探测") from exc
            project_assets.append(ProjectAsset(
                asset_id=item.asset_id,
                source_ref=str(media_path),
                source_identity_state="locally_verified",
                source_content_hash=content_hash,
                size_bytes=size_bytes,
                order=item.order,
                duration_us=int(metadata["duration_us"]),
                fps=float(metadata["fps"]) if metadata["fps"] else None,
                r_frame_rate=metadata.get("r_frame_rate"),
                avg_frame_rate=metadata.get("avg_frame_rate"),
                stream_time_base=metadata.get("stream_time_base"),
                has_audio=(
                    bool(metadata["has_audio"])
                    if metadata["probe_ok"] else None
                ),
                time_map=(
                    metadata.get("time_map")
                    if metadata["probe_ok"] else None
                ),
                probe_ok=bool(metadata["probe_ok"]),
                rights=item.rights,
            ))

        asset_ids_by_hash: dict[str, list[str]] = {}
        for asset in project_assets:
            if asset.source_content_hash is not None:
                asset_ids_by_hash.setdefault(
                    asset.source_content_hash.lower(), []).append(asset.asset_id)
        duplicate_asset_groups = [
            asset_ids for asset_ids in asset_ids_by_hash.values()
            if len(asset_ids) > 1
        ]
        if duplicate_asset_groups:
            duplicate_ids = "; ".join(
                ", ".join(asset_ids) for asset_ids in duplicate_asset_groups)
            raise HTTPException(
                422,
                detail=("项目清单不允许同一内容重复登记；冲突素材 ID："
                        f"{duplicate_ids}"),
            )

        repo = _project_manifest_repository()
        revision = req.expected_revision + 1
        from director_brain.utils import short_hash
        manifest = FilmProjectManifest(
            manifest_id=f"manifest_{short_hash(project_id)}_r{revision:08d}",
            revision=revision,
            boundary_basis=req.boundary_basis,
            boundary_source_ref=req.boundary_source_ref,
            boundary_evidence_refs=req.boundary_evidence_refs,
            boundary_state="declared_unverified",
            analysis_profile=req.analysis_profile,
            assets=project_assets,
            schema_version="1.4",
            project_id=project_id,
            created_at=int(time.time()),
            producer="director_brain_project_manifest_api",
            source_ref=req.boundary_source_ref,
        )
        try:
            repo.save_project_manifest(manifest, req.expected_revision)
        except ValueError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        return _envelope(corr, {
            "manifest": _public_manifest_payload(manifest),
            "boundary_state": "declared_unverified",
            "rights_state": "declared_per_asset",
            "current_revision": revision,
        })
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(422, detail="项目清单字段无效") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/observations/{observation_id}")
def project_observation_get_endpoint(
    project_id: str,
    observation_id: str,
    request: Request,
):
    """Resolve one evidence reference only within its owning project."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models.film_observation import FilmObservation

        repo = _project_manifest_repository()
        observation = repo.get(FilmObservation, observation_id)
        if observation is None or observation.project_id != project_id:
            raise HTTPException(404, detail="project observation 不存在")
        return _envelope(corr, {
            "observation": _public_observation_payload(observation),
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.post("/v1/projects/{project_id}/story-graph:build")
def project_story_graph_build_endpoint(project_id: str, request: Request):
    """Persist an asset-local graph collection from exact saved project evidence."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import FilmObservation, ProjectStoryGraph
        from director_brain.project_story_graph import (
            build_project_story_graph,
            build_project_story_graph_view,
            project_story_graph_id,
        )

        repo = _project_manifest_repository()
        manifest, context = _latest_project_context(repo, project_id)
        graph_id = project_story_graph_id(context)
        existing = repo.get(ProjectStoryGraph, graph_id)
        if existing is not None:
            mention_history, link_history = repo.list_project_story_review_histories(
                project_id, existing.graph_id)
            active_mention_review = mention_history[-1] if mention_history else None
            active_link_review = link_history[-1] if link_history else None
            graph_view = build_project_story_graph_view(
                existing, active_link_review, active_mention_review)
            return _envelope(corr, {
                "story_graph": existing.model_dump(mode="json"),
                "story_graph_view": graph_view.model_dump(mode="json"),
                "persisted": True,
                "reused": True,
            })

        observations = []
        for evidence_ref in context.evidence_refs:
            observation = repo.get(FilmObservation, evidence_ref)
            if observation is None or observation.project_id != project_id:
                raise HTTPException(
                    409, detail="project context 引用的观测证据缺失或越界")
            observations.append(observation)

        graph = build_project_story_graph(manifest, context, observations)
        try:
            repo.save_project_story_graph(graph)
        except ValueError as exc:
            raise HTTPException(409, detail="project StoryGraph 证据或版本已过期") from exc
        mention_history, link_history = repo.list_project_story_review_histories(
            project_id, graph.graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        active_link_review = link_history[-1] if link_history else None
        graph_view = build_project_story_graph_view(
            graph, active_link_review, active_mention_review)
        return _envelope(corr, {
            "story_graph": graph.model_dump(mode="json"),
            "story_graph_view": graph_view.model_dump(mode="json"),
            "persisted": True,
            "reused": False,
        })
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(409, detail="project StoryGraph 证据不一致") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/story-graph")
def project_story_graph_get_endpoint(
    project_id: str,
    request: Request,
    revision: int | None = Query(default=None, ge=1),
    context_id: str | None = Query(
        default=None,
        min_length=1,
        description="Optional exact persisted Context version for this manifest revision",
    ),
):
    """Read a persisted graph without requiring the active analysis provider."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import ProjectStoryGraph
        from director_brain.project_story_graph import (
            build_project_story_graph_view,
            project_story_graph_id,
        )
        from director_brain.models.film_context import FilmContextSnapshot

        repo = _project_manifest_repository()
        if revision is None:
            manifest = repo.latest_project_manifest(project_id)
            if manifest is None:
                raise HTTPException(404, detail="project manifest 不存在")
        else:
            manifest = repo.get_project_manifest(project_id, revision)
            if manifest is None:
                raise HTTPException(404, detail="指定 project manifest revision 不存在")

        graphs = [
            item for item in repo.list(ProjectStoryGraph, project_id=project_id)
            if (
                item.project_manifest_id == manifest.manifest_id
                and item.project_revision == manifest.revision
                and (context_id is None or item.context_id == context_id)
            )
        ]
        if not graphs:
            raise HTTPException(404, detail="当前 project StoryGraph 尚未生成")
        if context_id is None:
            newest_created_at = max(item.created_at for item in graphs)
            graphs = [item for item in graphs if item.created_at == newest_created_at]
        if len(graphs) != 1:
            raise HTTPException(
                409,
                detail=(
                    "multiple persisted StoryGraphs match this manifest; "
                    "specify context_id"
                ),
            )
        graph = graphs[0]
        context = repo.get(FilmContextSnapshot, graph.context_id)
        if context is None:
            raise HTTPException(409, detail="project StoryGraph context 不存在")
        latest_manifest = repo.latest_project_manifest(project_id)
        stale_reason = (
            "manifest_revision_changed"
            if latest_manifest is not None
            and (
                latest_manifest.manifest_id != manifest.manifest_id
                or latest_manifest.revision != manifest.revision
            )
            else "invalidated"
            if context.invalidated_at is not None
            else None
        )
        if (
            graph.project_id != project_id
            or graph.project_manifest_id != manifest.manifest_id
            or graph.project_revision != manifest.revision
            or graph.context_id != context.context_id
            or graph.analysis_fingerprint != context.analysis_fingerprint
            or project_story_graph_id(context) != graph.graph_id
            or context.project_id != project_id
            or context.project_manifest_id != manifest.manifest_id
            or context.project_revision != manifest.revision
            or (
                latest_manifest is not None
                and latest_manifest.manifest_id == manifest.manifest_id
                and context.invalidated_at is not None
            )
        ):
            raise HTTPException(409, detail="project StoryGraph or Context is stale or inconsistent")
        mention_history, link_history = repo.list_project_story_review_histories(
            project_id, graph.graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        active_link_review = link_history[-1] if link_history else None
        graph_view = build_project_story_graph_view(
            graph, active_link_review, active_mention_review)
        return _envelope(corr, {
            "story_graph": graph.model_dump(mode="json"),
            "story_graph_view": graph_view.model_dump(mode="json"),
            "persisted": True,
            "currentness": "stale" if stale_reason is not None else "current",
            "stale_reason": stale_reason,
            "usable_for_new_plan": stale_reason is None,
            "source_hash_validation_state": "not_revalidated_by_read",
        })
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(409, detail="project StoryGraph view evidence is inconsistent") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


def _build_project_story_mention_preview(
    *,
    project_id: str,
    manifest: FilmProjectManifest,
    graph: ProjectStoryGraph,
    anchor: ProjectStoryLinkAnchor,
    relation_kind: Literal["person_identity", "event_identity"],
) -> ProjectStoryMentionPreview:
    """Extract one exact, authorized mention preview without persistence."""
    from director_brain.utils import file_sha256
    from observation_service.keyframe import (
        MULTI_FRAME_SAMPLE_POSITIONS,
        extract_keyframes,
    )

    asset = next((
        item for item in manifest.assets
        if item.asset_id == anchor.project_asset_id
    ), None)
    if asset is None or asset.source_content_hash is None:
        raise HTTPException(409, detail="mention source asset is not hash-bound")
    media_path = _authorized_local_media_path(asset.source_ref)
    try:
        source_hash_before = file_sha256(str(media_path)).lower()
    except OSError as exc:
        raise HTTPException(409, detail="mention source asset is unavailable") from exc
    if source_hash_before != asset.source_content_hash.lower():
        raise HTTPException(409, detail="mention source no longer matches its manifest hash")

    frames: list[ProjectStoryMentionPreviewFrame] = []
    try:
        import base64
        import cv2

        with extract_keyframes(
            str(media_path), anchor.source_start, anchor.source_end,
            positions=MULTI_FRAME_SAMPLE_POSITIONS,
            local_only=True,
        ) as frame_paths:
            if len(frame_paths) != len(MULTI_FRAME_SAMPLE_POSITIONS):
                raise HTTPException(503, detail="preview frame extraction incomplete")
            for relative_position, frame_path in zip(
                MULTI_FRAME_SAMPLE_POSITIONS, frame_paths, strict=True
            ):
                image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    raise HTTPException(503, detail="preview frame decode failed")
                height, width = image.shape[:2]
                scale = min(1280 / max(height, width), 1.0)
                if scale < 1.0:
                    image = cv2.resize(
                        image,
                        (max(1, round(width * scale)),
                         max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                encoded, jpeg = cv2.imencode(
                    ".jpg", image,
                    [cv2.IMWRITE_JPEG_QUALITY, 82],
                )
                if not encoded:
                    raise HTTPException(503, detail="preview frame encoding failed")
                jpeg_bytes = jpeg.tobytes()
                if len(jpeg_bytes) > 1_500_000:
                    raise HTTPException(413, detail="preview frame exceeds the response limit")
                out_height, out_width = image.shape[:2]
                frames.append(ProjectStoryMentionPreviewFrame(
                    relative_position=relative_position,
                    sha256=hashlib.sha256(jpeg_bytes).hexdigest(),
                    width=out_width,
                    height=out_height,
                    jpeg_base64=base64.b64encode(jpeg_bytes).decode("ascii"),
                ))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, detail="preview frame processing failed") from exc

    try:
        source_hash_after = file_sha256(str(media_path)).lower()
    except OSError as exc:
        raise HTTPException(409, detail="mention source became unavailable") from exc
    if source_hash_after != asset.source_content_hash.lower():
        raise HTTPException(409, detail="mention source changed during preview extraction")

    return ProjectStoryMentionPreview(
        project_id=project_id,
        project_manifest_id=manifest.manifest_id,
        project_revision=manifest.revision,
        story_graph_id=graph.graph_id,
        relation_kind=relation_kind,
        anchor=anchor,
        frames=frames,
    )


@app.get("/v1/projects/{project_id}/story-graph/{story_graph_id}/mentions/{story_graph_node_id}/preview")
def project_story_mention_preview_get_endpoint(
    project_id: str,
    story_graph_id: str,
    story_graph_node_id: str,
    request: Request,
    response: Response,
    manifest_revision: int = Query(ge=1),
    relation_kind: Literal["person_identity", "event_identity"] = Query(),
):
    """Return three bounded local preview frames for one exact eligible mention.

    The endpoint is loopback-and-token protected, reads only a currently
    authorized manifest asset, and returns non-cacheable preview bytes without
    persisting them or making a provider call.
    """
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.context_gateway import PROJECT_CONTEXT_SCHEMA_VERSION
        from director_brain.models import ProjectStoryGraph
        from director_brain.models.film_context import FilmContextSnapshot
        from director_brain.project_story_link_candidates import (
            CandidateAnchorNotEligibleError,
            resolve_project_story_link_mention_anchor,
        )
        from director_brain.project_story_graph import project_story_graph_id as expected_graph_id

        repo = _project_manifest_repository()
        manifest = repo.latest_project_manifest(project_id)
        if manifest is None:
            raise HTTPException(404, detail="project manifest 不存在")
        if manifest.revision != manifest_revision:
            raise HTTPException(409, detail="mention preview requires the current manifest revision")
        graph = repo.get(ProjectStoryGraph, story_graph_id)
        if graph is None:
            raise HTTPException(404, detail="当前 project StoryGraph 不存在")
        context = repo.get(FilmContextSnapshot, graph.context_id)
        if context is None:
            raise HTTPException(409, detail="mention preview StoryGraph context 不存在")
        if (
            expected_graph_id(context) != story_graph_id
            or context.schema_version != PROJECT_CONTEXT_SCHEMA_VERSION
            or context.invalidated_at is not None
            or graph.project_id != project_id
            or graph.project_manifest_id != manifest.manifest_id
            or graph.project_revision != manifest.revision
            or graph.context_id != context.context_id
            or graph.analysis_fingerprint != context.analysis_fingerprint
            or context.project_id != project_id
            or context.project_manifest_id != manifest.manifest_id
            or context.project_revision != manifest.revision
        ):
            raise HTTPException(409, detail="mention preview graph or context is stale")

        mention_history = repo.list_project_story_mention_reviews(
            project_id, graph.graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        _validate_project_story_mention_review_snapshot(
            repo, active_mention_review, manifest, context, graph)
        try:
            anchor = resolve_project_story_link_mention_anchor(
                manifest,
                graph,
                relation_kind=relation_kind,
                story_graph_node_id=story_graph_node_id,
                mention_review=active_mention_review,
            )
        except CandidateAnchorNotEligibleError as exc:
            raise HTTPException(404, detail="eligible source mention not found") from exc
        preview = _build_project_story_mention_preview(
            project_id=project_id,
            manifest=manifest,
            graph=graph,
            anchor=anchor,
            relation_kind=relation_kind,
        )
        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return _envelope(corr, preview.model_dump(mode="json"))
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(409, detail="mention preview evidence is stale or inconsistent") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get(
    "/v1/projects/{project_id}/story-graph/{story_graph_id}/link-candidates/"
    "{candidate_id}/preview"
)
def project_story_link_candidate_preview_get_endpoint(
    project_id: str,
    story_graph_id: str,
    candidate_id: str,
    request: Request,
    response: Response,
    manifest_revision: int = Query(ge=1),
    relation_kind: Literal["person_identity", "event_identity"] = Query(),
    left_story_graph_node_id: str = Query(min_length=1),
    right_story_graph_node_id: str = Query(min_length=1),
):
    """Return both source previews for one exact current candidate pair.

    The caller supplies the candidate and mention IDs returned by candidate
    paging. The pair is reconstructed from current eligible graph evidence;
    no ranking score, provider assessment, link, or disposition is produced.
    """
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.context_gateway import PROJECT_CONTEXT_SCHEMA_VERSION
        from director_brain.models import ProjectStoryGraph
        from director_brain.models.film_context import FilmContextSnapshot
        from director_brain.project_story_graph import (
            project_story_graph_id as expected_graph_id,
        )
        from director_brain.project_story_link_candidates import (
            CandidateAnchorNotEligibleError,
            CandidateScopeNotEligibleError,
            resolve_project_story_link_candidate_from_mentions,
        )
        from director_brain.utils import file_sha256

        repo = _project_manifest_repository()
        manifest = repo.latest_project_manifest(project_id)
        if manifest is None:
            raise HTTPException(404, detail="project manifest 不存在")
        if manifest.revision != manifest_revision:
            raise HTTPException(
                409, detail="candidate preview requires the current manifest revision")
        graph = repo.get(ProjectStoryGraph, story_graph_id)
        if graph is None:
            raise HTTPException(404, detail="当前 project StoryGraph 不存在")
        context = repo.get(FilmContextSnapshot, graph.context_id)
        if context is None:
            raise HTTPException(409, detail="candidate preview StoryGraph context 不存在")
        if (
            expected_graph_id(context) != story_graph_id
            or context.schema_version != PROJECT_CONTEXT_SCHEMA_VERSION
            or context.invalidated_at is not None
            or graph.project_id != project_id
            or graph.project_manifest_id != manifest.manifest_id
            or graph.project_revision != manifest.revision
            or graph.context_id != context.context_id
            or graph.analysis_fingerprint != context.analysis_fingerprint
            or context.project_id != project_id
            or context.project_manifest_id != manifest.manifest_id
            or context.project_revision != manifest.revision
        ):
            raise HTTPException(
                409, detail="candidate preview graph or context is stale")

        mention_history = repo.list_project_story_mention_reviews(
            project_id, graph.graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        _validate_project_story_mention_review_snapshot(
            repo, active_mention_review, manifest, context, graph)
        try:
            candidate = resolve_project_story_link_candidate_from_mentions(
                manifest,
                graph,
                relation_kind=relation_kind,
                left_story_graph_node_id=left_story_graph_node_id,
                right_story_graph_node_id=right_story_graph_node_id,
                mention_review=active_mention_review,
            )
        except CandidateAnchorNotEligibleError as exc:
            raise HTTPException(
                404, detail="current eligible cross-asset candidate not found") from exc
        except CandidateScopeNotEligibleError as exc:
            raise HTTPException(422, detail="candidate preview scope is invalid") from exc
        if candidate.candidate_id != candidate_id:
            raise HTTPException(
                409, detail="candidate ID is stale or does not match the selected mentions")

        left_preview = _build_project_story_mention_preview(
            project_id=project_id,
            manifest=manifest,
            graph=graph,
            anchor=candidate.left_anchor,
            relation_kind=relation_kind,
        )
        right_preview = _build_project_story_mention_preview(
            project_id=project_id,
            manifest=manifest,
            graph=graph,
            anchor=candidate.right_anchor,
            relation_kind=relation_kind,
        )
        # Both sources must still match after the complete pair extraction, not
        # merely after each side was read independently.
        for anchor in (candidate.left_anchor, candidate.right_anchor):
            asset = next((
                item for item in manifest.assets
                if item.asset_id == anchor.project_asset_id
            ), None)
            if asset is None or asset.source_content_hash is None:
                raise HTTPException(409, detail="candidate preview source is no longer bound")
            media_path = _authorized_local_media_path(asset.source_ref)
            try:
                source_hash_after_pair = file_sha256(str(media_path)).lower()
            except OSError as exc:
                raise HTTPException(409, detail="candidate preview source became unavailable") from exc
            if source_hash_after_pair != asset.source_content_hash.lower():
                raise HTTPException(409, detail="candidate preview source changed during pair extraction")

        preview = ProjectStoryLinkCandidatePreview(
            project_id=project_id,
            project_manifest_id=manifest.manifest_id,
            project_revision=manifest.revision,
            story_graph_id=graph.graph_id,
            candidate_id=candidate.candidate_id,
            relation_kind=relation_kind,
            left=left_preview,
            right=right_preview,
        )
        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return _envelope(corr, preview.model_dump(mode="json"))
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(409, detail="candidate preview evidence is stale or inconsistent") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.put("/v1/projects/{project_id}/story-mention-reviews")
def project_story_mention_review_put_endpoint(
    project_id: str,
    request: Request,
    req: ProjectStoryMentionReviewRequest,
):
    """Append a caller correction snapshot without rewriting source observations."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import FilmObservation, ProjectStoryGraph
        from director_brain.project_story_graph import project_story_graph_id
        from director_brain.project_story_mention_review import (
            build_project_story_mention_review,
        )

        repo = _project_manifest_repository()
        manifest, context = _latest_project_context(repo, project_id)
        if manifest.revision != req.manifest_revision:
            raise HTTPException(
                409, detail="mention review requires the current manifest revision")
        _require_current_project_sources(manifest)
        expected_graph_id = project_story_graph_id(context)
        if req.story_graph_id != expected_graph_id:
            raise HTTPException(409, detail="mention review StoryGraph is stale")
        graph = repo.get(ProjectStoryGraph, expected_graph_id)
        if graph is None:
            raise HTTPException(
                409, detail="current project StoryGraph must exist before mention review")

        observations: dict[str, FilmObservation] = {}
        for observation_id in {
            item.anchor.observation_id for item in req.decisions
        }:
            observation = repo.get(FilmObservation, observation_id)
            if observation is None or observation.project_id != project_id:
                raise HTTPException(
                    409, detail="mention review cites missing or out-of-project evidence")
            observations[observation_id] = observation

        review = build_project_story_mention_review(
            manifest,
            context,
            graph,
            observations,
            [item.model_dump(mode="python") for item in req.decisions],
            review_revision=req.expected_review_revision + 1,
            created_at=int(time.time()),
        )
        request_identity = {
            "project_id": project_id,
            "manifest_revision": req.manifest_revision,
            "story_graph_id": req.story_graph_id,
            "expected_review_revision": req.expected_review_revision,
            "decisions": [item.model_dump(mode="json") for item in req.decisions],
        }
        request_fingerprint = hashlib.sha256(json.dumps(
            request_identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        _require_current_project_sources(manifest)
        try:
            stored_review, reused = repo.save_project_story_mention_review(
                project_id,
                review,
                expected_review_revision=req.expected_review_revision,
                idempotency_key=req.idempotency_key,
                request_fingerprint=request_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(
                409, detail="mention review rejected; reload current project evidence") from exc
        history = repo.list_project_story_mention_reviews(
            project_id, expected_graph_id)
        active_review = history[-1] if history else None
        submitted_review_is_current = (
            active_review is not None
            and active_review.review_id == stored_review.review_id
            and active_review.review_revision == stored_review.review_revision
        )
        return _envelope(corr, {
            "review": stored_review.model_dump(mode="json"),
            "review_revision": stored_review.review_revision,
            "currentness_state": (
                "current" if submitted_review_is_current else "superseded"),
            "active_review_id": (
                active_review.review_id if active_review else None),
            "active_review_revision": (
                active_review.review_revision if active_review else 0),
            "review_history_revisions": [item.review_revision for item in history],
            "replacement_semantics": "complete_current_mention_disposition_set",
            "decision_count": len(stored_review.decisions),
            "source_graph_mutated": False,
            "automatic_inference_state": "not_attempted",
            "quality_acceptance": "not_proven",
            "reused": reused,
            "persisted": True,
        })
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            409, detail="mention review evidence is stale or inconsistent") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/story-mention-reviews")
def project_story_mention_review_get_endpoint(
    project_id: str,
    request: Request,
):
    """Read the active source mention review and immutable revision history."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import ProjectStoryGraph
        from director_brain.project_story_graph import project_story_graph_id

        repo = _project_manifest_repository()
        _manifest, context = _latest_project_context(repo, project_id)
        graph_id = project_story_graph_id(context)
        graph = repo.get(ProjectStoryGraph, graph_id)
        if graph is None:
            raise HTTPException(404, detail="当前 revision 尚无 project StoryGraph")
        history = repo.list_project_story_mention_reviews(project_id, graph_id)
        active_review = history[-1] if history else None
        _validate_project_story_mention_review_snapshot(
            repo, active_review, _manifest, context, graph)
        return _envelope(corr, {
            "project_id": project_id,
            "story_graph_id": graph_id,
            "review": active_review.model_dump(mode="json") if active_review else None,
            "review_revision": active_review.review_revision if active_review else 0,
            "review_history": [item.model_dump(mode="json") for item in history],
            "replacement_semantics": "complete_current_mention_disposition_set",
            "source_graph_mutated": False,
            "source_hash_validation_state": "not_revalidated_by_read",
            "quality_acceptance": "not_proven",
        })
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            409, detail="mention review history is stale or inconsistent") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.put("/v1/projects/{project_id}/story-links")
def project_story_link_review_put_endpoint(
    project_id: str,
    request: Request,
    req: ProjectStoryLinkReviewRequest,
):
    """Append a caller-asserted link/disposition snapshot for the current graph."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import (
            FilmObservation,
            ProjectStoryGraph,
            ProjectStoryLinkComparison,
        )
        from director_brain.project_story_graph import project_story_graph_id
        from director_brain.project_story_link_review import (
            build_project_story_link_review,
            project_story_link_review_is_stale,
            validate_project_story_comparison_dispositions,
        )

        repo = _project_manifest_repository()
        manifest, context = _latest_project_context(repo, project_id)
        if manifest.revision != req.manifest_revision:
            raise HTTPException(409, detail="story link review requires the current manifest revision")
        _require_current_project_sources(manifest)
        expected_graph_id = project_story_graph_id(context)
        if req.story_graph_id != expected_graph_id:
            raise HTTPException(409, detail="story link review graph is stale")
        graph = repo.get(ProjectStoryGraph, expected_graph_id)
        if graph is None:
            raise HTTPException(409, detail="current project StoryGraph must exist before link review")

        request_identity = {
            "project_id": project_id,
            "manifest_revision": req.manifest_revision,
            "story_graph_id": req.story_graph_id,
            "expected_review_revision": req.expected_review_revision,
            "links": [link.model_dump(mode="json") for link in req.links],
            "relations": [relation.model_dump(mode="json") for relation in req.relations],
            "comparison_dispositions": [
                item.model_dump(mode="json")
                for item in req.comparison_dispositions
            ],
        }
        request_fingerprint = hashlib.sha256(json.dumps(
            request_identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

        def write_response(stored_review, reused):
            mention_history, history = repo.list_project_story_review_histories(
                project_id, expected_graph_id)
            active_mention_review = mention_history[-1] if mention_history else None
            active_review = history[-1] if history else None
            submitted_review_is_stale = project_story_link_review_is_stale(
                stored_review, active_mention_review)
            submitted_review_is_latest = (
                active_review is not None
                and active_review.review_id == stored_review.review_id
                and active_review.review_revision == stored_review.review_revision
            )
            submitted_currentness_state = (
                "stale" if submitted_review_is_stale
                else "current" if submitted_review_is_latest
                else "superseded"
            )
            active_review_is_current = (
                active_review is not None
                and not project_story_link_review_is_stale(
                    active_review, active_mention_review)
            )
            return _envelope(corr, {
                "review": stored_review.model_dump(mode="json"),
                "review_revision": stored_review.review_revision,
                "currentness_state": submitted_currentness_state,
                "active_link_count": (
                    len(active_review.links) if active_review_is_current else 0),
                "active_relation_count": (
                    len(active_review.relations) if active_review_is_current else 0),
                "active_link_review_id": (
                    active_review.review_id if active_review_is_current else None),
                "active_link_review_revision": (
                    active_review.review_revision if active_review_is_current else 0),
                "source_mention_review_id": (
                    active_mention_review.review_id if active_mention_review else None),
                "source_mention_review_revision": (
                    active_mention_review.review_revision if active_mention_review else 0),
                "review_history_revisions": [item.review_revision for item in history],
                "replacement_semantics": "complete_current_link_and_relation_set",
                "comparison_disposition_count": len(stored_review.comparison_dispositions),
                "comparison_results_remain_unreviewed": True,
                "actor_identity_state": "caller_asserted",
                "rights_evidence_state": "declared_unverified",
                "automatic_inference_state": "not_attempted",
                "quality_acceptance": "not_proven",
                "reused": reused,
                "persisted": True,
            })

        try:
            prior_review = repo.get_idempotent_project_story_link_review(
                project_id,
                expected_graph_id,
                idempotency_key=req.idempotency_key,
                request_fingerprint=request_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(409, detail="story link review idempotency conflict") from exc
        if prior_review is not None:
            return write_response(prior_review, True)

        mention_history = repo.list_project_story_mention_reviews(
            project_id, expected_graph_id)
        active_mention_review = mention_history[-1] if mention_history else None

        observation_ids = {
            anchor.observation_id
            for link in req.links
            for anchor in link.anchors
        }
        observation_ids.update(
            endpoint.observation_id
            for relation in req.relations
            for endpoint in (relation.from_anchor, relation.to_anchor)
        )
        observations: dict[str, FilmObservation] = {}
        for observation_id in observation_ids:
            observation = repo.get(FilmObservation, observation_id)
            if observation is None or observation.project_id != project_id:
                raise HTTPException(409, detail="story link review cites missing or out-of-project evidence")
            observations[observation_id] = observation

        comparisons = {}
        for disposition in req.comparison_dispositions:
            comparison = repo.get(
                ProjectStoryLinkComparison, disposition.comparison_id)
            if comparison is None or comparison.project_id != project_id:
                raise HTTPException(
                    409, detail="comparison disposition cites missing or out-of-project evidence")
            comparisons[disposition.comparison_id] = comparison

        review = build_project_story_link_review(
            manifest,
            context,
            graph,
            observations,
            [link.model_dump(mode="python") for link in req.links],
            review_revision=req.expected_review_revision + 1,
            relation_selections=[
                relation.model_dump(mode="python") for relation in req.relations
            ],
            comparison_dispositions=[
                item.model_dump(mode="python")
                for item in req.comparison_dispositions
            ],
            mention_review=active_mention_review,
        )
        try:
            validate_project_story_comparison_dispositions(review, comparisons)
        except ValueError as exc:
            raise HTTPException(
                409, detail="comparison disposition does not match exact project evidence"
            ) from exc
        _require_current_project_sources(manifest)
        try:
            stored_review, reused = repo.save_project_story_link_review(
                project_id,
                review,
                expected_review_revision=req.expected_review_revision,
                idempotency_key=req.idempotency_key,
                request_fingerprint=request_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(409, detail="story link review rejected; reload the current project evidence") from exc
        return write_response(stored_review, reused)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(409, detail="story link review evidence is stale or inconsistent") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/story-links")
def project_story_link_review_get_endpoint(
    project_id: str,
    request: Request,
    revision: int | None = Query(default=None, ge=1),
):
    """Read the active link overlay and immutable correction history."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import FilmContextSnapshot, ProjectStoryGraph
        from director_brain.project_story_graph import project_story_graph_id
        from director_brain.context_gateway import project_context_id
        from director_brain.project_story_link_review import (
            project_story_link_review_is_stale,
        )

        repo = _project_manifest_repository()
        if revision is None:
            _manifest, context = _latest_project_context(repo, project_id)
        else:
            manifest = repo.get_project_manifest(project_id, revision)
            if manifest is None:
                raise HTTPException(404, detail="指定 project manifest revision 不存在")
            context = repo.get(FilmContextSnapshot, project_context_id(manifest))
            if (
                context is None
                or context.project_id != project_id
                or context.project_manifest_id != manifest.manifest_id
                or context.project_revision != manifest.revision
            ):
                raise HTTPException(404, detail="指定 revision 尚无 project context")
        graph_id = project_story_graph_id(context)
        graph = repo.get(ProjectStoryGraph, graph_id)
        if graph is None:
            raise HTTPException(404, detail="指定 revision 尚无 project StoryGraph")
        mention_history, history = repo.list_project_story_review_histories(
            project_id, graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        active_review = history[-1] if history else None
        currentness_state = (
            "absent" if active_review is None
            else "stale" if project_story_link_review_is_stale(
                active_review, active_mention_review)
            else "current"
        )
        return _envelope(corr, {
            "project_id": project_id,
            "story_graph_id": graph_id,
            "review": active_review.model_dump(mode="json") if active_review else None,
            "review_revision": active_review.review_revision if active_review else 0,
            "currentness_state": currentness_state,
            "active_link_count": (
                len(active_review.links)
                if active_review is not None and currentness_state == "current"
                else 0
            ),
            "active_relation_count": (
                len(active_review.relations)
                if active_review is not None and currentness_state == "current"
                else 0
            ),
            "source_mention_review_id": (
                active_mention_review.review_id if active_mention_review else None),
            "source_mention_review_revision": (
                active_mention_review.review_revision if active_mention_review else 0),
            "review_history_revisions": [item.review_revision for item in history],
            "replacement_semantics": "complete_current_link_and_relation_set",
            "source_hash_validation_state": "not_revalidated_by_read",
            "rights_evidence_state": "declared_unverified",
            "automatic_inference_state": "not_attempted",
            "quality_acceptance": "not_proven",
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/story-link-candidates")
def project_story_link_candidates_get_endpoint(
    project_id: str,
    request: Request,
    manifest_revision: int = Query(ge=1),
    story_graph_id: str = Query(min_length=1),
    relation_kind: Literal["person_identity", "event_identity"] = Query(),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    selected_anchor_story_graph_node_id: str | None = Query(default=None, min_length=1),
    selected_counterpart_project_asset_id: str | None = Query(
        default=None,
        min_length=1,
        description=(
            "Optional project asset to pair with the selected mention anchor. "
            "Requires selected_anchor_story_graph_node_id. Ranking remains "
            "off unless ranking_profile is explicitly requested; no identity "
            "inference is performed."),
    ),
    selected_project_asset_pair: list[str] | None = Query(
        default=None,
        min_length=2,
        max_length=2,
        description=(
            "Optional pair of project asset IDs. Returns every eligible pair "
            "between those assets in manifest order, without ranking or "
            "identity inference; mutually exclusive with anchor scope."),
    ),
    expected_candidate_set_id: str | None = Query(
        default=None,
        min_length=1,
        description=(
            "Candidate-set identity returned by the first page. Required for "
            "offset greater than zero; a changed set returns HTTP 409."),
    ),
    ranking_profile: Literal["bge_m3_shadow_v1"] | None = Query(
        default=None,
        description=(
            "Optional local shadow ranking over the complete candidate scope. "
            "Requires a selected mention anchor. Scores are uncalibrated text "
            "similarities; all candidates remain and no identity is inferred."),
    ),
):
    """Page exhaustive candidate pairs, optionally shadow-ranked for one anchor."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import ProjectStoryGraph
        from director_brain.models.film_context import FilmContextSnapshot
        from director_brain.project_story_link_candidates import (
            CandidateAnchorNotEligibleError,
            CandidateScopeNotEligibleError,
            bind_project_story_link_candidate_progress,
            build_project_story_link_candidate_page,
        )
        from director_brain.utils import project_story_link_comparison_id

        if offset > 0 and expected_candidate_set_id is None:
            raise HTTPException(
                422,
                detail=(
                    "candidate continuation requires expected_candidate_set_id "
                    "from the first page"),
            )
        if ranking_profile is not None and selected_anchor_story_graph_node_id is None:
            raise HTTPException(
                422,
                detail="shadow ranking requires selected_anchor_story_graph_node_id",
            )

        repo = _project_manifest_repository()
        manifest = repo.latest_project_manifest(project_id)
        if manifest is None:
            raise HTTPException(404, detail="project manifest 不存在")
        if manifest.revision != manifest_revision:
            raise HTTPException(409, detail="candidate request requires the current manifest revision")
        if manifest.analysis_profile != ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1:
            raise HTTPException(
                409, detail="cross-asset candidate enumeration requires local_vlm_shadow_v1")
        from director_brain.context_gateway import PROJECT_CONTEXT_SCHEMA_VERSION

        graph = repo.get(ProjectStoryGraph, story_graph_id)
        if graph is None:
            raise HTTPException(409, detail="current project StoryGraph must exist before candidate enumeration")
        context = repo.get(FilmContextSnapshot, graph.context_id)
        if context is None:
            raise HTTPException(409, detail="candidate StoryGraph context does not exist")
        if context.schema_version != PROJECT_CONTEXT_SCHEMA_VERSION:
            raise HTTPException(
                409, detail="candidate StoryGraph context schema is stale")
        if (
            graph.project_id != project_id
            or graph.project_manifest_id != manifest.manifest_id
            or graph.project_revision != manifest.revision
            or graph.context_id != context.context_id
            or graph.analysis_fingerprint != context.analysis_fingerprint
            or context.project_id != project_id
            or context.project_manifest_id != manifest.manifest_id
            or context.project_revision != manifest.revision
            or context.invalidated_at is not None
        ):
            raise HTTPException(409, detail="candidate StoryGraph is stale or inconsistent")

        # This read-only listing is a view over persisted, hash-bound evidence.
        # It does not re-read potentially large media files. Operations that
        # consume source media or create a review/Plan revalidate source hashes.
        mention_history = repo.list_project_story_mention_reviews(
            project_id, graph.graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        _validate_project_story_mention_review_snapshot(
            repo, active_mention_review, manifest, context, graph)
        def build_candidate_page(page_limit: int, page_offset: int):
            return build_project_story_link_candidate_page(
                manifest,
                graph,
                relation_kind=relation_kind,
                limit=page_limit,
                offset=page_offset,
                mention_review=active_mention_review,
                selected_anchor_story_graph_node_id=(
                    selected_anchor_story_graph_node_id),
                selected_counterpart_project_asset_id=(
                    selected_counterpart_project_asset_id),
                selected_project_asset_pair=(
                    tuple(selected_project_asset_pair)
                    if selected_project_asset_pair is not None else None),
            )

        provider_invocation_count = 0
        embedding_invocation_count = 0
        if ranking_profile is None:
            page = build_candidate_page(limit, offset)
        else:
            base_page = build_candidate_page(200, 0)
            all_candidates = list(base_page.candidates)
            while len(all_candidates) < base_page.total:
                next_page = build_candidate_page(200, len(all_candidates))
                if next_page.candidate_set_id != base_page.candidate_set_id:
                    raise HTTPException(
                        409, detail="candidate set changed during shadow ranking")
                if not next_page.candidates:
                    raise HTTPException(
                        409, detail="candidate scope could not be read completely")
                all_candidates.extend(next_page.candidates)
            from director_brain.config import load_settings

            ranking_base_url = load_settings().ollama_base_url
            binding = resolve_project_story_link_ranking_binding(
                ranking_base_url)
            try:
                ranking_snapshot = repo.get_project_story_link_ranking_snapshot(
                    project_id,
                    base_page.candidate_set_id,
                    RANKING_PROFILE_ID,
                    RANKING_MODEL,
                    binding.digest,
                    RANKING_METRIC,
                )
            except ValueError as exc:
                raise ProjectStoryLinkRankingError(
                    "ranking_cache_invalid") from exc
            if ranking_snapshot is None:
                ranking = rank_project_story_link_candidates(
                    all_candidates,
                    base_url=ranking_base_url,
                    binding=binding,
                )
                try:
                    ranking_snapshot = project_story_link_ranking_snapshot_entries(
                        ranking.candidates)
                    repo.save_project_story_link_ranking_snapshot(
                        project_id,
                        base_page.candidate_set_id,
                        RANKING_PROFILE_ID,
                        RANKING_MODEL,
                        binding.digest,
                        RANKING_METRIC,
                        ranking_snapshot,
                    )
                    # Read the canonical immutable row back after save. This also
                    # detects a conflicting concurrent result for the same key.
                    ranking_snapshot = repo.get_project_story_link_ranking_snapshot(
                        project_id,
                        base_page.candidate_set_id,
                        RANKING_PROFILE_ID,
                        RANKING_MODEL,
                        binding.digest,
                        RANKING_METRIC,
                    )
                except ValueError as exc:
                    raise ProjectStoryLinkRankingError(
                        "ranking_cache_conflict") from exc
                if ranking_snapshot is None:
                    raise ProjectStoryLinkRankingError("ranking_cache_write_failed")
                provider_invocation_count = ranking.provider_invocation_count
                embedding_invocation_count = ranking.embedding_invocation_count
            else:
                # The model inventory call above validates the digest; cached
                # pagination requires no embedding calls.
                provider_invocation_count = 1
                embedding_invocation_count = 0
            try:
                ranked_candidates = apply_project_story_link_ranking_snapshot(
                    all_candidates, ranking_snapshot)
            except ProjectStoryLinkRankingError:
                raise
            ranked_set_id = ranked_candidate_set_id(
                base_page.candidate_set_id, binding.digest)
            ranked_candidates = ranked_candidates[offset:offset + limit]
            page_data = base_page.model_dump(mode="python")
            page_data.update({
                "candidate_set_id": ranked_set_id,
                "candidates": [item.model_dump(mode="python")
                               for item in ranked_candidates],
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(ranked_candidates) < base_page.total,
                "ranking_state": RANKING_STATE,
                "ranking_profile_id": RANKING_PROFILE_ID,
                "ranking_model": RANKING_MODEL,
                "ranking_model_digest": binding.digest,
                "ranking_metric": RANKING_METRIC,
                "ranking_score_semantics": "uncalibrated_similarity_only",
                "selection_semantics": (
                    "all_eligible_pairs_for_anchor_and_counterpart_asset_ranked_by_shadow_similarity"
                    if selected_counterpart_project_asset_id is not None
                    else "all_eligible_cross_asset_pairs_for_selected_mention_ranked_by_shadow_similarity"
                ),
            })
            from director_brain.models.project_story_link_candidates import (
                ProjectStoryLinkCandidatePage,
            )

            page = ProjectStoryLinkCandidatePage.model_validate(page_data)
        if (
            expected_candidate_set_id is not None
            and expected_candidate_set_id != page.candidate_set_id
        ):
            raise HTTPException(
                409,
                detail=(
                    "candidate set changed; restart pagination from the first page"),
            )
        comparison_ids = [
            project_story_link_comparison_id(project_id, candidate.candidate_id)
            for candidate in page.candidates
        ]
        comparisons_by_candidate = repo.get_project_story_link_comparisons_by_candidate_ids(
            project_id, [candidate.candidate_id for candidate in page.candidates])
        legacy_comparisons_by_id = repo.get_project_story_link_comparisons_by_ids(
            project_id, comparison_ids)
        review_history = (
            repo.list_project_story_link_reviews(project_id, graph.graph_id)
            if page.candidates else []
        )
        page = bind_project_story_link_candidate_progress(
            page, comparisons_by_candidate, legacy_comparisons_by_id,
            review_history)
        return _envelope(corr, {
            "candidate_page": page.model_dump(mode="json"),
            "derived_from_persisted_evidence": True,
            "provider_invocation_count": provider_invocation_count,
            "embedding_invocation_count": embedding_invocation_count,
            "automatic_link_created": False,
            "relation_inference_state": "EXPERIMENTAL",
            "quality_acceptance": "not_proven",
            "identity_decision_state": "not_attempted",
        })
    except HTTPException:
        raise
    except CandidateAnchorNotEligibleError as exc:
        raise HTTPException(
            422,
            detail=(
                "selected anchor must be an eligible current mention of the "
                "requested relation kind"),
        ) from exc
    except CandidateScopeNotEligibleError as exc:
        raise HTTPException(
            422,
            detail=(
                "selected asset-pair or anchor/counterpart scope is invalid "
                "for the current manifest"),
        ) from exc
    except ProjectStoryLinkRankingError as exc:
        status_code = 503 if exc.code in {
            "provider_unavailable", "ranking_model_unavailable"
        } else 502
        raise HTTPException(
            status_code,
            detail=f"local shadow ranking unavailable ({exc.code})",
        ) from exc
    except ValueError as exc:
        raise HTTPException(409, detail="candidate evidence is stale or inconsistent") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.post("/v1/projects/{project_id}/story-link-comparisons")
def project_story_link_comparison_post_endpoint(
    project_id: str,
    request: Request,
    req: ProjectStoryLinkComparisonRequest,
):
    """Persist one local-only, human-unreviewed cross-asset VLM comparison."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.config import load_settings
        from director_brain.models import (
            ClaimKind,
            FILM_OBSERVATION_SCHEMA_VERSION,
            FilmContextSnapshot,
            FilmObservation,
            ProjectStoryGraph,
            ProjectStoryLinkAnchor,
            ProjectStoryLinkComparison,
            ProjectStoryLinkEventEvidence,
            TimebaseUnit,
        )
        from director_brain.project_story_graph import project_story_graph_id
        from director_brain.project_story_link_candidates import (
            project_story_link_candidate_id_from_pair,
        )
        from director_brain.context_gateway import project_context_id
        from director_brain.utils import (
            file_sha256,
            project_story_link_comparison_id,
        )
        from observation_service.keyframe import (
            MULTI_FRAME_SAMPLE_POSITIONS,
            extract_keyframes,
        )
        from observation_service.ollama_vlm_adapter import (
            LocalVLMRuntimeBindingError,
            OllamaVLMAdapter,
            PROJECT_LINK_COMPARISON_GENERATION_PROFILE,
            PROJECT_LINK_COMPARISON_PROMPT_VERSION,
            PROJECT_LINK_EVENT_COMPARISON_PROMPT_VERSION,
            PROJECT_LINK_COMPARISON_SAMPLING_PROFILE,
            build_project_link_comparison_prompt,
        )

        repo = _project_manifest_repository()
        manifest, context = _latest_project_context(repo, project_id)
        if manifest.revision != req.manifest_revision:
            raise HTTPException(409, detail="comparison requires the current manifest revision")
        if manifest.analysis_profile != ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1:
            raise HTTPException(
                409, detail="cross-asset comparison requires local_vlm_shadow_v1")
        _require_current_project_sources(manifest)
        expected_graph_id = project_story_graph_id(context)
        if req.story_graph_id != expected_graph_id:
            raise HTTPException(409, detail="comparison StoryGraph is stale")
        graph = repo.get(ProjectStoryGraph, expected_graph_id)
        if graph is None:
            raise HTTPException(409, detail="current project StoryGraph must exist before comparison")
        if (
            graph.project_id != project_id
            or graph.project_manifest_id != manifest.manifest_id
            or graph.project_revision != manifest.revision
            or graph.context_id != context.context_id
            or graph.analysis_fingerprint != context.analysis_fingerprint
        ):
            raise HTTPException(409, detail="comparison StoryGraph is stale or inconsistent")

        mention_history = repo.list_project_story_mention_reviews(
            project_id, graph.graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        _validate_project_story_mention_review_snapshot(
            repo, active_mention_review, manifest, context, graph)
        expected_review_id = (
            active_mention_review.review_id
            if active_mention_review is not None else None)
        expected_review_revision = (
            active_mention_review.review_revision
            if active_mention_review is not None else 0)
        if (
            req.source_mention_review_id != expected_review_id
            or req.source_mention_review_revision != expected_review_revision
        ):
            raise HTTPException(
                409, detail="comparison source mention review is stale")
        if active_mention_review is not None and (
            req.left_story_graph_node_id is None
            or req.right_story_graph_node_id is None
        ):
            raise HTTPException(
                409,
                detail=(
                    "comparison under an active source mention review requires "
                    "exact source mention nodes"),
            )

        left = repo.get(FilmObservation, req.left_observation_id)
        right = repo.get(FilmObservation, req.right_observation_id)
        if left is None or right is None:
            raise HTTPException(404, detail="comparison observation does not exist")
        if (
            left.observation_id not in context.evidence_refs
            or right.observation_id not in context.evidence_refs
            or left.project_id != project_id
            or right.project_id != project_id
            or left.claim_kind != ClaimKind.MODEL_OBSERVATION
            or right.claim_kind != ClaimKind.MODEL_OBSERVATION
            or left.provider != "ollama_qwen3_vl"
            or right.provider != "ollama_qwen3_vl"
            or left.timebase_unit != TimebaseUnit.MICROSECONDS
            or right.timebase_unit != TimebaseUnit.MICROSECONDS
            or left.timebase != 1_000_000
            or right.timebase != 1_000_000
            or left.end_frame <= left.start_frame
            or right.end_frame <= right.start_frame
        ):
            raise HTTPException(409, detail="comparison requires current local-VLM observations")
        if left.project_asset_id == right.project_asset_id:
            raise HTTPException(422, detail="comparison requires observations from different assets")

        assets = {item.asset_id: item for item in manifest.assets}
        left_asset = assets.get(left.project_asset_id or "")
        right_asset = assets.get(right.project_asset_id or "")
        for asset, observation in ((left_asset, left), (right_asset, right)):
            if (
                asset is None
                or asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED
                or asset.source_identity_state != "locally_verified"
                or asset.source_content_hash is None
                or observation.media_hash.lower() != asset.source_content_hash.lower()
            ):
                raise HTTPException(403, detail="comparison source is not locally authorized and verified")

        def _stored_mention_node(observation, node_id: str | None, relation_kind: str):
            if node_id is None:
                return None
            expected_type = (
                "person_mention" if relation_kind == "person_identity"
                else "event_mention"
            )
            asset_graph = next(
                (item for item in graph.assets
                 if item.asset_id == observation.project_asset_id),
                None,
            )
            node = next(
                (item for item in asset_graph.story_graph.nodes
                 if item.node_id == node_id),
                None,
            ) if asset_graph and asset_graph.story_graph else None
            anchor = node.attributes.get("source_anchor") if node else None
            if (
                node is None
                or node.node_type.value != expected_type
                or node.ref_id != observation.observation_id
                or not isinstance(anchor, dict)
                or anchor.get("project_asset_id") != observation.project_asset_id
                or str(anchor.get("source_content_hash", "")).lower()
                != observation.media_hash.lower()
                or anchor.get("source_start") != observation.start_frame
                or anchor.get("source_end") != observation.end_frame
                or anchor.get("timebase") != observation.timebase
                or anchor.get("timebase_unit") != observation.timebase_unit.value
            ):
                raise HTTPException(
                    409, detail="comparison mention node is stale or outside its source evidence")
            return node

        def _stored_person_description(
            observation, requested: str | None, node_id: str | None,
        ):
            if requested is None:
                return None
            node = _stored_mention_node(
                observation, node_id, "person_identity")
            if node is not None:
                decision = next((
                    item for item in (
                        active_mention_review.decisions
                        if active_mention_review is not None else []
                    )
                    if item.relation_kind == "person_identity"
                    and item.anchor.project_asset_id == observation.project_asset_id
                    and item.anchor.story_graph_node_id == node_id
                    and item.anchor.observation_id == observation.observation_id
                ), None)
                if decision is not None and decision.decision == "rejected":
                    raise HTTPException(
                        409, detail="person mention was rejected in the active caller review")
                effective_description = (
                    decision.corrected_person_description
                    if decision is not None
                    and decision.decision == "corrected_by_caller"
                    else node.attributes.get("description")
                )
                if effective_description != requested:
                    raise HTTPException(
                        422, detail="person description differs from the active source mention review")
                return requested
            try:
                payload = json.loads(observation.claim)
            except (TypeError, json.JSONDecodeError) as exc:
                raise HTTPException(409, detail="observation person evidence is malformed") from exc
            people = payload.get("people") if isinstance(payload, dict) else None
            if not isinstance(people, list) or requested not in people:
                raise HTTPException(
                    422, detail="person description must exactly match stored observation evidence")
            return requested

        def _stored_event_evidence(observation, node_id: str | None):
            try:
                payload = json.loads(observation.claim)
            except (TypeError, json.JSONDecodeError) as exc:
                raise HTTPException(409, detail="observation event evidence is malformed") from exc
            if not isinstance(payload, dict):
                raise HTTPException(409, detail="observation event evidence is malformed")
            fields = {
                key: payload[key]
                for key in ("action_type", "scene_description", "temporal_notes")
                if isinstance(payload.get(key), str) and payload[key].strip()
            }
            node = _stored_mention_node(
                observation, node_id, "event_identity")
            if node is not None and node.attributes.get("semantic_fields") != fields:
                raise HTTPException(
                    409, detail="event mention node does not match its stored evidence")
            try:
                source_evidence = ProjectStoryLinkEventEvidence.model_validate(fields)
            except ValidationError as exc:
                raise HTTPException(
                    422,
                    detail=(
                        "event comparison requires specific stored scene or "
                        "temporal evidence from both observations"),
                ) from exc
            if node is None:
                return source_evidence
            decision = next((
                item for item in (
                    active_mention_review.decisions
                    if active_mention_review is not None else []
                )
                if item.relation_kind == "event_identity"
                and item.anchor.project_asset_id == observation.project_asset_id
                and item.anchor.story_graph_node_id == node_id
                and item.anchor.observation_id == observation.observation_id
            ), None)
            if decision is not None and decision.decision == "rejected":
                raise HTTPException(
                    409, detail="event mention was rejected in the active caller review")
            if decision is not None and decision.decision == "corrected_by_caller":
                if decision.corrected_event_evidence is None:
                    raise HTTPException(
                        409, detail="active event mention correction is malformed")
                return decision.corrected_event_evidence
            return source_evidence

        left_person = _stored_person_description(
            left, req.left_person_description, req.left_story_graph_node_id)
        right_person = _stored_person_description(
            right, req.right_person_description, req.right_story_graph_node_id)
        left_event = right_event = None
        if req.relation_kind == "event_identity":
            left_event = _stored_event_evidence(left, req.left_story_graph_node_id)
            right_event = _stored_event_evidence(right, req.right_story_graph_node_id)
        left_anchor = ProjectStoryLinkAnchor(
            project_asset_id=left_asset.asset_id,
            source_asset_id=left.media_asset_id,
            story_graph_node_id=req.left_story_graph_node_id,
            observation_id=left.observation_id,
            source_content_hash=left_asset.source_content_hash.lower(),
            source_start=left.start_frame,
            source_end=left.end_frame,
            timebase=left.timebase,
            timebase_unit=left.timebase_unit,
        )
        right_anchor = ProjectStoryLinkAnchor(
            project_asset_id=right_asset.asset_id,
            source_asset_id=right.media_asset_id,
            story_graph_node_id=req.right_story_graph_node_id,
            observation_id=right.observation_id,
            source_content_hash=right_asset.source_content_hash.lower(),
            source_start=right.start_frame,
            source_end=right.end_frame,
            timebase=right.timebase,
            timebase_unit=right.timebase_unit,
        )
        if req.candidate_id is not None:
            try:
                expected_candidate_id = project_story_link_candidate_id_from_pair(
                    graph.graph_id,
                    req.relation_kind,
                    left_anchor,
                    right_anchor,
                    left_person_description=left_person,
                    right_person_description=right_person,
                    left_event_evidence=left_event,
                    right_event_evidence=right_event,
                    source_mention_review_id=expected_review_id,
                    source_mention_review_revision=expected_review_revision,
                )
            except ValueError as exc:
                raise HTTPException(
                    409, detail="candidate comparison requires exact graph mention nodes"
                ) from exc
            if expected_candidate_id != req.candidate_id:
                raise HTTPException(
                    409, detail="candidate id does not match the exact graph mention pair")

        settings = load_settings()
        adapter = OllamaVLMAdapter(
            model=settings.project_local_vlm_model,
            base_url=settings.ollama_base_url,
            model_digest=settings.project_local_vlm_digest,
            runtime_version=settings.project_local_vlm_runtime_version,
            enforce_loopback=True,
        )
        model_version = adapter.model
        if ":" not in model_version:
            model_version += ":latest"
        if adapter.model_digest:
            model_version += (
                "@sha256:" + adapter.model_digest.removeprefix("sha256:").lower())
        if adapter.runtime_version:
            model_version += "|ollama:" + adapter.runtime_version

        exact_prompt = build_project_link_comparison_prompt(
            req.relation_kind,
            left_person,
            right_person,
            left_event_evidence=(
                left_event.model_dump(mode="json", exclude_none=True)
                if left_event else None
            ),
            right_event_evidence=(
                right_event.model_dump(mode="json", exclude_none=True)
                if right_event else None
            ),
        )
        prompt_version = (
            PROJECT_LINK_EVENT_COMPARISON_PROMPT_VERSION
            if req.relation_kind == "event_identity"
            else PROJECT_LINK_COMPARISON_PROMPT_VERSION
        )
        prompt_sha256 = hashlib.sha256(exact_prompt.encode("utf-8")).hexdigest()
        try:
            ffmpeg_version = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.splitlines()[0].strip()[:120]
        except Exception:  # noqa: BLE001
            ffmpeg_version = "unavailable"
        sampling_profile = (
            PROJECT_LINK_COMPARISON_SAMPLING_PROFILE
            + "|ffmpeg=" + ffmpeg_version
        )

        request_identity = {
            "project_id": project_id,
            "manifest_id": manifest.manifest_id,
            "manifest_revision": manifest.revision,
            "context_id": context.context_id,
            "analysis_fingerprint": context.analysis_fingerprint,
            "story_graph_id": graph.graph_id,
            "source_mention_review_id": (
                active_mention_review.review_id
                if active_mention_review is not None else None),
            "source_mention_review_revision": (
                active_mention_review.review_revision
                if active_mention_review is not None else 0),
            "candidate_id": req.candidate_id,
            "attempt_number": req.attempt_number,
            "relation_kind": req.relation_kind,
            "left_anchor": left_anchor.model_dump(mode="json"),
            "right_anchor": right_anchor.model_dump(mode="json"),
            "left_person_description": left_person,
            "right_person_description": right_person,
            "left_event_evidence": (
                left_event.model_dump(mode="json", exclude_none=True)
                if left_event else None
            ),
            "right_event_evidence": (
                right_event.model_dump(mode="json", exclude_none=True)
                if right_event else None
            ),
            "provider": "ollama_qwen3_vl",
            "model_version": model_version,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "generation_profile": PROJECT_LINK_COMPARISON_GENERATION_PROFILE,
            "sampling_profile": sampling_profile,
            "frame_extractor_version": ffmpeg_version,
            "observation_schema_version": FILM_OBSERVATION_SCHEMA_VERSION,
        }
        request_fingerprint = hashlib.sha256(json.dumps(
            request_identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        key_hash = hashlib.sha256(req.idempotency_key.encode("utf-8")).hexdigest()
        comparison_id = project_story_link_comparison_id(
            project_id, req.idempotency_key)
        existing = repo.get(ProjectStoryLinkComparison, comparison_id)
        if existing is not None:
            if (
                existing.project_id != project_id
                or existing.idempotency_key_sha256 != key_hash
                or existing.request_fingerprint != request_fingerprint
            ):
                raise HTTPException(409, detail="idempotency key is bound to another comparison")
            return _envelope(corr, {
                "comparison": existing.model_dump(mode="json"),
                "reused": True,
                "automatic_link_created": False,
                "director_plan_updated": False,
                "relation_inference_state": "EXPERIMENTAL",
                "quality_acceptance": "not_proven",
            })

        if req.candidate_id is not None:
            candidate_history = repo.get_project_story_link_comparisons_by_candidate_ids(
                project_id, [req.candidate_id]).get(req.candidate_id, [])
            candidate_history.sort(key=lambda item: (
                item.attempt_number or 0,
                item.created_at or 0,
                item.comparison_id,
            ))
            if candidate_history:
                previous = candidate_history[-1]
                expected_attempt_number = (previous.attempt_number or 1) + 1
                if (
                    req.attempt_number != expected_attempt_number
                    or previous.run_state != "failed"
                ):
                    raise HTTPException(
                        409,
                        detail="candidate retry must follow its latest failed attempt",
                    )
            elif req.attempt_number != 1:
                legacy_first_id = project_story_link_comparison_id(
                    project_id, req.candidate_id)
                legacy_first = repo.get(ProjectStoryLinkComparison, legacy_first_id)
                legacy_is_same_candidate = False
                if legacy_first is not None and legacy_first.candidate_id is None:
                    try:
                        legacy_is_same_candidate = (
                            project_story_link_candidate_id_from_pair(
                                legacy_first.story_graph_id,
                                legacy_first.relation_kind,
                                legacy_first.left_anchor,
                                legacy_first.right_anchor,
                                left_person_description=(
                                    legacy_first.left_person_description),
                                right_person_description=(
                                    legacy_first.right_person_description),
                                left_event_evidence=legacy_first.left_event_evidence,
                                right_event_evidence=legacy_first.right_event_evidence,
                                source_mention_review_id=(
                                    legacy_first.source_mention_review_id),
                                source_mention_review_revision=(
                                    legacy_first.source_mention_review_revision),
                            ) == req.candidate_id
                        )
                    except ValueError:
                        legacy_is_same_candidate = False
                if req.attempt_number != 2 or (
                    legacy_is_same_candidate
                    and legacy_first is not None
                    and legacy_first.run_state != "failed"
                ):
                    raise HTTPException(
                        409,
                        detail="candidate retry number is not the next failed attempt",
                    )

        failure_code = None
        result = None
        left_frame_hashes = []
        right_frame_hashes = []
        try:
            if ffmpeg_version == "unavailable":
                failure_code = "sampling_failed"
            else:
                adapter.verify_runtime_binding()
                left_path = _authorized_local_media_path(left_asset.source_ref)
                right_path = _authorized_local_media_path(right_asset.source_ref)
                with extract_keyframes(
                    str(left_path), left.start_frame, left.end_frame,
                    positions=MULTI_FRAME_SAMPLE_POSITIONS,
                    local_only=True,
                ) as left_frames:
                    with extract_keyframes(
                        str(right_path), right.start_frame, right.end_frame,
                        positions=MULTI_FRAME_SAMPLE_POSITIONS,
                        local_only=True,
                    ) as right_frames:
                        left_frame_hashes = [
                            file_sha256(frame).lower() for frame in left_frames]
                        right_frame_hashes = [
                            file_sha256(frame).lower() for frame in right_frames]
                        if len(left_frames) != 3 or len(right_frames) != 3:
                            failure_code = "sampling_failed"
                        else:
                            result = adapter.compare_cross_asset_frames(
                                left_frames,
                                right_frames,
                                relation_kind=req.relation_kind,
                                left_person_description=left_person,
                                right_person_description=right_person,
                                left_event_evidence=(
                                    left_event.model_dump(
                                        mode="json", exclude_none=True)
                                    if left_event else None
                                ),
                                right_event_evidence=(
                                    right_event.model_dump(
                                        mode="json", exclude_none=True)
                                    if right_event else None
                                ),
                            )
        except LocalVLMRuntimeBindingError:
            failure_code = "runtime_binding_failed"
        except OSError:
            failure_code = "sampling_failed"
        except Exception:  # noqa: BLE001
            failure_code = "provider_unavailable"

        try:
            current_left_hash = file_sha256(
                str(_authorized_local_media_path(left_asset.source_ref))).lower()
            current_right_hash = file_sha256(
                str(_authorized_local_media_path(right_asset.source_ref))).lower()
        except OSError:
            current_left_hash = current_right_hash = ""
        if (
            current_left_hash != left_asset.source_content_hash.lower()
            or current_right_hash != right_asset.source_content_hash.lower()
        ):
            failure_code = "source_changed"
            result = None

        if failure_code is None:
            if not isinstance(result, dict) or result.get("status") != "OBSERVED":
                failure_type = result.get("failure_type") if isinstance(result, dict) else None
                failure_code = {
                    "DECODE": "sampling_failed",
                    "NETWORK": "provider_unavailable",
                    "RATE_LIMIT": "provider_rate_limited",
                    "REQUEST": "provider_request_rejected",
                }.get(failure_type, "provider_invalid")

        comparison = ProjectStoryLinkComparison(
            comparison_id=comparison_id,
            project_manifest_id=manifest.manifest_id,
            project_revision=manifest.revision,
            context_id=context.context_id,
            story_graph_id=graph.graph_id,
            analysis_fingerprint=context.analysis_fingerprint,
            source_mention_review_id=(
                active_mention_review.review_id
                if active_mention_review is not None else None),
            source_mention_review_revision=(
                active_mention_review.review_revision
                if active_mention_review is not None else 0),
            request_fingerprint=request_fingerprint,
            idempotency_key_sha256=key_hash,
            candidate_id=req.candidate_id,
            attempt_number=req.attempt_number,
            relation_kind=req.relation_kind,
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            left_person_description=left_person,
            right_person_description=right_person,
            left_event_evidence=left_event,
            right_event_evidence=right_event,
            run_state="failed" if failure_code else "completed_unreviewed",
            assessment=(result.get("assessment") if failure_code is None else None),
            evidence_for=(result.get("evidence_for", []) if failure_code is None else []),
            evidence_against=(result.get("evidence_against", []) if failure_code is None else []),
            limitation=(result.get("limitation") if failure_code is None else None),
            left_frame_sha256=left_frame_hashes,
            right_frame_sha256=right_frame_hashes,
            frame_extractor_version=ffmpeg_version,
            failure_code=failure_code,
            project_id=project_id,
            created_at=int(time.time()),
            producer="director_brain_project_story_link_comparison_shadow_v1",
            source_ref=graph.graph_id,
            model_version=model_version,
            prompt_version=prompt_version,
            prompt_sha256=prompt_sha256,
            generation_profile=PROJECT_LINK_COMPARISON_GENERATION_PROFILE,
            sampling_profile=sampling_profile,
            observation_schema_version=FILM_OBSERVATION_SCHEMA_VERSION,
        )
        try:
            stored, reused = repo.save_project_story_link_comparison(comparison)
        except ValueError as exc:
            raise HTTPException(409, detail="comparison idempotency conflict") from exc
        return _envelope(corr, {
            "comparison": stored.model_dump(mode="json"),
            "reused": reused,
            "automatic_link_created": False,
            "director_plan_updated": False,
            "relation_inference_state": "EXPERIMENTAL",
            "quality_acceptance": "not_proven",
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/story-link-comparisons/{comparison_id}")
def project_story_link_comparison_get_endpoint(
    project_id: str,
    comparison_id: str,
    request: Request,
):
    """Read a retained unreviewed pairwise comparison by exact project ID."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import ProjectStoryLinkComparison

        repo = _project_manifest_repository()
        comparison = repo.get(ProjectStoryLinkComparison, comparison_id)
        if comparison is None or comparison.project_id != project_id:
            raise HTTPException(404, detail="project comparison does not exist")
        return _envelope(corr, {
            "comparison": comparison.model_dump(mode="json"),
            "automatic_link_created": False,
            "director_plan_updated": False,
            "quality_acceptance": "not_proven",
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/story-link-comparisons")
def project_story_link_comparison_list_endpoint(
    project_id: str,
    request: Request,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Page newest immutable comparison records for one project."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        repo = _project_manifest_repository()
        comparisons, total = repo.list_project_story_link_comparisons(
            project_id, limit=limit, offset=offset)
        return _envelope(corr, {
            "project_id": project_id,
            "comparisons": [item.model_dump(mode="json") for item in comparisons],
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": offset + len(comparisons) < total,
            "quality_acceptance": "not_proven",
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.post(
    "/v1/projects/{project_id}/director-plans:generate",
    responses=_PROJECT_REASONER_FAILURE_RESPONSES,
)
def project_director_plan_generate_endpoint(
    project_id: str,
    request: Request,
    req: ProjectPlanRequest,
):
    """Generate a project Plan using only current, locally persisted evidence."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import FilmObservation, ProjectStoryGraph
        from director_brain.project_story_graph import project_story_graph_id

        repo = _project_manifest_repository()
        manifest, context = _latest_project_context(repo, project_id)
        if manifest.revision != req.manifest_revision:
            raise HTTPException(
                409, detail="project plan requires the current manifest revision")
        _require_current_project_sources(manifest)
        graph = repo.get(ProjectStoryGraph, project_story_graph_id(context))
        if graph is None:
            raise HTTPException(
                409, detail="current project StoryGraph must be built before planning")

        if req.supersedes_plan_id is not None:
            try:
                predecessor = repo.get_project_plan_bundle(
                    project_id, req.supersedes_plan_id)
            except ValueError as exc:
                raise HTTPException(
                    409, detail="superseded project Plan evidence is inconsistent") from exc
            if predecessor is None:
                raise HTTPException(409, detail="superseded project Plan does not exist")
            _, predecessor_edl, predecessor_plan = predecessor
            if (
                predecessor_plan.project_manifest_id != manifest.manifest_id
                or predecessor_plan.project_revision != manifest.revision
                or predecessor_plan.project_context_id != context.context_id
                or predecessor_plan.project_story_graph_id != graph.graph_id
                or compute_plan_hash(predecessor_plan)
                != req.expected_superseded_plan_hash
                or compute_edl_hash(predecessor_edl)
                != req.expected_superseded_edl_hash
            ):
                raise HTTPException(
                    409,
                    detail="superseded Plan is stale or its reviewed hashes changed",
                )

        observations = []
        for evidence_ref in context.evidence_refs:
            observation = repo.get(FilmObservation, evidence_ref)
            if observation is None or observation.project_id != project_id:
                raise HTTPException(
                    409, detail="project context references missing or out-of-scope evidence")
            observations.append(observation)

        mention_history, link_review_history = repo.list_project_story_review_histories(
            project_id, graph.graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        active_link_review = link_review_history[-1] if link_review_history else None
        _validate_project_story_mention_review_snapshot(
            repo, active_mention_review, manifest, context, graph)

        brief = compile_project_brief(
            project_id,
            f"manifest://{manifest.manifest_id}",
            observations,
            target_duration_us=req.target_duration_us,
            intent_text=req.intent_text,
        )
        from director_brain.utils import short_hash

        shadow_candidate = None
        if req.reasoner_strategy == "llm":
            candidate_request_identity = (
                _project_director_candidate_request_identity(
                    project_id,
                    manifest,
                    context,
                    graph,
                    active_link_review,
                    active_mention_review,
                    req,
                    brief,
                )
            )
            candidate_request_fingerprint = (
                _fingerprint_project_director_candidate_request(
                    candidate_request_identity)
            )
            candidate_request_key = short_hash(json.dumps(
                candidate_request_identity, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ))
            brief = brief.model_copy(update={
                "brief_id": f"brief_candidate_{candidate_request_key}"
            })
            request_identity = {
                "identity_version": "project-plan-request-v3",
                "candidate_request_identity": candidate_request_identity,
                "reasoner_strategy": req.reasoner_strategy,
                "selected_strategy_hypothesis_id": (
                    req.selected_strategy_hypothesis_id),
                "selected_candidate_binding_digest": (
                    req.selected_candidate_binding_digest),
                "selected_comparison_id": req.selected_comparison_id,
            }
        else:
            request_identity = {
                "identity_version": "project-plan-request-v2",
                "project_id": project_id,
                "manifest_id": manifest.manifest_id,
                "context_id": context.context_id,
                "story_graph_id": graph.graph_id,
                "story_link_review_id": (
                    active_link_review.review_id if active_link_review else None),
                "story_link_review_revision": (
                    active_link_review.review_revision if active_link_review else None),
                "target_duration_us": req.target_duration_us,
                "intent_text": req.intent_text,
                "voice_led": req.voice_led,
                "audio_style": req.audio_style,
                "pacing_style": req.pacing_style,
                **({"transition_policy": req.transition_policy}
                   if req.transition_policy != "none" else {}),
                "brief_producer": brief.producer,
                "director_reasoner_producer": DIRECTOR_REASONER_PRODUCER,
            }
        if req.supersedes_plan_id is not None:
            request_identity["supersession"] = {
                "supersedes_plan_id": req.supersedes_plan_id,
                "expected_plan_hash": req.expected_superseded_plan_hash,
                "expected_edl_hash": req.expected_superseded_edl_hash,
                "clarified_by": req.clarified_by,
            }
        request_key = short_hash(json.dumps(
            request_identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ))
        if req.reasoner_strategy == "heuristic":
            brief = brief.model_copy(update={"brief_id": f"brief_{request_key}"})
        if req.reasoner_strategy == "llm":
            shadow_record = repo.get(
                ProjectDirectorShadowComparisonRecord,
                req.selected_comparison_id,
            )
            if (shadow_record is None
                    or shadow_record.project_id != project_id
                    or shadow_record.request_fingerprint
                    != candidate_request_fingerprint):
                raise HTTPException(
                    409, detail="selected Director comparison is missing or stale")
            selected = [
                item for item in shadow_record.comparison.strategy_candidates
                if item.hypothesis.hypothesis_id
                == req.selected_strategy_hypothesis_id
                and item.candidate_binding_digest
                == req.selected_candidate_binding_digest
            ]
            if len(selected) != 1:
                raise HTTPException(
                    409, detail="selected Director candidate is missing or stale")
            shadow_candidate = selected[0]
            edl, plan = materialize_project_plan_from_candidate(
                brief,
                manifest,
                context,
                graph,
                shadow_candidate,
                strategy="llm",
                selected_strategy_hypothesis_id=(
                    req.selected_strategy_hypothesis_id),
                selected_candidate_binding_digest=(
                    req.selected_candidate_binding_digest),
                project_story_link_review=active_link_review,
                project_story_mention_review=active_mention_review,
            )
        else:
            edl, plan = generate_project_plan(
                brief,
                manifest,
                context,
                graph,
                observations,
                strategy=req.reasoner_strategy,
                voice_led=req.voice_led,
                transition_policy=req.transition_policy,
                audio_style=req.audio_style,
                pacing_style=req.pacing_style,
                project_story_link_review=active_link_review,
                project_story_mention_review=active_mention_review,
            )
        plan.supersedes_plan_id = req.supersedes_plan_id
        if req.supersedes_plan_id is not None:
            successor_suffix = short_hash(
                f"{req.supersedes_plan_id}:{req.expected_superseded_plan_hash}:"
                f"{plan.plan_id}:{edl.edl_id}"
            )
            plan.plan_id = f"{plan.plan_id}_sup_{successor_suffix}"
            edl.edl_id = f"{edl.edl_id}_sup_{successor_suffix}"
            plan.edl_id = edl.edl_id
        valid, errors = validate_plan(edl, plan, observations)
        unresolved_constraints = interpret_constraints(brief).unverifiable
        requires_input = bool(unresolved_constraints)
        plan.validation_status = (
            "invalid" if not valid
            else "needs_input" if requires_input
            else "valid"
        )
        context_state = (
            PlanState.CONTEXT_READY
            if context.coverage.endswith("_observed")
            else PlanState.CONTEXT_PARTIAL
        )
        transition_plan(plan, context_state)
        transition_plan(plan, PlanState.VALIDATING)
        transition_plan(
            plan,
            PlanState.FAILED_VALIDATION if not valid
            else PlanState.NEEDS_INPUT if requires_input
            else PlanState.READY_FOR_STRATEGY_CONFIRMATION,
        )
        _require_current_project_sources(manifest)
        try:
            brief, edl, plan, reused = repo.save_project_plan_bundle(
                brief, edl, plan,
                expected_superseded_plan_hash=req.expected_superseded_plan_hash,
                expected_superseded_edl_hash=req.expected_superseded_edl_hash,
                clarified_by=req.clarified_by,
            )
        except ValueError as exc:
            raise HTTPException(
                409,
                detail="project draft persistence found stale or conflicting evidence",
            ) from exc
        return _envelope(corr, {
            "brief": brief.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "edl": edl.model_dump(mode="json"),
            "plan_hash": compute_plan_hash(plan),
            "edl_hash": compute_edl_hash(edl),
            "validation": {
                "valid": valid,
                "requires_input": requires_input,
                "errors": errors,
            },
            "persisted": True,
            "reused": reused,
            "evidence_scope": {
                "manifest_id": manifest.manifest_id,
                "manifest_revision": manifest.revision,
                "context_id": context.context_id,
                "story_graph_id": graph.graph_id,
                "timeline_scope": "project_per_asset",
                "cross_asset_relations_state": "not_attempted",
                "project_story_mention_review_id": (
                    active_mention_review.review_id
                    if active_mention_review else None),
                "project_story_mention_review_revision": (
                    active_mention_review.review_revision
                    if active_mention_review else 0),
                "project_story_mention_review_state": (
                    "caller_asserted" if active_mention_review else "not_provided"),
                "project_story_link_review_id": (
                    active_link_review.review_id if active_link_review else None),
                "project_story_link_review_revision": (
                    active_link_review.review_revision if active_link_review else None),
                "project_story_link_review_state": (
                    "caller_asserted" if active_link_review else "not_provided"),
                "automatic_cross_asset_inference_state": "not_attempted",
                "quality_acceptance": "not_proven",
            },
        })
    except HTTPException:
        raise
    except PathwayNotActiveError as exc:
        raise HTTPException(
            status_code=409,
            detail="project Director Reasoner pathway is not admitted",
        ) from exc
    except EvidenceTooPoorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)[:300]) from exc
    except LLMInputCapacityError as exc:
        raise HTTPException(
            status_code=413,
            detail="local project Director Reasoner input exceeds its safe capacity",
        ) from exc
    except LLMTransportError as exc:
        raise HTTPException(
            status_code=503,
            detail="local project Director Reasoner provider is unavailable",
            headers=_safe_project_reasoner_transport_headers(exc),
        ) from exc
    except LLMStructuredOutputError as exc:
        raise HTTPException(
            status_code=502,
            detail="local project Director Reasoner returned invalid structured output",
            headers=_safe_project_reasoner_failure_headers(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="project planning evidence is inconsistent") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.post(
    "/v1/projects/{project_id}/director-plans:compare-shadow-strategies",
    responses=_PROJECT_SHADOW_COMPARISON_RESPONSES,
)
@_serialize_project_director_shadow_request
def project_director_shadow_strategy_comparison_endpoint(
    project_id: str,
    request: Request,
    req: ProjectShadowStrategyComparisonRequest,
):
    """Persist a local SHADOW strategy comparison for exact readback only."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models import FilmObservation, ProjectStoryGraph
        from director_brain.project_story_graph import project_story_graph_id
        from director_brain.utils import short_hash

        repo = _project_manifest_repository()
        manifest, context = _latest_project_context(repo, project_id)
        if manifest.revision != req.manifest_revision:
            raise HTTPException(
                409, detail="strategy comparison requires the current manifest revision")
        _require_current_project_sources(manifest)
        graph = repo.get(ProjectStoryGraph, project_story_graph_id(context))
        if graph is None:
            raise HTTPException(
                409, detail="current project StoryGraph must be built before comparison")

        observations = []
        for evidence_ref in context.evidence_refs:
            observation = repo.get(FilmObservation, evidence_ref)
            if observation is None or observation.project_id != project_id:
                raise HTTPException(
                    409, detail="project context references missing or out-of-scope evidence")
            observations.append(observation)

        mention_history, link_review_history = repo.list_project_story_review_histories(
            project_id, graph.graph_id)
        active_mention_review = mention_history[-1] if mention_history else None
        active_link_review = link_review_history[-1] if link_review_history else None
        _validate_project_story_mention_review_snapshot(
            repo, active_mention_review, manifest, context, graph)
        brief = compile_project_brief(
            project_id,
            f"manifest://{manifest.manifest_id}",
            observations,
            target_duration_us=req.target_duration_us,
            intent_text=req.intent_text,
        )
        request_identity = _project_director_candidate_request_identity(
            project_id,
            manifest,
            context,
            graph,
            active_link_review,
            active_mention_review,
            req,
            brief,
        )
        request_fingerprint = _fingerprint_project_director_candidate_request(
            request_identity)
        idempotency_key = req.idempotency_key
        if not idempotency_key.strip():
            raise HTTPException(422, detail="idempotency_key must not be blank")
        idempotency_key_sha256 = hashlib.sha256(
            idempotency_key.encode("utf-8")
        ).hexdigest()
        comparison_id = _project_director_shadow_comparison_id(
            project_id, idempotency_key)
        brief_key = short_hash(json.dumps(
            request_identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ))
        brief = brief.model_copy(update={"brief_id": f"brief_candidate_{brief_key}"})

        existing_record = repo.get(
            ProjectDirectorShadowComparisonRecord, comparison_id)
        if existing_record is not None:
            if (existing_record.project_id != project_id
                    or existing_record.idempotency_key_sha256
                    != idempotency_key_sha256
                    or existing_record.request_fingerprint != request_fingerprint):
                raise HTTPException(
                    409, detail="idempotency key is already bound to another comparison")
            return _envelope(corr, _project_director_shadow_comparison_payload(
                existing_record,
                observations,
                active_link_review=active_link_review,
                active_mention_review=active_mention_review,
                idempotent_replay=True,
            ))
        _replay_project_director_shadow_failure(
            repo,
            project_id=project_id,
            comparison_id=comparison_id,
            request_fingerprint=request_fingerprint,
            idempotency_key_sha256=idempotency_key_sha256,
        )

        _authorize_project_director_shadow_provider_call(comparison_id)

        try:
            comparison = generate_project_shadow_strategy_options(
                brief,
                manifest,
                context,
                graph,
                observations,
                voice_led=req.voice_led,
                audio_style=req.audio_style,
                pacing_style=req.pacing_style,
                transition_policy=req.transition_policy,
                project_story_link_review=active_link_review,
                project_story_mention_review=active_mention_review,
            )
        except HTTPException:
            raise
        except Exception as exc:
            _record_project_director_shadow_failure(
                repo,
                project_id=project_id,
                comparison_id=comparison_id,
                request_fingerprint=request_fingerprint,
                idempotency_key_sha256=idempotency_key_sha256,
                manifest=manifest,
                context=context,
                graph=graph,
                exc=exc,
            )
            if (isinstance(exc, ValueError)
                    and not isinstance(
                        exc, (LLMInputCapacityError, LLMStructuredOutputError))):
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "local project Director Reasoner failed with an "
                        "unclassified error"
                    ),
                    headers={
                        "X-Director-Brain-Failure-Code": (
                            _UNCLASSIFIED_REASONER_VALUE_ERROR_CODE
                        ),
                    },
                ) from None
            raise
        _require_current_project_sources(manifest)

        requires_input = bool(interpret_constraints(brief).unverifiable)

        def comparison_plan_validation_status(valid: bool) -> str:
            if not valid:
                return "invalid"
            return "needs_input" if requires_input else "valid"

        baseline_valid, _baseline_errors = validate_plan(
            comparison.baseline_edl, comparison.baseline_plan, observations)
        comparison.baseline_plan.validation_status = (
            comparison_plan_validation_status(baseline_valid))
        for candidate in comparison.strategy_candidates:
            valid, _errors = validate_plan(candidate.edl, candidate.plan, observations)
            candidate.plan.validation_status = (
                comparison_plan_validation_status(valid))
        for candidate in comparison.strategy_candidates:
            candidate.candidate_binding_digest = (
                compute_project_strategy_candidate_binding_digest(
                    brief,
                    manifest,
                    context,
                    graph,
                    candidate.hypothesis,
                    candidate.edl,
                    candidate.plan,
                    project_story_link_review=active_link_review,
                    project_story_mention_review=active_mention_review,
                )
            )
        comparison = comparison.model_copy(update={
            "persistence_state": "PERSISTED_LOCAL",
        })
        record = ProjectDirectorShadowComparisonRecord(
            comparison_id=comparison_id,
            project_id=project_id,
            created_at=int(time.time()),
            producer="director_reasoner_strategy_comparison_shadow",
            source_ref=f"manifest://{manifest.manifest_id}",
            request_fingerprint=request_fingerprint,
            idempotency_key_sha256=idempotency_key_sha256,
            project_manifest_id=manifest.manifest_id,
            project_revision=manifest.revision,
            context_id=context.context_id,
            story_graph_id=graph.graph_id,
            project_story_link_review_id=(
                active_link_review.review_id if active_link_review else None),
            project_story_link_review_revision=(
                active_link_review.review_revision if active_link_review else None),
            project_story_mention_review_id=(
                active_mention_review.review_id if active_mention_review else None),
            project_story_mention_review_revision=(
                active_mention_review.review_revision if active_mention_review else 0),
            comparison=comparison,
        )
        claim = _project_director_shadow_current_claim.get()
        if claim is not None:
            claim.assert_owner()
        record, replayed = repo.save_project_director_shadow_comparison(record)
        _complete_project_director_shadow_run()
        return _envelope(corr, _project_director_shadow_comparison_payload(
            record,
            observations,
            active_link_review=active_link_review,
            active_mention_review=active_mention_review,
            idempotent_replay=replayed,
        ))
    except HTTPException:
        raise
    except ProjectDirectorShadowRunLeaseLostError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "comparison ownership changed; read back its status before "
                "requesting another attempt"
            ),
            headers={"X-Director-Brain-Run-State": "ownership_lost"},
        ) from exc
    except EvidenceTooPoorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)[:300]) from exc
    except LLMInputCapacityError as exc:
        raise HTTPException(
            status_code=413,
            detail="local project Director Reasoner input exceeds its safe capacity",
        ) from exc
    except LLMTransportError as exc:
        raise HTTPException(
            status_code=503,
            detail="local project Director Reasoner provider is unavailable",
            headers=_safe_project_reasoner_transport_headers(exc),
        ) from exc
    except LLMStructuredOutputError as exc:
        raise HTTPException(
            status_code=502,
            detail="local project Director Reasoner returned invalid structured output",
            headers=_safe_project_reasoner_failure_headers(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/director-plan-shadow-comparisons")
def project_director_shadow_comparison_list_endpoint(
    project_id: str,
    request: Request,
    response: Response,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List local non-confirmable SHADOW comparisons without narrative text."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        repo = _project_manifest_repository()
        records, total = repo.list_project_director_shadow_comparisons(
            project_id, limit=limit, offset=offset)
        currentness_state = (
            _project_director_shadow_currentness_state(repo, project_id)
            if records else None)
        items = []
        for item in records:
            currentness = _project_director_shadow_comparison_currentness(
                repo, item, currentness_state)
            items.append({
                "comparison_id": item.comparison_id,
                "created_at": item.created_at,
                "project_manifest_id": item.project_manifest_id,
                "project_revision": item.project_revision,
                "context_id": item.context_id,
                "story_graph_id": item.story_graph_id,
                "state": item.comparison.state,
                "persistence_state": item.comparison.persistence_state,
                "strategy_hypothesis_ids": [
                    candidate.hypothesis.hypothesis_id
                    for candidate in item.comparison.strategy_candidates
                ],
                "confirmable": False,
                "quality_acceptance": "NOT_PROVEN",
                **currentness,
            })
        response.headers["Cache-Control"] = "no-store"
        return _envelope(corr, {
            "items": items,
            "count": total,
            "limit": limit,
            "offset": offset,
        })
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="project comparison page is invalid") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="project comparison lookup failed") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get(
    "/v1/projects/{project_id}/director-plan-shadow-comparisons/{comparison_id}",
    responses={
        202: {
            "description": (
                "The exact same-key Reasoner comparison is still in progress "
                "or is safe to resume before provider invocation."
            ),
            "headers": {
                "Retry-After": {
                    "description": "Suggested delay before the next readback.",
                    "schema": {"type": "integer"},
                },
                "X-Director-Brain-Run-State": {
                    "description": "Fixed, non-semantic lifecycle state.",
                    "schema": {
                        "type": "string",
                        "enum": ["running", "recoverable"],
                    },
                },
            },
        },
        409: {
            "description": (
                "The prior provider outcome is uncertain and must not be "
                "replayed with the same idempotency key."
            ),
            "headers": {
                "X-Director-Brain-Run-State": {
                    "description": "Fixed, non-semantic lifecycle state.",
                    "schema": {
                        "type": "string",
                        "enum": ["outcome_unknown"],
                    },
                },
            },
        },
    },
)
def project_director_shadow_comparison_get_endpoint(
    project_id: str,
    comparison_id: str,
    request: Request,
    response: Response,
):
    """Read one exact local SHADOW candidate and disclose source staleness."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        repo = _project_manifest_repository()
        record = repo.get(ProjectDirectorShadowComparisonRecord, comparison_id)
        if record is None:
            run_state = _project_director_shadow_run_store().get_state(
                comparison_id, project_id)
            response.headers["Cache-Control"] = "no-store"
            if run_state in {"running", "recoverable"}:
                response.status_code = 202
                response.headers["Retry-After"] = "2"
                response.headers["X-Director-Brain-Run-State"] = (
                    "running" if run_state == "running" else "recoverable")
                return _envelope(corr, {
                    "comparison_id": comparison_id,
                    "run_state": (
                        "running" if run_state == "running"
                        else "awaiting_retry_before_provider"),
                    "persisted": False,
                    "confirmable": False,
                    "quality_acceptance": "NOT_PROVEN",
                })
            if run_state == "uncertain":
                response.status_code = 409
                response.headers["X-Director-Brain-Run-State"] = (
                    "outcome_unknown")
                return _envelope(corr, {
                    "comparison_id": comparison_id,
                    "run_state": "outcome_unknown",
                    "retry_action": "new_idempotency_key_required",
                    "persisted": False,
                    "confirmable": False,
                    "quality_acceptance": "NOT_PROVEN",
                })
            raise HTTPException(404, detail="Director shadow comparison does not exist")
        if record.project_id != project_id:
            raise HTTPException(404, detail="Director shadow comparison does not exist")
        currentness = _project_director_shadow_comparison_currentness(
            repo, record,
            _project_director_shadow_currentness_state(repo, project_id),
        )
        response.headers["Cache-Control"] = "no-store"
        return _envelope(corr, {
            "comparison_record": record.model_dump(mode="json"),
            "strategy_asset_coverage": [
                _project_director_strategy_asset_coverage(candidate)
                for candidate in record.comparison.strategy_candidates
            ],
            "persisted": True,
            "confirmable": False,
            "quality_acceptance": "NOT_PROVEN",
            **currentness,
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="project comparison readback failed") from exc
    finally:
        if repo is not None:
            repo.close()


@app.post(
    "/v1/projects/{project_id}/director-plan-shadow-comparisons/"
    "{comparison_id}:reject-constraint-assessment"
)
def project_constraint_assessment_rejection_endpoint(
    project_id: str,
    comparison_id: str,
    request: Request,
    response: Response,
    req: ProjectConstraintAssessmentRejectionRequest,
):
    """Persist a caller's rejection of one exact unverified model suggestion."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        repo = _project_manifest_repository()
        record = repo.get(ProjectDirectorShadowComparisonRecord, comparison_id)
        if record is None or record.project_id != project_id:
            raise HTTPException(404, detail="Director shadow comparison does not exist")
        try:
            receipt, reused = repo.reject_project_constraint_assessment(
                project_id,
                comparison_id,
                hypothesis_id=req.hypothesis_id,
                candidate_binding_digest=req.candidate_binding_digest,
                constraint_kind=req.constraint_kind,
                brief_index=req.brief_index,
                expected_assessment=req.expected_assessment,
                idempotency_key=req.idempotency_key,
                rejected_by=req.rejected_by,
                reason_code=req.reason_code,
            )
        except ValueError as exc:
            raise HTTPException(
                409,
                detail=(
                    "constraint assessment rejection failed; reload the exact "
                    "comparison and verify its candidate binding"
                ),
            ) from exc
        response.headers["Cache-Control"] = "no-store"
        return _envelope(corr, {
            "assessment_rejection": receipt,
            "persisted": True,
            "reused": reused,
            "actor_identity_state": "caller_asserted",
            "candidate_mutated": False,
            "plan_state_unchanged": True,
            "quality_acceptance": "NOT_PROVEN",
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail="constraint assessment rejection failed") from exc
    finally:
        if repo is not None:
            repo.close()


@app.post("/v1/projects/{project_id}/director-plans/{plan_id}:confirm-strategy")
def project_strategy_confirmation_endpoint(
    project_id: str,
    plan_id: str,
    request: Request,
    req: ProjectStrategyConfirmationRequest,
):
    """Persist an exact user strategy confirmation without enabling execution."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        repo = _project_manifest_repository()
        bundle = repo.get_project_plan_bundle(project_id, plan_id)
        if bundle is None:
            raise HTTPException(404, detail="project director plan 不存在")
        _brief, _edl, plan = bundle
        if plan.state == PlanState.READY_FOR_STRATEGY_CONFIRMATION.value:
            manifest = repo.latest_project_manifest(project_id)
            if (
                manifest is None
                or manifest.manifest_id != plan.project_manifest_id
                or manifest.revision != plan.project_revision
            ):
                raise HTTPException(409, detail="project Plan is no longer on the current revision")
            _require_current_project_sources(manifest)
        elif plan.state != PlanState.STRATEGY_CONFIRMED.value:
            raise HTTPException(409, detail="project Plan is not ready for confirmation")

        try:
            confirmed_brief, edl, confirmed_plan, confirmation, reused = (
                repo.confirm_project_strategy(
                    project_id,
                    plan_id,
                    expected_plan_hash=req.expected_plan_hash,
                    expected_edl_hash=req.expected_edl_hash,
                    idempotency_key=req.idempotency_key,
                    confirmed_by=req.confirmed_by,
                    output_target=req.output_target,
                    notes=req.notes,
                )
            )
        except ValueError as exc:
            raise HTTPException(
                409,
                detail="project strategy confirmation rejected; reload the current Plan and verify its hashes",
            ) from exc
        receipt = repo.get_project_strategy_confirmation(project_id, plan_id)
        return _envelope(corr, {
            "brief": confirmed_brief.model_dump(mode="json"),
            "plan": confirmed_plan.model_dump(mode="json"),
            "edl": edl.model_dump(mode="json"),
            "plan_hash": compute_plan_hash(confirmed_plan),
            "edl_hash": compute_edl_hash(edl),
            "confirmation": confirmation,
            "confirmation_receipt": receipt,
            "persisted": True,
            "reused": reused,
            "state": confirmed_plan.state,
            "dispatch_eligible": False,
            "dispatch_block_reason": "P3 ExecutionPort is not admitted",
            "actor_identity_state": "caller_asserted",
            "quality_acceptance": "not_proven",
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.post("/v1/projects/{project_id}/director-plans/{plan_id}:reject-strategy")
def project_strategy_rejection_endpoint(
    project_id: str,
    plan_id: str,
    request: Request,
    req: ProjectStrategyRejectionRequest,
):
    """Persist an exact user rejection of a reviewable project Plan."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        repo = _project_manifest_repository()
        bundle = repo.get_project_plan_bundle(project_id, plan_id)
        if bundle is None:
            raise HTTPException(404, detail="project director plan 不存在")
        _brief, _edl, plan = bundle
        if plan.state not in {
            PlanState.READY_FOR_STRATEGY_CONFIRMATION.value,
            PlanState.NEEDS_INPUT.value,
            PlanState.REJECTED.value,
        }:
            raise HTTPException(409, detail="project Plan is not reviewable for rejection")

        try:
            rejected_brief, edl, rejected_plan, rejection, reused = (
                repo.reject_project_strategy(
                    project_id,
                    plan_id,
                    expected_plan_hash=req.expected_plan_hash,
                    expected_edl_hash=req.expected_edl_hash,
                    idempotency_key=req.idempotency_key,
                    rejected_by=req.rejected_by,
                    reason=req.reason,
                )
            )
        except ValueError as exc:
            raise HTTPException(
                409,
                detail="project strategy rejection failed; reload the current Plan and verify its hashes",
            ) from exc
        receipt = repo.get_project_strategy_rejection(project_id, plan_id)
        return _envelope(corr, {
            "brief": rejected_brief.model_dump(mode="json"),
            "plan": rejected_plan.model_dump(mode="json"),
            "edl": edl.model_dump(mode="json"),
            "plan_hash": compute_plan_hash(rejected_plan),
            "edl_hash": compute_edl_hash(edl),
            "rejection": rejection,
            "rejection_receipt": receipt,
            "persisted": True,
            "reused": reused,
            "state": rejected_plan.state,
            "dispatch_eligible": False,
            "actor_identity_state": "caller_asserted",
            "quality_acceptance": "not_proven",
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


def _project_plan_link_currentness(repo, plan) -> str:
    """Report whether a stored Plan still binds the current link/mention reviews."""
    from director_brain.project_story_link_review import (
        project_story_link_review_is_stale,
    )

    graph_id = plan.project_story_graph_id
    if graph_id is None:
        return "not_applicable"
    mention_history, link_history = repo.list_project_story_review_histories(
        plan.project_id, graph_id)
    active_mention = mention_history[-1] if mention_history else None
    active_link = link_history[-1] if link_history else None
    expected_binding = (
        (active_link.review_id, active_link.review_revision)
        if active_link is not None else (None, None)
    )
    actual_binding = (
        plan.project_story_link_review_id,
        plan.project_story_link_review_revision,
    )
    if actual_binding != expected_binding:
        return "stale"
    if active_link is not None and project_story_link_review_is_stale(
        active_link, active_mention
    ):
        return "stale"
    return "current"


@app.get("/v1/projects/{project_id}/director-plans/{plan_id}")
def project_director_plan_get_endpoint(
    project_id: str,
    plan_id: str,
    request: Request,
):
    """Read one persisted Plan/EDL/Brief bundle within its owning project."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        repo = _project_manifest_repository()
        try:
            bundle = repo.get_project_plan_bundle(project_id, plan_id)
        except ValueError as exc:
            raise HTTPException(409, detail="persisted project draft is inconsistent") from exc
        if bundle is None:
            raise HTTPException(404, detail="project director plan 不存在")
        brief, edl, plan = bundle
        confirmation_receipt = repo.get_project_strategy_confirmation(
            project_id, plan_id)
        rejection_receipt = repo.get_project_strategy_rejection(
            project_id, plan_id)
        return _envelope(corr, {
            "brief": brief.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "edl": edl.model_dump(mode="json"),
            "plan_hash": compute_plan_hash(plan),
            "edl_hash": compute_edl_hash(edl),
            "persisted": True,
            "confirmation_receipt": confirmation_receipt,
            "rejection_receipt": rejection_receipt,
            "evidence_scope": {
                "manifest_id": plan.project_manifest_id,
                "manifest_revision": plan.project_revision,
                "context_id": plan.project_context_id,
                "story_graph_id": plan.project_story_graph_id,
                "timeline_scope": "project_per_asset",
                "cross_asset_relations_state": "not_attempted",
                "project_story_link_currentness": _project_plan_link_currentness(
                    repo, plan),
                "quality_acceptance": "not_proven",
            },
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/director-plans")
def project_director_plan_list_endpoint(project_id: str, request: Request):
    """List this project's persisted Plan versions for local recovery/discovery."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models.director_plan import DirectorDecisionPlan

        repo = _project_manifest_repository()
        latest_manifest = repo.latest_project_manifest(project_id)
        plans = repo.list(DirectorDecisionPlan, project_id=project_id)
        plans = [
            item for item in plans
            if item.project_manifest_id is not None
            and item.project_context_id is not None
            and item.project_story_graph_id is not None
        ]
        plans.sort(
            key=lambda item: (item.project_revision or 0, item.created_at, item.plan_id),
            reverse=True,
        )
        entries = []
        for plan in plans:
            try:
                bundle = repo.get_project_plan_bundle(project_id, plan.plan_id)
            except ValueError as exc:
                raise HTTPException(
                    409, detail="persisted project plan index contains an inconsistent bundle"
                ) from exc
            if bundle is None:
                continue
            _brief, edl, stored_plan = bundle
            entries.append({
                "plan_id": stored_plan.plan_id,
                "brief_id": stored_plan.brief_id,
                "edl_id": stored_plan.edl_id,
                "supersedes_plan_id": stored_plan.supersedes_plan_id,
                "manifest_id": stored_plan.project_manifest_id,
                "manifest_revision": stored_plan.project_revision,
                "context_id": stored_plan.project_context_id,
                "story_graph_id": stored_plan.project_story_graph_id,
                "created_at": stored_plan.created_at,
                "state": stored_plan.state,
                "validation_status": stored_plan.validation_status,
                "approval_state": stored_plan.approval_state,
                "plan_hash": compute_plan_hash(stored_plan),
                "edl_hash": compute_edl_hash(edl),
                "revision_state": (
                    "latest_manifest_revision"
                    if latest_manifest is not None
                    and stored_plan.project_revision == latest_manifest.revision
                    else "historical_manifest_revision"
                ),
                "project_story_link_currentness": _project_plan_link_currentness(
                    repo, stored_plan),
                "quality_acceptance": "not_proven",
            })
        return _envelope(corr, {"plans": entries, "count": len(entries)})
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/film-manifest")
def project_film_manifest_get_endpoint(
    project_id: str,
    request: Request,
    revision: int | None = Query(default=None, ge=1),
):
    """Read an exact or latest persisted project manifest revision."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        repo = _project_manifest_repository()
        manifest = (
            repo.get_project_manifest(project_id, revision)
            if revision is not None
            else repo.latest_project_manifest(project_id)
        )
        if manifest is None:
            raise HTTPException(404, detail="project manifest 不存在")
        return _envelope(corr, {
            "manifest": _public_manifest_payload(manifest),
            "boundary_state": manifest.boundary_state,
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/film-context/{context_id}")
def project_film_context_get_endpoint(
    project_id: str,
    context_id: str,
    request: Request,
    layer: str | None = Query(
        default=None, description="按需请求的层：project|asset|scene|evidence"),
    reason: str = Query(
        default="", description="EVIDENCE 层用途理由（方案 §6：必须）"),
    include_stale: bool = Query(
        default=False,
        description=(
            "显式只读访问该项目已失效的历史 Context；历史版本不可用于新 Plan"),
    ),
):
    """Read current Context, or explicitly include a stale audit snapshot."""
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from director_brain.models.film_context import FilmContextSnapshot

        repo = _project_manifest_repository()
        snapshot = repo.get(FilmContextSnapshot, context_id)
        if snapshot is None or snapshot.project_id != project_id:
            raise HTTPException(404, detail="项目上下文不存在")
        manifest = repo.latest_project_manifest(project_id)
        if manifest is None:
            raise HTTPException(404, detail="project manifest 不存在")
        stale_reason = (
            "invalidated" if snapshot.invalidated_at is not None else
            "manifest_revision_changed"
            if (
                snapshot.project_manifest_id != manifest.manifest_id
                or snapshot.project_revision != manifest.revision
            ) else None
        )
        if stale_reason is not None and not include_stale:
            raise HTTPException(410, detail="项目上下文已因 manifest 更新而失效")
        if layer:
            want = layer.lower()
            have = [str(value) for value in snapshot.layers]
            if want not in have:
                raise HTTPException(
                    400, detail=f"快照层 {have} 不包含请求层 {want}")
            if want == "evidence" and not reason.strip():
                raise HTTPException(
                    400, detail="EVIDENCE 层按需展开必须提供 reason（方案 §6）")
        return _envelope(corr, {
            "project_id": project_id,
            "snapshot": snapshot.model_dump(mode="json"),
            "currentness": "stale" if stale_reason is not None else "current",
            "stale_reason": stale_reason,
            "usable_for_new_plan": stale_reason is None,
            "requested_layer": layer,
            "expansion_reason": reason or None,
            "source_hash_validation_state": "not_revalidated_by_read",
        })
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.post("/v1/projects/{project_id}/film-context:snapshot")
def project_film_context_snapshot_endpoint(
    project_id: str,
    request: Request,
    req: ProjectContextSnapshotRequest,
):
    """Build a project snapshot from one explicit, ordered asset manifest."""
    _require_project_api_access(request)
    return _build_project_film_context_snapshot(project_id, req)


def _project_context_job_store():
    from storage.project_context_jobs import SqliteProjectContextJobStore

    return SqliteProjectContextJobStore(
        _project_context_job_database_path()
    )


def _project_context_job_database_path() -> Path:
    from director_brain.config import load_settings

    configured = load_settings().sqlite_path
    if configured == ":memory:":
        raise HTTPException(
            503, detail="持久项目任务需要文件型 SQLite 数据库")
    return Path(configured).expanduser().resolve()


def _public_project_context_job(job) -> dict[str, Any]:
    return job.model_dump(
        mode="json",
        exclude={"request_fingerprint", "idempotency_key", "worker_id"},
    )


@app.post("/v1/projects/{project_id}/film-context:jobs")
def submit_project_film_context_job_endpoint(
    project_id: str,
    request: Request,
    req: ProjectContextJobRequest,
    response: Response,
    idempotency_key: str = Header(
        ..., min_length=1, max_length=128, alias="Idempotency-Key"),
):
    """Submit/replay one durable project-scoped Context analysis job."""
    _require_project_api_access(request)
    if idempotency_key.strip() != idempotency_key:
        raise HTTPException(422, detail="Idempotency-Key 不得含首尾空白")
    from director_brain.context_gateway import (
        project_analysis_profile,
        project_asset_analysis_fingerprint,
        project_context_id,
    )
    from director_brain.models.project_context_job import (
        ProjectContextJob,
        ProjectContextJobState,
    )
    from storage.project_context_jobs import IdempotencyConflictError

    corr = _new_correlation_id()
    repo = _project_manifest_repository()
    try:
        manifest = repo.get_project_manifest(project_id, req.manifest_revision)
        if manifest is None:
            raise HTTPException(404, detail="指定 project manifest revision 不存在")
        latest = repo.latest_project_manifest(project_id)
        if latest is None or latest.revision != manifest.revision:
            raise HTTPException(
                409, detail="只有最新 manifest revision 可以生成当前项目上下文")
        try:
            profile = project_analysis_profile(manifest.analysis_profile)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                503, detail="项目分析档案缺少有效的本地模型绑定") from exc
        source_bindings = [
            {
                "asset_id": asset.asset_id,
                "order": asset.order,
                "source_content_hash": (
                    asset.source_content_hash.lower()
                    if asset.source_content_hash is not None else None
                ),
                "analysis_fingerprint": project_asset_analysis_fingerprint(
                    asset, manifest.analysis_profile),
            }
            for asset in manifest.assets
        ]
        request_payload = {
            "project_id": project_id,
            "manifest_id": manifest.manifest_id,
            "manifest_revision": manifest.revision,
            "analysis_profile": manifest.analysis_profile.value,
            "provider_profile": profile,
            "source_bindings": source_bindings,
        }
        request_fingerprint = hashlib.sha256(
            json.dumps(
                request_payload, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        now = int(time.time())
        job = ProjectContextJob(
            job_id=f"context_job_{uuid.uuid4().hex}",
            project_id=project_id,
            manifest_id=manifest.manifest_id,
            manifest_revision=manifest.revision,
            context_id=project_context_id(manifest),
            analysis_profile=manifest.analysis_profile.value,
            request_fingerprint=request_fingerprint,
            idempotency_key=idempotency_key,
            state=ProjectContextJobState.QUEUED,
            created_at=now,
            updated_at=now,
        )
        store = _project_context_job_store()
        try:
            stored, created = store.create(job)
        except IdempotencyConflictError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        if stored.state == ProjectContextJobState.QUEUED:
            db_path = str(_project_context_job_database_path())
            try:
                from director_brain.project_context_job_queue import (
                    ensure_project_context_worker,
                )

                ensure_project_context_worker(db_path, stored.job_id)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    503,
                    detail=(
                        "任务已持久化但本地队列暂不可用；使用相同 Idempotency-Key "
                        "重试可安全恢复"
                    ),
                ) from exc
        response.status_code = 202 if created else 200
        return _envelope(corr, {
            "job": _public_project_context_job(stored),
            "replayed": not created,
            "status_path": (
                f"/v1/projects/{project_id}/film-context-jobs/{stored.job_id}"
            ),
        })
    finally:
        repo.close()


@app.get("/v1/projects/{project_id}/film-context-jobs/{job_id}")
def get_project_film_context_job_endpoint(
    project_id: str, job_id: str, request: Request,
):
    _require_project_api_access(request)
    corr = _new_correlation_id()
    store = _project_context_job_store()
    job = store.get(job_id, project_id)
    if job is None:
        raise HTTPException(404, detail="项目任务不存在")
    if job.state.value in {"queued", "running", "cancel_requested"}:
        from director_brain.project_context_job_queue import (
            ensure_project_context_worker,
        )

        ensure_project_context_worker(_project_context_job_database_path())
        job = store.get(job_id, project_id) or job
    return _envelope(corr, {"job": _public_project_context_job(job)})


@app.get("/v1/projects/{project_id}/film-context-jobs/{job_id}/events")
def list_project_film_context_job_events_endpoint(
    project_id: str, job_id: str, request: Request,
):
    _require_project_api_access(request)
    corr = _new_correlation_id()
    store = _project_context_job_store()
    if store.get(job_id, project_id) is None:
        raise HTTPException(404, detail="项目任务不存在")
    return _envelope(corr, {
        "project_id": project_id,
        "job_id": job_id,
        "events": store.list_events(job_id, project_id),
    })


@app.post("/v1/projects/{project_id}/film-context-jobs/{job_id}:cancel")
def cancel_project_film_context_job_endpoint(
    project_id: str, job_id: str, request: Request,
):
    _require_project_api_access(request)
    corr = _new_correlation_id()
    job = _project_context_job_store().request_cancel(job_id, project_id)
    if job is None:
        raise HTTPException(404, detail="项目任务不存在")
    if job.state.value in {"running", "cancel_requested"}:
        from director_brain.config import load_settings
        from director_brain.project_context_job_queue import (
            ensure_project_context_worker,
        )

        ensure_project_context_worker(_project_context_job_database_path())
    return _envelope(corr, {"job": _public_project_context_job(job)})


def _build_project_film_context_snapshot(
    project_id: str,
    req: ProjectContextSnapshotRequest,
    progress_callback: Callable[
        [str, str | None, int, int, int, int | None], None
    ] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
):
    """Build and persist one exact multi-asset context, optionally as a job."""
    from observation_service.analysis_control import AnalysisCancelledError
    from storage.project_context_jobs import ProjectContextJobLeaseLostError

    def check_cancellation() -> None:
        if cancellation_check is not None and cancellation_check():
            raise AnalysisCancelledError("analysis cancellation requested")

    corr = _new_correlation_id()
    repo = None
    try:
        check_cancellation()
        repo = _project_manifest_repository()
        manifest = repo.get_project_manifest(project_id, req.manifest_revision)
        if manifest is None:
            raise HTTPException(404, detail="指定 project manifest revision 不存在")
        latest_manifest = repo.latest_project_manifest(project_id)
        if latest_manifest is None or latest_manifest.revision != manifest.revision:
            raise HTTPException(
                409, detail="只有最新 manifest revision 可以生成当前项目上下文")

        if progress_callback is not None:
            progress_callback(
                "validating_manifest", None, 0, len(manifest.assets), 0, None)

        from director_brain.models.film_context import AssetAnalysisState
        from director_brain.models.film_observation import ClaimKind
        from director_brain.utils import file_sha256

        source_state_by_asset: dict[str, AssetAnalysisState] = {}
        blocked_reason_by_asset: dict[str, str] = {}
        for asset in manifest.assets:
            check_cancellation()
            if asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED:
                source_state_by_asset[asset.asset_id] = (
                    AssetAnalysisState.NOT_AUTHORIZED)
                blocked_reason_by_asset[asset.asset_id] = (
                    "rights_unverified"
                    if asset.rights.state == RightsState.UNVERIFIED
                    else "local_processing_not_authorized"
                )
                continue
            media_path = _authorized_local_media_path(
                asset.source_ref, require_file=False)
            if not media_path.is_file():
                source_state_by_asset[asset.asset_id] = (
                    AssetAnalysisState.SOURCE_UNAVAILABLE)
                continue
            try:
                current_hash = file_sha256(asset.source_ref)
            except OSError:
                source_state_by_asset[asset.asset_id] = (
                    AssetAnalysisState.SOURCE_UNAVAILABLE)
                continue
            if current_hash.lower() != asset.source_content_hash.lower():
                source_state_by_asset[asset.asset_id] = AssetAnalysisState.SOURCE_CHANGED
            else:
                assert asset.source_content_hash is not None
                source_state_by_asset[asset.asset_id] = AssetAnalysisState.OBSERVED

        from director_brain.context_gateway import (
            build_multi_asset_project_context,
            project_analysis_profile,
            project_asset_analysis_fingerprint,
            project_context_id,
        )
        from director_brain.models.film_context import FilmContextSnapshot

        try:
            profile = project_analysis_profile(manifest.analysis_profile)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(503, detail="项目分析档案缺少有效的本地模型绑定") from exc
        context_id = project_context_id(manifest)
        cached = repo.get(FilmContextSnapshot, context_id)
        cached_coverage_by_asset = (
            {item.asset_id: item for item in cached.asset_coverage}
            if cached is not None else {}
        )
        cached_coverage_stable = (
            cached is not None
            and len(cached_coverage_by_asset) == len(manifest.assets)
            and all(
                (coverage := cached_coverage_by_asset.get(asset.asset_id)) is not None
                and coverage.order == asset.order
                and coverage.rights_state == asset.rights.state.value
                and coverage.has_audio == asset.has_audio
                and coverage.probe_ok == asset.probe_ok
                and coverage.time_map == asset.time_map
                and coverage.source_content_hash == (
                    asset.source_content_hash.lower()
                    if asset.source_content_hash is not None else None
                )
                and (
                    (
                    asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED
                    and coverage.analysis_state == AssetAnalysisState.NOT_AUTHORIZED
                    and coverage.blocked_reason == (
                        "rights_unverified"
                        if asset.rights.state == RightsState.UNVERIFIED
                        else "local_processing_not_authorized"
                    )
                )
                or (
                    asset.rights.state == RightsState.LOCAL_PROCESSING_ALLOWED
                        and coverage.analysis_state in (
                            AssetAnalysisState.OBSERVED,
                            AssetAnalysisState.COMPLETED_EMPTY,
                        )
                        and coverage.blocked_reason is None
                    )
                )
                for asset in manifest.assets
            )
        )
        source_states_stable = all(
            source_state_by_asset[asset.asset_id] == (
                AssetAnalysisState.OBSERVED
                if asset.rights.state == RightsState.LOCAL_PROCESSING_ALLOWED
                else AssetAnalysisState.NOT_AUTHORIZED
            )
            for asset in manifest.assets
        )
        if (
            cached is not None
            and cached.invalidated_at is None
            and not source_states_stable
        ):
            changed_assets = [
                asset.asset_id
                for asset in manifest.assets
                if source_state_by_asset[asset.asset_id] != (
                    AssetAnalysisState.OBSERVED
                    if asset.rights.state == RightsState.LOCAL_PROCESSING_ALLOWED
                    else AssetAnalysisState.NOT_AUTHORIZED
                )
            ]
            raise HTTPException(
                409,
                detail=(
                    "已有项目上下文绑定的来源素材已变化或不可用: "
                    + ", ".join(changed_assets)
                    + "。请提交新的 manifest revision 后重试；现有上下文已保留。"
                ),
            )
        if req.observations_by_asset is not None and not all(
            state == AssetAnalysisState.OBSERVED
            for state in source_state_by_asset.values()
        ):
            unready_assets = [
                asset_id for asset_id, state in source_state_by_asset.items()
                if state != AssetAnalysisState.OBSERVED
            ]
            raise HTTPException(
                409,
                detail=("外部提供的观测要求所有来源素材当前可读且 hash 匹配: "
                        + ", ".join(unready_assets)),
            )
        if (
            cached is not None
            and cached.invalidated_at is None
            and cached_coverage_stable
            and source_states_stable
            and req.observations_by_asset is None
        ):
            check_cancellation()
            if progress_callback is not None:
                progress_callback(
                    "context_persisted", None, len(manifest.assets),
                    len(manifest.assets), 0, None)
            return _envelope(corr, {
                "snapshot": cached.model_dump(mode="json"),
                "context_id": context_id,
                "coverage": cached.coverage,
                "evidence_refs": cached.evidence_refs,
                "cache_hit": True,
            })

        analysis_cache_state_by_asset: dict[str, str] = {}
        analysis_state_by_asset: dict[str, AssetAnalysisState] = {}
        failure_code_by_asset: dict[str, str] = {}
        if req.observations_by_asset is not None:
            if manifest.analysis_profile in (
                ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1,
                ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1,
            ):
                raise HTTPException(
                    422,
                    detail="本地模型分析 profile 必须由绑定的本地管线生成观测",
                )
            expected_asset_ids = {asset.asset_id for asset in manifest.assets}
            if set(req.observations_by_asset) != expected_asset_ids:
                raise HTTPException(
                    422, detail="observations_by_asset 必须精确覆盖 manifest 中的全部素材")
            observations_by_asset = {}
            from director_brain.models.film_observation import FilmObservation

            for asset in manifest.assets:
                observations = [
                    FilmObservation(**item)
                    for item in req.observations_by_asset[asset.asset_id]
                ]
                if asset.source_content_hash is None:
                    raise HTTPException(
                        409, detail=f"素材 {asset.asset_id} 未获准观测注入")
                if any(obs.project_id != project_id for obs in observations):
                    raise HTTPException(
                        422, detail=f"素材 {asset.asset_id} 的观测 project_id 不匹配")
                if any(obs.media_hash.lower() != asset.source_content_hash.lower()
                       for obs in observations):
                    raise HTTPException(
                        409, detail=f"素材 {asset.asset_id} 的观测 source hash 不匹配")
                analysis_cache_state_by_asset[asset.asset_id] = "provided"
                observations_by_asset[asset.asset_id] = observations
                analysis_state_by_asset[asset.asset_id] = (
                    AssetAnalysisState.PROVIDED
                    if observations else AssetAnalysisState.PROVIDED_EMPTY
                )
        else:
            from observation_service.pipeline import analyze_media

            observations_by_asset = {}
            pending_analysis_cache = {}
            for asset_index, asset in enumerate(manifest.assets):
                check_cancellation()
                if progress_callback is not None:
                    progress_callback(
                        "deterministic_analysis", asset.asset_id, asset_index,
                        len(manifest.assets), 0, None)
                source_state = source_state_by_asset[asset.asset_id]
                if source_state != AssetAnalysisState.OBSERVED:
                    observations_by_asset[asset.asset_id] = []
                    analysis_state_by_asset[asset.asset_id] = source_state
                    analysis_cache_state_by_asset[asset.asset_id] = "not_run"
                    if source_state != AssetAnalysisState.NOT_AUTHORIZED:
                        failure_code_by_asset[asset.asset_id] = (
                            "source_unavailable"
                            if source_state == AssetAnalysisState.SOURCE_UNAVAILABLE
                            else "source_changed"
                        )
                    if progress_callback is not None:
                        progress_callback(
                            "asset_complete", asset.asset_id, asset_index + 1,
                            len(manifest.assets), 0, None)
                    continue
                assert asset.source_content_hash is not None
                fingerprint = project_asset_analysis_fingerprint(
                    asset, manifest.analysis_profile)
                cached_analysis = repo.get_analysis_result(
                    fingerprint, asset.source_content_hash, profile)
                asset_progress_callback = None
                if progress_callback is not None:
                    asset_progress_callback = (
                        lambda phase, completed, total, a=asset, i=asset_index:
                        progress_callback(
                            phase, a.asset_id, i, len(manifest.assets),
                            completed, total
                        )
                    )
                if cached_analysis is None:
                    speech_analysis_failure = None
                    try:
                        from director_brain.analysis_cache import (
                            RepositoryAnalysisCache,
                        )

                        shot_cache = RepositoryAnalysisCache(
                            repo,
                            asset.source_content_hash,
                            profile,
                        )
                        if (manifest.analysis_profile
                                == ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1):
                            from director_brain.config import load_settings
                            from observation_service.ollama_vlm_adapter import (
                                OllamaVLMAdapter,
                            )
                            from observation_service.pipeline import analyze_media_full

                            settings = load_settings()
                            adapter = OllamaVLMAdapter(
                                model=settings.project_local_vlm_model,
                                base_url=settings.ollama_base_url,
                                model_digest=settings.project_local_vlm_digest,
                                runtime_version=(
                                    settings.project_local_vlm_runtime_version),
                                enforce_loopback=True,
                            )
                            analysis_options = {}
                            if asset_progress_callback is not None:
                                analysis_options["progress_callback"] = (
                                    asset_progress_callback)
                            if cancellation_check is not None:
                                analysis_options["cancellation_check"] = (
                                    cancellation_check)
                            source_observations = analyze_media_full(
                                asset.source_ref,
                                vlm=True,
                                asr=False,
                                vlm_adapter=adapter,
                                vlm_cache=shot_cache,
                                **analysis_options,
                            )
                        elif (manifest.analysis_profile
                              == ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1):
                            from observation_service.pipeline import (
                                analyze_media_with_project_asr,
                            )

                            model_path, model_digest, model_label, _runtime = (
                                project_local_asr_model_binding())
                            outcome = analyze_media_with_project_asr(
                                asset.source_ref,
                                cache=shot_cache,
                                model_path=str(model_path),
                                model_digest=model_digest,
                                model_version=(
                                    f"{model_label}@sha256:{model_digest}"),
                                source_has_audio=asset.has_audio,
                                time_map=asset.time_map,
                            )
                            check_cancellation()
                            source_observations = outcome.observations
                            speech_analysis_failure = outcome.failure_code
                        else:
                            analysis_options = {}
                            if asset_progress_callback is not None:
                                analysis_options["progress_callback"] = (
                                    asset_progress_callback)
                            if cancellation_check is not None:
                                analysis_options["cancellation_check"] = (
                                    cancellation_check)
                            source_observations = analyze_media(
                                asset.source_ref,
                                cache=shot_cache,
                                **analysis_options,
                            )
                    except (AnalysisCancelledError, ProjectContextJobLeaseLostError):
                        raise
                    except Exception:  # noqa: BLE001
                        observations_by_asset[asset.asset_id] = []
                        analysis_state_by_asset[asset.asset_id] = AssetAnalysisState.FAILED
                        analysis_cache_state_by_asset[asset.asset_id] = "failed"
                        failure_code_by_asset[asset.asset_id] = "analyzer_failed"
                        if progress_callback is not None:
                            progress_callback(
                                "asset_complete", asset.asset_id,
                                asset_index + 1, len(manifest.assets), 0, None)
                        continue
                    try:
                        current_hash = file_sha256(asset.source_ref).lower()
                    except OSError:
                        current_hash = ""
                    if current_hash != asset.source_content_hash.lower():
                        state = (
                            AssetAnalysisState.SOURCE_CHANGED
                            if current_hash else AssetAnalysisState.SOURCE_UNAVAILABLE
                        )
                        observations_by_asset[asset.asset_id] = []
                        analysis_state_by_asset[asset.asset_id] = state
                        analysis_cache_state_by_asset[asset.asset_id] = "not_cached"
                        failure_code_by_asset[asset.asset_id] = (
                            "source_changed" if current_hash else "source_unavailable"
                        )
                        continue
                    if any(obs.media_hash.lower() != asset.source_content_hash.lower()
                           for obs in source_observations):
                        raise HTTPException(
                            409,
                            detail=f"素材 {asset.asset_id} 分析结果 source hash 不匹配",
                        )
                    local_vlm_partial = (
                        manifest.analysis_profile
                        == ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1
                        and any(
                            observation.observation_type == "vlm_semantic"
                            and observation.claim_kind == ClaimKind.NOT_DETERMINED
                            for observation in source_observations
                        )
                    )
                    if local_vlm_partial:
                        analysis_state_by_asset[asset.asset_id] = (
                            AssetAnalysisState.PARTIAL)
                        failure_code_by_asset[asset.asset_id] = (
                            "semantic_analysis_partial")
                        analysis_cache_state_by_asset[asset.asset_id] = "partial"
                    elif speech_analysis_failure is not None:
                        analysis_state_by_asset[asset.asset_id] = (
                            AssetAnalysisState.PARTIAL
                            if source_observations
                            else AssetAnalysisState.FAILED
                        )
                        failure_code_by_asset[asset.asset_id] = (
                            "speech_analysis_partial"
                            if source_observations
                            else "speech_analysis_failed"
                        )
                        analysis_cache_state_by_asset[asset.asset_id] = (
                            "partial" if source_observations else "failed")
                    else:
                        pending_analysis_cache[asset.asset_id] = source_observations
                        analysis_cache_state_by_asset[asset.asset_id] = "computed"
                else:
                    source_observations = cached_analysis
                    analysis_cache_state_by_asset[asset.asset_id] = "reused"
                observations_by_asset[asset.asset_id] = source_observations
                if asset.asset_id not in analysis_state_by_asset:
                    analysis_state_by_asset[asset.asset_id] = (
                        AssetAnalysisState.OBSERVED
                        if source_observations
                        else AssetAnalysisState.COMPLETED_EMPTY
                    )
                if progress_callback is not None:
                    progress_callback(
                        "asset_complete", asset.asset_id, asset_index + 1,
                        len(manifest.assets), 0, None)

            for asset in manifest.assets:
                if analysis_state_by_asset.get(asset.asset_id) not in (
                    AssetAnalysisState.OBSERVED,
                    AssetAnalysisState.COMPLETED_EMPTY,
                    AssetAnalysisState.PARTIAL,
                ):
                    continue
                try:
                    current_hash = file_sha256(asset.source_ref).lower()
                except OSError:
                    current_hash = ""
                if current_hash != asset.source_content_hash.lower():
                    state = (
                        AssetAnalysisState.SOURCE_CHANGED
                        if current_hash else AssetAnalysisState.SOURCE_UNAVAILABLE
                    )
                    observations_by_asset[asset.asset_id] = []
                    analysis_state_by_asset[asset.asset_id] = state
                    analysis_cache_state_by_asset[asset.asset_id] = "not_cached"
                    failure_code_by_asset[asset.asset_id] = (
                        "source_changed" if current_hash else "source_unavailable"
                    )
                    pending_analysis_cache.pop(asset.asset_id, None)

            for asset in manifest.assets:
                check_cancellation()
                if asset.asset_id in pending_analysis_cache:
                    fingerprint = project_asset_analysis_fingerprint(
                        asset, manifest.analysis_profile)
                    repo.save_analysis_result(
                        fingerprint,
                        asset.source_content_hash,
                        profile,
                        pending_analysis_cache[asset.asset_id],
                    )

        # Namespace evidence identities by project asset while preserving the
        # upstream observation id for exact lineage and downstream audit.
        from director_brain.utils import short_hash
        from director_brain.models.film_observation import FilmObservation

        project_observations_by_asset: dict[str, list[FilmObservation]] = {}
        for asset in manifest.assets:
            project_observations: list[FilmObservation] = []
            for observation in observations_by_asset[asset.asset_id]:
                if observation.project_id != project_id:
                    if req.observations_by_asset is not None:
                        raise HTTPException(
                            422,
                            detail=f"素材 {asset.asset_id} 的观测 project_id 不匹配",
                        )
                source_observation_id = (
                    observation.source_observation_id or observation.observation_id
                )
                evidence_payload = observation.model_dump(
                    mode="json",
                    exclude={
                        "observation_id",
                        "project_id",
                        "project_asset_id",
                        "source_observation_id",
                        "source_ref",
                    },
                )
                project_observation_id = "projobs_" + short_hash(
                    json.dumps({
                        "project_id": project_id,
                        "project_asset_id": asset.asset_id,
                        "source_content_hash": asset.source_content_hash,
                        "source_observation_id": source_observation_id,
                        "evidence": evidence_payload,
                    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
                project_observations.append(observation.model_copy(update={
                    "observation_id": project_observation_id,
                    "project_asset_id": asset.asset_id,
                    "source_observation_id": source_observation_id,
                    "project_id": project_id,
                    "source_ref": asset.source_ref,
                }))
            project_observations_by_asset[asset.asset_id] = project_observations

        snapshot = build_multi_asset_project_context(
            manifest,
            project_observations_by_asset,
            analysis_cache_state_by_asset,
            analysis_state_by_asset,
            failure_code_by_asset,
            blocked_reason_by_asset,
        )
        all_project_observations = [
            observation
            for asset in manifest.assets
            for observation in project_observations_by_asset[asset.asset_id]
        ]
        try:
            check_cancellation()
            repo.save_project_context(
                snapshot,
                all_project_observations,
                expected_manifest_revision=manifest.revision,
            )
            if progress_callback is not None:
                progress_callback(
                    "context_persisted", None, len(manifest.assets),
                    len(manifest.assets), 0, None)
        except ValueError as exc:
            raise HTTPException(409, detail=str(exc)[:300]) from exc
        return _envelope(corr, {
            "snapshot": snapshot.model_dump(mode="json"),
            "context_id": snapshot.context_id,
            "coverage": snapshot.coverage,
            "evidence_refs": snapshot.evidence_refs,
            "cache_hit": False,
        })
    except (AnalysisCancelledError, ProjectContextJobLeaseLostError):
        raise
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(422, detail="项目上下文请求字段无效") from exc
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)[:300]) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, detail=f"项目请求失败，关联编号 {corr}") from exc
    finally:
        if repo is not None:
            repo.close()


@app.get("/v1/projects/{project_id}/decision-ledger")
def decision_ledger_endpoint(project_id: str, request: Request):
    corr = _new_correlation_id()
    repo = None
    try:
        _require_project_api_access(request)
        from storage.repository import DecisionLedgerEntry
        repo = _project_manifest_repository()
        entries = repo.list(DecisionLedgerEntry, project_id=project_id)
        return _envelope(corr, {
            "project_id": project_id,
            "entries": [
                {"ledger_id": e.ledger_id, "decision_id": e.decision_id,
                 "project_id": e.project_id, "action": e.action,
                 "timestamp": e.timestamp, "detail": e.detail}
                for e in entries
            ],
        })
    except HTTPException:
        raise
    except FileNotFoundError:
        return _envelope(corr, {"entries": [], "note": "账本数据库尚未创建"})
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])
    finally:
        if repo is not None:
            repo.close()


# 导入延迟引用（避免循环 import）
from director_brain.story_graph_builder import build_story_graph  # noqa: E402
from director_brain.plan_state import compute_plan_hash, compute_edl_hash  # noqa: E402
