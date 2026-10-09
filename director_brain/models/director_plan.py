"""导演决策计划 DirectorDecisionPlan 与 Decision 模型。"""
from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_serializer,
    model_validator,
)

from director_brain.models.base import BaseRecord
from director_brain.models.edl import EditorialDecisionList


class Decision(BaseModel):
    """单条编辑决策理由（嵌入式组件，非顶层记录）。"""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    purpose: str
    shot_refs: list[str] = Field(default_factory=list)
    project_asset_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    alternative_project_asset_ids: list[str | None] = Field(default_factory=list)
    confidence: float | None = None
    requires_approval: bool = False

    @model_validator(mode="after")
    def validate_alternative_asset_refs(self):
        if (self.alternative_project_asset_ids
                and len(self.alternative_project_asset_ids) != len(self.alternatives)):
            raise ValueError(
                "alternative_project_asset_ids must align with alternatives")
        return self


class ProjectNarrativeEvidenceRef(BaseModel):
    """Exact semantic observation supplied to a project narrative hypothesis."""

    model_config = ConfigDict(extra="forbid")

    project_asset_id: str = Field(min_length=1)
    source_media_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    source_asset_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)


def derive_candidate_edl_source_asset_alignment(
    source_refs: list[ProjectNarrativeEvidenceRef],
    selected_edl_source_assets: set[tuple[str | None, str, str]],
) -> str:
    """Return asset-identity overlap only; never infer time or semantic support."""
    cited_assets = {
        (
            source.project_asset_id,
            source.source_media_hash.lower(),
            source.source_asset_id,
        )
        for source in source_refs
    }
    if not cited_assets:
        return "no_cited_sources"
    selected_assets = {
        (project_asset_id, media_hash.lower(), source_asset_id)
        for project_asset_id, media_hash, source_asset_id
        in selected_edl_source_assets
    }
    selected_count = len(cited_assets & selected_assets)
    if selected_count == len(cited_assets):
        return "all_cited_assets_selected"
    if selected_count:
        return "some_cited_assets_selected"
    return "no_cited_assets_selected"


class ProjectNarrativeEvidenceClaim(BaseModel):
    """One strategy rationale statement tied to exact project evidence."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=1000)
    source_refs: list[ProjectNarrativeEvidenceRef] = Field(min_length=1, max_length=8)


class ProjectNarrativeSourceRationale(BaseModel):
    """One strategy-specific include/exclude decision tied to evidence."""

    model_config = ConfigDict(extra="forbid")

    focus_source: ProjectNarrativeEvidenceRef
    disposition: Literal["include", "exclude"]
    statement: str = Field(min_length=1, max_length=240)
    source_refs: list[ProjectNarrativeEvidenceRef] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_focus_source_is_cited(self):
        focus = (
            self.focus_source.project_asset_id,
            self.focus_source.source_media_hash.lower(),
            self.focus_source.source_asset_id,
            self.focus_source.observation_id,
        )
        cited = {
            (
                source.project_asset_id,
                source.source_media_hash.lower(),
                source.source_asset_id,
                source.observation_id,
            )
            for source in self.source_refs
        }
        if focus not in cited:
            raise ValueError("source rationale must cite its focus source")
        if len(cited) != len(self.source_refs):
            raise ValueError("source rationale evidence references must be unique")
        return self


class ProjectNarrativeConstraintAssessment(BaseModel):
    """Unverified per-strategy review suggestion for one exact Brief entry."""

    model_config = ConfigDict(extra="forbid")

    constraint_kind: Literal["must_include", "must_avoid"]
    brief_index: StrictInt = Field(ge=0)
    constraint_text: str = Field(min_length=1)
    assessment: Literal[
        "candidate_supported", "candidate_conflicted", "unresolved",
    ]
    statement: str = Field(min_length=1, max_length=240)
    source_refs: list[ProjectNarrativeEvidenceRef] = Field(default_factory=list)
    candidate_edl_source_asset_alignment: Literal[
        "not_evaluated",
        "no_cited_sources",
        "all_cited_assets_selected",
        "some_cited_assets_selected",
        "no_cited_assets_selected",
    ] = Field(
        default="not_evaluated",
        description=(
            "Mechanical source-asset identity overlap with the candidate EDL; "
            "does not establish cited time-range overlap or semantic constraint truth."
        ),
    )

    @model_validator(mode="after")
    def validate_source_refs(self):
        identities = [(
            item.project_asset_id,
            item.source_media_hash.lower(),
            item.source_asset_id,
            item.observation_id,
        ) for item in self.source_refs]
        if len(identities) != len(set(identities)):
            raise ValueError("constraint assessment source refs must be unique")
        if self.assessment != "unresolved" and not self.source_refs:
            raise ValueError(
                "supported or conflicted constraint assessments require evidence")
        return self


class ProjectNarrativeShotHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: ProjectNarrativeEvidenceRef
    label: str = Field(min_length=1)
    rationale: str | None = None


class ProjectNarrativeActBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act: Literal["hook", "develop", "peak", "resolve"]
    source_refs: list[ProjectNarrativeEvidenceRef] = Field(min_length=1)


class ProjectNarrativeStrategyHypothesis(BaseModel):
    """One unranked editorial strategy hypothesis bound to project evidence."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}$")
    label: str = Field(min_length=1, max_length=120)
    editorial_intent: str = Field(min_length=1, max_length=1000)
    emotional_arc: ProjectNarrativeEvidenceClaim
    editing_language_choice: Literal[
        "fast_cut", "slow_paced", "montage", "jump_cut",
    ] | None = None
    editing_language_rationale: ProjectNarrativeEvidenceClaim | None = None
    audio_style_choice: Literal["none", "j_cut", "l_cut"] | None = None
    audio_style_rationale: ProjectNarrativeEvidenceClaim | None = None
    transition_policy_choice: Literal[
        "none", "dissolve_act_boundary",
    ] | None = None
    transition_policy_rationale: ProjectNarrativeEvidenceClaim | None = None
    transition_duration_us: int | None = Field(default=None, gt=0)
    ordered_sources: list[ProjectNarrativeEvidenceRef] = Field(min_length=2)
    source_rationales: list[ProjectNarrativeSourceRationale] = Field(min_length=2)
    act_boundaries: list[ProjectNarrativeActBoundary] = Field(default_factory=list)
    tradeoffs: list[ProjectNarrativeEvidenceClaim] = Field(min_length=1, max_length=8)
    uncertainties: list[ProjectNarrativeEvidenceClaim] = Field(min_length=1, max_length=8)
    constraint_assessments: list[ProjectNarrativeConstraintAssessment] = Field(
        default_factory=list)

    @model_serializer(mode="wrap")
    def omit_empty_constraint_assessments(self, handler):
        """Keep legacy traces byte-shape compatible when no review was requested."""
        data = handler(self)
        if not self.constraint_assessments:
            data.pop("constraint_assessments", None)
        return data

    @model_validator(mode="after")
    def validate_source_rationale_coverage(self):
        if (self.editing_language_choice is None) != (
            self.editing_language_rationale is None
        ):
            raise ValueError(
                "strategy editing language and rationale must be provided together")
        if (self.audio_style_choice is None) != (
            self.audio_style_rationale is None
        ):
            raise ValueError(
                "strategy audio style and rationale must be provided together")
        if (self.transition_policy_choice is None) != (
            self.transition_policy_rationale is None
        ):
            raise ValueError(
                "strategy transition policy and rationale must be provided together")
        if (self.transition_duration_us is not None
                and self.transition_policy_choice != "dissolve_act_boundary"):
            raise ValueError(
                "transition duration requires a dissolve transition policy")
        def identity(source: ProjectNarrativeEvidenceRef):
            return (
                source.project_asset_id,
                source.source_media_hash.lower(),
                source.source_asset_id,
                source.observation_id,
            )

        ordered = [identity(source) for source in self.ordered_sources]
        rationales = [identity(item.focus_source) for item in self.source_rationales]
        if len(set(ordered)) != len(ordered):
            raise ValueError("strategy ordered sources must be unique")
        if len(set(rationales)) != len(rationales) or set(rationales) != set(ordered):
            raise ValueError(
                "strategy source rationales must cover each ordered source exactly once")
        assessment_refs = [
            (item.constraint_kind, item.brief_index)
            for item in self.constraint_assessments
        ]
        if len(assessment_refs) != len(set(assessment_refs)):
            raise ValueError("strategy constraint assessments must be unique")
        return self


class ProjectNarrativeCallProvenance(BaseModel):
    """Content-free request scope for one successful project Reasoner call."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    stage: Literal["flat_project", "segment", "project_synthesis"]
    model: str = Field(min_length=1, max_length=256)
    prompt_version: str = Field(min_length=1, max_length=120)
    system_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_evidence_ref_indexes: list[StrictInt] = Field(
        min_length=1,
        description=(
            "Zero-based positions in ProjectNarrativeReasoningTrace.input_evidence_refs"
        ),
    )
    temperature: float = Field(ge=0.0, le=2.0)
    timeout_seconds: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    provider_identity_state: Literal["reported", "not_reported"] = Field(
        default="not_reported",
        description=(
            "Whether the response envelope reported a model identifier or system fingerprint; "
            "this is not an immutable model revision or quality admission."
        ),
    )
    provider_reported_model: str | None = Field(
        default=None,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$",
        description="Model identifier returned by the provider response envelope, if any.",
    )
    provider_reported_system_fingerprint: str | None = Field(
        default=None,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$",
        description="System fingerprint returned by the provider response envelope, if any.",
    )
    runtime_binding_state: Literal["unbound", "verified"] = "unbound"
    verified_model_digest: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
        description=(
            "Exact Ollama model digest verified against the local model catalog "
            "immediately before this provider call."
        ),
    )
    verified_runtime_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$",
        description=(
            "Ollama runtime version verified immediately before this provider call."
        ),
    )

    @model_validator(mode="after")
    def validate_provider_identity_state(self):
        has_identity = (
            self.provider_reported_model is not None
            or self.provider_reported_system_fingerprint is not None
        )
        if (self.provider_identity_state == "reported") != has_identity:
            raise ValueError(
                "provider identity state must match reported response identifiers")
        return self

    @model_validator(mode="after")
    def validate_runtime_binding_state(self):
        has_complete_binding = (
            self.verified_model_digest is not None
            and self.verified_runtime_version is not None
        )
        has_partial_binding = (
            (self.verified_model_digest is None)
            != (self.verified_runtime_version is None)
        )
        if has_partial_binding or (
            (self.runtime_binding_state == "verified") != has_complete_binding
        ):
            raise ValueError(
                "verified runtime binding requires a model digest and runtime version")
        return self

    @model_validator(mode="after")
    def validate_input_evidence_ref_indexes(self):
        if (
            any(index < 0 for index in self.input_evidence_ref_indexes)
            or self.input_evidence_ref_indexes != sorted(
                set(self.input_evidence_ref_indexes))
        ):
            raise ValueError(
                "provider call evidence reference indexes must be ordered and unique")
        return self


class ProjectDirectorSourceSelectionAuditEntry(BaseModel):
    """Bind one model disposition to the planner's final SHADOW EDL outcome."""

    model_config = ConfigDict(extra="forbid")

    strategy_rank: int = Field(ge=0)
    strategy_disposition: Literal["include", "exclude"]
    selected_in_edl: bool
    reason_codes: list[Literal[
        "present_in_candidate_edl",
        "excluded_by_director_strategy",
        "excluded_by_must_avoid",
        "source_has_no_technical_data",
        "source_is_dark_shot",
        "source_role_discarded",
        "technical_eligibility_not_met",
        "source_interval_below_minimum",
        "act_duration_target_reached",
        "act_remaining_capacity_below_minimum",
    ]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reason_codes(self):
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("selection audit reason codes must be unique")
        if self.selected_in_edl:
            if (self.strategy_disposition != "include"
                    or self.reason_codes != ["present_in_candidate_edl"]):
                raise ValueError(
                    "selected sources must be included and record their candidate EDL outcome")
        elif "present_in_candidate_edl" in self.reason_codes:
            raise ValueError(
                "non-selected sources cannot record candidate EDL presence")
        if (self.strategy_disposition == "exclude"
                and "excluded_by_director_strategy" not in self.reason_codes):
            raise ValueError(
                "strategy-excluded sources must retain their disposition reason")
        return self


class ProjectNarrativeReasoningTrace(BaseModel):
    """Unverified project narrative hypotheses with exact input lineage."""

    model_config = ConfigDict(extra="forbid")

    state: Literal[
        "SHADOW_UNVERIFIED", "ACTIVE_PATHWAY_UNVERIFIED",
    ] = "SHADOW_UNVERIFIED"
    evidence_meaning: Literal["input_lineage_only_not_claim_verification"] = (
        "input_lineage_only_not_claim_verification"
    )
    brief_id: str = Field(min_length=1)
    provider: Literal["ollama"] = "ollama"
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    provider_call_provenance: list[ProjectNarrativeCallProvenance] = Field(
        default_factory=list)
    provider_call_provenance_state: Literal[
        "captured", "legacy_not_captured",
    ] = "legacy_not_captured"
    story_arc: str = Field(min_length=1)
    input_evidence_refs: list[ProjectNarrativeEvidenceRef] = Field(min_length=2)
    emotional_hypotheses: list[ProjectNarrativeShotHypothesis] = Field(min_length=2)
    key_moments: list[ProjectNarrativeShotHypothesis] = Field(default_factory=list)
    act_boundaries: list[ProjectNarrativeActBoundary] = Field(default_factory=list)
    suggested_order: list[ProjectNarrativeEvidenceRef] = Field(min_length=2)
    strategy_hypotheses: list[ProjectNarrativeStrategyHypothesis] = Field(min_length=2)
    active_strategy_hypothesis_id: str | None = Field(default=None, min_length=1)
    limitations: list[str] = Field(default_factory=list)
    caller_asserted_link_hints_supplied: int = Field(
        default=0,
        ge=0,
        description=(
            "Caller-asserted link hints supplied as unverified context; this does "
            "not establish that the model used or accepted them."
        ),
    )
    caller_asserted_link_hint_ids_supplied: list[StrictStr] = Field(
        default_factory=list,
        max_length=500,
        description="Stable IDs of caller link hints supplied to the provider.",
    )
    caller_asserted_link_hint_ids_omitted: list[StrictStr] = Field(
        default_factory=list,
        max_length=500,
        description=(
            "Stable IDs of current review links omitted because they could not be "
            "mapped to the active semantic evidence supplied to the provider."
        ),
    )
    caller_asserted_link_review_link_count: int = Field(default=0, ge=0, le=500)
    caller_asserted_link_provenance_state: Literal[
        "captured", "legacy_not_captured",
    ] = "legacy_not_captured"

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_caller_link_count(cls, value):
        if not isinstance(value, dict) or "caller_asserted_link_hints_used" not in value:
            return value
        migrated = dict(value)
        legacy_count = migrated.pop("caller_asserted_link_hints_used")
        current_count = migrated.get("caller_asserted_link_hints_supplied")
        if current_count is not None and current_count != legacy_count:
            raise ValueError(
                "legacy and current caller link hint counts disagree")
        migrated.setdefault("caller_asserted_link_hints_supplied", legacy_count)
        return migrated

    @model_validator(mode="after")
    def validate_caller_link_hint_provenance(self):
        supplied_ids = self.caller_asserted_link_hint_ids_supplied
        omitted_ids = self.caller_asserted_link_hint_ids_omitted
        all_ids = supplied_ids + omitted_ids
        if any(not link_id or len(link_id) > 128 for link_id in all_ids):
            raise ValueError("caller link hint IDs must be bounded non-empty identifiers")
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("caller link hint IDs must be unique and disjoint")
        if self.caller_asserted_link_provenance_state == "captured":
            if self.caller_asserted_link_hints_supplied != len(supplied_ids):
                raise ValueError(
                    "captured caller link hint count must match supplied IDs")
            if self.caller_asserted_link_review_link_count != len(all_ids):
                raise ValueError(
                    "captured caller link IDs must partition the current review")
        return self

    @model_validator(mode="after")
    def validate_provider_call_provenance(self):
        if self.provider_call_provenance_state == "captured":
            if not self.provider_call_provenance:
                raise ValueError(
                    "captured project narrative trace requires provider call provenance")
            if [item.sequence for item in self.provider_call_provenance] != list(
                range(1, len(self.provider_call_provenance) + 1)
            ):
                raise ValueError(
                    "provider call provenance sequence must be contiguous")
            if any(
                item.model != self.model
                or item.prompt_version != self.prompt_version
                or item.temperature != self.temperature
                for item in self.provider_call_provenance
            ):
                raise ValueError(
                    "provider call provenance must match trace model settings")
            runtime_bindings = {
                (
                    item.runtime_binding_state,
                    item.verified_model_digest,
                    item.verified_runtime_version,
                )
                for item in self.provider_call_provenance
            }
            if len(runtime_bindings) != 1:
                raise ValueError(
                    "provider calls must share one verified runtime model binding")
            source_count = len(self.input_evidence_refs)
            if any(
                index >= source_count
                for item in self.provider_call_provenance
                for index in item.input_evidence_ref_indexes
            ):
                raise ValueError(
                    "provider call evidence reference index is outside the trace")
            stages = [item.stage for item in self.provider_call_provenance]
            if stages == ["flat_project"]:
                if self.provider_call_provenance[0].input_evidence_ref_indexes != list(
                    range(source_count)
                ):
                    raise ValueError(
                        "flat project call must cover every trace evidence reference")
            elif "flat_project" in stages:
                raise ValueError(
                    "flat project call cannot be mixed with hierarchical calls")
            else:
                segment_calls = [
                    item for item in self.provider_call_provenance
                    if item.stage == "segment"
                ]
                if (
                    not segment_calls
                    or not any(stage == "project_synthesis" for stage in stages)
                ):
                    raise ValueError(
                        "hierarchical provenance requires segment and synthesis calls")
                segment_indexes = [
                    index for item in segment_calls
                    for index in item.input_evidence_ref_indexes
                ]
                if sorted(segment_indexes) != list(range(source_count)):
                    raise ValueError(
                        "segment calls must cover each trace evidence reference once")
        elif self.provider_call_provenance:
            raise ValueError(
                "legacy provider provenance state cannot include call records")
        return self

    @model_validator(mode="after")
    def validate_constraint_assessment_scope(self):
        input_refs = {
            (
                item.project_asset_id,
                item.source_media_hash.lower(),
                item.source_asset_id,
                item.observation_id,
            )
            for item in self.input_evidence_refs
        }
        scopes = []
        for strategy in self.strategy_hypotheses:
            scope = tuple(sorted((
                item.constraint_kind,
                item.brief_index,
                item.constraint_text,
            ) for item in strategy.constraint_assessments))
            scopes.append(scope)
            for assessment in strategy.constraint_assessments:
                if any((
                    source.project_asset_id,
                    source.source_media_hash.lower(),
                    source.source_asset_id,
                    source.observation_id,
                ) not in input_refs for source in assessment.source_refs):
                    raise ValueError(
                        "constraint assessment cites evidence outside the trace")
        if scopes and any(scope != scopes[0] for scope in scopes[1:]):
            raise ValueError(
                "all strategy hypotheses must assess the same Brief constraints")
        return self


class DirectorDecisionPlan(BaseRecord):
    """面向执行的导演决策计划。"""

    plan_id: str
    version: str
    brief_id: str | None = None
    edl_id: str | None = None
    project_manifest_id: str | None = None
    project_revision: int | None = Field(default=None, ge=1)
    project_context_id: str | None = None
    project_story_graph_id: str | None = None
    project_story_link_review_id: str | None = Field(default=None, min_length=1)
    project_story_link_review_revision: int | None = Field(default=None, ge=1)
    brief_version: str
    film_state_version: str
    sequence: list[str] = Field(default_factory=list)
    sequence_project_asset_ids: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    project_narrative_reasoning: ProjectNarrativeReasoningTrace | None = None
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    #: T2：是否存在任何判据放宽/兜底/借用（静默降级显式化；正常输入为 False）
    degraded: bool = False
    #: T2：降级事件列表（append-only，受控前缀：relax_technical_usable /
    #: borrowed_shot / fallback_selection）
    degradation_events: list[str] = Field(default_factory=list)
    validation_status: str
    approval_state: str
    supersedes_plan_id: str | None = None
    #: Plan 状态机当前态（候选③执法点：plan_state.PlanState 的值；
    #: 渲染/执行闸门据此执法——未 STRATEGY_CONFIRMED 不得渲染）。
    state: str = "draft"

    @model_validator(mode="after")
    def validate_sequence_asset_refs(self):
        if (self.sequence_project_asset_ids
                and len(self.sequence_project_asset_ids) != len(self.sequence)):
            raise ValueError(
                "sequence_project_asset_ids must align with sequence")
        return self

    @model_validator(mode="after")
    def validate_project_artifact_binding(self):
        project_fields = (
            self.project_manifest_id,
            self.project_revision,
            self.project_context_id,
            self.project_story_graph_id,
        )
        if any(value is not None for value in project_fields):
            if any(value is None for value in project_fields):
                raise ValueError("project plan binding fields must be complete")
            if not self.brief_id or not self.edl_id:
                raise ValueError("project plan requires brief_id and edl_id bindings")
            if self.film_state_version != self.project_context_id:
                raise ValueError("project plan film_state_version must match context_id")
        if (self.project_story_link_review_id is None) != (
            self.project_story_link_review_revision is None
        ):
            raise ValueError(
                "project story link review id and revision must be provided together")
        if (self.project_story_link_review_id is not None
                and self.project_story_graph_id is None):
            raise ValueError(
                "project story link review binding requires a project StoryGraph")
        return self


class ProjectDirectorStrategyCandidate(BaseModel):
    """A non-confirmable project Plan/EDL generated from one SHADOW hypothesis."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: ProjectNarrativeStrategyHypothesis
    candidate_binding_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "SHA-256 binding to the exact project evidence scope, stable DirectorBrief, "
            "Plan and EDL state, strategy hypothesis, reasoning trace, decisions, and "
            "materialized EDL behavior. Generated artifact IDs and timestamps are "
            "excluded. A matching digest is required before a later formal generation "
            "can reuse a candidate."
        ),
    )
    edl: EditorialDecisionList
    plan: DirectorDecisionPlan
    selection_audit: list[ProjectDirectorSourceSelectionAuditEntry] = Field(
        min_length=2)
    sequence_changed_vs_heuristic: bool = Field(
        description=(
            "Whether the ordered source identities differ from the heuristic baseline; "
            "does not compare trims, transitions, audio, or effects."
        ),
    )
    distinct_edl_sequence_from_baseline: bool = Field(
        description=(
            "Whether the materialized EDL execution signature differs from the "
            "heuristic baseline, including source intervals, transitions, audio "
            "offsets, effects, and output tracks; provenance-only metadata is excluded."
        ),
    )
    distinct_edl_sequence_from_other_candidates: bool = Field(
        description=(
            "Whether this candidate's materialized EDL execution signature is unique "
            "among the strategy candidates, including source intervals, transitions, "
            "audio offsets, effects, and output tracks."
        ),
    )

    @model_validator(mode="after")
    def validate_shadow_candidate_binding(self):
        trace = self.plan.project_narrative_reasoning
        if (trace is None
                or trace.active_strategy_hypothesis_id != self.hypothesis.hypothesis_id):
            raise ValueError(
                "strategy candidate Plan must bind its active narrative hypothesis")
        if self.plan.edl_id != self.edl.edl_id:
            raise ValueError("strategy candidate Plan and EDL IDs must match")
        if self.plan.state != "draft":
            raise ValueError("SHADOW strategy candidates must remain draft")
        if "director_reasoner_shadow_candidate=not_confirmable" not in self.plan.constraints:
            raise ValueError("SHADOW strategy candidate Plan must be non-confirmable")

        ordered_sources = self.hypothesis.ordered_sources
        if len(self.selection_audit) != len(ordered_sources):
            raise ValueError(
                "strategy selection audit must cover every ordered source")

        def source_identity(source: ProjectNarrativeEvidenceRef):
            return (
                source.project_asset_id,
                source.source_media_hash.lower(),
                source.source_asset_id,
            )

        ordered_identities = [source_identity(source) for source in ordered_sources]
        if len(set(ordered_identities)) != len(ordered_identities):
            raise ValueError("strategy hypothesis contains duplicate ordered sources")
        if [entry.strategy_rank for entry in self.selection_audit] != list(
            range(len(ordered_sources))
        ):
            raise ValueError(
                "strategy selection audit must preserve exact hypothesis order")

        disposition_by_source = {
            source_identity(item.focus_source): item.disposition
            for item in self.hypothesis.source_rationales
        }
        if any(
            entry.strategy_disposition
            != disposition_by_source.get(
                source_identity(ordered_sources[entry.strategy_rank]))
            for entry in self.selection_audit
        ):
            raise ValueError(
                "strategy selection audit must match each source disposition")

        selected_by_audit = {
            source_identity(ordered_sources[entry.strategy_rank])
            for entry in self.selection_audit
            if entry.selected_in_edl
        }
        selected_by_edl = {
            (
                edit.project_asset_id,
                edit.source_media_hash.lower(),
                edit.source_asset_id,
            )
            for edit in self.edl.ordered_edits
        }
        if selected_by_audit != selected_by_edl:
            raise ValueError(
                "strategy selection audit must match final EDL source identities")
        active_hypothesis = next(
            (
                item for item in trace.strategy_hypotheses
                if item.hypothesis_id == self.hypothesis.hypothesis_id
            ),
            None,
        )
        if active_hypothesis is None:
            raise ValueError(
                "strategy candidate hypothesis is missing from its reasoning trace")
        if active_hypothesis != self.hypothesis:
            raise ValueError(
                "strategy candidate hypothesis must match its reasoning trace")
        for assessment in active_hypothesis.constraint_assessments:
            if assessment.candidate_edl_source_asset_alignment == "not_evaluated":
                continue
            expected_alignment = derive_candidate_edl_source_asset_alignment(
                assessment.source_refs, selected_by_edl)
            if assessment.candidate_edl_source_asset_alignment != expected_alignment:
                raise ValueError(
                    "constraint source alignment must match the candidate EDL assets")
        return self


class ProjectDirectorShadowComparison(BaseModel):
    """Unranked SHADOW strategy comparison; it is never confirmable."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["SHADOW_UNVERIFIED"] = "SHADOW_UNVERIFIED"
    baseline_edl: EditorialDecisionList
    baseline_plan: DirectorDecisionPlan
    strategy_candidates: list[ProjectDirectorStrategyCandidate] = Field(min_length=2)
    persistence_state: Literal["NOT_PERSISTED", "PERSISTED_LOCAL"] = "NOT_PERSISTED"
    confirmable: Literal[False] = False
    quality_acceptance: Literal["NOT_PROVEN"] = "NOT_PROVEN"

    @model_validator(mode="after")
    def validate_distinct_hypothesis_ids(self):
        hypothesis_ids = [
            item.hypothesis.hypothesis_id for item in self.strategy_candidates
        ]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("shadow comparison strategy hypothesis IDs must be unique")
        return self
