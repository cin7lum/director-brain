"""Version-bound, unranked cross-asset mention pairs for review workflows."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from director_brain.models.project_story_link_comparison import (
    ProjectStoryLinkEventEvidence,
)
from director_brain.models.project_story_link_review import (
    ProjectStoryLinkAnchor,
    ProjectStoryLinkComparisonDisposition,
)
from director_brain.models.project_story_mention_review import (
    ProjectStoryMentionReview,
)


class ProjectStoryLinkCandidateCallerDisposition(BaseModel):
    """Latest caller-entered disposition, including whether it is still current."""

    model_config = ConfigDict(extra="forbid")

    disposition: ProjectStoryLinkComparisonDisposition
    review_id: str = Field(min_length=1)
    review_revision: int = Field(ge=1)
    is_current: bool


class ProjectStoryLinkCandidateComparisonAttempt(BaseModel):
    """One immutable local comparison attempt for the candidate."""

    model_config = ConfigDict(extra="forbid")

    attempt_number: int = Field(ge=1)
    comparison_id: str = Field(min_length=1)
    run_state: Literal[
        "idempotency_conflict", "completed_unreviewed", "failed"
    ]
    failure_code: Literal[
        "runtime_binding_failed",
        "sampling_failed",
        "provider_unavailable",
        "provider_rate_limited",
        "provider_request_rejected",
        "provider_invalid",
        "source_changed",
    ] | None = None

    @model_validator(mode="after")
    def validate_attempt_outcome(self) -> Self:
        if (self.run_state == "failed") != (self.failure_code is not None):
            raise ValueError("comparison attempt failure code must match its run state")
        return self


class ProjectStoryLinkComparisonRequestData(BaseModel):
    """Exact request body for comparing one discovered candidate pair."""

    model_config = ConfigDict(extra="forbid")

    manifest_revision: int = Field(ge=1)
    story_graph_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1, max_length=128)
    candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    attempt_number: int | None = Field(default=None, ge=1)
    source_mention_review_id: str | None = Field(default=None, min_length=1)
    source_mention_review_revision: int = Field(default=0, ge=0)
    relation_kind: Literal["person_identity", "event_identity"]
    left_observation_id: str = Field(min_length=1)
    right_observation_id: str = Field(min_length=1)
    left_story_graph_node_id: str | None = Field(default=None, min_length=1)
    right_story_graph_node_id: str | None = Field(default=None, min_length=1)
    left_person_description: str | None = Field(
        default=None, min_length=1, max_length=240)
    right_person_description: str | None = Field(
        default=None, min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if (self.source_mention_review_id is None) != (
            self.source_mention_review_revision == 0
        ):
            raise ValueError("source mention review identity and revision must be bound")
        if self.left_observation_id == self.right_observation_id:
            raise ValueError("comparison requires two distinct observations")
        if self.candidate_id is None:
            if self.attempt_number is not None:
                raise ValueError("attempt_number requires a candidate_id")
        else:
            if self.attempt_number is None:
                raise ValueError("candidate comparison requires an attempt_number")
            expected_key = (
                self.candidate_id
                if self.attempt_number == 1
                else f"{self.candidate_id}:retry:{self.attempt_number}"
            )
            if self.idempotency_key != expected_key:
                raise ValueError(
                    "candidate idempotency key must bind the exact attempt number")
        if self.relation_kind == "person_identity":
            if not self.left_person_description or not self.right_person_description:
                raise ValueError("person comparison requires both stored descriptions")
        elif (self.left_person_description is not None
              or self.right_person_description is not None):
            raise ValueError("event comparison cannot include person descriptions")
        return self


class ProjectStoryLinkCandidateComparisonProgress(BaseModel):
    """Persisted comparison attempts and the stable next retry, if needed."""

    model_config = ConfigDict(extra="forbid")

    comparison_run_state: Literal[
        "not_started",
        "idempotency_conflict",
        "failed",
        "completed_unreviewed",
    ] = "not_started"
    comparison_id: str | None = None
    attempts: list[ProjectStoryLinkCandidateComparisonAttempt] = Field(
        default_factory=list)
    retry_comparison_request: ProjectStoryLinkComparisonRequestData | None = None
    latest_caller_disposition: ProjectStoryLinkCandidateCallerDisposition | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if self.comparison_run_state == "not_started":
            if (
                self.comparison_id is not None
                or self.attempts
                or self.retry_comparison_request is not None
                or self.latest_caller_disposition is not None
            ):
                raise ValueError("not-started candidate cannot cite comparison progress")
        elif self.comparison_run_state == "idempotency_conflict":
            if (
                self.comparison_id is None
                or self.attempts
                or self.retry_comparison_request is None
                or self.latest_caller_disposition is not None
            ):
                raise ValueError("idempotency conflict must offer a separate retry request")
        else:
            if not self.attempts or self.comparison_id is None:
                raise ValueError("recorded comparison progress requires an attempt")
            latest = self.attempts[-1]
            if latest.comparison_id != self.comparison_id:
                raise ValueError("latest comparison id must match the last attempt")
            if latest.run_state != self.comparison_run_state:
                raise ValueError("candidate run state must match the latest attempt")
            if self.comparison_run_state == "failed":
                if self.retry_comparison_request is None:
                    raise ValueError("failed candidate must expose a retry request")
            elif self.retry_comparison_request is not None:
                raise ValueError("completed candidate cannot expose a failure retry")
        if self.attempts:
            if self.attempts[0].attempt_number != 1:
                raise ValueError("candidate comparison history must begin at attempt one")
            if any(
                right.attempt_number != left.attempt_number + 1
                for left, right in zip(self.attempts, self.attempts[1:])
            ):
                raise ValueError("candidate comparison attempts must be consecutive")
            if any(
                item.run_state == "idempotency_conflict"
                for item in self.attempts[1:]
            ):
                raise ValueError("idempotency conflict can only occupy attempt one")
        return self


class ProjectStoryLinkCandidate(BaseModel):
    """One exact eligible pair; it is not a match hypothesis or inferred link."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    relation_kind: Literal["person_identity", "event_identity"]
    left_anchor: ProjectStoryLinkAnchor
    right_anchor: ProjectStoryLinkAnchor
    left_person_description: str | None = Field(default=None, max_length=240)
    right_person_description: str | None = Field(default=None, max_length=240)
    left_event_evidence: ProjectStoryLinkEventEvidence | None = None
    right_event_evidence: ProjectStoryLinkEventEvidence | None = None
    comparison_request: ProjectStoryLinkComparisonRequestData
    candidate_basis: Literal["exhaustive_cross_asset_pair"] = (
        "exhaustive_cross_asset_pair"
    )
    ranking_state: Literal["unranked", "shadow_ranked_unadmitted"] = "unranked"
    ranking_position: int | None = Field(default=None, ge=1)
    ranking_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, allow_inf_nan=False,
        description=(
            "Raw, uncalibrated text-embedding cosine similarity. It is not an "
            "identity probability, confidence, or match decision."
        ),
    )
    automatic_inference_state: Literal["not_attempted"] = "not_attempted"
    comparison_progress: ProjectStoryLinkCandidateComparisonProgress = Field(
        default_factory=ProjectStoryLinkCandidateComparisonProgress)

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.left_anchor.project_asset_id == self.right_anchor.project_asset_id:
            raise ValueError("a cross-asset candidate must span two project assets")
        request = self.comparison_request
        if (
            request.relation_kind != self.relation_kind
            or request.candidate_id != self.candidate_id
            or request.left_observation_id != self.left_anchor.observation_id
            or request.right_observation_id != self.right_anchor.observation_id
            or request.left_story_graph_node_id
            != self.left_anchor.story_graph_node_id
            or request.right_story_graph_node_id
            != self.right_anchor.story_graph_node_id
            or request.left_person_description != self.left_person_description
            or request.right_person_description != self.right_person_description
        ):
            raise ValueError("candidate comparison request must match its exact anchors")
        if self.relation_kind == "person_identity":
            if self.left_event_evidence is not None or self.right_event_evidence is not None:
                raise ValueError("person candidates cannot carry event evidence")
        elif self.left_person_description is not None or self.right_person_description is not None:
            raise ValueError("event candidates cannot carry person descriptions")
        if self.ranking_state == "unranked":
            if self.ranking_position is not None or self.ranking_score is not None:
                raise ValueError("unranked candidate cannot carry ranking output")
        elif self.ranking_position is None or self.ranking_score is None:
            raise ValueError("shadow-ranked candidate requires position and raw score")
        return self


class ProjectStoryLinkCandidateAssetCoverage(BaseModel):
    """Coverage disclosure; rights remain caller-declared, not independently verified."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    state: Literal[
        "eligible_under_manifest_declaration",
        "no_eligible_mentions",
        "not_locally_authorized",
        "source_identity_unverified",
        "analysis_incomplete",
    ]
    rights_state: Literal[
        "unverified", "local_processing_allowed", "third_party_processing_allowed"
    ]
    rights_evidence_state: Literal["declared_unverified"] = "declared_unverified"
    source_identity_state: Literal["locally_verified", "declared_unverified"]
    eligible_mention_count: int = Field(ge=0)


class ProjectStoryLinkCandidatePage(BaseModel):
    """Stable page over a graph-revision-bound, unranked caller scope."""

    model_config = ConfigDict(extra="forbid")

    candidate_set_id: str = Field(min_length=1)
    candidate_algorithm_version: Literal["exhaustive_cross_asset_pairs_v1"] = (
        "exhaustive_cross_asset_pairs_v1"
    )
    project_id: str = Field(min_length=1)
    project_manifest_id: str = Field(min_length=1)
    project_revision: int = Field(ge=1)
    source_hash_validation_state: Literal["not_revalidated_by_read"] = Field(
        default="not_revalidated_by_read",
        description=(
            "This read uses the persisted manifest and evidence without hashing "
            "current source files. Any operation that consumes source media or "
            "creates a review or Plan must revalidate the registered source hashes."
        ),
    )
    context_id: str = Field(min_length=1)
    analysis_fingerprint: str = Field(min_length=1)
    story_graph_id: str = Field(min_length=1)
    source_mention_review_id: str | None = Field(default=None, min_length=1)
    source_mention_review_revision: int = Field(default=0, ge=0)
    source_mention_review_state: Literal["absent", "caller_asserted"] = Field(
        default="absent",
        description=(
            "Whether a caller-asserted source mention review snapshot is bound. "
            "caller_asserted does not verify reviewer identity or prove that "
            "the source media was viewed."
        ),
    )
    relation_kind: Literal["person_identity", "event_identity"]
    selected_anchor_story_graph_node_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional caller-selected current mention. When present, candidates "
            "are complete for this mention across eligible assets, unless a "
            "counterpart asset is also selected; no ranking or identity "
            "inference is performed."
        ),
    )
    selected_counterpart_project_asset_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional project asset paired with the selected mention. This "
            "narrows the complete unranked page to that asset only; it does "
            "not infer or score identity. Requires a selected anchor."
        ),
    )
    selected_project_asset_pair: tuple[str, str] | None = Field(
        default=None,
        description=(
            "Optional pair of caller-selected project assets. When present, "
            "the page contains every eligible cross-asset pair within those "
            "assets, in manifest order, without ranking or identity inference. "
            "It cannot be combined with a selected mention anchor or counterpart asset."
        ),
    )
    asset_coverage: list[ProjectStoryLinkCandidateAssetCoverage]
    eligible_asset_count: int = Field(ge=0)
    eligible_mention_count: int = Field(ge=0)
    candidates: list[ProjectStoryLinkCandidate]
    limit: int = Field(ge=1, le=200)
    offset: int = Field(ge=0)
    total: int = Field(ge=0)
    has_more: bool
    selection_semantics: Literal[
        "all_eligible_cross_asset_pairs_in_manifest_order",
        "all_eligible_cross_asset_pairs_for_selected_mention_in_manifest_order",
        "all_eligible_pairs_for_anchor_and_counterpart_asset_in_manifest_order",
        "all_eligible_cross_asset_pairs_for_selected_asset_pair_in_manifest_order",
        "all_eligible_cross_asset_pairs_for_selected_mention_ranked_by_shadow_similarity",
        "all_eligible_pairs_for_anchor_and_counterpart_asset_ranked_by_shadow_similarity",
    ] = "all_eligible_cross_asset_pairs_in_manifest_order"
    ranking_state: Literal["unranked", "shadow_ranked_unadmitted"] = "unranked"
    ranking_profile_id: Literal["bge_m3_shadow_v1"] | None = None
    ranking_model: Literal["bge-m3:latest"] | None = None
    ranking_model_digest: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$")
    ranking_metric: Literal["cosine_similarity"] | None = None
    ranking_score_semantics: Literal["uncalibrated_similarity_only"] | None = None
    automatic_inference_state: Literal["not_attempted"] = "not_attempted"
    relation_inference_state: Literal["EXPERIMENTAL"] = "EXPERIMENTAL"
    quality_acceptance: Literal["not_proven"] = "not_proven"

    @model_validator(mode="after")
    def validate_page(self) -> Self:
        if (self.source_mention_review_id is None) != (
            self.source_mention_review_revision == 0
        ):
            raise ValueError("mention review identity and revision must be bound together")
        if (self.source_mention_review_id is None) != (
            self.source_mention_review_state == "absent"
        ):
            raise ValueError("mention review state must match its identity")
        if self.eligible_asset_count > len(self.asset_coverage):
            raise ValueError("eligible asset count exceeds coverage")
        if self.eligible_asset_count != sum(
            item.state == "eligible_under_manifest_declaration"
            for item in self.asset_coverage
        ):
            raise ValueError("eligible asset count must match disclosed coverage")
        if self.eligible_mention_count != sum(
            item.eligible_mention_count for item in self.asset_coverage
        ):
            raise ValueError("eligible mention count must match disclosed coverage")
        if (
            self.selected_counterpart_project_asset_id is not None
            and self.selected_anchor_story_graph_node_id is None
        ):
            raise ValueError("counterpart asset scope requires a selected anchor")
        if self.selected_project_asset_pair is not None:
            if self.selected_project_asset_pair[0] == self.selected_project_asset_pair[1]:
                raise ValueError("selected asset pair must contain different assets")
            if (
                self.selected_anchor_story_graph_node_id is not None
                or self.selected_counterpart_project_asset_id is not None
            ):
                raise ValueError("selected asset pair cannot be combined with anchor scope")
            expected_selection_semantics = (
                "all_eligible_cross_asset_pairs_for_selected_asset_pair_in_manifest_order")
        elif self.selected_counterpart_project_asset_id is not None:
            expected_selection_semantics = (
                "all_eligible_pairs_for_anchor_and_counterpart_asset_ranked_by_shadow_similarity"
                if self.ranking_state == "shadow_ranked_unadmitted"
                else "all_eligible_pairs_for_anchor_and_counterpart_asset_in_manifest_order"
            )
        elif self.selected_anchor_story_graph_node_id is not None:
            expected_selection_semantics = (
                "all_eligible_cross_asset_pairs_for_selected_mention_ranked_by_shadow_similarity"
                if self.ranking_state == "shadow_ranked_unadmitted"
                else "all_eligible_cross_asset_pairs_for_selected_mention_in_manifest_order"
            )
        else:
            expected_selection_semantics = (
                "all_eligible_cross_asset_pairs_in_manifest_order")
        if self.selection_semantics != expected_selection_semantics:
            raise ValueError("candidate selection semantics must match its anchor scope")
        if self.total and self.eligible_asset_count < 2:
            raise ValueError("cross-asset pairs require at least two eligible assets")
        if len(self.candidates) > self.limit:
            raise ValueError("candidate page exceeds its requested limit")
        if self.has_more != (self.offset + len(self.candidates) < self.total):
            raise ValueError("has_more must match the candidate page boundary")
        if len({item.candidate_id for item in self.candidates}) != len(self.candidates):
            raise ValueError("candidate ids must be unique within a page")
        if self.ranking_state == "unranked":
            if any((
                self.ranking_profile_id is not None,
                self.ranking_model is not None,
                self.ranking_model_digest is not None,
                self.ranking_metric is not None,
                self.ranking_score_semantics is not None,
            )) or any(item.ranking_state != "unranked" for item in self.candidates):
                raise ValueError("unranked page cannot carry ranking output")
        else:
            if self.selected_anchor_story_graph_node_id is None:
                raise ValueError("shadow ranking requires a caller-selected anchor")
            if (
                self.ranking_profile_id != "bge_m3_shadow_v1"
                or self.ranking_model != "bge-m3:latest"
                or self.ranking_model_digest is None
                or self.ranking_metric != "cosine_similarity"
                or self.ranking_score_semantics != "uncalibrated_similarity_only"
            ):
                raise ValueError("shadow ranking metadata is incomplete")
            if any(item.ranking_state != "shadow_ranked_unadmitted"
                   for item in self.candidates):
                raise ValueError("ranked page candidates must carry ranking output")
            expected_positions = list(range(self.offset + 1,
                                            self.offset + len(self.candidates) + 1))
            if [item.ranking_position for item in self.candidates] != expected_positions:
                raise ValueError("ranking positions must preserve the global page order")
        if self.selected_project_asset_pair is not None and any(
            {
                item.left_anchor.project_asset_id,
                item.right_anchor.project_asset_id,
            } != set(self.selected_project_asset_pair)
            for item in self.candidates
        ):
            raise ValueError("asset-pair candidate page contains a pair outside its scope")
        if self.selected_anchor_story_graph_node_id is not None and any(
            self.selected_anchor_story_graph_node_id not in {
                item.left_anchor.story_graph_node_id,
                item.right_anchor.story_graph_node_id,
            }
            for item in self.candidates
        ):
            raise ValueError("anchor-scoped candidate page contains a non-anchor pair")
        if self.selected_counterpart_project_asset_id is not None and any(
            (
                item.right_anchor.project_asset_id
                if item.left_anchor.story_graph_node_id
                == self.selected_anchor_story_graph_node_id
                else item.left_anchor.project_asset_id
            ) != self.selected_counterpart_project_asset_id
            for item in self.candidates
        ):
            raise ValueError(
                "counterpart-scoped candidate page contains a different asset pair")
        if any(
            item.comparison_request.source_mention_review_id
            != self.source_mention_review_id
            or item.comparison_request.source_mention_review_revision
            != self.source_mention_review_revision
            for item in self.candidates
        ):
            raise ValueError("candidate request must bind the page source mention review")
        return self
