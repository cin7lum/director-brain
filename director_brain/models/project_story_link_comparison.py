"""Unreviewed local-VLM comparison evidence for one cross-asset pair."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from director_brain.models.base import BaseRecord
from director_brain.models.film_observation import TimebaseUnit
from director_brain.models.project_story_link_review import ProjectStoryLinkAnchor


class ProjectStoryLinkEventEvidence(BaseModel):
    """Exact source-local semantic fields identifying one event mention."""

    model_config = ConfigDict(extra="forbid")

    action_type: str | None = Field(default=None, max_length=40)
    scene_description: str | None = Field(default=None, max_length=1000)
    temporal_notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_specific_event_evidence(self) -> Self:
        if (not (self.scene_description or "").strip()
                and not (self.temporal_notes or "").strip()):
            raise ValueError(
                "event comparison requires a stored scene or temporal description")
        return self


class ProjectStoryLinkComparison(BaseRecord):
    """Persist one local shadow comparison without asserting an identity link.

    This review aid is separate from ``ProjectStoryLinkReview`` and cannot be
    consumed by a Director Plan.
    """

    comparison_id: str
    project_manifest_id: str
    project_revision: int = Field(ge=1)
    context_id: str
    story_graph_id: str
    analysis_fingerprint: str
    source_mention_review_id: str | None = Field(default=None, min_length=1)
    source_mention_review_revision: int = Field(default=0, ge=0)
    request_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    idempotency_key_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    attempt_number: int | None = Field(default=None, ge=1)
    relation_kind: Literal["person_identity", "event_identity"]
    left_anchor: ProjectStoryLinkAnchor
    right_anchor: ProjectStoryLinkAnchor
    left_person_description: str | None = Field(default=None, max_length=240)
    right_person_description: str | None = Field(default=None, max_length=240)
    left_event_evidence: ProjectStoryLinkEventEvidence | None = None
    right_event_evidence: ProjectStoryLinkEventEvidence | None = None
    run_state: Literal["completed_unreviewed", "failed"]
    assessment: Literal[
        "possible_match", "visually_distinct", "insufficient_evidence"
    ] | None = None
    evidence_for: list[str] = Field(default_factory=list, max_length=5)
    evidence_against: list[str] = Field(default_factory=list, max_length=5)
    limitation: str | None = Field(default=None, max_length=400)
    left_frame_sha256: list[str] = Field(default_factory=list, max_length=3)
    right_frame_sha256: list[str] = Field(default_factory=list, max_length=3)
    frame_extractor_version: str = Field(min_length=1, max_length=120)
    failure_code: Literal[
        "runtime_binding_failed",
        "sampling_failed",
        "provider_unavailable",
        "provider_rate_limited",
        "provider_request_rejected",
        "provider_invalid",
        "source_changed",
    ] | None = None
    review_state: Literal["unreviewed"] = "unreviewed"
    confidence_type: Literal[
        "uncalibrated_model_assessment"
    ] = "uncalibrated_model_assessment"
    provider: Literal["ollama_qwen3_vl"] = "ollama_qwen3_vl"
    model_version: str = Field(min_length=1, max_length=240)
    prompt_version: str = Field(min_length=1, max_length=120)
    prompt_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    generation_profile: str = Field(min_length=1, max_length=240)
    sampling_profile: str = Field(min_length=1, max_length=240)
    observation_schema_version: str = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def validate_pair_and_outcome(self) -> Self:
        if (self.source_mention_review_id is None) != (
            self.source_mention_review_revision == 0
        ):
            raise ValueError("comparison must bind its source mention review revision")
        if (self.candidate_id is None) != (self.attempt_number is None):
            raise ValueError("candidate-bound comparison requires its attempt number")
        if self.left_anchor.project_asset_id == self.right_anchor.project_asset_id:
            raise ValueError("cross-asset comparison requires two distinct assets")
        if self.left_anchor.timebase_unit != TimebaseUnit.MICROSECONDS:
            raise ValueError("left comparison anchor must use microseconds")
        if self.right_anchor.timebase_unit != TimebaseUnit.MICROSECONDS:
            raise ValueError("right comparison anchor must use microseconds")
        if self.relation_kind == "person_identity":
            if not self.left_person_description or not self.right_person_description:
                raise ValueError(
                    "person comparison requires exact stored person descriptions")
            if self.left_event_evidence is not None or self.right_event_evidence is not None:
                raise ValueError("person comparison cannot carry event evidence")
        elif (self.left_person_description is not None
              or self.right_person_description is not None):
            raise ValueError("event comparison cannot carry person descriptions")
        elif ((self.left_event_evidence is None)
              != (self.right_event_evidence is None)):
            raise ValueError("event comparison requires evidence from both observations")
        elif (self.prompt_version == "project_cross_asset_pair_v2"
              and (self.left_event_evidence is None
                   or self.right_event_evidence is None)):
            raise ValueError("v2 event comparison requires exact stored event evidence")

        if self.run_state == "completed_unreviewed":
            if self.assessment is None or self.failure_code is not None:
                raise ValueError("completed comparison requires an assessment")
            if len(self.left_frame_sha256) != 3 or len(self.right_frame_sha256) != 3:
                raise ValueError("completed comparison requires six input-frame hashes")
        elif (self.assessment is not None or self.failure_code is None
              or self.evidence_for or self.evidence_against):
            raise ValueError("failed comparison must carry only a failure code")
        if any(
            len(value) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in value)
            for value in (*self.left_frame_sha256, *self.right_frame_sha256)
        ):
            raise ValueError("input frame hashes must be SHA-256 hex values")
        return self
