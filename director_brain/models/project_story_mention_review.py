"""Caller review of source-local Person/Event StoryGraph mentions."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from director_brain.models.base import BaseRecord
from director_brain.models.project_story_link_comparison import (
    ProjectStoryLinkEventEvidence,
)
from director_brain.models.project_story_link_review import ProjectStoryLinkAnchor


class ProjectStoryMentionDecision(BaseModel):
    """One exact mention disposition; source observations remain immutable."""

    model_config = ConfigDict(extra="forbid")

    relation_kind: Literal["person_identity", "event_identity"]
    anchor: ProjectStoryLinkAnchor
    decision: Literal["accepted_as_observed", "corrected_by_caller", "rejected"]
    corrected_person_description: str | None = Field(
        default=None, min_length=1, max_length=240)
    corrected_event_evidence: ProjectStoryLinkEventEvidence | None = None
    rationale: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_correction_fields(self) -> Self:
        if self.decision != "accepted_as_observed" and not self.rationale.strip():
            raise ValueError("a correction or rejection requires caller rationale")
        if self.decision == "corrected_by_caller":
            if self.relation_kind == "person_identity":
                if not self.corrected_person_description:
                    raise ValueError("corrected person mention requires a description")
                if self.corrected_event_evidence is not None:
                    raise ValueError("person mention cannot carry event corrections")
            else:
                if self.corrected_event_evidence is None:
                    raise ValueError("corrected event mention requires event evidence")
                if self.corrected_person_description is not None:
                    raise ValueError("event mention cannot carry person corrections")
        elif (
            self.corrected_person_description is not None
            or self.corrected_event_evidence is not None
        ):
            raise ValueError("only a corrected mention can carry replacement content")
        return self


class ProjectStoryMentionReview(BaseRecord):
    """Immutable full snapshot of caller dispositions for one graph revision."""

    schema_version: Literal["1.0"] = "1.0"
    review_id: str = Field(min_length=1)
    project_manifest_id: str = Field(min_length=1)
    project_revision: int = Field(ge=1)
    context_id: str = Field(min_length=1)
    story_graph_id: str = Field(min_length=1)
    analysis_fingerprint: str = Field(min_length=1)
    review_revision: int = Field(ge=1)
    review_scope: Literal["source_local_person_event_mentions"] = (
        "source_local_person_event_mentions"
    )
    actor_identity_state: Literal["caller_asserted"] = "caller_asserted"
    decisions: list[ProjectStoryMentionDecision] = Field(max_length=10_000)

    @model_validator(mode="after")
    def validate_unique_targets(self) -> Self:
        keys = [
            (item.relation_kind, item.anchor.project_asset_id,
             item.anchor.story_graph_node_id)
            for item in self.decisions
        ]
        if any(node_id is None for _kind, _asset, node_id in keys):
            raise ValueError("source mention review requires exact graph node anchors")
        if len(keys) != len(set(keys)):
            raise ValueError("a review snapshot cannot repeat an exact source mention")
        return self
