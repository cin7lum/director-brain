"""Immutable local record for one unranked Director Reasoner comparison."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from director_brain.models.base import BaseRecord
from director_brain.models.director_plan import ProjectDirectorShadowComparison


class ProjectDirectorShadowComparisonRecord(BaseRecord):
    """Persist a SHADOW comparison for audit and restart readback only.

    The stored candidate text remains local to the project's SQLite repository.
    This record is immutable, cannot be confirmed, and does not imply quality
    acceptance or pathway admission.
    """

    comparison_id: str = Field(min_length=1, max_length=128)
    request_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    idempotency_key_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    project_manifest_id: str = Field(min_length=1)
    project_revision: int = Field(ge=1)
    context_id: str = Field(min_length=1)
    story_graph_id: str = Field(min_length=1)
    project_story_link_review_id: str | None = Field(default=None, min_length=1)
    project_story_link_review_revision: int | None = Field(default=None, ge=1)
    project_story_mention_review_id: str | None = Field(default=None, min_length=1)
    project_story_mention_review_revision: int = Field(default=0, ge=0)
    comparison: ProjectDirectorShadowComparison
    confirmable: Literal[False] = False
    quality_acceptance: Literal["NOT_PROVEN"] = "NOT_PROVEN"

    @model_validator(mode="after")
    def validate_scope_and_shadow_state(self) -> Self:
        if (self.project_story_link_review_id is None) != (
            self.project_story_link_review_revision is None
        ):
            raise ValueError("link review ID and revision must be provided together")
        if (self.project_story_mention_review_id is None) != (
            self.project_story_mention_review_revision == 0
        ):
            raise ValueError("mention review ID and revision binding is inconsistent")
        if (self.comparison.confirmable
                or self.comparison.quality_acceptance != "NOT_PROVEN"
                or self.comparison.persistence_state != "PERSISTED_LOCAL"):
            raise ValueError("stored Director comparison must remain local SHADOW")
        if self.comparison.baseline_plan.project_id != self.project_id:
            raise ValueError("Director comparison belongs to a different project")
        for candidate in self.comparison.strategy_candidates:
            plan = candidate.plan
            if (plan.project_id != self.project_id
                    or plan.project_manifest_id != self.project_manifest_id
                    or plan.project_revision != self.project_revision
                    or plan.project_context_id != self.context_id
                    or plan.project_story_graph_id != self.story_graph_id):
                raise ValueError("Director candidate source scope is inconsistent")
        return self
