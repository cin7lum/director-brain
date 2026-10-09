"""Ephemeral, locally extracted evidence for one cross-asset candidate pair."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from director_brain.models.project_story_mention_preview import (
    ProjectStoryMentionPreview,
)


class ProjectStoryLinkCandidatePreview(BaseModel):
    """Two exact mention previews for caller review; no link is inferred."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_id: str = Field(min_length=1)
    project_manifest_id: str = Field(min_length=1)
    project_revision: int = Field(ge=1)
    story_graph_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    relation_kind: Literal["person_identity", "event_identity"]
    source_hash_validation_state: Literal["verified_before_and_after"] = (
        "verified_before_and_after"
    )
    source_time_alignment_state: Literal["not_assumed"] = "not_assumed"
    preview_profile: Literal[
        "ffmpeg_relative_0.15_0.50_0.85_cv2_jpeg_1280_v1"
    ] = "ffmpeg_relative_0.15_0.50_0.85_cv2_jpeg_1280_v1"
    cache_policy: Literal["no_store"] = "no_store"
    left: ProjectStoryMentionPreview
    right: ProjectStoryMentionPreview

    @model_validator(mode="after")
    def validate_pair_binding(self) -> Self:
        for preview in (self.left, self.right):
            if (
                preview.project_id != self.project_id
                or preview.project_manifest_id != self.project_manifest_id
                or preview.project_revision != self.project_revision
                or preview.story_graph_id != self.story_graph_id
                or preview.relation_kind != self.relation_kind
            ):
                raise ValueError("candidate pair previews must share one current project graph")
            if preview.anchor.story_graph_node_id is None:
                raise ValueError("candidate pair preview requires exact mention nodes")
        if self.left.anchor.project_asset_id == self.right.anchor.project_asset_id:
            raise ValueError("candidate pair preview must span two project assets")
        return self
