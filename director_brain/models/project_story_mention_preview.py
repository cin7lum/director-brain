"""Ephemeral, locally extracted visual evidence for one StoryGraph mention."""
from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from director_brain.models.project_story_link_review import ProjectStoryLinkAnchor


class ProjectStoryMentionPreviewFrame(BaseModel):
    """One bounded JPEG frame returned for an authorized source mention."""

    model_config = ConfigDict(extra="forbid")

    relative_position: Literal[0.15, 0.5, 0.85]
    media_type: Literal["image/jpeg"] = "image/jpeg"
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    width: int = Field(gt=0, le=1280)
    height: int = Field(gt=0, le=1280)
    jpeg_base64: str = Field(min_length=4, max_length=2_000_000)

    @model_validator(mode="after")
    def validate_encoded_frame(self) -> Self:
        try:
            payload = base64.b64decode(self.jpeg_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("preview frame must contain valid base64 JPEG data") from exc
        if not payload.startswith(b"\xff\xd8\xff"):
            raise ValueError("preview frame must be a JPEG image")
        if hashlib.sha256(payload).hexdigest() != self.sha256.lower():
            raise ValueError("preview frame checksum does not match its payload")
        return self


class ProjectStoryMentionPreview(BaseModel):
    """Read-only visual review aid; frame data is never persisted by this model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_id: str = Field(min_length=1)
    project_manifest_id: str = Field(min_length=1)
    project_revision: int = Field(ge=1)
    story_graph_id: str = Field(min_length=1)
    relation_kind: Literal["person_identity", "event_identity"]
    anchor: ProjectStoryLinkAnchor
    source_hash_validation_state: Literal["verified_before_and_after"] = (
        "verified_before_and_after"
    )
    preview_profile: Literal[
        "ffmpeg_relative_0.15_0.50_0.85_cv2_jpeg_1280_v1"
    ] = "ffmpeg_relative_0.15_0.50_0.85_cv2_jpeg_1280_v1"
    cache_policy: Literal["no_store"] = "no_store"
    frames: list[ProjectStoryMentionPreviewFrame] = Field(
        min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_frame_order(self) -> Self:
        if self.anchor.story_graph_node_id is None:
            raise ValueError("mention preview requires an exact StoryGraph node binding")
        if [frame.relative_position for frame in self.frames] != [0.15, 0.5, 0.85]:
            raise ValueError("preview frames must preserve the frozen sample positions")
        return self
