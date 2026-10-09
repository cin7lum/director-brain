"""Project and asset manifest contracts for multi-asset Film Context.

The manifest records caller-provided project boundaries and rights evidence as
claims. It does not independently certify either claim or assign evaluation
splits.
"""
from __future__ import annotations

import enum
import re
from fractions import Fraction
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from director_brain.models.base import BaseRecord


class ProjectBoundaryBasis(str, enum.Enum):
    USER_PROJECT_MANIFEST = "user_project_manifest"
    DATASET_PROJECT_ID = "dataset_project_id"
    DATASET_EVENT_ID = "dataset_event_id"
    DATASET_RECORDING_ID = "dataset_recording_id"
    PRODUCTION_PACKAGE = "production_package"


class RightsBasis(str, enum.Enum):
    PUBLIC_LICENSE = "public_license"
    OWNER_PERMISSION = "owner_permission"
    OTHER_DOCUMENTED_AUTHORITY = "other_documented_authority"


class RightsState(str, enum.Enum):
    UNVERIFIED = "unverified"
    LOCAL_PROCESSING_ALLOWED = "local_processing_allowed"
    THIRD_PARTY_PROCESSING_ALLOWED = "third_party_processing_allowed"


class ProjectAnalysisProfile(str, enum.Enum):
    """Versioned analysis path selected as part of a manifest revision."""

    DETERMINISTIC_V1 = "deterministic_v1"
    LOCAL_VLM_SHADOW_V1 = "local_vlm_shadow_v1"
    LOCAL_ASR_SHADOW_V1 = "local_asr_shadow_v1"


_DECIMAL_SECONDS_PATTERN = r"^-?[0-9]+(?:\.[0-9]+)?$"


class ProjectStreamTiming(BaseModel):
    """One source stream's native clock and offset from the container origin."""

    model_config = ConfigDict(extra="forbid")

    stream_index: int = Field(ge=0)
    codec_type: Literal["video", "audio"]
    codec_name: str | None = None
    time_base: str | None = Field(
        default=None, pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    start_pts: int | None = None
    start_time_seconds: str | None = Field(
        default=None, pattern=_DECIMAL_SECONDS_PATTERN)
    duration_ts: int | None = None
    duration_seconds: str | None = Field(
        default=None, pattern=r"^[0-9]+(?:\.[0-9]+)?$")
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    channel_layout: str | None = None
    language: str | None = None
    is_default: bool | None = None
    source_start_offset_numerator: int | None = None
    source_start_offset_denominator: int | None = Field(default=None, gt=0)
    source_start_offset_state: Literal[
        "mapped_from_pts", "mapped_from_start_time", "unavailable"
    ] = "unavailable"

    @model_validator(mode="after")
    def validate_source_clock_mapping(self) -> Self:
        has_numerator = self.source_start_offset_numerator is not None
        has_denominator = self.source_start_offset_denominator is not None
        if has_numerator != has_denominator:
            raise ValueError("source start offset numerator and denominator must be paired")
        if self.source_start_offset_state == "unavailable":
            if has_numerator:
                raise ValueError("unavailable source clock cannot carry a mapped offset")
        elif not has_numerator:
            raise ValueError("mapped source clock requires an exact rational offset")
        elif (self.source_start_offset_state == "mapped_from_pts"
                and (self.start_pts is None or self.time_base is None)):
            raise ValueError("PTS mapping requires start_pts and time_base")
        elif (self.source_start_offset_state == "mapped_from_start_time"
                and self.start_time_seconds is None):
            raise ValueError("start_time mapping requires start_time_seconds")
        return self


class ProjectAssetTimeMap(BaseModel):
    """Source video/audio clocks without implying a shared project timeline."""

    model_config = ConfigDict(extra="forbid")

    container_start_time_seconds: str | None = Field(
        default=None, pattern=_DECIMAL_SECONDS_PATTERN)
    video_stream: ProjectStreamTiming | None = None
    audio_streams: list[ProjectStreamTiming] = Field(default_factory=list)
    mapping_state: Literal["complete", "partial", "unavailable"]

    @model_validator(mode="after")
    def validate_stream_set_and_mapping_state(self) -> Self:
        streams = ([self.video_stream] if self.video_stream is not None else [])
        streams.extend(self.audio_streams)
        if self.video_stream is not None and self.video_stream.codec_type != "video":
            raise ValueError("video_stream must contain a video stream")
        if any(stream.codec_type != "audio" for stream in self.audio_streams):
            raise ValueError("audio_streams may contain only audio streams")
        if len({stream.stream_index for stream in streams}) != len(streams):
            raise ValueError("stream indexes must be unique within a time map")

        mapped = [
            stream for stream in streams
            if stream.source_start_offset_state != "unavailable"
        ]
        if mapped and self.container_start_time_seconds is None:
            raise ValueError("mapped stream offsets require a container start time")
        video_mapped = (
            self.video_stream is not None
            and self.video_stream.source_start_offset_state != "unavailable"
        )
        all_required_mapped = video_mapped and all(
            stream.source_start_offset_state != "unavailable"
            for stream in self.audio_streams
        )
        expected_state = (
            "complete" if all_required_mapped
            else "partial" if mapped
            else "unavailable"
        )
        if self.mapping_state != expected_state:
            raise ValueError(
                f"mapping_state must be {expected_state} for the available stream clocks")
        return self

    def first_audio_stream_clock_from_video(self) -> tuple[int, Fraction] | None:
        """Return the decoder's first audio stream and its offset from video time.

        faster-whisper's file decoder selects audio ordinal zero and discards
        frame PTS while assembling samples. The project adapter therefore adds
        this exact rational start-offset difference to transcript timestamps.
        """
        if not self.audio_streams:
            return None
        video = self.video_stream
        audio = self.audio_streams[0]
        if (video is None
                or video.source_start_offset_numerator is None
                or video.source_start_offset_denominator is None
                or audio.source_start_offset_numerator is None
                or audio.source_start_offset_denominator is None):
            raise ValueError("audio-to-video source clock mapping is unavailable")
        video_offset = Fraction(
            video.source_start_offset_numerator,
            video.source_start_offset_denominator,
        )
        audio_offset = Fraction(
            audio.source_start_offset_numerator,
            audio.source_start_offset_denominator,
        )
        return audio.stream_index, audio_offset - video_offset


class ProcessingRights(BaseModel):
    """Per-asset processing declaration; evidence is retained, never inferred."""

    model_config = ConfigDict(extra="forbid")

    state: RightsState = RightsState.UNVERIFIED
    basis: RightsBasis | None = None
    evidence_ref: str | None = None
    license_id: str | None = None
    permission_id: str | None = None
    evidence_state: Literal["declared_unverified"] = "declared_unverified"

    @model_validator(mode="after")
    def validate_authorization_evidence(self) -> Self:
        if self.evidence_state != "declared_unverified":
            raise ValueError("this API cannot mark rights evidence as independently verified")
        if self.state == RightsState.UNVERIFIED:
            if (self.basis is not None or self.evidence_ref or self.license_id
                    or self.permission_id):
                raise ValueError(
                    "unverified rights must not carry authorization claims")
            return self
        if self.basis is None or not self.evidence_ref or not self.evidence_ref.strip():
            raise ValueError(
                "processing authorization requires a basis and evidence_ref")
        if self.basis == RightsBasis.PUBLIC_LICENSE and not self.license_id:
            raise ValueError("public-license authority requires license_id")
        if (self.state == RightsState.THIRD_PARTY_PROCESSING_ALLOWED
                and not (self.license_id or self.permission_id)):
            raise ValueError(
                "third-party processing requires an explicit license_id or permission_id")
        return self


class ProjectAsset(BaseModel):
    """One immutable source identity in a project manifest revision."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    source_identity_state: Literal["locally_verified", "declared_unverified"] = (
        "locally_verified"
    )
    source_content_hash: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$")
    size_bytes: int | None = Field(default=None, gt=0)
    order: int = Field(ge=0)
    duration_us: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, gt=0)
    # Raw positive ffprobe rationals preserve source-clock provenance. These
    # do not by themselves define a VFR frame-index-to-time conversion.
    r_frame_rate: str | None = Field(
        default=None, pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    avg_frame_rate: str | None = Field(
        default=None, pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    stream_time_base: str | None = Field(
        default=None, pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    has_audio: bool | None = None
    time_map: ProjectAssetTimeMap | None = None
    probe_ok: bool | None = None
    rights: ProcessingRights

    @model_validator(mode="after")
    def validate_source_binding(self) -> Self:
        if self.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED:
            if self.source_identity_state != "declared_unverified":
                raise ValueError(
                    "assets without local processing rights require an unverified source identity")
            if any(value is not None for value in (
                self.source_content_hash,
                self.size_bytes,
                self.duration_us,
                self.fps,
                self.r_frame_rate,
                self.avg_frame_rate,
                self.stream_time_base,
                self.has_audio,
                self.time_map,
                self.probe_ok,
            )):
                raise ValueError(
                    "assets without local processing rights cannot claim media-derived identity or metadata")
            if not re.fullmatch(
                r"(?:dataset|source|asset)://[A-Za-z0-9._/-]+", self.source_ref
            ):
                raise ValueError(
                    "assets without local processing rights require a non-filesystem opaque source reference")
            return self
        if (self.source_identity_state != "locally_verified"
                or self.source_content_hash is None
                or self.size_bytes is None
                or self.probe_ok is None):
            raise ValueError(
                "authorized local assets require a locally verified content identity")
        if self.time_map is not None:
            if self.probe_ok is not True:
                raise ValueError("source stream timing requires a successful media probe")
            if self.has_audio != bool(self.time_map.audio_streams):
                raise ValueError("has_audio must match the mapped audio stream inventory")
        return self


class FilmProjectManifest(BaseRecord):
    """Versioned project boundary and ordered, content-addressed asset set."""

    manifest_id: str
    revision: int = Field(ge=1)
    boundary_basis: ProjectBoundaryBasis
    boundary_source_ref: str = Field(min_length=1)
    boundary_evidence_refs: list[str] = Field(min_length=1)
    boundary_state: str = "declared_unverified"
    analysis_profile: ProjectAnalysisProfile = ProjectAnalysisProfile.DETERMINISTIC_V1
    assets: list[ProjectAsset] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_asset_set(self) -> Self:
        if self.boundary_state != "declared_unverified":
            raise ValueError("project boundary cannot be marked verified by this API")
        if not self.project_id.strip() or not self.boundary_source_ref.strip():
            raise ValueError("project_id and boundary_source_ref must be non-empty")
        if any(not ref.strip() for ref in self.boundary_evidence_refs):
            raise ValueError("boundary evidence references must be non-empty")
        ids = [asset.asset_id for asset in self.assets]
        orders = [asset.order for asset in self.assets]
        hashes = [asset.source_content_hash.lower() for asset in self.assets
                  if asset.source_content_hash is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("asset_id values must be unique within a manifest")
        if len(orders) != len(set(orders)):
            raise ValueError("asset order values must be unique within a manifest")
        if len(hashes) != len(set(hashes)):
            raise ValueError(
                "duplicate source content hashes must be represented once per project")
        if not re.fullmatch(r"manifest_[a-f0-9]{16}_r[0-9]{8}", self.manifest_id):
            raise ValueError("manifest_id must be generated from project_id and revision")
        return self
