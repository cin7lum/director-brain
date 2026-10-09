"""影片上下文快照 FilmContextSnapshot 模型。

一次分析调用产生的、可被下游引用与缓存的影片状态快照。
"""
from __future__ import annotations

import enum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from director_brain.models.base import BaseRecord
from director_brain.models.film_observation import TimebaseUnit
from director_brain.models.project import ProjectAssetTimeMap


class ContextLayer(str, enum.Enum):
    """上下文覆盖层级。"""

    PROJECT = "project"
    ASSET = "asset"
    SCENE = "scene"
    EVIDENCE = "evidence"


class AssetAnalysisState(str, enum.Enum):
    """Per-asset state; a project context may retain usable assets on failure."""

    OBSERVED = "observed"
    COMPLETED_EMPTY = "completed_empty"
    PARTIAL = "partial"
    FAILED = "failed"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_CHANGED = "source_changed"
    NOT_AUTHORIZED = "not_authorized"
    PROVIDED = "provided"
    PROVIDED_EMPTY = "provided_empty"


class ObservationTimebase(BaseModel):
    """A source timebase descriptor; projects never merge these implicitly."""

    model_config = ConfigDict(extra="forbid")

    value: int = Field(gt=0)
    unit: TimebaseUnit


class ProjectAssetCoverage(BaseModel):
    """Typed coverage/provenance summary for one ordered source asset."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    order: int = Field(ge=0)
    source_content_hash: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$")
    duration_us: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, gt=0)
    # Raw source clock metadata retained from ffprobe. A constant nominal rate
    # is not sufficient to map variable-frame-rate frames to timestamps.
    r_frame_rate: str | None = Field(
        default=None, pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    avg_frame_rate: str | None = Field(
        default=None, pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    stream_time_base: str | None = Field(
        default=None, pattern=r"^[1-9][0-9]*/[1-9][0-9]*$")
    has_audio: bool | None = None
    time_map: ProjectAssetTimeMap | None = None
    probe_ok: bool | None = None
    observation_count: int = Field(ge=0)
    evidence_refs: list[str] = Field(default_factory=list)
    observation_timebases: list[ObservationTimebase] = Field(default_factory=list)
    analysis_state: AssetAnalysisState
    analysis_cache_state: str
    failure_code: Literal[
        "analyzer_failed", "source_unavailable", "source_changed",
        "semantic_analysis_partial", "speech_analysis_partial",
        "speech_analysis_failed",
    ] | None = None
    blocked_reason: Literal[
        "rights_unverified", "local_processing_not_authorized"
    ] | None = None
    rights_state: str
    rights_evidence_state: str

    @staticmethod
    def derive_state(
        observation_count: int,
        provided: bool,
    ) -> AssetAnalysisState:
        if provided:
            return (
                AssetAnalysisState.PROVIDED
                if observation_count
                else AssetAnalysisState.PROVIDED_EMPTY
            )
        return (
            AssetAnalysisState.OBSERVED
            if observation_count
            else AssetAnalysisState.COMPLETED_EMPTY
        )

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        if (self.rights_state != "local_processing_allowed"
                and self.time_map is not None):
            raise ValueError("unapproved media cannot expose source stream timing")
        if (self.time_map is not None
                and self.has_audio != bool(self.time_map.audio_streams)):
            raise ValueError("has_audio must match the source audio stream inventory")
        if len(self.evidence_refs) != self.observation_count:
            raise ValueError("evidence_refs must match observation_count")
        if self.analysis_state in (
            AssetAnalysisState.FAILED,
            AssetAnalysisState.SOURCE_UNAVAILABLE,
            AssetAnalysisState.SOURCE_CHANGED,
        ):
            if self.observation_count != 0 or self.failure_code is None:
                raise ValueError(
                    "incomplete asset coverage requires a failure code and no evidence")
        elif self.analysis_state == AssetAnalysisState.PARTIAL:
            if (self.observation_count == 0
                    or self.failure_code not in (
                        "semantic_analysis_partial",
                        "speech_analysis_partial",
                    )):
                raise ValueError(
                    "partial asset coverage requires evidence and a stage failure code")
        elif self.failure_code is not None:
            raise ValueError("complete asset coverage cannot carry a failure code")
        if self.analysis_state == AssetAnalysisState.NOT_AUTHORIZED:
            if self.observation_count != 0 or self.blocked_reason is None:
                raise ValueError(
                    "unauthorized asset coverage requires a block reason and no evidence")
        elif self.blocked_reason is not None:
            raise ValueError("only unauthorized asset coverage can carry a block reason")
        if self.analysis_state == AssetAnalysisState.FAILED:
            if self.failure_code not in (
                "analyzer_failed", "speech_analysis_failed"):
                raise ValueError("failed asset coverage requires an analyzer failure code")
        elif self.analysis_state == AssetAnalysisState.PARTIAL:
            if self.failure_code not in (
                "semantic_analysis_partial", "speech_analysis_partial"):
                raise ValueError("partial asset coverage requires a partial failure code")
        else:
            expected_failure_code = {
                AssetAnalysisState.SOURCE_UNAVAILABLE: "source_unavailable",
                AssetAnalysisState.SOURCE_CHANGED: "source_changed",
            }.get(self.analysis_state)
            if (expected_failure_code is not None
                    and self.failure_code != expected_failure_code):
                raise ValueError("failure_code does not match analysis_state")
        if self.analysis_state in (
            AssetAnalysisState.COMPLETED_EMPTY,
            AssetAnalysisState.PROVIDED_EMPTY,
        ) and self.observation_count != 0:
            raise ValueError("empty analysis state cannot carry observations")
        if self.analysis_state in (
            AssetAnalysisState.OBSERVED,
            AssetAnalysisState.PROVIDED,
        ) and self.observation_count == 0:
            raise ValueError("observed state requires evidence references")
        return self


class FilmContextSnapshot(BaseRecord):
    """某次分析调用产生的影片上下文快照（project_id / created_at 由基类提供）。"""

    context_id: str
    project_manifest_id: str | None = None
    project_revision: int | None = None
    asset_refs: list[str] = Field(default_factory=list)
    source_content_hashes: list[str | None] = Field(default_factory=list)
    layers: list[ContextLayer] = Field(default_factory=list)
    analysis_fingerprint: str
    provider: str
    model: str
    prompt_version: str
    sampling_config: dict[str, Any] = Field(default_factory=dict)
    # A multi-asset project has no shared timeline; its observations retain
    # their own units in asset_coverage and FilmObservation records.
    timeline_scope: Literal[
        "single_asset", "project_per_asset", "mixed_or_unknown", "not_applicable"
    ] = "not_applicable"
    timebase: int | None = None
    timebase_unit: TimebaseUnit | None = None
    asset_coverage: list[ProjectAssetCoverage] = Field(default_factory=list)
    coverage: str
    rights_scope: str
    evidence_refs: list[str] = Field(default_factory=list)
    cache_state: str
    invalidated_at: int | None = None
