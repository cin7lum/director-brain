"""Durable lifecycle contract for project Film Context analysis jobs."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProjectContextJobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ProjectContextJobProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: Literal[
        "queued",
        "validating_manifest",
        "deterministic_analysis",
        "semantic_analysis",
        "asset_complete",
        "context_persisted",
    ] = "queued"
    current_asset_id: str | None = None
    assets_completed: int = Field(default=0, ge=0)
    assets_total: int = Field(default=0, ge=0)
    units_completed: int = Field(default=0, ge=0)
    units_total: int | None = Field(default=None, ge=0)


class ProjectContextJob(BaseModel):
    """Project-scoped state; never contains media paths, auth tokens, or outputs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    job_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    manifest_id: str = Field(min_length=1)
    manifest_revision: int = Field(ge=1)
    context_id: str = Field(min_length=1)
    analysis_profile: str = Field(min_length=1)
    request_fingerprint: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=1, max_length=128)
    state: ProjectContextJobState
    progress: ProjectContextJobProgress = Field(
        default_factory=ProjectContextJobProgress)
    attempt: int = Field(default=0, ge=0)
    worker_id: str | None = None
    created_at: int = Field(ge=0)
    updated_at: int = Field(ge=0)
    started_at: int | None = Field(default=None, ge=0)
    heartbeat_at: int | None = Field(default=None, ge=0)
    finished_at: int | None = Field(default=None, ge=0)
    cancel_requested_at: int | None = Field(default=None, ge=0)
    result_context_id: str | None = None
    failure_code: str | None = None

    @model_validator(mode="after")
    def validate_lifecycle_state(self) -> "ProjectContextJob":
        if self.state in (
            ProjectContextJobState.RUNNING,
            ProjectContextJobState.CANCEL_REQUESTED,
        ):
            if self.worker_id is None or self.heartbeat_at is None:
                raise ValueError("active context job requires a worker lease")
        if self.state in (
            ProjectContextJobState.SUCCEEDED,
            ProjectContextJobState.CANCELLED,
            ProjectContextJobState.FAILED,
        ) and self.finished_at is None:
            raise ValueError("terminal context job requires finished_at")
        if self.state == ProjectContextJobState.SUCCEEDED:
            if self.result_context_id is None or self.failure_code is not None:
                raise ValueError("successful context job requires only a result id")
        if self.state == ProjectContextJobState.FAILED and self.failure_code is None:
            raise ValueError("failed context job requires a failure code")
        if self.state in (
            ProjectContextJobState.QUEUED,
            ProjectContextJobState.CANCELLED,
            ProjectContextJobState.FAILED,
        ) and self.worker_id is not None:
            raise ValueError("inactive context job cannot hold a worker lease")
        return self
