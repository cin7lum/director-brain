"""Canonical Parameterization models for Director Brain.

ParameterizationDecision: execution parameter + feasibility + evidence + status.
ParameterizationContext: provider-neutral facts needed to compute parameters.

These are SEPARATE from DirectorDecision (semantic intent).
DirectorDecision = WHAT the user wants.
ParameterizationDecision = HOW to execute it (or why not).

extra="forbid" on all models — no silent field additions.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParameterizationStatus(str, Enum):
    """Execution readiness status for a parameter."""
    READY = "READY"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    NEEDS_DECISION = "NEEDS_DECISION"
    CONFLICT = "CONFLICT"
    UNSATISFIABLE = "UNSATISFIABLE"
    UNAVAILABLE = "UNAVAILABLE"


class CandidateSource(str, Enum):
    """Where a parameter candidate comes from."""
    EXPLICIT_USER = "explicit_user"
    OBSERVED_EVENT = "observed_event"        # dialogue onset, silence boundary
    FEASIBLE_BOUNDARY = "feasible_boundary"  # min/max of feasible range
    CONVENTION = "convention"                 # frame-rate derived increments (EXPERIMENTAL)
    HANDLE_FRACTION = "handle_fraction"       # fraction of available audio handle


class AudioEventType(str, Enum):
    DIALOGUE_ONSET = "dialogue_onset"
    SILENCE_START = "silence_start"
    SILENCE_END = "silence_end"
    SPEECH_TAIL = "speech_tail"


class AudioEvent(BaseModel):
    """An observed audio event with provenance."""
    model_config = ConfigDict(extra="forbid")

    event_type: AudioEventType
    frame: int = Field(description="Frame position relative to picture cut (positive = after cut)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(default="", description="How this event was detected (asr, waveform, manual)")
    evidence_ref: str = Field(default="", description="Reference to raw evidence")


class ClipBoundary(BaseModel):
    """A clip's timeline and source boundary in frames."""
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(description="Stable clip identity, not positional index")
    media_name: str = ""
    start_frame: int
    end_frame: int
    duration_frames: int
    source_start_frame: int = Field(default=0, description="Source frame at timeline start")
    source_end_frame: int = Field(default=0)
    source_duration_frames: int = Field(default=0)


class FeasibleRange(BaseModel):
    """Hard feasibility bounds for a parameter."""
    model_config = ConfigDict(extra="forbid")

    parameter_name: str
    min_value: float
    max_value: float
    unit: str = Field(description="frames or seconds")
    rationale: str = ""
    constraints_checked: list[str] = Field(default_factory=list)


class ParameterCandidate(BaseModel):
    """A candidate parameter value with evidence."""
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str
    source: CandidateSource
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ParameterizationContext(BaseModel):
    """Provider-neutral context for parameterization.

    Contains ONLY film-semantic and technical facts any provider could use.
    NO recordFrame, trackIndex, Resolve API, MCP fields, run_script.
    """
    model_config = ConfigDict(extra="forbid")

    frame_rate: float
    picture_cut_frame: int = Field(description="The video cut point being worked on")
    incoming_clip: ClipBoundary | None = None
    outgoing_clip: ClipBoundary | None = None
    audio_events: list[AudioEvent] = Field(default_factory=list)
    available_audio_handle_before: int | None = Field(
        default=None,
        description="Frames of incoming audio available before picture cut (from source range)",
    )
    has_dialogue: bool | None = Field(default=None, description="None = unknown")
    timeline_clip_count: int = 0
    provenance: dict[str, str] = Field(
        default_factory=dict,
        description="Where each fact came from: e.g. {'audio_handle': 'timeline_readback', 'dialogue_onset': 'external_asr'}",
    )
    notes: list[str] = Field(default_factory=list)


class ParameterizationDecision(BaseModel):
    """Output of the context parameterizer.

    Separates execution parameter from semantic intent (DirectorDecision).
    READY only when exact_value is set, feasible, and evidence-backed.
    """
    model_config = ConfigDict(extra="forbid")

    parameter_name: str = Field(description="e.g. 'audio_offset_frames'")
    exact_value: float | None = Field(default=None)
    unit: str | None = Field(default=None, description="frames or seconds")
    status: ParameterizationStatus
    feasible_range: FeasibleRange | None = None
    candidates: list[ParameterCandidate] = Field(default_factory=list)
    selected_candidate: ParameterCandidate | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    reason_code: str = Field(default="", description="Machine-readable reason for status")
    evidence_refs: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    conflict_detail: str = ""
