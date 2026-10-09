"""Canonical DirectorDecision model.

Single source of truth for film-level director decisions.
Tool-agnostic, NLE-agnostic, provider-agnostic.

Expresses WHAT the director wants to happen at the film-semantic level.
Does NOT express HOW to implement it (no MCP/DaVinci/OTIO/tool fields).

This is the production formalization of the POC schema proven in
DIRECTOR_BRAIN_SEMANTIC_DECISION_POC.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class DecisionStatus(str, Enum):
    """Readiness status of a semantic director decision."""
    READY = "READY"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    UNDERSPECIFIED = "UNDERSPECIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    CONFLICTING_CONSTRAINTS = "CONFLICTING_CONSTRAINTS"


class Parameterization(BaseModel):
    """Parameter information for a decision.

    Vague language (\"一点\", \"稍微\") → magnitude only, exact_value=null.
    exact_value is set ONLY when the user explicitly provides a number.
    """
    model_config = ConfigDict(extra="forbid")

    magnitude: str | None = Field(
        default=None,
        description="Qualitative magnitude if given (slight, moderate, significant). Null if not specified.",
    )
    exact_value: float | None = Field(
        default=None,
        description="Exact numeric value ONLY if user explicitly provided one. Never infer from vague language.",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of exact_value (seconds, frames, percent). Null if exact_value is null.",
    )
    certainty: str = Field(
        default="unknown",
        description="How certain the parameter is: explicit, inferred, unknown.",
    )


class DirectorDecision(BaseModel):
    """Film-level director decision. Tool/NLE/provider-agnostic.

    Expresses WHAT the director wants, not HOW to implement it.
    Downstream (Arsenal → Film Capability → Execution) derives implementation.
    """
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(description="Unique decision identifier.")

    creative_intent: str = Field(
        description="Natural language summary of the creative goal. Film semantics only.",
    )

    target: list[str] = Field(
        default_factory=list,
        description=(
            "Film semantic objects this decision targets. "
            "Examples: incoming_dialogue, incoming_picture, current_picture_cut, "
            "outgoing_audio, reaction_shot, shot_order, transition_point. "
            "NOT tool IDs, NOT API parameters."
        ),
    )

    desired_relation_or_change: list[str] = Field(
        default_factory=list,
        description=(
            "Semantic relations or changes desired. "
            "Controlled film vocabulary: audio_precedes_picture, "
            "outgoing_audio_continues_after_cut, extend_visible_duration, "
            "shorten_visible_duration, reorder_story_beat, "
            "preserve_picture_cut, avoid_transition, etc. "
            "Never tool/capability IDs like J_CUT or HOLD."
        ),
    )

    must_preserve: list[str] = Field(
        default_factory=list,
        description="What must NOT change. Film semantics (picture_cut_position, shot_identity, source_media).",
    )

    must_avoid: list[str] = Field(
        default_factory=list,
        description="What must be avoided. Film semantics (transition, audio_lead, shot_substitution).",
    )

    parameterization: Parameterization = Field(
        default_factory=Parameterization,
        description="Parameter info. Vague language = magnitude only, exact_value=null.",
    )

    required_context: list[str] = Field(
        default_factory=list,
        description=(
            "What context is needed to finalize parameters. "
            "Examples: dialogue_onset_timing, current_cut_point, audio_waveform, "
            "shot_duration, available_media_range. Empty if status=READY."
        ),
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Confidence in semantic understanding ONLY. Not quality, not executability.",
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Excerpts from user input that support each part of the decision.",
    )

    source_evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Exact source evidence IDs from the caller-provided allowlist. "
            "These identify project/asset/observation evidence; never invent IDs."
        ),
    )

    status: DecisionStatus = Field(
        description="Readiness status. READY only means semantic info is sufficient, not that execution is guaranteed.",
    )

    user_terminology: list[str] = Field(
        default_factory=list,
        description="Original user terms preserved verbatim (e.g. if user says 'J-cut'). Traceability only.",
    )
