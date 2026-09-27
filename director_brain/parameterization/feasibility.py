"""Deterministic feasibility bounds for J_CUT audio offset.

Feasibility is computed BEFORE any creative selection.
A value outside feasible range is rejected regardless of aesthetics.
"""
from __future__ import annotations

from director_brain.models.parameterization import (
    FeasibleRange,
    ParameterizationContext,
)


def compute_jcut_feasible_range(ctx: ParameterizationContext) -> FeasibleRange:
    """Compute hard bounds for audio lead (frames before picture cut).

    Constraints:
    1. Source range: incoming clip source_start_frame determines available handle
    2. Available audio handle: if explicitly provided, max <= handle
    3. Non-negative: audio cannot start before timeline frame 0
    4. Picture cut is fixed: we only move audio, not video
    """
    constraints: list[str] = []
    max_lead = float("inf")
    min_lead = 0.0

    # Constraint 1: source range bound
    if ctx.incoming_clip is not None:
        src_start = ctx.incoming_clip.source_start_frame
        if src_start > 0:
            max_lead = min(max_lead, float(src_start))
            constraints.append(f"source_start_frame={src_start}")
        else:
            # Full source range: no audio available before cut
            max_lead = 0.0
            constraints.append("full_source_range_no_handle")

    # Constraint 2: explicit available handle
    if ctx.available_audio_handle_before is not None:
        max_lead = min(max_lead, float(ctx.available_audio_handle_before))
        constraints.append(f"available_handle={ctx.available_audio_handle_before}")

    # Constraint 3: timeline boundary
    if ctx.incoming_clip is not None and ctx.incoming_clip.start_frame > 0:
        max_lead = min(max_lead, float(ctx.incoming_clip.start_frame))
        constraints.append("timeline_start_boundary")

    if max_lead == float("inf"):
        max_lead = 0.0
        constraints.append("no_handle_information_default_zero")

    rationale = (
        f"Audio lead bounded by: {', '.join(constraints)}. "
        f"Picture cut position is fixed (J_CUT only moves incoming audio earlier)."
    )

    return FeasibleRange(
        parameter_name="audio_offset_frames",
        min_value=min_lead,
        max_value=max_lead,
        unit="frames",
        rationale=rationale,
        constraints_checked=constraints,
    )


def validate_explicit_value(
    value: float, feasible: FeasibleRange
) -> tuple[bool, str]:
    """Check if an explicit user value is within feasible range.

    Returns (is_valid, reason). Never clamps.
    """
    if value < feasible.min_value:
        return False, f"value {value} < min {feasible.min_value}"
    if value > feasible.max_value:
        return False, f"value {value} > max {feasible.max_value}"
    return True, "within_feasible_range"
