"""Creative selection / abstention for J_CUT audio offset.

Technical feasibility ≠ creative preference.
Abstention (NEEDS_DECISION) is correct behavior when multiple candidates exist
without a clear winner.
"""
from __future__ import annotations

from director_brain.models.parameterization import (
    CandidateSource,
    ParameterCandidate,
    ParameterizationDecision,
    ParameterizationStatus,
)


def select_or_abstain(
    candidates: list[ParameterCandidate],
    feasible_max: float,
    user_exact_value: float | None = None,
    user_value_valid: bool = True,
) -> ParameterizationDecision:
    """Select a candidate or abstain.

    Rules:
    - feasible_max = 0 → UNSATISFIABLE (no audio lead possible)
    - User explicit value, valid → READY (confidence 1.0)
    - User explicit value, invalid → CONFLICT (never clamp)
    - 1 strong candidate (OBSERVED_EVENT) → READY
    - Multiple strong candidates → NEEDS_DECISION
    - Only weak candidates (HANDLE_FRACTION, CONVENTION) → NEEDS_DECISION
    - 0 candidates, range > 0 → NEEDS_CONTEXT
    """
    base = {
        "parameter_name": "audio_offset_frames",
        "unit": "frames",
        "candidates": candidates,
    }

    # User explicit value (check BEFORE feasible_max=0 — explicit request for
    # impossible value is CONFLICT, not UNSATISFIABLE)
    if user_exact_value is not None:
        if user_value_valid:
            return ParameterizationDecision(
                **base,
                status=ParameterizationStatus.READY,
                exact_value=user_exact_value,
                selected_candidate=ParameterCandidate(
                    value=user_exact_value,
                    unit="frames",
                    source=CandidateSource.EXPLICIT_USER,
                    rationale="User explicitly specified this value",
                    confidence=1.0,
                ),
                confidence=1.0,
                rationale=f"User explicit value {user_exact_value}f is within feasible range.",
                reason_code="explicit_user_value_valid",
                evidence_refs=["user_input"],
            )
        else:
            return ParameterizationDecision(
                **base,
                status=ParameterizationStatus.CONFLICT,
                exact_value=None,
                confidence=0.0,
                rationale=f"User requested {user_exact_value}f but feasible max is {feasible_max}f. Not clamped.",
                reason_code="explicit_value_exceeds_feasible_range",
                conflict_detail=f"requested={user_exact_value}, feasible_max={feasible_max}",
            )

    # No feasible lead at all (and no explicit user value)
    if feasible_max <= 0:
        return ParameterizationDecision(
            **base,
            status=ParameterizationStatus.UNSATISFIABLE,
            exact_value=None,
            confidence=0.0,
            rationale="No feasible audio lead available (max=0). Check source range / audio handle.",
            reason_code="no_feasible_handle",
            missing_context=["available_audio_handle", "source_range"],
        )

    # No candidates but feasible range exists
    if not candidates:
        return ParameterizationDecision(
            **base,
            status=ParameterizationStatus.NEEDS_CONTEXT,
            exact_value=None,
            confidence=0.0,
            rationale="Feasible range exists but no candidates generated. Need audio signal or explicit value.",
            reason_code="no_candidates_need_context",
            missing_context=["dialogue_onset", "audio_signal", "explicit_parameter"],
        )

    # Classify candidates by strength
    strong = [c for c in candidates if c.source == CandidateSource.OBSERVED_EVENT]
    weak = [c for c in candidates if c.source in (
        CandidateSource.HANDLE_FRACTION, CandidateSource.CONVENTION
    )]

    # 1 strong candidate → READY
    if len(strong) == 1:
        chosen = strong[0]
        return ParameterizationDecision(
            **base,
            status=ParameterizationStatus.READY,
            exact_value=chosen.value,
            selected_candidate=chosen,
            confidence=max(chosen.confidence, 0.5),
            rationale=f"Selected {chosen.value}f from {chosen.source.value}: {chosen.rationale}",
            reason_code="single_strong_candidate",
            evidence_refs=chosen.evidence_refs,
        )

    # Multiple strong candidates → NEEDS_DECISION
    if len(strong) > 1:
        return ParameterizationDecision(
            **base,
            status=ParameterizationStatus.NEEDS_DECISION,
            exact_value=None,
            confidence=0.0,
            rationale=f"{len(strong)} strong candidates from observed events. Need creative selection.",
            reason_code="multiple_strong_candidates",
            missing_context=["creative_selection"],
        )

    # Only weak candidates → NEEDS_DECISION
    if weak and not strong:
        return ParameterizationDecision(
            **base,
            status=ParameterizationStatus.NEEDS_DECISION,
            exact_value=None,
            confidence=0.0,
            rationale=f"Only weak candidates (handle fractions / conventions). No real audio signal. Need creative decision or context.",
            reason_code="only_weak_candidates",
            missing_context=["dialogue_onset", "audio_signal"],
        )

    # Fallback (shouldn't reach here)
    return ParameterizationDecision(
        **base,
        status=ParameterizationStatus.NEEDS_CONTEXT,
        exact_value=None,
        confidence=0.0,
        rationale="Unable to determine parameter from available context.",
        reason_code="undetermined",
        missing_context=["context"],
    )
