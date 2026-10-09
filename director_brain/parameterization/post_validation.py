"""Post-validation invariants for DirectorDecision semantic status.

DirectorDecision.status describes SEMANTIC READINESS only:
  - Is the user's intent understood clearly enough to act on?
  - Are there semantic conflicts?

It does NOT describe parameterization readiness.
exact_value=None is NOT a semantic incompleteness — it means the parameterizer
should determine the exact execution value.

This validator only REJECTS or DOWNGRADES status based on SEMANTIC issues.
It never creates values, never checks parameter completeness.
"""
from __future__ import annotations

import re

from director_brain.models.director_decision import DecisionStatus, DirectorDecision


# These equivalence classes are limited to the controlled film-language terms
# already defined in llm_adapter.SYSTEM_PROMPT. Unknown free text is never
# guessed or rewritten here.
_KNOWN_CONSTRAINT_EQUIVALENTS = {
    "audio_precedes_picture": "audio_precedes_picture",
    "audio_lead": "audio_precedes_picture",
    "outgoing_audio_continues_after_cut": "outgoing_audio_continues_after_cut",
    "audio_tail": "outgoing_audio_continues_after_cut",
    "extend_visible_duration": "extend_visible_duration",
    "duration_extension": "extend_visible_duration",
    "shorten_visible_duration": "shorten_visible_duration",
    "duration_shortening": "shorten_visible_duration",
    "allow_transition": "transition",
    "transition": "transition",
}


def _constraint_key(value: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
    return _KNOWN_CONSTRAINT_EQUIVALENTS.get(normalized, normalized)


def _known_constraint_conflicts(
    desired: list[str], must_avoid: list[str],
) -> list[tuple[str, str]]:
    """Return stable source terms whose known meanings directly conflict."""
    desired_by_key: dict[str, list[str]] = {}
    avoided_by_key: dict[str, list[str]] = {}
    for value in desired:
        desired_by_key.setdefault(_constraint_key(value), []).append(value)
    for value in must_avoid:
        avoided_by_key.setdefault(_constraint_key(value), []).append(value)

    return sorted(
        (positive, negative)
        for key in desired_by_key.keys() & avoided_by_key.keys()
        for positive in desired_by_key[key]
        for negative in avoided_by_key[key]
    )


def validate_director_decision(decision: DirectorDecision) -> tuple[DirectorDecision, list[str]]:
    """Validate DirectorDecision semantic consistency.

    Returns (possibly_corrected_decision, list_of_violations).
    Only downgrades status for semantic reasons — never for missing parameters.

    Checks:
    1. READY but no desired_relation_or_change → NEEDS_CONTEXT (intent unclear)
    2. READY but semantic conflict (desired relation contradicts must_avoid) → CONFLICTING_CONSTRAINTS
    3. CONFLICTING_CONSTRAINTS with exact_value → warning (value should be ignored)
    """
    violations: list[str] = []

    if decision.status == "READY":
        # Check 1: READY requires at least one desired relation/change
        if not decision.desired_relation_or_change:
            violations.append(
                "READY status with no desired_relation_or_change. "
                "Semantic intent unclear. Downgrading to NEEDS_CONTEXT."
            )
            decision = decision.model_copy(update={"status": DecisionStatus.NEEDS_CONTEXT})

        # Check 2: READY should not have semantic contradiction between
        # desired_relation_or_change and must_avoid
        elif decision.must_avoid:
            contradiction = _known_constraint_conflicts(
                decision.desired_relation_or_change,
                decision.must_avoid,
            )
            if contradiction:
                violations.append(
                    "Semantic contradiction: desired relation conflicts with "
                    f"must_avoid terms: {contradiction}. "
                    "Downgrading to CONFLICTING_CONSTRAINTS."
                )
                decision = decision.model_copy(
                    update={"status": DecisionStatus.CONFLICTING_CONSTRAINTS}
                )

    # Check 3: CONFLICTING_CONSTRAINTS should not have exact_value
    if decision.status == "CONFLICTING_CONSTRAINTS":
        if decision.parameterization and decision.parameterization.exact_value is not None:
            violations.append(
                "CONFLICTING_CONSTRAINTS with exact_value set — value should be ignored."
            )

    return decision, violations
