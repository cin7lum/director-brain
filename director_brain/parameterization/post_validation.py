"""Post-validation invariants for DirectorDecision status.

Fixes the bug where LLM returns READY with exact_value=None.
This validator only REJECTS or DOWNGRADES status — it never creates values.
"""
from __future__ import annotations

from typing import Any

from director_brain.models.director_decision import DecisionStatus, DirectorDecision


# Capabilities that require an exact execution parameter
_PARAMETER_REQUIRED_CAPABILITIES = {
    "audio_precedes_picture",
    "J_CUT",
    "j_cut",
}


def validate_director_decision(decision: DirectorDecision) -> tuple[DirectorDecision, list[str]]:
    """Validate DirectorDecision status vs parameterization consistency.

    Returns (possibly_corrected_decision, list_of_violations).
    Only downgrades status — never fills in missing values.
    """
    violations: list[str] = []

    # Check: READY + exact_value=None for parameter-required capability
    if decision.status == "READY":
        param = decision.parameterization
        needs_param = any(
            cap in _PARAMETER_REQUIRED_CAPABILITIES
            for cap in (decision.desired_relation_or_change or [])
        )
        if needs_param and param is not None and param.exact_value is None:
            violations.append(
                "READY status with exact_value=None for parameter-required capability. "
                "Downgrading to NEEDS_CONTEXT."
            )
            # Downgrade status (immutable model — create copy)
            decision = decision.model_copy(update={"status": DecisionStatus.NEEDS_CONTEXT})

    # Check: CONFLICTING_CONSTRAINTS should not have exact_value
    if decision.status == "CONFLICTING_CONSTRAINTS" and decision.parameterization and decision.parameterization.exact_value is not None:
        violations.append("CONFLICTING_CONSTRAINTS with exact_value set — value should be ignored.")

    return decision, violations
