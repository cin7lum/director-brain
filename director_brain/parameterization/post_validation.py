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

from director_brain.models.director_decision import DecisionStatus, DirectorDecision


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
            desired_set = set(decision.desired_relation_or_change)
            avoid_set = set(decision.must_avoid)
            contradiction = desired_set & avoid_set
            if contradiction:
                violations.append(
                    f"Semantic contradiction: desired and must_avoid overlap: {contradiction}. "
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
