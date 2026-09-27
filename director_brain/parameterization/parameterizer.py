"""J_CUT parameterizer: context → feasibility → candidates → selection.

Orchestrates the deterministic pipeline:
1. Compute feasible range from context
2. Validate user explicit value (if any)
3. Generate candidates from real signals
4. Select or abstain
"""
from __future__ import annotations

from director_brain.models.parameterization import (
    ParameterizationContext,
    ParameterizationDecision,
)
from director_brain.parameterization.candidates import generate_jcut_candidates
from director_brain.parameterization.feasibility import (
    compute_jcut_feasible_range,
    validate_explicit_value,
)
from director_brain.parameterization.selector import select_or_abstain


class JCutParameterizer:
    """Deterministic parameterizer for J_CUT audio offset."""

    def parameterize(
        self,
        ctx: ParameterizationContext,
        user_exact_value: float | None = None,
    ) -> ParameterizationDecision:
        """Run the full parameterization pipeline."""
        # Step 1: Feasibility (always first)
        feasible = compute_jcut_feasible_range(ctx)

        # Step 2: Validate user explicit value
        user_value_valid = True
        if user_exact_value is not None:
            user_value_valid, _ = validate_explicit_value(user_exact_value, feasible)

        # Step 3: Generate candidates (only if feasible range exists)
        candidates: list = []
        if feasible.max_value > 0:
            candidates = generate_jcut_candidates(ctx, feasible)

        # Step 4: Select or abstain
        decision = select_or_abstain(
            candidates=candidates,
            feasible_max=feasible.max_value,
            user_exact_value=user_exact_value,
            user_value_valid=user_value_valid,
        )
        decision.feasible_range = feasible

        return decision
