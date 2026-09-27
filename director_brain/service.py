"""Thin application entrypoint for Semantic Director Reasoning.

This is the formal 02 runtime wiring:
  user direction → SemanticDirectorService → SemanticDirectorReasoner → LLMAdapter → DirectorDecision
  → (optional) ParameterizationContext → JCutParameterizer → ParameterizationDecision

It does NOT build a workflow engine or agent framework.
It only composes the existing components and exposes a clean entry path.

NEEDS_CONTEXT and SEMANTIC_REASONER_UNAVAILABLE propagate to the caller.
No silent heuristic fallback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from director_brain.llm_adapter import LLMAdapter, LLMResult
from director_brain.models.director_decision import DirectorDecision
from director_brain.models.parameterization import (
    ParameterizationContext,
    ParameterizationDecision,
)
from director_brain.parameterization.parameterizer import JCutParameterizer
from director_brain.parameterization.post_validation import validate_director_decision
from director_brain.semantic_reasoner import SemanticDirectorReasoner, SemanticReasonerResult


@dataclass
class DirectorRequestResult:
    """Result returned by the application entrypoint.

    Separates three concerns:
    - semantic_decision: WHAT the user wants (DirectorDecision)
    - parameterization_decision: HOW to execute / why not (ParameterizationDecision)
    - overall_status: can this be handed to 03?
    """
    semantic_decision: DirectorDecision | None
    parameterization_decision: ParameterizationDecision | None = None
    status: str = ""  # overall execution readiness
    error: str | None = None
    trace_id: str = ""
    model: str = ""
    latency_ms: int = 0
    post_validation_violations: list[str] = field(default_factory=list)
    # Backward compat: decision alias for semantic_decision
    @property
    def decision(self) -> DirectorDecision | None:
        return self.semantic_decision


class SemanticDirectorService:
    """Formal 02 application entrypoint for semantic director reasoning.

    Wires: user input → SemanticDirectorReasoner → LLMAdapter → DirectorDecision
           → post-validation → (optional) parameterizer → ParameterizationDecision

    Usage:
        service = SemanticDirectorService()
        result = service.process_direction("让下一句声音提前一点进入...")
        if result.status == "READY":
            arsenal.execute(result.semantic_decision, result.parameterization_decision)
        elif result.status == "NEEDS_CONTEXT":
            gather_context(...)
        else:
            report_to_user(result.status, result.error)
    """

    def __init__(self, reasoner: SemanticDirectorReasoner | None = None):
        self.reasoner = reasoner or SemanticDirectorReasoner()
        self._parameterizer = JCutParameterizer()

    def process_direction(
        self,
        user_direction: str,
        context: str | None = None,
        decision_id: str | None = None,
        parameterization_context: ParameterizationContext | None = None,
    ) -> DirectorRequestResult:
        """Process a natural-language director direction.

        Steps:
        1. Semantic reasoning (LLM) → DirectorDecision
        2. Post-validation: fix READY+exact_value=None bug (downgrade, never fill)
        3. If parameterization_context provided: run deterministic parameterizer
        4. Compute overall status from both decisions

        Fail-closed: any failure returns SEMANTIC_REASONER_UNAVAILABLE.
        No silent heuristic fallback.

        Args:
            user_direction: Natural language director request.
            context: Optional available context string for LLM.
            decision_id: Optional explicit decision ID.
            parameterization_context: Optional provider-neutral context for parameterization.
        """
        trace_id = f"svc_{int(time.time() * 1000)}"

        result: SemanticReasonerResult = self.reasoner.reason(
            user_direction=user_direction,
            context=context,
            decision_id=decision_id,
        )

        if result.decision is None:
            return DirectorRequestResult(
                semantic_decision=None,
                parameterization_decision=None,
                status="SEMANTIC_REASONER_UNAVAILABLE",
                error=result.error,
                trace_id=trace_id,
                model=result.model,
                latency_ms=result.latency_ms,
            )

        # Step 2: Post-validation (fix READY+exact_value=None bug)
        decision, violations = validate_director_decision(result.decision)

        # Step 3: Parameterization (if context available)
        param_decision: ParameterizationDecision | None = None
        if parameterization_context is not None:
            user_exact = None
            if decision.parameterization and decision.parameterization.exact_value is not None:
                user_exact = float(decision.parameterization.exact_value)
            param_decision = self._parameterizer.parameterize(
                ctx=parameterization_context,
                user_exact_value=user_exact,
            )

        # Step 4: Overall status
        overall_status = self._compute_overall_status(decision, param_decision)

        return DirectorRequestResult(
            semantic_decision=decision,
            parameterization_decision=param_decision,
            status=overall_status,
            error=None,
            trace_id=trace_id,
            model=result.model,
            latency_ms=result.latency_ms,
            post_validation_violations=violations,
        )

    @staticmethod
    def _compute_overall_status(
        decision: DirectorDecision,
        param_decision: ParameterizationDecision | None,
    ) -> str:
        """Compute overall execution readiness from semantic + parameterization decisions.

        Only READY if BOTH semantic and parameterization are READY.
        Otherwise propagate the most specific non-READY status.
        """
        semantic_status = decision.status.value if decision.status else "UNDERSPECIFIED"

        # If semantic isn't READY, that takes priority
        if semantic_status != "READY":
            return semantic_status

        # Semantic is READY. Check parameterization.
        if param_decision is None:
            # No parameterization context — semantic READY is the best we have
            return "READY"

        param_status = param_decision.status.value
        if param_status == "READY":
            return "READY"
        # Parameterization not ready — propagate its status
        return param_status
