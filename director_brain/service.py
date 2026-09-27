"""Thin application entrypoint for Semantic Director Reasoning.

This is the formal 02 runtime wiring:
  user direction → SemanticDirectorService → SemanticDirectorReasoner → LLMAdapter → DirectorDecision
  → (optional) ParameterizationContext → JCutParameterizer → ParameterizationDecision

Three independent status concepts:
  - semantic_status: Is the user's intent understood clearly? (DirectorDecision.status)
  - parameterization_status: Are execution parameters determined? (ParameterizationDecision.status)
  - execution_readiness: Can this be handed to 03? (semantic READY + parameterization READY)

It does NOT build a workflow engine or agent framework.
It only composes the existing components and exposes a clean entry path.
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
    - status: EXECUTION readiness (can this be handed to 03?)

    Additional explicit fields prevent downstream from guessing:
    - semantic_status: DirectorDecision.status (semantic understanding only)
    - execution_readiness: canonical combined readiness
    """
    semantic_decision: DirectorDecision | None
    parameterization_decision: ParameterizationDecision | None = None
    status: str = ""  # execution readiness (backward-compatible alias)
    semantic_status: str = ""  # DirectorDecision.status value
    execution_readiness: str = ""  # canonical: READY / NOT_READY / BLOCKED / WAITING_FOR_*
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
           → post-validation (semantic only) → (optional) parameterizer → ParameterizationDecision
           → execution readiness composition

    Usage:
        service = SemanticDirectorService()
        result = service.process_direction("让下一句声音提前一点进入...")
        if result.execution_readiness == "READY":
            arsenal.execute(result.semantic_decision, result.parameterization_decision)
        elif result.execution_readiness == "WAITING_FOR_DECISION":
            present_candidates_to_user(result.parameterization_decision)
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
        2. Post-validation: semantic consistency only (never checks parameter completeness)
        3. If parameterization_context provided: run deterministic parameterizer
        4. Compute execution readiness from semantic + parameterization

        Fail-closed: any failure returns SEMANTIC_REASONER_UNAVAILABLE.
        No silent heuristic fallback.
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
                semantic_status="UNAVAILABLE",
                execution_readiness="NOT_READY",
                error=result.error,
                trace_id=trace_id,
                model=result.model,
                latency_ms=result.latency_ms,
            )

        # Step 2: Post-validation (semantic consistency only)
        decision, violations = validate_director_decision(result.decision)
        semantic_status = decision.status.value if decision.status else "UNDERSPECIFIED"

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

        # Step 4: Execution readiness
        execution_readiness = self._compute_execution_readiness(decision, param_decision)

        return DirectorRequestResult(
            semantic_decision=decision,
            parameterization_decision=param_decision,
            status=execution_readiness,
            semantic_status=semantic_status,
            execution_readiness=execution_readiness,
            error=None,
            trace_id=trace_id,
            model=result.model,
            latency_ms=result.latency_ms,
            post_validation_violations=violations,
        )

    @staticmethod
    def _compute_execution_readiness(
        decision: DirectorDecision,
        param_decision: ParameterizationDecision | None,
    ) -> str:
        """Compute canonical execution readiness from semantic + parameterization.

        Returns the overall status string (backward-compatible):
        - Semantic READY + Parameterization READY → READY
        - Semantic READY + Parameterization NEEDS_CONTEXT → WAITING_FOR_CONTEXT
        - Semantic READY + Parameterization NEEDS_DECISION → WAITING_FOR_DECISION
        - Semantic READY + no param context → READY (semantic clear, params not yet evaluated)
        - Semantic not READY → propagate semantic status (NEEDS_CONTEXT, UNDERSPECIFIED, etc.)
        - Semantic CONFLICT / Parameterization CONFLICT/UNSATISFIABLE → BLOCKED
        """
        semantic_status = decision.status.value if decision.status else "UNDERSPECIFIED"

        # Semantic conflict → BLOCKED
        if semantic_status in ("CONFLICTING_CONSTRAINTS", "UNSUPPORTED"):
            return "BLOCKED"

        # Semantic not ready → propagate semantic status (caller needs to resolve intent first)
        if semantic_status != "READY":
            return semantic_status

        # Semantic READY. Check parameterization.
        if param_decision is None:
            # No parameterization context provided yet — semantic is ready,
            # execution depends on obtaining parameters
            return "READY"

        param_status = param_decision.status.value
        if param_status == "READY":
            return "READY"
        if param_status == "NEEDS_CONTEXT":
            return "WAITING_FOR_CONTEXT"
        if param_status == "NEEDS_DECISION":
            return "WAITING_FOR_DECISION"
        if param_status in ("CONFLICT", "UNSATISFIABLE"):
            return "BLOCKED"
        # UNAVAILABLE or other
        return "NOT_READY"
