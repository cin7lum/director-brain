"""Thin application entrypoint for Semantic Director Reasoning.

This is the formal 02 runtime wiring:
  user direction → SemanticDirectorService → SemanticDirectorReasoner → LLMAdapter → DirectorDecision

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
from director_brain.semantic_reasoner import SemanticDirectorReasoner, SemanticReasonerResult


@dataclass
class DirectorRequestResult:
    """Result returned by the application entrypoint."""
    decision: DirectorDecision | None
    status: str  # READY / NEEDS_CONTEXT / UNDERSPECIFIED / UNSUPPORTED /
                 # CONFLICTING_CONSTRAINTS / SEMANTIC_REASONER_UNAVAILABLE
    error: str | None = None
    trace_id: str = ""
    model: str = ""
    latency_ms: int = 0
    # Caller should inspect these to decide next steps:
    # - status=READY → decision is semantically complete, proceed to Arsenal
    # - status=NEEDS_CONTEXT → decision has required_context, gather context then retry
    # - status=UNDERSPECIFIED → request too vague, ask user to clarify
    # - status=UNSUPPORTED → not an editing decision, don't proceed
    # - status=CONFLICTING_CONSTRAINTS → ask user to resolve conflict
    # - status=SEMANTIC_REASONER_UNAVAILABLE → model failed, do NOT proceed


class SemanticDirectorService:
    """Formal 02 application entrypoint for semantic director reasoning.

    Wires: user input → SemanticDirectorReasoner → LLMAdapter → DirectorDecision.

    Usage:
        service = SemanticDirectorService()
        result = service.process_direction("让下一句声音提前一点进入...")
        if result.status == "READY":
            arsenal.execute(result.decision)
        elif result.status == "NEEDS_CONTEXT":
            gather_context(result.decision.required_context)
        else:
            # fail closed — no silent fallback
            report_to_user(result.status, result.error)
    """

    def __init__(self, reasoner: SemanticDirectorReasoner | None = None):
        self.reasoner = reasoner or SemanticDirectorReasoner()

    def process_direction(
        self,
        user_direction: str,
        context: str | None = None,
        decision_id: str | None = None,
    ) -> DirectorRequestResult:
        """Process a natural-language director direction.

        This is the formal entrypoint. It calls SemanticDirectorReasoner,
        which calls LLMAdapter, which uses canonical DirectorDecision schema.

        Fail-closed: any failure returns SEMANTIC_REASONER_UNAVAILABLE.
        No silent heuristic fallback.

        Args:
            user_direction: Natural language director request.
            context: Optional available context.
            decision_id: Optional explicit decision ID.

        Returns:
            DirectorRequestResult with decision and status.
        """
        trace_id = f"svc_{int(time.time() * 1000)}"

        result: SemanticReasonerResult = self.reasoner.reason(
            user_direction=user_direction,
            context=context,
            decision_id=decision_id,
        )

        if result.decision is None:
            return DirectorRequestResult(
                decision=None,
                status="SEMANTIC_REASONER_UNAVAILABLE",
                error=result.error,
                trace_id=trace_id,
                model=result.model,
                latency_ms=result.latency_ms,
            )

        return DirectorRequestResult(
            decision=result.decision,
            status=result.decision.status.value,
            trace_id=trace_id,
            model=result.model,
            latency_ms=result.latency_ms,
        )
