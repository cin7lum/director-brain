"""SemanticDirectorReasoner — natural language → DirectorDecision.

Responsibility: user direction + available context → DirectorDecision.

NOT responsible for:
- Shot ranking (HeuristicDirectorReasoner = SHOT_SELECTION_SPECIALIST)
- Tool selection / Film Capability resolution (Arsenal)
- DaVinci execution
- OTIO projection
- FQL scoring

Fail-closed: if LLM is unavailable or output is invalid, returns
SEMANTIC_REASONER_UNAVAILABLE. No silent fallback to keyword parser.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from director_brain.llm_adapter import LLMAdapter, LLMResult
from director_brain.models.director_decision import DirectorDecision


@dataclass
class SemanticReasonerResult:
    """Result of semantic reasoning."""
    decision: DirectorDecision | None
    error: str | None = None
    model: str = ""
    prompt_version: str = "1.0"
    latency_ms: int = 0
    trace_id: str = ""


class SemanticDirectorReasoner:
    """Converts natural-language director requests into structured DirectorDecision.

    Uses mature LLM + Structured Output. Thin film-specific schema.
    HeuristicDirectorReasoner remains as SHOT_SELECTION_SPECIALIST.
    """

    def __init__(self, llm_adapter: LLMAdapter | None = None):
        self.llm = llm_adapter or LLMAdapter()

    def reason(
        self,
        user_direction: str,
        context: str | None = None,
        decision_id: str | None = None,
    ) -> SemanticReasonerResult:
        """Produce a DirectorDecision from natural language.

        Args:
            user_direction: The director's natural language request.
            context: Optional available context (timeline state, observations).
            decision_id: Optional explicit ID.

        Returns:
            SemanticReasonerResult with decision or error.
            Fail-closed — no keyword fallback.
        """
        if not user_direction or not user_direction.strip():
            return SemanticReasonerResult(
                decision=None,
                error="SEMANTIC_REASONER_UNAVAILABLE: empty user direction",
            )

        trace_id = f"sem_{int(time.time() * 1000)}"

        result: LLMResult = self.llm.generate_decision(
            user_input=user_direction,
            context=context,
            decision_id=decision_id or trace_id,
        )

        if result.decision is None:
            return SemanticReasonerResult(
                decision=None,
                error=result.error or "SEMANTIC_REASONER_UNAVAILABLE",
                model=result.model,
                prompt_version=result.prompt_version,
                latency_ms=result.latency_ms,
                trace_id=trace_id,
            )

        return SemanticReasonerResult(
            decision=result.decision,
            model=result.model,
            prompt_version=result.prompt_version,
            latency_ms=result.latency_ms,
            trace_id=trace_id,
        )
