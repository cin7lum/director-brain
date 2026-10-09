"""SemanticDirectorReasoner — natural language → DirectorDecision.

Responsibility: user direction + available context → DirectorDecision.
**链 B 唯一的"一句话→决策"路径**（架构体检候选②合并）：生产消费侧
（semantic_shadow 影子对账）与组合入口（service，03 交接参数化）都必须
经此 reasoner；ollama/ark 等传输以可注入 adapter 换入。

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

import json
import time
from dataclasses import dataclass

from director_brain.llm_adapter import LLMAdapter, LLMResult
from director_brain.models.director_decision import DirectorDecision
from director_brain.parameterization.provenance import has_explicit_numeric_unit_quote


# Stable, non-provider-derived categories for failures caused by invalid input
# or untrusted model claims. The composition service uses this set to preserve
# its INVALID vs UNAVAILABLE status contract without parsing free text.
INVALID_REASONER_FAILURE_CODES = frozenset({
    "source_evidence_allowlist_not_list",
    "malformed_source_evidence_allowlist",
    "duplicate_source_evidence_refs",
    "source_reference_not_allowlisted",
    "evidence_excerpt_unbound",
    "exact_parameterization_unverified",
})


@dataclass
class SemanticReasonerResult:
    """Result of semantic reasoning."""
    decision: DirectorDecision | None
    error: str | None = None
    model: str = ""
    prompt_version: str = "1.2"
    schema_version: str = ""
    latency_ms: int = 0
    trace_id: str = ""
    exact_parameterization_source_verified: bool = False
    failure_code: str | None = None


class SemanticDirectorReasoner:
    """Converts natural-language director requests into structured DirectorDecision.

    Uses mature LLM + Structured Output. Thin film-specific schema.
    HeuristicDirectorReasoner remains as SHOT_SELECTION_SPECIALIST.
    """

    def __init__(self, llm_adapter=None):
        # llm_adapter: 任何暴露 generate_decision(user_input, context=...,
        # decision_id=...) -> LLMResult 的传输（ollama/ark 可换；候选②）。
        self.llm = llm_adapter or LLMAdapter()

    def reason(
        self,
        user_direction: str,
        context: str | None = None,
        decision_id: str | None = None,
        available_source_evidence_refs: list[str] | None = None,
    ) -> SemanticReasonerResult:
        """Produce a semantic decision with source references restricted to a caller allowlist.

        Context remains descriptive input. Source evidence IDs are separately
        allowlisted and any model citation outside that exact set fails closed.
        """
        if not isinstance(user_direction, str) or not user_direction.strip():
            return SemanticReasonerResult(
                decision=None,
                error="SEMANTIC_REASONER_UNAVAILABLE: empty user direction",
                failure_code="empty_user_direction",
            )

        trace_id = f"sem_{int(time.time() * 1000)}"
        raw_refs = (
            [] if available_source_evidence_refs is None
            else available_source_evidence_refs
        )
        if not isinstance(raw_refs, list):
            return SemanticReasonerResult(
                decision=None,
                error="SEMANTIC_REASONER_INVALID: source evidence allowlist must be a list",
                trace_id=trace_id,
                failure_code="source_evidence_allowlist_not_list",
            )
        allowed_refs: list[str] = []
        for ref in raw_refs:
            if (
                not isinstance(ref, str)
                or not ref
                or ref != ref.strip()
                or any(ord(char) < 32 for char in ref)
            ):
                return SemanticReasonerResult(
                    decision=None,
                    error="SEMANTIC_REASONER_INVALID: malformed source evidence allowlist",
                    trace_id=trace_id,
                    failure_code="malformed_source_evidence_allowlist",
                )
            allowed_refs.append(ref)
        if len(allowed_refs) != len(set(allowed_refs)):
            return SemanticReasonerResult(
                decision=None,
                error="SEMANTIC_REASONER_INVALID: duplicate source evidence IDs",
                trace_id=trace_id,
                failure_code="duplicate_source_evidence_refs",
            )

        reasoning_context = context
        if allowed_refs:
            allowlist_text = (
                "Authorized source evidence reference IDs (JSON; cite only exact IDs "
                "from this list): "
                + json.dumps(allowed_refs, ensure_ascii=False)
            )
            reasoning_context = (
                f"{context}\n\n{allowlist_text}" if context else allowlist_text
            )

        try:
            result: LLMResult = self.llm.generate_decision(
                user_input=user_direction,
                context=reasoning_context,
                decision_id=decision_id or trace_id,
            )
        except Exception:  # noqa: BLE001
            return SemanticReasonerResult(
                decision=None,
                error="SEMANTIC_REASONER_UNAVAILABLE: adapter exception",
                trace_id=trace_id,
                failure_code="adapter_exception",
            )

        if not isinstance(result, LLMResult):
            return SemanticReasonerResult(
                decision=None,
                error="SEMANTIC_REASONER_UNAVAILABLE: adapter returned an invalid result",
                trace_id=trace_id,
                failure_code="adapter_invalid_result",
            )
        if result.decision is None:
            return SemanticReasonerResult(
                decision=None,
                # Provider strings may include response fragments, request
                # data, or implementation details. Expose only a fixed code.
                error="SEMANTIC_REASONER_UNAVAILABLE: provider failure",
                model=result.model,
                prompt_version=result.prompt_version,
                latency_ms=result.latency_ms,
                trace_id=trace_id,
                failure_code="provider_failure",
            )

        if not isinstance(result.decision, DirectorDecision):
            return SemanticReasonerResult(
                decision=None,
                error="SEMANTIC_REASONER_UNAVAILABLE: adapter returned an invalid decision",
                model=result.model,
                prompt_version=result.prompt_version,
                latency_ms=result.latency_ms,
                trace_id=trace_id,
                failure_code="adapter_invalid_decision",
            )

        cited_refs = result.decision.source_evidence_refs
        if len(cited_refs) != len(set(cited_refs)):
            return SemanticReasonerResult(
                decision=None,
                error="SEMANTIC_REASONER_INVALID: duplicate source evidence IDs",
                model=result.model,
                prompt_version=result.prompt_version,
                schema_version=result.schema_version,
                latency_ms=result.latency_ms,
                trace_id=trace_id,
                failure_code="duplicate_source_evidence_refs",
            )
        if not set(cited_refs).issubset(set(allowed_refs)):
            return SemanticReasonerResult(
                decision=None,
                error="SEMANTIC_REASONER_INVALID: model cited unavailable source evidence",
                model=result.model,
                prompt_version=result.prompt_version,
                schema_version=result.schema_version,
                latency_ms=result.latency_ms,
                trace_id=trace_id,
                failure_code="source_reference_not_allowlisted",
            )

        # ``evidence`` is contractually an excerpt from the director's request,
        # not a model-generated paraphrase. Keep it separate from source refs:
        # project observations must use the exact caller allowlist above.
        for excerpt in result.decision.evidence:
            if not excerpt.strip() or excerpt not in user_direction:
                return SemanticReasonerResult(
                    decision=None,
                    error=(
                        "SEMANTIC_REASONER_INVALID: evidence excerpt is not an "
                        "exact substring of the director request"
                    ),
                    model=result.model,
                    prompt_version=result.prompt_version,
                    schema_version=result.schema_version,
                    latency_ms=result.latency_ms,
                    trace_id=trace_id,
                    failure_code="evidence_excerpt_unbound",
                )

        exact_parameterization_source_verified = False
        parameterization = result.decision.parameterization
        if parameterization is not None and parameterization.exact_value is not None:
            if (
                parameterization.certainty != "explicit"
                or not has_explicit_numeric_unit_quote(
                    parameterization.exact_value,
                    parameterization.unit,
                    result.decision.evidence,
                )
            ):
                return SemanticReasonerResult(
                    decision=None,
                    error=(
                        "SEMANTIC_REASONER_INVALID: exact parameter value lacks "
                        "an explicit numeric-and-unit request excerpt"
                    ),
                    model=result.model,
                    prompt_version=result.prompt_version,
                    schema_version=result.schema_version,
                    latency_ms=result.latency_ms,
                    trace_id=trace_id,
                    failure_code="exact_parameterization_unverified",
                )
            exact_parameterization_source_verified = True

        return SemanticReasonerResult(
            decision=result.decision,
            model=result.model,
            prompt_version=result.prompt_version,
            schema_version=result.schema_version,
            latency_ms=result.latency_ms,
            trace_id=trace_id,
            exact_parameterization_source_verified=(
                exact_parameterization_source_verified
            ),
        )
