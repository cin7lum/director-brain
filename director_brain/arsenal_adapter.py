"""Thin 02→03 adapter.

Maps Director Brain (02) outputs to Arsenal (03) inputs.
Only produces 03 execution parameters when execution is READY.
Otherwise returns None — caller must not invoke 03 execution.

This is the ONLY integration point between 02 and 03.
02 does NOT import Blackmagic Provider internals.
03 does NOT import Director Brain models.
"""
from __future__ import annotations

from typing import Any

from director_brain.models.director_decision import DirectorDecision
from director_brain.models.parameterization import ParameterizationDecision


def to_arsenal_parameterization(
    semantic: DirectorDecision,
    parameterization: ParameterizationDecision | None,
    *,
    exact_parameterization_source_verified: bool = False,
) -> dict[str, Any] | None:
    """Convert 02 decisions to 03 DirectorDecision.parameterization dict.

    Returns None if not ready for execution (caller must NOT invoke 03).
    Returns a dict suitable for 03's dataclass DirectorDecision.parameterization field.

    Gating:
    - Semantic must be READY (intent understood clearly)
    - Parameterization must be READY (exact execution parameter determined)
    - Neither condition alone is sufficient.
    - Without a ParameterizationDecision, the exact value and unit must have
      been source-verified by SemanticDirectorReasoner; no default unit is inferred.
    """
    # Semantic must be READY
    if semantic.status is None or semantic.status.value != "READY":
        return None

    # If no parameterization decision, pass through semantic parameterization
    if parameterization is None:
        if (
            semantic.parameterization is not None
            and semantic.parameterization.exact_value is not None
            and semantic.parameterization.unit is not None
            and semantic.parameterization.certainty == "explicit"
            and exact_parameterization_source_verified
        ):
            return {
                "exact_value": semantic.parameterization.exact_value,
                "unit": semantic.parameterization.unit,
                "magnitude": semantic.parameterization.magnitude,
                "certainty": semantic.parameterization.certainty,
            }
        return None

    # Parameterization must be READY
    if parameterization.status.value != "READY":
        return None

    # Must have exact_value
    if parameterization.exact_value is None:
        return None

    return {
        "exact_value": parameterization.exact_value,
        "unit": parameterization.unit or "frames",
        "parameter_name": parameterization.parameter_name,
        "confidence": parameterization.confidence,
        "evidence_refs": parameterization.evidence_refs,
        "feasible_min": parameterization.feasible_range.min_value if parameterization.feasible_range else None,
        "feasible_max": parameterization.feasible_range.max_value if parameterization.feasible_range else None,
    }


def from_service_result(
    service_result: Any,
) -> dict[str, Any] | None:
    """Convert a SemanticDirectorService result to 03 parameterization.

    Uses the canonical execution_readiness from the service — does NOT
    duplicate status logic. Only produces payload when execution_readiness == "READY";
    without parameterizer context, the service requires a source-verified exact
    numeric/unit quote from the original director request.

    Args:
        service_result: DirectorRequestResult from SemanticDirectorService.process_direction()

    Returns:
        03 parameterization dict if READY, None otherwise.
    """
    if service_result is None:
        return None
    if getattr(service_result, "execution_readiness", "") != "READY":
        return None
    if service_result.semantic_decision is None:
        return None
    return to_arsenal_parameterization(
        service_result.semantic_decision,
        service_result.parameterization_decision,
        exact_parameterization_source_verified=getattr(
            service_result, "exact_parameterization_source_verified", False
        ),
    )


def is_execution_ready(
    semantic: DirectorDecision,
    parameterization: ParameterizationDecision | None,
    *,
    exact_parameterization_source_verified: bool = False,
) -> bool:
    """Check if the combined decisions are ready for 03 execution."""
    return to_arsenal_parameterization(
        semantic,
        parameterization,
        exact_parameterization_source_verified=exact_parameterization_source_verified,
    ) is not None
