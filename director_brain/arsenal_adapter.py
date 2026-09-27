"""Thin 02→03 adapter.

Maps Director Brain (02) outputs to Arsenal (03) inputs.
Only produces 03 execution parameters when ParameterizationDecision.status == READY.
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
) -> dict[str, Any] | None:
    """Convert 02 decisions to 03 DirectorDecision.parameterization dict.

    Returns None if not ready for execution (caller must NOT invoke 03).
    Returns a dict suitable for 03's dataclass DirectorDecision.parameterization field.
    """
    # Semantic must be READY
    if semantic.status is None or semantic.status.value != "READY":
        return None

    # If no parameterization decision, pass through semantic parameterization
    if parameterization is None:
        if semantic.parameterization and semantic.parameterization.exact_value is not None:
            return {
                "exact_value": semantic.parameterization.exact_value,
                "unit": semantic.parameterization.unit or "frames",
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


def is_execution_ready(
    semantic: DirectorDecision,
    parameterization: ParameterizationDecision | None,
) -> bool:
    """Check if the combined decisions are ready for 03 execution."""
    return to_arsenal_parameterization(semantic, parameterization) is not None
