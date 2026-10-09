"""Candidate generation for J_CUT audio offset.

Candidates come from REAL context signals, never random numbers or keyword lookups.
Only observed boundaries before the picture cut can supply a J-cut lead.

Convention candidates are EXPERIMENTAL — they have no professional corpus citation
and are only included when no audio signal is available.
"""
from __future__ import annotations

from director_brain.models.parameterization import (
    AudioEventType,
    CandidateSource,
    FeasibleRange,
    ParameterCandidate,
    ParameterizationContext,
)

# Convention increments in seconds. These are EXPERIMENTAL_CANDIDATE, not product defaults.
# Source: POC author's heuristic, no professional corpus citation.
_CONVENTION_SECONDS = [0.25, 0.33, 0.5]


def generate_jcut_candidates(
    ctx: ParameterizationContext,
    feasible: FeasibleRange,
) -> list[ParameterCandidate]:
    """Generate candidate audio offsets from real context signals.

    Only candidates within feasible range are returned.
    No random numbers. No keyword lookups.
    """
    candidates: list[ParameterCandidate] = []
    seen_values: set[float] = set()

    def _add(value: float, source: CandidateSource, rationale: str,
             confidence: float = 0.0, evidence_refs: list[str] | None = None) -> None:
        if value < feasible.min_value or value > feasible.max_value:
            return
        if value in seen_values:
            return
        seen_values.add(value)
        candidates.append(ParameterCandidate(
            value=value,
            unit="frames",
            source=source,
            rationale=rationale,
            confidence=confidence,
            evidence_refs=evidence_refs or [],
        ))

    # AudioEvent.frame is signed relative to the picture cut: positive means
    # after it. A J-cut can only start from an observed event before that cut.
    # DIALOGUE_ONSET and SILENCE_END identify a possible incoming speech
    # boundary; SILENCE_START does not.
    for event in ctx.audio_events:
        if (
            event.event_type not in (
                AudioEventType.DIALOGUE_ONSET,
                AudioEventType.SILENCE_END,
            )
            or event.frame >= 0
        ):
            continue
        lead = float(-event.frame)
        _add(
            lead,
            CandidateSource.OBSERVED_EVENT,
            f"{event.event_type.value} {lead:g} frames before picture cut",
            confidence=event.confidence,
            evidence_refs=[event.evidence_ref] if event.evidence_ref else [],
        )

    # 3. Handle fraction candidates (weak signal, only if handle known)
    if ctx.available_audio_handle_before is not None and ctx.available_audio_handle_before > 0:
        handle = float(ctx.available_audio_handle_before)
        for frac, label in [(0.25, "quarter"), (0.5, "half"), (0.75, "three_quarter")]:
            _add(
                round(handle * frac),
                CandidateSource.HANDLE_FRACTION,
                f"{label} of available handle ({handle}f)",
                confidence=0.3,
            )

    # 4. Convention candidates (EXPERIMENTAL, only if no audio signal)
    has_audio_signal = any(
        e.event_type in (
            AudioEventType.DIALOGUE_ONSET,
            AudioEventType.SILENCE_END,
        ) and e.frame < 0
        for e in ctx.audio_events
    )
    if not has_audio_signal and feasible.max_value > 0:
        fps = ctx.frame_rate if ctx.frame_rate > 0 else 24.0
        for sec in _CONVENTION_SECONDS:
            frames = round(sec * fps)
            _add(
                float(frames),
                CandidateSource.CONVENTION,
                f"EXPERIMENTAL: {sec}s at {fps}fps = {frames}f (no audio signal available)",
                confidence=0.1,
            )

    return candidates
