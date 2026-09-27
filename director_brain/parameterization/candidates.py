"""Candidate generation for J_CUT audio offset.

Candidates come from REAL context signals, never random numbers or keyword lookups.
Signal priority: dialogue_onset > silence_boundary > handle_fraction > convention.

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

    # 1. Dialogue onset candidates (strong signal)
    for event in ctx.audio_events:
        if event.event_type == AudioEventType.DIALOGUE_ONSET:
            # Dialogue onset frame is relative to picture cut
            # Positive frame = after cut; for J_CUT we want audio BEFORE cut
            # If onset is at source frame N, audio lead = N (start audio N frames before cut)
            lead = float(event.frame)
            _add(
                lead,
                CandidateSource.OBSERVED_EVENT,
                f"dialogue_onset at source frame {event.frame} (confidence={event.confidence})",
                confidence=event.confidence,
                evidence_refs=[event.evidence_ref] if event.evidence_ref else [],
            )

    # 2. Silence boundary candidates (strong signal)
    for event in ctx.audio_events:
        if event.event_type in (AudioEventType.SILENCE_END, AudioEventType.SILENCE_START):
            lead = float(event.frame)
            _add(
                lead,
                CandidateSource.OBSERVED_EVENT,
                f"{event.event_type.value} at frame {event.frame}",
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
        e.event_type in (AudioEventType.DIALOGUE_ONSET, AudioEventType.SILENCE_END)
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
