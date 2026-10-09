"""Tests for Director Brain Context Parameterization Phase 1.

Covers:
- 16-case POC regression (expected results unchanged from POC)
- Status invariant tests (READY + exact_value=None = invalid)
- Exact parameter provenance tests
- Metamorphic tests (M1-M5)
- Evidence removal tests
- Conflict tests (no clamping)
- No-hallucination tests
- 02→03 adapter tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure 02 package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from director_brain.models.director_decision import DirectorDecision, Parameterization
from director_brain.models.parameterization import (
    AudioEvent,
    AudioEventType,
    CandidateSource,
    ClipBoundary,
    ParameterizationContext,
    ParameterizationDecision,
    ParameterizationStatus,
)
from director_brain.parameterization.candidates import generate_jcut_candidates
from director_brain.parameterization.feasibility import (
    compute_jcut_feasible_range,
    validate_explicit_value,
)
from director_brain.parameterization.parameterizer import JCutParameterizer
from director_brain.parameterization.post_validation import validate_director_decision
from director_brain.parameterization.selector import select_or_abstain
from director_brain.arsenal_adapter import is_execution_ready, to_arsenal_parameterization


# ============================================================
# Helpers
# ============================================================

def _make_ctx(
    *,
    handle: int | None = None,
    source_start: int = 0,
    dialogue_onsets: list[int] | None = None,
    fps: float = 24.0,
    clip_count: int = 3,
) -> ParameterizationContext:
    """Build a context; each dialogue offset is frames before the cut."""
    incoming = ClipBoundary(
        clip_id="clip_B",
        media_name="video_B.mp4",
        start_frame=120,
        end_frame=240,
        duration_frames=120,
        source_start_frame=source_start,
        source_end_frame=120,
        source_duration_frames=120,
    )
    outgoing = ClipBoundary(
        clip_id="clip_A",
        media_name="video_A.mp4",
        start_frame=0,
        end_frame=120,
        duration_frames=120,
        source_start_frame=0,
        source_end_frame=120,
        source_duration_frames=120,
    )
    audio_events = []
    if dialogue_onsets:
        for frame in dialogue_onsets:
            audio_events.append(AudioEvent(
                event_type=AudioEventType.DIALOGUE_ONSET,
                frame=-frame,
                confidence=0.85,
                source="test_fixture",
                evidence_ref=f"test_dialogue_{-frame}f",
            ))
    provenance = {}
    if handle is not None:
        provenance["available_audio_handle_before"] = "test_fixture"
    return ParameterizationContext(
        frame_rate=fps,
        picture_cut_frame=120,
        incoming_clip=incoming,
        outgoing_clip=outgoing,
        audio_events=audio_events,
        available_audio_handle_before=handle,
        has_dialogue=bool(dialogue_onsets) if dialogue_onsets else None,
        timeline_clip_count=clip_count,
        provenance=provenance,
    )


def _make_director_decision(
    *,
    status: str = "READY",
    exact_value: float | None = None,
    magnitude: str | None = None,
    desired_relation: list[str] | None = None,
) -> DirectorDecision:
    """Build a minimal DirectorDecision for testing."""
    return DirectorDecision(
        decision_id="test-decision",
        creative_intent="Test creative intent for parameterization",
        status=status,
        desired_relation_or_change=desired_relation or ["audio_precedes_picture"],
        must_preserve=["picture_cut_position"],
        must_avoid=["transition"],
        parameterization=Parameterization(
            magnitude=magnitude,
            exact_value=exact_value,
            unit="frames",
            certainty="explicit" if exact_value is not None else "unknown",
        ),
        confidence=0.7,
        evidence=[],
        user_terminology=["test"],
    )


# ============================================================
# 16-Case POC Regression (expected results unchanged)
# ============================================================

class TestPOC16Regression:
    """Migrate POC 16 cases. Expected results must match POC exactly."""

    def test_b01_explicit_10f_zero_handle_conflict(self):
        ctx = _make_ctx(handle=0, source_start=0)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=10.0)
        assert d.status == ParameterizationStatus.CONFLICT

    def test_b02_explicit_10f_24handle_ready(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=10.0)
        assert d.status == ParameterizationStatus.READY
        assert d.exact_value == 10.0

    def test_b03_explicit_30f_24handle_conflict(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=30.0)
        assert d.status == ParameterizationStatus.CONFLICT

    def test_b04_explicit_5f_60handle_ready(self):
        ctx = _make_ctx(handle=60, source_start=60)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=5.0)
        assert d.status == ParameterizationStatus.READY
        assert d.exact_value == 5.0

    def test_b05_vague_zero_handle_unsatisfiable(self):
        ctx = _make_ctx(handle=0, source_start=0)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.UNSATISFIABLE

    def test_b06_vague_24handle_no_dialogue_needs_decision(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=None)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.NEEDS_DECISION

    def test_b07_strong_vague_60handle_no_signal_needs_decision(self):
        ctx = _make_ctx(handle=60, source_start=60, dialogue_onsets=None)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.NEEDS_DECISION

    def test_b08_single_dialogue_onset_ready(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.READY
        assert d.exact_value == 10.0

    def test_b09_multiple_dialogue_onsets_needs_decision(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[6, 18])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.NEEDS_DECISION

    def test_b10_no_dialogue_24handle_needs_decision(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.NEEDS_DECISION

    def test_b11_no_dialogue_zero_handle_unsatisfiable(self):
        ctx = _make_ctx(handle=0, source_start=0, dialogue_onsets=[])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.UNSATISFIABLE

    def test_b12_explicit_0f_ready(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=0.0)
        assert d.status == ParameterizationStatus.READY
        assert d.exact_value == 0.0

    def test_b13_negative_value_conflict(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=-5.0)
        assert d.status == ParameterizationStatus.CONFLICT

    def test_b14_explicit_24f_boundary_ready(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=24.0)
        assert d.status == ParameterizationStatus.READY
        assert d.exact_value == 24.0

    def test_b15_explicit_25f_over_boundary_conflict(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=25.0)
        assert d.status == ParameterizationStatus.CONFLICT

    def test_b16_dialogue_onset_over_handle_needs_decision(self):
        # Dialogue onset at 30f but only 24f handle → candidate rejected,
        # remaining convention candidates → NEEDS_DECISION
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[30])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.NEEDS_DECISION

    def test_jcut_uses_dialogue_onset_before_cut_as_lead_distance(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        d = JCutParameterizer().parameterize(ctx)
        assert d.status == ParameterizationStatus.READY
        assert d.exact_value == 10.0
        assert d.selected_candidate.evidence_refs == ["test_dialogue_-10f"]

    def test_jcut_uses_pre_cut_silence_end_as_speech_boundary(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[])
        ctx = ctx.model_copy(update={"audio_events": [AudioEvent(
            event_type=AudioEventType.SILENCE_END,
            frame=-8,
            confidence=0.85,
            source="test_fixture",
            evidence_ref="silence_end_before_cut",
        )]})
        d = JCutParameterizer().parameterize(ctx)
        assert d.status == ParameterizationStatus.READY
        assert d.exact_value == 8.0
        assert d.selected_candidate.evidence_refs == ["silence_end_before_cut"]

    def test_jcut_does_not_treat_dialogue_after_cut_as_lead(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[])
        ctx = ctx.model_copy(update={
            "audio_events": [AudioEvent(
                event_type=AudioEventType.DIALOGUE_ONSET,
                frame=10,
                confidence=0.85,
                source="test_fixture",
                evidence_ref="dialogue_after_cut",
            )],
            "has_dialogue": True,
        })
        d = JCutParameterizer().parameterize(ctx)
        assert d.status == ParameterizationStatus.NEEDS_DECISION
        assert all(
            candidate.source != CandidateSource.OBSERVED_EVENT
            for candidate in d.candidates
        )

    def test_jcut_does_not_use_silence_start_as_incoming_audio_cue(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[])
        ctx = ctx.model_copy(update={"audio_events": [AudioEvent(
            event_type=AudioEventType.SILENCE_START,
            frame=-8,
            confidence=0.85,
            source="test_fixture",
            evidence_ref="silence_start_before_cut",
        )]})
        d = JCutParameterizer().parameterize(ctx)
        assert d.status == ParameterizationStatus.NEEDS_DECISION
        assert all(
            candidate.source != CandidateSource.OBSERVED_EVENT
            for candidate in d.candidates
        )


# ============================================================
# Status Invariant Tests
# ============================================================

class TestStatusInvariants:
    """Semantic status invariants after decoupling from parameterization.

    DirectorDecision.status describes SEMANTIC readiness only.
    exact_value=None is NOT a semantic incompleteness — parameterizer handles it.
    """

    def test_ready_with_none_exact_stays_ready(self):
        """Semantic clear + exact_value=None → READY (parameterizer fills value)."""
        d = _make_director_decision(status="READY", exact_value=None)
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "READY"
        assert len(violations) == 0

    def test_ready_with_exact_kept(self):
        d = _make_director_decision(status="READY", exact_value=10.0)
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "READY"
        assert len(violations) == 0

    def test_needs_context_unchanged(self):
        d = _make_director_decision(status="NEEDS_CONTEXT", exact_value=None)
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "NEEDS_CONTEXT"

    def test_post_validator_never_fills_value(self):
        """Post-validator must NOT create exact_value — only downgrade for semantic issues."""
        d = _make_director_decision(status="READY", exact_value=None)
        fixed, _ = validate_director_decision(d)
        assert fixed.parameterization is not None
        assert fixed.parameterization.exact_value is None

    def test_ready_without_desired_relation_downgraded(self):
        """READY but no desired_relation_or_change → NEEDS_CONTEXT (intent unclear)."""
        d = DirectorDecision(
            decision_id="test-no-relation",
            creative_intent="No relation test",
            status="READY",
            desired_relation_or_change=[],
            must_preserve=["picture_cut_position"],
            must_avoid=["transition"],
            parameterization=Parameterization(exact_value=None, unit="frames", certainty="unknown"),
            confidence=0.7,
            evidence=[],
            user_terminology=[],
        )
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "NEEDS_CONTEXT"
        assert len(violations) > 0

    def test_ready_with_semantic_conflict_downgraded(self):
        """READY but desired_relation contradicts must_avoid → CONFLICTING_CONSTRAINTS."""
        d = DirectorDecision(
            decision_id="test-conflict",
            creative_intent="Conflict test",
            status="READY",
            desired_relation_or_change=["audio_precedes_picture"],
            must_preserve=["picture_cut_position"],
            must_avoid=["audio_precedes_picture"],
            parameterization=Parameterization(exact_value=10.0, unit="frames", certainty="explicit"),
            confidence=0.7,
            evidence=[],
            user_terminology=[],
        )
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "CONFLICTING_CONSTRAINTS"
        assert len(violations) > 0

    def test_ready_with_audio_lead_alias_conflict_downgraded(self):
        """Known film-language aliases must not bypass negative constraints."""
        d = DirectorDecision(
            decision_id="test-audio-alias-conflict",
            creative_intent="Keep incoming sound before picture but avoid audio lead",
            status="READY",
            desired_relation_or_change=["audio_precedes_picture"],
            must_avoid=["audio_lead"],
            parameterization=Parameterization(exact_value=None),
        )
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "CONFLICTING_CONSTRAINTS"
        assert any("audio_precedes_picture" in item for item in violations)
        assert any("audio_lead" in item for item in violations)

    def test_ready_with_transition_alias_conflict_downgraded(self):
        d = DirectorDecision(
            decision_id="test-transition-alias-conflict",
            creative_intent="Allow a transition while avoiding transitions",
            status="READY",
            desired_relation_or_change=["allow_transition"],
            must_avoid=["transition"],
        )
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "CONFLICTING_CONSTRAINTS"
        assert violations

    @pytest.mark.parametrize(("desired", "must_avoid"), [
        ("outgoing_audio_continues_after_cut", "audio-tail"),
        ("extend_visible_duration", "duration extension"),
        ("shorten_visible_duration", "duration_shortening"),
    ])
    def test_other_documented_constraint_aliases_conflict(
        self, desired: str, must_avoid: str,
    ):
        d = DirectorDecision(
            decision_id="test-known-constraint-alias",
            creative_intent="A positive relation conflicts with an avoided effect",
            status="READY",
            desired_relation_or_change=[desired],
            must_avoid=[must_avoid],
        )
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "CONFLICTING_CONSTRAINTS"
        assert violations

    def test_opposite_audio_direction_is_not_a_conflict(self):
        d = DirectorDecision(
            decision_id="test-audio-direction-is-distinct",
            creative_intent="Carry outgoing sound across the picture cut",
            status="READY",
            desired_relation_or_change=["outgoing_audio_continues_after_cut"],
            must_avoid=["audio_lead"],
        )
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "READY"
        assert not violations

    def test_conflicting_with_exact_value_warned(self):
        d = DirectorDecision(
            decision_id="test-conflict2",
            creative_intent="Conflict test",
            status="CONFLICTING_CONSTRAINTS",
            desired_relation_or_change=["audio_precedes_picture"],
            must_preserve=[],
            must_avoid=["audio_precedes_picture"],
            parameterization=Parameterization(exact_value=10.0, unit="frames", certainty="explicit"),
            confidence=0.7,
            evidence=[],
            user_terminology=[],
        )
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "CONFLICTING_CONSTRAINTS"
        assert any("exact_value" in v for v in violations)


# ============================================================
# Semantic ↔ Parameterization ↔ Execution Decoupling Tests (S1-S7)
# ============================================================

class TestStatusDecoupling:
    """Three independent status concepts must not be conflated.

    S1: Semantic complete + no parameter → Semantic READY
    S2: Semantic complete + Parameterizer NEEDS_DECISION → execution not ready
    S3: Semantic complete + Parameterizer READY → execution ready
    S4: Semantic incomplete + Parameterizer READY → blocked
    S5: Semantic conflict + Parameterizer READY → blocked
    S6: Semantic ready + parameter infeasible → blocked
    S7: Removing parameter evidence → semantic remains READY → execution NOT_READY
    """

    def test_s1_semantic_complete_no_param_stays_ready(self):
        """S1: Semantic clear + exact_value=None → Semantic READY (not NEEDS_CONTEXT)."""
        d = _make_director_decision(status="READY", exact_value=None)
        fixed, _ = validate_director_decision(d)
        assert fixed.status.value == "READY"

    def test_s2_semantic_ready_param_needs_decision_not_execution_ready(self):
        """S2: Semantic READY + Parameterizer NEEDS_DECISION → adapter returns None."""
        semantic = _make_director_decision(status="READY", exact_value=None)
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=None)
        param = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert param.status == ParameterizationStatus.NEEDS_DECISION
        assert to_arsenal_parameterization(semantic, param) is None
        assert is_execution_ready(semantic, param) is False

    def test_s3_semantic_ready_param_ready_execution_ready(self):
        """S3: Semantic READY + Parameterizer READY → adapter produces payload."""
        semantic = _make_director_decision(status="READY", exact_value=None)
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        param = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert param.status == ParameterizationStatus.READY
        assert param.exact_value == 10.0
        result = to_arsenal_parameterization(semantic, param)
        assert result is not None
        assert result["exact_value"] == 10.0

    def test_s4_semantic_incomplete_param_ready_blocked(self):
        """S4: Semantic NEEDS_CONTEXT + Parameterizer READY → adapter blocks."""
        semantic = _make_director_decision(status="NEEDS_CONTEXT", exact_value=None)
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        param = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert param.status == ParameterizationStatus.READY
        # Parameterization READY cannot rescue semantic incompleteness
        assert to_arsenal_parameterization(semantic, param) is None

    def test_s5_semantic_conflict_param_ready_blocked(self):
        """S5: Semantic CONFLICTING_CONSTRAINTS + Parameterizer READY → blocked."""
        semantic = DirectorDecision(
            decision_id="test-s5",
            creative_intent="Conflict",
            status="CONFLICTING_CONSTRAINTS",
            desired_relation_or_change=["audio_precedes_picture"],
            must_preserve=[],
            must_avoid=["audio_precedes_picture"],
            parameterization=Parameterization(exact_value=10.0, unit="frames", certainty="explicit"),
            confidence=0.7,
            evidence=[],
            user_terminology=[],
        )
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        param = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert param.status == ParameterizationStatus.READY
        assert to_arsenal_parameterization(semantic, param) is None

    def test_s6_semantic_ready_param_infeasible_blocked(self):
        """S6: Semantic READY + Parameterizer UNSATISFIABLE → blocked."""
        semantic = _make_director_decision(status="READY", exact_value=None)
        ctx = _make_ctx(handle=0, source_start=0)
        param = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert param.status == ParameterizationStatus.UNSATISFIABLE
        assert to_arsenal_parameterization(semantic, param) is None

    def test_s7_remove_param_evidence_semantic_stays_ready_execution_not_ready(self):
        """S7: Removing parameter evidence → semantic stays READY, execution becomes NOT_READY.

        This is the key metamorphic test: semantic understanding does not depend on
        parameter evidence. Execution readiness does.
        """
        semantic = _make_director_decision(status="READY", exact_value=None)

        # With dialogue evidence → parameter READY → execution ready
        ctx_with = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        param_with = JCutParameterizer().parameterize(ctx_with, user_exact_value=None)
        assert param_with.status == ParameterizationStatus.READY
        assert to_arsenal_parameterization(semantic, param_with) is not None

        # Semantic stays READY regardless of parameter evidence
        fixed_semantic, _ = validate_director_decision(semantic)
        assert fixed_semantic.status.value == "READY"

        # Remove dialogue evidence → parameter NEEDS_DECISION → execution NOT ready
        ctx_without = _make_ctx(handle=24, source_start=24, dialogue_onsets=None)
        param_without = JCutParameterizer().parameterize(ctx_without, user_exact_value=None)
        assert param_without.status == ParameterizationStatus.NEEDS_DECISION
        assert to_arsenal_parameterization(semantic, param_without) is None

        # Semantic still READY (it was never dependent on parameter evidence)
        assert fixed_semantic.status.value == "READY"


# ============================================================
# Provenance Tests
# ============================================================

class TestProvenance:
    """Every READY exact_value must have evidence_refs."""

    def test_explicit_user_value_has_provenance(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=10.0)
        assert d.status == ParameterizationStatus.READY
        assert "user_input" in d.evidence_refs

    def test_dialogue_onset_has_provenance(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.READY
        assert len(d.evidence_refs) > 0
        assert d.selected_candidate is not None
        assert d.selected_candidate.source == CandidateSource.OBSERVED_EVENT

    def test_needs_decision_has_no_exact_value(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=None)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.exact_value is None

    def test_context_provenance_recorded(self):
        ctx = _make_ctx(handle=24, source_start=24)
        assert "available_audio_handle_before" in ctx.provenance


# ============================================================
# Metamorphic Tests (M1-M5)
# ============================================================

class TestMetamorphic:
    """M1-M5 metamorphic property tests."""

    def test_m1_handle_decrease_feasible_max_not_increase(self):
        """M1: audio handle decrease → feasible max cannot increase."""
        ctx24 = _make_ctx(handle=24, source_start=24)
        ctx12 = _make_ctx(handle=12, source_start=12)
        f24 = compute_jcut_feasible_range(ctx24)
        f12 = compute_jcut_feasible_range(ctx12)
        assert f12.max_value <= f24.max_value

    def test_m2_exact_value_change_not_stale(self):
        """M2: changing exact value from 10f to 25f must not return old 10f."""
        ctx = _make_ctx(handle=24, source_start=24)
        d10 = JCutParameterizer().parameterize(ctx, user_exact_value=10.0)
        d25 = JCutParameterizer().parameterize(ctx, user_exact_value=25.0)
        assert d10.exact_value == 10.0
        assert d25.status == ParameterizationStatus.CONFLICT
        assert d25.exact_value is None

    def test_m3_remove_dialogue_evidence_loses_ready(self):
        """M3: removing dialogue evidence must lose the READY it supported."""
        ctx_with = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        ctx_without = _make_ctx(handle=24, source_start=24, dialogue_onsets=None)
        d_with = JCutParameterizer().parameterize(ctx_with, user_exact_value=None)
        d_without = JCutParameterizer().parameterize(ctx_without, user_exact_value=None)
        assert d_with.status == ParameterizationStatus.READY
        assert d_without.status != ParameterizationStatus.READY
        assert d_without.exact_value is None

    def test_m4_frame_rate_change_recomputes(self):
        """M4: changing frame rate must recompute convention candidates."""
        ctx24 = _make_ctx(handle=60, source_start=60, fps=24.0, dialogue_onsets=None)
        ctx30 = _make_ctx(handle=60, source_start=60, fps=30.0, dialogue_onsets=None)
        f24 = compute_jcut_feasible_range(ctx24)
        c24 = generate_jcut_candidates(ctx24, f24)
        c30 = generate_jcut_candidates(ctx30, compute_jcut_feasible_range(ctx30))
        # Convention candidates differ by fps
        vals24 = {c.value for c in c24 if c.source == CandidateSource.CONVENTION}
        vals30 = {c.value for c in c30 if c.source == CandidateSource.CONVENTION}
        assert vals24 != vals30  # 0.25*24=6 vs 0.25*30=7.5→8

    def test_m5_conflicting_preserve_downgrades(self):
        """M5: adding conflicting must_preserve must downgrade READY.
        (Simulated by making context unsatisfiable — zero handle.)"""
        ctx_ok = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        ctx_conflict = _make_ctx(handle=0, source_start=0, dialogue_onsets=[10])
        d_ok = JCutParameterizer().parameterize(ctx_ok, user_exact_value=None)
        d_conflict = JCutParameterizer().parameterize(ctx_conflict, user_exact_value=None)
        assert d_ok.status == ParameterizationStatus.READY
        assert d_conflict.status != ParameterizationStatus.READY


# ============================================================
# Evidence Removal Test (anti-self-certification)
# ============================================================

class TestEvidenceRemoval:
    """If READY depends on evidence, removing evidence must remove READY."""

    def test_remove_dialogue_evidence_ready_disappears(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.READY
        assert d.exact_value == 10.0

        # Remove the evidence
        ctx_no_evidence = ctx.model_copy(update={"audio_events": []})
        d2 = JCutParameterizer().parameterize(ctx_no_evidence, user_exact_value=None)
        assert d2.status != ParameterizationStatus.READY
        assert d2.exact_value is None

    def test_remove_handle_ready_disappears(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.status == ParameterizationStatus.READY

        # Remove handle (set to 0)
        ctx_no_handle = ctx.model_copy(update={
            "available_audio_handle_before": 0,
            "incoming_clip": ctx.incoming_clip.model_copy(update={"source_start_frame": 0}),
        })
        d2 = JCutParameterizer().parameterize(ctx_no_handle, user_exact_value=None)
        assert d2.status == ParameterizationStatus.UNSATISFIABLE


# ============================================================
# Conflict / No-Clamping Tests
# ============================================================

class TestNoClamping:
    """User values outside feasible range → CONFLICT, never clamped."""

    def test_value_over_max_conflict_not_clamped(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=30.0)
        assert d.status == ParameterizationStatus.CONFLICT
        assert d.exact_value is None
        assert "24" in d.conflict_detail

    def test_value_under_min_conflict(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=-1.0)
        assert d.status == ParameterizationStatus.CONFLICT

    def test_boundary_value_accepted(self):
        ctx = _make_ctx(handle=24, source_start=24)
        d = JCutParameterizer().parameterize(ctx, user_exact_value=24.0)
        assert d.status == ParameterizationStatus.READY
        assert d.exact_value == 24.0

    def test_validate_explicit_value_helper(self):
        ctx = _make_ctx(handle=24, source_start=24)
        f = compute_jcut_feasible_range(ctx)
        ok, _ = validate_explicit_value(10.0, f)
        assert ok is True
        ok, reason = validate_explicit_value(30.0, f)
        assert ok is False
        assert "max" in reason


# ============================================================
# No-Hallucination Tests
# ============================================================

class TestNoHallucination:
    """Parameterizer must never invent values without evidence."""

    def test_no_context_no_value(self):
        ctx = _make_ctx(handle=0, source_start=0, dialogue_onsets=[])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.exact_value is None

    def test_vague_no_signal_no_value(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[])
        d = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert d.exact_value is None
        assert d.status == ParameterizationStatus.NEEDS_DECISION

    def test_convention_candidates_marked_experimental(self):
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[])
        f = compute_jcut_feasible_range(ctx)
        cands = generate_jcut_candidates(ctx, f)
        conventions = [c for c in cands if c.source == CandidateSource.CONVENTION]
        for c in conventions:
            assert "EXPERIMENTAL" in c.rationale

    def test_no_random_values(self):
        """Run parameterizer 10 times — same context must give same result."""
        ctx = _make_ctx(handle=24, source_start=24, dialogue_onsets=[10])
        results = [JCutParameterizer().parameterize(ctx, user_exact_value=None) for _ in range(10)]
        values = {r.exact_value for r in results}
        assert len(values) == 1  # deterministic
        assert 10.0 in values


# ============================================================
# 02→03 Adapter Tests
# ============================================================

class TestArsenalAdapter:
    """Thin adapter: only READY produces 03 parameters."""

    def test_ready_semantic_and_param_produces_dict(self):
        semantic = _make_director_decision(status="READY", exact_value=10.0)
        ctx = _make_ctx(handle=24, source_start=24)
        param = JCutParameterizer().parameterize(ctx, user_exact_value=10.0)
        result = to_arsenal_parameterization(semantic, param)
        assert result is not None
        assert result["exact_value"] == 10.0

    def test_param_not_ready_returns_none(self):
        semantic = _make_director_decision(status="READY", exact_value=None)
        ctx = _make_ctx(handle=0, source_start=0)
        param = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        assert param.status == ParameterizationStatus.UNSATISFIABLE
        result = to_arsenal_parameterization(semantic, param)
        assert result is None

    def test_semantic_not_ready_returns_none(self):
        semantic = _make_director_decision(status="NEEDS_CONTEXT", exact_value=None)
        ctx = _make_ctx(handle=24, source_start=24)
        param = JCutParameterizer().parameterize(ctx, user_exact_value=None)
        result = to_arsenal_parameterization(semantic, param)
        assert result is None

    def test_is_execution_ready_matches(self):
        semantic = _make_director_decision(status="READY", exact_value=10.0)
        ctx = _make_ctx(handle=24, source_start=24)
        param = JCutParameterizer().parameterize(ctx, user_exact_value=10.0)
        assert is_execution_ready(semantic, param) is True

        semantic_bad = _make_director_decision(status="NEEDS_CONTEXT")
        assert is_execution_ready(semantic_bad, param) is False

    def test_no_param_decision_semantic_ready_with_value(self):
        """A source-verified explicit value can pass through without context."""
        semantic = _make_director_decision(status="READY", exact_value=10.0)
        assert to_arsenal_parameterization(semantic, None) is None
        result = to_arsenal_parameterization(
            semantic, None, exact_parameterization_source_verified=True,
        )
        assert result is not None
        assert result["exact_value"] == 10.0

    def test_no_param_decision_does_not_invent_default_unit(self):
        semantic = _make_director_decision(status="READY", exact_value=10.0)
        semantic.parameterization.unit = None
        assert to_arsenal_parameterization(
            semantic, None, exact_parameterization_source_verified=True,
        ) is None


# ============================================================
# Model Schema Tests
# ============================================================

class TestModelSchemas:
    """extra=forbid on all models."""

    def test_parameterization_decision_rejects_extra(self):
        with pytest.raises(Exception):
            ParameterizationDecision(
                parameter_name="test",
                status=ParameterizationStatus.READY,
                extra_field="should fail",
            )

    def test_parameterization_context_rejects_provider_fields(self):
        """Context must NOT contain recordFrame, trackIndex, etc."""
        with pytest.raises(Exception):
            ParameterizationContext(
                frame_rate=24.0,
                picture_cut_frame=120,
                recordFrame=100,  # should be rejected
            )

    def test_context_no_resolve_api_fields(self):
        """Verify context schema doesn't include provider-specific fields."""
        fields = ParameterizationContext.model_fields.keys()
        assert "recordFrame" not in fields
        assert "trackIndex" not in fields
        assert "mediaType" not in fields
        assert "run_script" not in fields
        assert "mcp_tool" not in fields
