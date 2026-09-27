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
    """Build a minimal ParameterizationContext for testing."""
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
                frame=frame,
                confidence=0.85,
                source="test_fixture",
                evidence_ref=f"test_dialogue_{frame}f",
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


# ============================================================
# Status Invariant Tests
# ============================================================

class TestStatusInvariants:
    """READY + exact_value=None must be invalid for parameter-required capabilities."""

    def test_ready_with_none_exact_downgraded(self):
        d = _make_director_decision(status="READY", exact_value=None)
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "NEEDS_CONTEXT"
        assert len(violations) > 0

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
        """Post-validator must NOT create exact_value — only downgrade status."""
        d = _make_director_decision(status="READY", exact_value=None)
        fixed, _ = validate_director_decision(d)
        assert fixed.parameterization is not None
        assert fixed.parameterization.exact_value is None

    def test_non_parameter_capability_ready_kept(self):
        """READY without exact_value is fine if capability doesn't need parameter."""
        d = _make_director_decision(
            status="READY", exact_value=None,
            desired_relation=["some_other_capability"],
        )
        fixed, violations = validate_director_decision(d)
        assert fixed.status.value == "READY"


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
        """If no parameterization context but semantic has exact_value, pass through."""
        semantic = _make_director_decision(status="READY", exact_value=10.0)
        result = to_arsenal_parameterization(semantic, None)
        assert result is not None
        assert result["exact_value"] == 10.0


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
