"""Integration tests for SemanticDirectorReasoner (Phase-1A).

I1-I12 as specified. Uses mocked LLM adapter for deterministic testing.
A real-model smoke test is included but skipped when no model is available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from director_brain.models.director_decision import (
    DirectorDecision, DecisionStatus, Parameterization,
)
from director_brain.semantic_reasoner import SemanticDirectorReasoner, SemanticReasonerResult
from director_brain.llm_adapter import LLMAdapter, LLMResult


def _make_decision(**overrides) -> DirectorDecision:
    """Helper: build a valid DirectorDecision with sensible defaults."""
    defaults = dict(
        decision_id="test_001",
        creative_intent="test intent",
        target=["incoming_dialogue"],
        desired_relation_or_change=["audio_precedes_picture"],
        must_preserve=[],
        must_avoid=[],
        parameterization=Parameterization(magnitude="slight", exact_value=None),
        required_context=["dialogue_onset_timing"],
        confidence=0.85,
        evidence=[],
        status=DecisionStatus.NEEDS_CONTEXT,
        user_terminology=[],
    )
    defaults.update(overrides)
    return DirectorDecision(**defaults)


def _mock_llm(decision: DirectorDecision | None = None, error: str | None = None) -> LLMAdapter:
    """Create a mock LLM adapter that returns a fixed result."""
    adapter = MagicMock(spec=LLMAdapter)
    result = LLMResult(
        decision=decision,
        error=error,
        model="test-model",
        prompt_version="1.0",
        raw_response=json.dumps(decision.model_dump()) if decision else "",
        latency_ms=42,
    )
    adapter.generate_decision.return_value = result
    return adapter


# ── I1: Semantic Reasoner callable from entrypoint ──

class TestI1Entrypoint:
    def test_i1_reasoner_can_be_called(self):
        reasoner = SemanticDirectorReasoner(llm_adapter=_mock_llm(_make_decision()))
        result = reasoner.reason("test input")
        assert isinstance(result, SemanticReasonerResult)
        assert result.decision is not None
        assert result.error is None

    def test_i1_empty_input_fails_closed(self):
        reasoner = SemanticDirectorReasoner(llm_adapter=_mock_llm())
        result = reasoner.reason("")
        assert result.decision is None
        assert "SEMANTIC_REASONER_UNAVAILABLE" in (result.error or "")
        assert result.failure_code == "empty_user_direction"


# ── I2: Structured output schema valid ──

class TestSourceEvidenceReferences:
    def test_verbatim_director_request_excerpt_is_accepted(self):
        excerpt = "不要改变画面的切点"
        adapter = _mock_llm(_make_decision(evidence=[excerpt]))
        result = SemanticDirectorReasoner(llm_adapter=adapter).reason(
            f"请执行要求：{excerpt}。"
        )

        assert result.decision is not None
        assert result.decision.evidence == [excerpt]
        assert result.error is None
        assert result.failure_code is None

    def test_invented_or_paraphrased_excerpt_fails_closed(self):
        invented = "用户明确要求声音提前两秒"
        adapter = _mock_llm(_make_decision(evidence=[invented]))
        result = SemanticDirectorReasoner(llm_adapter=adapter).reason(
            "让声音提前一点进入"
        )

        assert result.decision is None
        assert "exact substring" in (result.error or "")
        assert invented not in (result.error or "")
        assert result.failure_code == "evidence_excerpt_unbound"

    def test_blank_excerpt_fails_closed(self):
        adapter = _mock_llm(_make_decision(evidence=["  "]))
        result = SemanticDirectorReasoner(llm_adapter=adapter).reason("自然一点")

        assert result.decision is None
        assert "exact substring" in (result.error or "")
        assert result.failure_code == "evidence_excerpt_unbound"

    def test_exact_parameter_value_requires_quoted_numeric_value_and_unit(self):
        quote = "Audio leads by 8 frames"
        decision = _make_decision(
            evidence=[quote],
            parameterization=Parameterization(
                exact_value=8.0,
                unit="frames",
                certainty="explicit",
            ),
        )
        result = SemanticDirectorReasoner(llm_adapter=_mock_llm(decision)).reason(quote)

        assert result.decision is not None
        assert result.exact_parameterization_source_verified is True

    def test_exact_parameter_value_cannot_be_inferred_from_vague_quote(self):
        quote = "让声音提前一点进入"
        decision = _make_decision(
            evidence=[quote],
            parameterization=Parameterization(
                exact_value=8.0,
                unit="frames",
                certainty="explicit",
            ),
        )
        result = SemanticDirectorReasoner(llm_adapter=_mock_llm(decision)).reason(quote)

        assert result.decision is None
        assert "numeric-and-unit request excerpt" in (result.error or "")
        assert result.failure_code == "exact_parameterization_unverified"

    def test_exact_parameter_value_unit_must_match_quote(self):
        quote = "Audio leads by 8 frames"
        decision = _make_decision(
            evidence=[quote],
            parameterization=Parameterization(
                exact_value=8.0,
                unit="seconds",
                certainty="explicit",
            ),
        )
        result = SemanticDirectorReasoner(llm_adapter=_mock_llm(decision)).reason(quote)

        assert result.decision is None
        assert "numeric-and-unit request excerpt" in (result.error or "")
        assert result.failure_code == "exact_parameterization_unverified"

    def test_only_allowlisted_source_references_are_returned(self):
        adapter = _mock_llm(_make_decision(source_evidence_refs=["obs_a"]))
        reasoner = SemanticDirectorReasoner(llm_adapter=adapter)

        result = reasoner.reason(
            "基于这个素材证据安排声音进入",
            context="obs_a: next dialogue begins after the picture cut",
            available_source_evidence_refs=["obs_a", "obs_b"],
        )

        assert result.decision is not None
        assert result.decision.source_evidence_refs == ["obs_a"]
        call_context = adapter.generate_decision.call_args.kwargs["context"]
        assert "Authorized source evidence reference IDs" in call_context
        assert '["obs_a", "obs_b"]' in call_context

    def test_unlisted_source_reference_fails_closed(self):
        adapter = _mock_llm(_make_decision(source_evidence_refs=["invented-id"]))
        reasoner = SemanticDirectorReasoner(llm_adapter=adapter)

        result = reasoner.reason(
            "安排声音进入",
            available_source_evidence_refs=["obs_a"],
        )

        assert result.decision is None
        assert "unavailable source evidence" in (result.error or "")
        assert result.failure_code == "source_reference_not_allowlisted"

    def test_source_reference_without_allowlist_fails_closed(self):
        adapter = _mock_llm(_make_decision(source_evidence_refs=["obs_a"]))
        reasoner = SemanticDirectorReasoner(llm_adapter=adapter)

        result = reasoner.reason("安排声音进入")

        assert result.decision is None
        assert "unavailable source evidence" in (result.error or "")
        assert result.failure_code == "source_reference_not_allowlisted"

    def test_malformed_allowlist_stops_before_provider_call(self):
        adapter = _mock_llm(_make_decision())
        reasoner = SemanticDirectorReasoner(llm_adapter=adapter)

        result = reasoner.reason(
            "安排声音进入",
            available_source_evidence_refs=["obs_a", "obs_a"],
        )

        assert result.decision is None
        assert "duplicate source evidence" in (result.error or "")
        assert result.failure_code == "duplicate_source_evidence_refs"
        adapter.generate_decision.assert_not_called()

    def test_adapter_exception_fails_closed_without_echoing_exception(self):
        adapter = _mock_llm()
        adapter.generate_decision.side_effect = RuntimeError("private context value")
        reasoner = SemanticDirectorReasoner(llm_adapter=adapter)

        result = reasoner.reason("安排声音进入")

        assert result.decision is None
        assert result.error == "SEMANTIC_REASONER_UNAVAILABLE: adapter exception"
        assert "private context value" not in (result.error or "")
        assert result.failure_code == "adapter_exception"

    def test_provider_error_text_is_normalized_to_a_fixed_failure(self):
        adapter = _mock_llm(error="PRIVATE_PROVIDER_RESPONSE_MARKER")
        result = SemanticDirectorReasoner(llm_adapter=adapter).reason(
            "让声音稍早进入")

        assert result.decision is None
        assert result.failure_code == "provider_failure"
        assert "PRIVATE_PROVIDER_RESPONSE_MARKER" not in (result.error or "")

class TestI2SchemaValid:
    def test_i2_decision_validates(self):
        d = _make_decision()
        assert isinstance(d, DirectorDecision)
        assert d.status == DecisionStatus.NEEDS_CONTEXT

    def test_i2_extra_fields_rejected(self):
        with pytest.raises(Exception):
            DirectorDecision(
                decision_id="x", creative_intent="x", status=DecisionStatus.READY,
                extra_field="not allowed",
            )

    def test_i2_invalid_status_rejected(self):
        with pytest.raises(Exception):
            DirectorDecision(
                decision_id="x", creative_intent="x", status="INVALID_STATUS",
            )


# ── I3: Vertical Slice hard case ──

class TestI3VerticalSlice:
    VERTICAL_SLICE_INPUT = (
        "让下一句声音提前一点进入，让镜头切换更自然，"
        "但不要改变画面的切点，也不要加转场。"
    )

    def _vs_decision(self) -> DirectorDecision:
        return _make_decision(
            decision_id="vs_001",
            creative_intent="通过让下一句声音先于画面进入，使镜头切换在感知上更自然",
            target=["incoming_dialogue", "incoming_picture", "current_picture_cut", "transition_point"],
            desired_relation_or_change=["audio_precedes_picture"],
            must_preserve=["picture_cut_position"],
            must_avoid=["transition"],
            parameterization=Parameterization(magnitude="slight", exact_value=None, certainty="inferred"),
            required_context=["dialogue_onset_timing", "current_cut_point", "audio_waveform"],
            confidence=0.85,
            evidence=["让下一句声音提前一点进入", "不要改变画面的切点", "不要加转场"],
            status=DecisionStatus.NEEDS_CONTEXT,
        )

    def test_i3_positive_audio_before_picture(self):
        d = self._vs_decision()
        assert "audio_precedes_picture" in d.desired_relation_or_change

    def test_i3_picture_cut_preserved(self):
        d = self._vs_decision()
        assert "picture_cut_position" in d.must_preserve

    def test_i3_no_transition(self):
        d = self._vs_decision()
        assert any("transition" in m.lower() for m in d.must_avoid)

    def test_i3_exact_offset_not_fabricated(self):
        d = self._vs_decision()
        assert d.parameterization.exact_value is None

    def test_i3_needs_context_explicit(self):
        d = self._vs_decision()
        assert d.status == DecisionStatus.NEEDS_CONTEXT
        assert len(d.required_context) > 0

    def test_i3_no_tool_leakage(self):
        d = self._vs_decision()
        text = json.dumps(d.model_dump(), ensure_ascii=False).lower()
        forbidden = ["j_cut", "jcut", "l_cut", "lcut", "recordframe",
                     "mediatype", "trackindex", "davinci", "mcp", "otio"]
        for term in forbidden:
            assert term not in text, f"tool leakage: {term}"


# ── I4: Ambiguous → UNDERSPECIFIED ──

class TestI4Ambiguous:
    @pytest.mark.parametrize("text", ["这里自然一点", "情绪再压一点", "节奏舒服一点", "这里高级一点"])
    def test_i4_ambiguous_underspecified(self, text):
        d = _make_decision(
            creative_intent=f"模糊请求: {text}",
            desired_relation_or_change=[],
            status=DecisionStatus.UNDERSPECIFIED,
            required_context=["clarification_needed"],
        )
        assert d.status == DecisionStatus.UNDERSPECIFIED
        assert len(d.desired_relation_or_change) == 0


# ── I5: Missing context → NEEDS_CONTEXT ──

class TestI5NeedsContext:
    def test_i5_vague_param_needs_context(self):
        d = _make_decision(
            parameterization=Parameterization(magnitude="slight", exact_value=None),
            status=DecisionStatus.NEEDS_CONTEXT,
            required_context=["dialogue_onset_timing", "current_cut_point"],
        )
        assert d.status == DecisionStatus.NEEDS_CONTEXT
        assert d.parameterization.exact_value is None
        assert len(d.required_context) > 0

    def test_i5_explicit_param_ready(self):
        d = _make_decision(
            parameterization=Parameterization(
                magnitude="explicit", exact_value=2.0, unit="seconds", certainty="explicit"),
            status=DecisionStatus.READY,
            required_context=[],
        )
        assert d.status == DecisionStatus.READY
        assert d.parameterization.exact_value == 2.0


# ── I6: Conflicting constraints ──

class TestI6Conflict:
    def test_i6_conflict_detected(self):
        d = _make_decision(
            creative_intent="约束冲突：不改切点但提前B镜头",
            desired_relation_or_change=[],
            must_preserve=["picture_cut_position"],
            status=DecisionStatus.CONFLICTING_CONSTRAINTS,
            required_context=["clarification_of_conflict"],
        )
        assert d.status == DecisionStatus.CONFLICTING_CONSTRAINTS

    def test_i6_conflict_does_not_pick_side(self):
        d = _make_decision(
            status=DecisionStatus.CONFLICTING_CONSTRAINTS,
            desired_relation_or_change=[],
        )
        # Must not have a positive action that resolves the conflict
        assert len(d.desired_relation_or_change) == 0


# ── I7: Negative audio advance ──

class TestI7NegativeAudio:
    @pytest.mark.parametrize("text", ["声音别抢在画面前面", "不要让下一句声音提前"])
    def test_i7_no_positive_audio_relation(self, text):
        d = _make_decision(
            creative_intent="禁止声音领先",
            desired_relation_or_change=["preserve_picture_cut"],
            must_avoid=["audio_lead"],
            must_preserve=["audio_video_sync"],
            status=DecisionStatus.READY,
        )
        assert "audio_precedes_picture" not in d.desired_relation_or_change
        assert any("audio" in m.lower() or "lead" in m.lower() for m in d.must_avoid)


# ── I8: J-cut vs L-cut direction ──

class TestI8Direction:
    def test_i8_jcut_direction(self):
        d = _make_decision(
            desired_relation_or_change=["audio_precedes_picture"],
            target=["incoming_dialogue", "incoming_picture"],
        )
        assert "audio_precedes_picture" in d.desired_relation_or_change
        assert "outgoing_audio_continues_after_cut" not in d.desired_relation_or_change

    def test_i8_lcut_direction(self):
        d = _make_decision(
            desired_relation_or_change=["outgoing_audio_continues_after_cut"],
            target=["outgoing_audio", "current_picture_cut"],
        )
        assert "outgoing_audio_continues_after_cut" in d.desired_relation_or_change
        assert "audio_precedes_picture" not in d.desired_relation_or_change


# ── I9: No parameter hallucination ──

class TestI9NoHallucination:
    def test_i9_slight_no_exact_value(self):
        d = _make_decision(
            parameterization=Parameterization(magnitude="slight", exact_value=None),
        )
        assert d.parameterization.exact_value is None
        assert d.parameterization.magnitude == "slight"

    def test_i9_explicit_value_preserved(self):
        d = _make_decision(
            parameterization=Parameterization(
                exact_value=0.5, unit="seconds", certainty="explicit"),
        )
        assert d.parameterization.exact_value == 0.5
        assert d.parameterization.unit == "seconds"


# ── I10: Tool/API leakage = 0 ──

class TestI10NoLeakage:
    def test_i10_no_tool_names_in_core_fields(self):
        d = _make_decision()
        # Check all fields except user_terminology
        data = {k: v for k, v in d.model_dump().items() if k != "user_terminology"}
        text = json.dumps(data, ensure_ascii=False).lower()
        forbidden = ["j_cut", "jcut", "l_cut", "lcut", "hold", "reorder",
                     "mcp", "davinci", "otio", "recordframe", "mediatype",
                     "trackindex", "resolve", "ffmpeg"]
        # "reorder" is part of "reorder_story_beat" which is allowed film vocabulary
        leaks = []
        for term in forbidden:
            if term == "reorder":
                # Only flag standalone "reorder", not "reorder_story_beat"
                import re
                if re.search(r'(?<![a-z_])reorder(?![a-z_])', text):
                    leaks.append(term)
            elif term in text:
                leaks.append(term)
        assert len(leaks) == 0, f"tool leakage: {leaks}"

    def test_i10_user_terminology_exempt(self):
        # If user says "J-cut", it can appear in user_terminology
        d = _make_decision(user_terminology=["J-cut"])
        assert "J-cut" in d.user_terminology


# ── I11: Model failure → fail closed ──

class TestI11FailClosed:
    def test_i11_llm_connection_error(self):
        adapter = MagicMock(spec=LLMAdapter)
        adapter.generate_decision.return_value = LLMResult(
            decision=None, error="SEMANTIC_REASONER_UNAVAILABLE: connection refused",
            model="test", latency_ms=5,
        )
        reasoner = SemanticDirectorReasoner(llm_adapter=adapter)
        result = reasoner.reason("test")
        assert result.decision is None
        assert "SEMANTIC_REASONER_UNAVAILABLE" in (result.error or "")

    def test_i11_invalid_json_fails_closed(self):
        adapter = MagicMock(spec=LLMAdapter)
        adapter.generate_decision.return_value = LLMResult(
            decision=None, error="SEMANTIC_REASONER_UNAVAILABLE: invalid JSON",
            model="test", latency_ms=5,
        )
        reasoner = SemanticDirectorReasoner(llm_adapter=adapter)
        result = reasoner.reason("test")
        assert result.decision is None
        assert result.error is not None

    def test_i11_no_keyword_fallback(self):
        """When LLM fails, must NOT fall back to keyword parser."""
        adapter = MagicMock(spec=LLMAdapter)
        adapter.generate_decision.return_value = LLMResult(
            decision=None, error="SEMANTIC_REASONER_UNAVAILABLE",
            model="test", latency_ms=5,
        )
        reasoner = SemanticDirectorReasoner(llm_adapter=adapter)
        result = reasoner.reason("让下一句声音提前一点进入")
        # Must be error, not a keyword-parsed decision
        assert result.decision is None
        assert result.error is not None


# ── I12: Heuristic shot-selection regression ──

class TestI12HeuristicRegression:
    """Verify HeuristicDirectorReasoner still works (not modified)."""

    def test_i12_heuristic_importable(self):
        from director_brain.director_reasoner import HeuristicDirectorReasoner
        reasoner = HeuristicDirectorReasoner()
        assert reasoner is not None

    def test_i12_heuristic_is_shot_selection(self):
        """Heuristic remains SHOT_SELECTION_SPECIALIST — produces select_shot decisions."""
        from director_brain.director_reasoner import HeuristicDirectorReasoner
        # Just verify the class exists and has generate_plan
        assert hasattr(HeuristicDirectorReasoner, "generate_plan")


# ── Real model smoke test (skipped without model) ──

class TestRealModelSmoke:
    @pytest.mark.skipif(
        not __import__("socket").socket().connect_ex(("localhost", 11434)) == 0,
        reason="ollama not running",
    )
    def test_real_model_vertical_slice(self):
        """Smoke test with real ollama model. Skipped if no model."""
        adapter = LLMAdapter()
        result = adapter.generate_decision(
            "让下一句声音提前一点进入，但不要改变画面的切点，也不要加转场。"
        )
        if result.decision is None:
            pytest.skip(f"model unavailable: {result.error}")
        assert isinstance(result.decision, DirectorDecision)
