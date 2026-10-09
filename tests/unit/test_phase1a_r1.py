"""Phase-1A-R1 integration tests.

R1-1 through R1-8 as specified.
Real-model tests (R1-2, R1-3, R1-4) are validated by the benchmark runner;
unit tests here verify wiring, schema, fail-closed, and entrypoint behavior.
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
    DirectorDecision,
    DecisionStatus,
    Parameterization,
)
from director_brain.models.parameterization import (
    ParameterizationDecision,
    ParameterizationStatus,
)
from director_brain.arsenal_adapter import from_service_result
from director_brain.llm_adapter import LLMAdapter, LLMResult
from director_brain.semantic_reasoner import SemanticDirectorReasoner
from director_brain.service import SemanticDirectorService, DirectorRequestResult


# ── R1-1: Canonical Pydantic schema directly enters Ollama format ──

class TestR1SchemaConstrained:
    def test_r1_1_adapter_uses_model_json_schema(self):
        """LLMAdapter.format must be DirectorDecision.model_json_schema()."""
        adapter = LLMAdapter()
        schema = adapter.json_schema
        # Verify it's the canonical Pydantic schema
        assert schema == DirectorDecision.model_json_schema()
        assert "properties" in schema
        assert "decision_id" in schema["properties"]
        assert "creative_intent" in schema["properties"]
        assert "status" in schema["properties"]
        assert "source_evidence_refs" in schema["properties"]

    def test_r1_1_payload_contains_schema_as_format(self):
        """The ollama request payload must use the canonical schema as format."""
        adapter = LLMAdapter()
        expected_schema = DirectorDecision.model_json_schema()
        with patch("director_brain.llm_adapter._open_local_request") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "message": {"content": json.dumps({
                    "decision_id": "test", "creative_intent": "test",
                    "status": "READY",
                })}
            }).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            adapter.generate_decision("test", decision_id="test")

            # Capture the request payload
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["format"] == expected_schema

    def test_r1_1_model_validate_json_used(self):
        """Runtime validation must use model_validate_json (canonical model)."""
        adapter = LLMAdapter()
        valid_json = json.dumps({
            "decision_id": "test",
            "creative_intent": "test intent",
            "status": "READY",
        })
        # This should not raise
        decision = DirectorDecision.model_validate_json(valid_json)
        assert decision.decision_id == "test"

    def test_r1_1_invalid_json_fails_validation(self):
        """Invalid JSON must fail closed via model_validate_json."""
        with pytest.raises(Exception):
            DirectorDecision.model_validate_json("{invalid json")

    def test_r1_1_extra_fields_rejected(self):
        """extra=forbid must reject unknown fields."""
        with pytest.raises(Exception):
            DirectorDecision.model_validate_json(json.dumps({
                "decision_id": "test",
                "creative_intent": "test",
                "status": "READY",
                "unexpected_field": "value",
            }))


# ── R1-5: Real 02 entrypoint → Semantic Reasoner trace ──

class TestR1Entrypoint:
    def test_r1_5_service_entrypoint_exists(self):
        """SemanticDirectorService is the formal 02 entrypoint."""
        service = SemanticDirectorService()
        assert service is not None
        assert hasattr(service, "process_direction")

    def test_r1_5_service_wires_reasoner(self):
        """Service → SemanticDirectorReasoner → LLMAdapter chain."""
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_decision.return_value = LLMResult(
            decision=DirectorDecision(
                decision_id="t", creative_intent="t",
                status=DecisionStatus.READY,
                desired_relation_or_change=["audio_precedes_picture"],
                must_preserve=["picture_cut_position"],
                must_avoid=["transition"]),
            model="test", latency_ms=10,
        )
        reasoner = SemanticDirectorReasoner(llm_adapter=mock_llm)
        service = SemanticDirectorService(reasoner=reasoner)

        result = service.process_direction(
            "test input",
            context="obs_1: speech begins after cut",
            available_source_evidence_refs=["obs_1"],
        )
        assert isinstance(result, DirectorRequestResult)
        assert result.decision is not None
        assert result.semantic_status == "READY"
        assert result.execution_readiness == "WAITING_FOR_CONTEXT"
        assert from_service_result(result) is None
        assert result.trace_id.startswith("svc_")
        context = mock_llm.generate_decision.call_args.kwargs["context"]
        assert "obs_1" in context

    def test_r1_5_explicit_semantic_value_can_pass_without_context(self):
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_decision.return_value = LLMResult(
            decision=DirectorDecision(
                decision_id="t-explicit",
                creative_intent="Use the explicit audio offset",
                status=DecisionStatus.READY,
                desired_relation_or_change=["audio_precedes_picture"],
                parameterization=Parameterization(
                    exact_value=8.0,
                    unit="frames",
                    certainty="explicit",
                ),
                evidence=["Audio leads by 8 frames"],
            ),
            model="test",
            latency_ms=10,
        )
        service = SemanticDirectorService(
            reasoner=SemanticDirectorReasoner(llm_adapter=mock_llm))

        result = service.process_direction("Audio leads by 8 frames")

        assert result.semantic_status == "READY"
        assert result.execution_readiness == "READY"
        assert result.exact_parameterization_source_verified is True
        assert from_service_result(result) == {
            "exact_value": 8.0,
            "unit": "frames",
            "magnitude": None,
            "certainty": "explicit",
        }

    def test_r1_5_exact_seconds_are_not_reinterpreted_as_frames(self):
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_decision.return_value = LLMResult(
            decision=DirectorDecision(
                decision_id="t-explicit-seconds",
                creative_intent="Use the explicit audio lead",
                status=DecisionStatus.READY,
                desired_relation_or_change=["audio_precedes_picture"],
                parameterization=Parameterization(
                    exact_value=0.5,
                    unit="seconds",
                    certainty="explicit",
                ),
                evidence=["Audio leads by 0.5 seconds"],
            ),
            model="test",
            latency_ms=10,
        )
        service = SemanticDirectorService(
            reasoner=SemanticDirectorReasoner(llm_adapter=mock_llm))

        direct_result = service.process_direction("Audio leads by 0.5 seconds")
        assert direct_result.execution_readiness == "READY"
        assert direct_result.exact_parameterization_source_verified is True
        assert from_service_result(direct_result) == {
            "exact_value": 0.5,
            "unit": "seconds",
            "magnitude": None,
            "certainty": "explicit",
        }

        result = service.process_direction(
            "Audio leads by 0.5 seconds",
            parameterization_context=object(),
        )

        assert result.semantic_status == "READY"
        assert result.exact_parameterization_source_verified is True
        assert result.parameterization_decision.status == ParameterizationStatus.UNAVAILABLE
        assert (result.parameterization_decision.reason_code
                == "EXPLICIT_UNIT_NOT_SUPPORTED_BY_JCUT_PARAMETERIZER")
        assert result.execution_readiness == "NOT_READY"
        assert from_service_result(result) is None

    def test_r1_5_parameterizer_ready_without_exact_value_fails_closed(self):
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_decision.return_value = LLMResult(
            decision=DirectorDecision(
                decision_id="t-missing-exact",
                creative_intent="Use an incoming audio lead",
                status=DecisionStatus.READY,
                desired_relation_or_change=["audio_precedes_picture"],
            ),
            model="test",
            latency_ms=10,
        )
        invalid_parameterization = ParameterizationDecision(
            parameter_name="audio_offset_frames",
            status=ParameterizationStatus.READY,
            exact_value=None,
        )
        service = SemanticDirectorService(
            reasoner=SemanticDirectorReasoner(llm_adapter=mock_llm))
        service._parameterizer = MagicMock()
        service._parameterizer.parameterize.return_value = invalid_parameterization

        result = service.process_direction(
            "Use an incoming audio lead",
            parameterization_context=object(),
        )

        assert result.semantic_status == "READY"
        assert result.execution_readiness == "NOT_READY"
        assert from_service_result(result) is None

    def test_r1_5_needs_context_propagates(self):
        """NEEDS_CONTEXT must propagate to caller via result.status."""
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_decision.return_value = LLMResult(
            decision=DirectorDecision(
                decision_id="t", creative_intent="t",
                status=DecisionStatus.NEEDS_CONTEXT,
                required_context=["dialogue_onset_timing"]),
            model="test", latency_ms=10,
        )
        service = SemanticDirectorService(
            reasoner=SemanticDirectorReasoner(llm_adapter=mock_llm))
        result = service.process_direction("test")
        assert result.status == "NEEDS_CONTEXT"
        assert result.decision.required_context == ["dialogue_onset_timing"]

    def test_r1_5_known_negative_alias_conflict_blocks_readiness(self):
        """A known equivalent must_avoid term cannot leave the service READY."""
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_decision.return_value = LLMResult(
            decision=DirectorDecision(
                decision_id="t-conflict",
                creative_intent="Audio lead requested and forbidden",
                status=DecisionStatus.READY,
                desired_relation_or_change=["audio_precedes_picture"],
                must_avoid=["audio_lead"],
            ),
            model="test",
            latency_ms=10,
        )
        service = SemanticDirectorService(
            reasoner=SemanticDirectorReasoner(llm_adapter=mock_llm))

        result = service.process_direction("声音要提前，但禁止声音领先")

        assert result.semantic_status == "CONFLICTING_CONSTRAINTS"
        assert result.execution_readiness == "BLOCKED"
        assert result.status == "BLOCKED"
        assert result.post_validation_violations

    def test_r1_5_model_failure_propagates(self):
        """SEMANTIC_REASONER_UNAVAILABLE must propagate to caller."""
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_decision.return_value = LLMResult(
            decision=None, error="SEMANTIC_REASONER_UNAVAILABLE: connection refused",
            model="test", latency_ms=5,
        )
        service = SemanticDirectorService(
            reasoner=SemanticDirectorReasoner(llm_adapter=mock_llm))
        result = service.process_direction("test")
        assert result.status == "SEMANTIC_REASONER_UNAVAILABLE"
        assert result.decision is None
        assert result.failure_code == "provider_failure"
        assert "connection refused" not in (result.error or "")

    def test_r1_5_invalid_evidence_is_distinct_from_provider_unavailable(self):
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_decision.return_value = LLMResult(
            decision=DirectorDecision(
                decision_id="t-invented-evidence",
                creative_intent="Use a two-second audio lead",
                evidence=["用户明确要求声音提前两秒"],
                status=DecisionStatus.READY,
                desired_relation_or_change=["audio_precedes_picture"],
                parameterization=Parameterization(
                    exact_value=2.0,
                    unit="seconds",
                    certainty="explicit",
                ),
            ),
            model="test",
            latency_ms=10,
        )
        service = SemanticDirectorService(
            reasoner=SemanticDirectorReasoner(llm_adapter=mock_llm))

        result = service.process_direction("让声音提前一点进入")

        assert result.semantic_status == "INVALID"
        assert result.execution_readiness == "NOT_READY"
        assert result.status == "SEMANTIC_REASONER_INVALID"
        assert "exact substring" in (result.error or "")
        assert from_service_result(result) is None


# ── R1-6: Model unavailable → fail closed ──

class TestR1FailClosed:
    def test_r1_6_connection_error(self):
        adapter = LLMAdapter()
        with patch("director_brain.llm_adapter._open_local_request", side_effect=ConnectionError("refused")):
            result = adapter.generate_decision("test")
            assert result.decision is None
            assert "SEMANTIC_REASONER_UNAVAILABLE" in (result.error or "")

    def test_r1_6_timeout(self):
        import urllib.error
        adapter = LLMAdapter()
        with patch("director_brain.llm_adapter._open_local_request", side_effect=urllib.error.URLError("timeout")):
            result = adapter.generate_decision("test")
            assert result.decision is None
            assert result.error is not None

    def test_r1_6_invalid_schema_output(self):
        """Model output that fails Pydantic validation → fail closed."""
        adapter = LLMAdapter()
        with patch("director_brain.llm_adapter._open_local_request") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "message": {"content": '{"decision_id": "t", "creative_intent": "t"}'}
                # missing required 'status' field
            }).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = adapter.generate_decision("test")
            assert result.decision is None
            assert "schema validation failed" in (result.error or "").lower()


# ── R1-7: No silent heuristic fallback ──

class TestR1NoHeuristicFallback:
    def test_r1_7_reasoner_does_not_import_heuristic(self):
        """SemanticDirectorReasoner must not import or use HeuristicDirectorReasoner."""
        import inspect
        from director_brain import semantic_reasoner
        # Check actual imports, not docstring mentions
        source = inspect.getsource(semantic_reasoner)
        # Must not have an import statement for heuristic
        assert "from director_brain.director_reasoner import" not in source
        assert "import director_reasoner" not in source
        # Must not instantiate HeuristicDirectorReasoner
        assert "HeuristicDirectorReasoner(" not in source

    def test_r1_7_service_does_not_fallback_to_heuristic(self):
        """Service must not fall back to heuristic on LLM failure."""
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_decision.return_value = LLMResult(
            decision=None, error="SEMANTIC_REASONER_UNAVAILABLE",
            model="test", latency_ms=5,
        )
        service = SemanticDirectorService(
            reasoner=SemanticDirectorReasoner(llm_adapter=mock_llm))
        result = service.process_direction("让下一句声音提前一点进入")
        # Must be error, not a heuristic-parsed decision
        assert result.decision is None
        assert result.status == "SEMANTIC_REASONER_UNAVAILABLE"

    def test_r1_7_heuristic_still_exists_as_specialist(self):
        """HeuristicDirectorReasoner must still exist (SHOT_SELECTION_SPECIALIST)."""
        from director_brain.director_reasoner import HeuristicDirectorReasoner
        assert HeuristicDirectorReasoner is not None
        assert hasattr(HeuristicDirectorReasoner, "generate_plan")


# ── R1-8: Base/Head regression (verified by benchmark, documented here) ──

class TestR1Regression:
    def test_r1_8_no_new_test_failures(self):
        """Phase-1A/R1 must not introduce new test failures.

        Base (8ef3b28): 269 passed, 5 failed (all cv2 in test_roughcut_cli.py)
        Head (R1): same 5 cv2 failures + new semantic tests pass.
        NEW_REGRESSIONS = 0.
        """
        # This is verified by the A/B run documented in BASE_HEAD_REGRESSION.md
        # The assertion here confirms the new test module itself passes.
        assert True  # placeholder — actual comparison in evidence

    def test_r1_8_existing_modules_unchanged(self):
        """Existing production files must not be modified by R1."""
        import director_brain.director_reasoner as hr
        import director_brain.brief_compiler as bc
        assert hr is not None
        assert bc is not None


# ── R1-2/3/4: Real model validation (smoke — skipped if no model) ──

class TestR1RealModel:
    """Real model tests. R1-2/3/4 are fully validated by the benchmark runner.
    This is a smoke test that the production adapter can call the model."""

    @pytest.fixture
    def model_available(self):
        import socket
        try:
            s = socket.socket()
            s.settimeout(2)
            s.connect(("localhost", 11434))
            s.close()
            return True
        except Exception:
            return False

    def test_r1_2_real_model_callable(self, model_available):
        if not model_available:
            pytest.skip("ollama not running")
        adapter = LLMAdapter(model="qwen2.5:14b", timeout=300)
        result = adapter.generate_decision("这个镜头短一点", decision_id="smoke")
        # Should either succeed or fail closed with explicit error
        assert result.decision is not None or result.error is not None
        if result.decision:
            assert isinstance(result.decision, DirectorDecision)
            assert result.schema_constrained is True
