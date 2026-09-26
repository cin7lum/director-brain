# Runtime Entrypoint Trace

## Formal entrypoint
`director_brain/service.py` → `SemanticDirectorService.process_direction()`

## Call chain
```
Caller (application/CLI/API)
  ↓
SemanticDirectorService.process_direction(user_direction, context, decision_id)
  ↓ constructs SemanticDirectorReasoner (or accepts injected)
SemanticDirectorReasoner.reason(user_direction, context, decision_id)
  ↓ calls LLMAdapter.generate_decision()
LLMAdapter.generate_decision(user_input, context, decision_id)
  ↓ builds payload with format=DirectorDecision.model_json_schema()
  ↓ POST http://localhost:11434/api/chat
  ↓ parses response, DirectorDecision.model_validate_json()
  ↓ returns LLMResult(decision=..., error=...)
SemanticDirectorReasoner.reason()
  ↓ wraps in SemanticReasonerResult, adds trace_id
  ↓ if decision is None → error propagates (no fallback)
SemanticDirectorService.process_direction()
  ↓ maps decision.status → DirectorRequestResult.status
  ↓ if decision is None → status="SEMANTIC_REASONER_UNAVAILABLE"
  ↓ returns DirectorRequestResult to caller
```

## How NEEDS_CONTEXT propagates
- `DirectorDecision.status = NEEDS_CONTEXT`
- → `SemanticReasonerResult.decision.status = NEEDS_CONTEXT`
- → `DirectorRequestResult.status = "NEEDS_CONTEXT"`
- → Caller inspects `result.decision.required_context` and gathers context
- No automatic parameterization, no silent retry

## How model failure propagates
- LLM connection error / invalid JSON / schema validation failure
- → `LLMResult.decision = None, error = "SEMANTIC_REASONER_UNAVAILABLE: ..."`
- → `SemanticReasonerResult.decision = None, error = ...`
- → `DirectorRequestResult.status = "SEMANTIC_REASONER_UNAVAILABLE"`
- → Caller must handle explicitly. No heuristic fallback, no keyword parser.

## Who consumes the result
- Currently: test suite (R1-5 tests verify propagation)
- Future: Arsenal (03) Film Capability Resolution layer
- Future: Context Parameterizer (next phase)
