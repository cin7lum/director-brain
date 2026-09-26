# Diff Summary (R1 vs base 2de3404)

## llm_adapter.py changes (~80 lines modified)
- DEFAULT_MODEL: llama3.2:3b → qwen2.5:14b
- format: "json" → DirectorDecision.model_json_schema()
- validation: json.loads + DirectorDecision(**data) → model_validate_json()
- decision_id: always override with caller-provided value
- LLMResult: added schema_constrained, schema_version fields
- prompt: added field usage rules + status definitions section
- error handling: added HTTP 400 fallback to plain json mode
- PROMPT_VERSION: 1.0 → 1.1

## service.py (new, ~90 lines)
- SemanticDirectorService class
- DirectorRequestResult dataclass
- process_direction() entrypoint
- Status propagation (READY/NEEDS_CONTEXT/UNDERSPECIFIED/UNSUPPORTED/
  CONFLICTING_CONSTRAINTS/SEMANTIC_REASONER_UNAVAILABLE)

## test_phase1a_r1.py (new, ~250 lines)
- R1-1: schema-constrained output (5 tests)
- R1-5: entrypoint wiring (4 tests)
- R1-6: fail-closed (3 tests)
- R1-7: no heuristic fallback (3 tests)
- R1-8: regression (2 tests)
- R1-2: real model smoke (1 test)
