# Structured Output Runtime

## Chain (single source of truth)
```
DirectorDecision (canonical Pydantic model, director_brain/models/director_decision.py)
  ↓ model_json_schema()
JSON Schema (generated, not hand-maintained)
  ↓ ollama format= param
LLM constrained generation (qwen2.5:14b)
  ↓ model_validate_json()
DirectorDecision (runtime validated, extra=forbid)
```

## Verification
- R1-1 tests confirm `LLMAdapter.json_schema == DirectorDecision.model_json_schema()`
- R1-1 tests confirm HTTP payload `format` field equals canonical schema
- R1-1 tests confirm `model_validate_json` rejects invalid JSON and extra fields
- Real model run: 32/32 outputs pass schema validation (100%)

## No second schema
- No hand-written JSON Schema file
- No prompt-embedded schema copy
- Schema is always derived from the canonical Pydantic model at runtime
