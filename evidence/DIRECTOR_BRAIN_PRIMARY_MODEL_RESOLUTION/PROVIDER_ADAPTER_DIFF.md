# Provider Adapter Diff

## New File: director_brain/zhipu_adapter.py

### Classification
NEW_MODEL_ADAPTER_ONLY — thin transport adapter only.

### What it does
- Provides ZhipuLLMAdapter class with same generate_decision() interface as LLMAdapter
- Calls Zhipu OpenAI-compatible API (https://open.bigmodel.cn/api/paas/v4/chat/completions)
- Uses response_format={"type": "json_object"} for JSON output mode
- Includes canonical DirectorDecision.model_json_schema() in system prompt
  (transport adaptation — Zhipu API does not support full JSON Schema constrained
  generation like ollama's format param)
- Runtime validation via DirectorDecision.model_validate_json()
- Fail-closed: any error returns SEMANTIC_REASONER_UNAVAILABLE, no silent fallback
- Records model identity, latency, raw response, schema_constrained flag

### What it does NOT do
- No semantic repair of model output
- No keyword patches
- No model-specific answer correction
- No benchmark-specific transformations
- No retry logic
- No model routing or fallback
- No modification to DirectorDecision schema
- No modification to SemanticDirectorReasoner
- No modification to SemanticDirectorService

### Interface compatibility
ZhipuLLMAdapter.generate_decision() returns LLMResult (same dataclass as LLMAdapter).
Can be injected into SemanticDirectorReasoner(llm_adapter=...) via duck typing.

### Schema source of truth
Uses DirectorDecision.model_json_schema() — the same canonical schema as LLMAdapter.
No second schema copy. Schema version string: "pydantic_DirectorDecision".

### Difference from ollama LLMAdapter
| Aspect | ollama LLMAdapter | ZhipuLLMAdapter |
|---|---|---|
| API | ollama /api/chat | Zhipu /chat/completions |
| Schema constraint | format=schema (full JSON Schema) | response_format=json_object + schema in prompt |
| Transport | urllib to localhost | urllib to external HTTPS |
| Auth | none | Bearer token |
| Max tokens | ollama default | 2048 explicit |
| Temperature | 0.1 (options) | 0.1 (top-level) |

### Files modified
- NEW: director_brain/zhipu_adapter.py
- No existing product files modified
