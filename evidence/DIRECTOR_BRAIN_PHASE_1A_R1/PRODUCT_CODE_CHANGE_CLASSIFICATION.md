# Product Code Change Classification

## EXISTING_PRODUCT_FILES_MODIFIED
- `director_brain/llm_adapter.py` — modified:
  - Changed DEFAULT_MODEL from llama3.2:3b to qwen2.5:7b → qwen2.5:14b
  - Changed format from "json" to DirectorDecision.model_json_schema()
  - Changed validation from json.loads + DirectorDecision(**) to model_validate_json()
  - Added decision_id override (caller-provided authoritative)
  - Added schema_constrained and schema_version to LLMResult
  - Improved prompt: added field usage rules + status definitions
  - Added fallback to "json" mode if schema format rejected (with schema_constrained=False)
  - Bumped PROMPT_VERSION to 1.1

## NEW_PRODUCT_RUNTIME_FILES_ADDED
- `director_brain/models/director_decision.py` — canonical DirectorDecision model
  (added in Phase-1A, not modified in R1)
- `director_brain/semantic_reasoner.py` — SemanticDirectorReasoner
  (added in Phase-1A, not modified in R1)
- `director_brain/service.py` — SemanticDirectorService (thin entrypoint, R1)

## TEST_FILES_ADDED_OR_MODIFIED
- `tests/unit/test_semantic_reasoner.py` — I1-I12 tests (Phase-1A)
- `tests/unit/test_phase1a_r1.py` — R1-1 through R1-8 tests (R1)

## NOT MODIFIED
- director_brain/director_reasoner.py (Heuristic) — unchanged
- director_brain/brief_compiler.py — unchanged
- director_brain/models/director_brief.py — unchanged
- director_brain/models/edl.py — unchanged
- director_brain/edl_generator.py — unchanged
- All other existing production files — unchanged
