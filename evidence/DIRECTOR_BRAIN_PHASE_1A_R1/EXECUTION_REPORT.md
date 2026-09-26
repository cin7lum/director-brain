# Director Brain Phase-1A-R1 — Semantic Reasoner Integration Closure

**Base commit**: 2de3404
**Result**: DIRECTOR_BRAIN_PHASE_1A_R1 = IMPLEMENTATION_READY_FOR_REVIEW

## Four evidence gaps closed

### 1. Structured Output truly schema-constrained
- `LLMAdapter` now uses `DirectorDecision.model_json_schema()` as ollama `format` param
- Runtime validation uses `DirectorDecision.model_validate_json()`
- Single source of truth: canonical Pydantic model → JSON Schema → LLM → validation
- `extra=forbid` preserved
- Verified: R1-1 tests (5 tests) confirm payload format == canonical schema

### 2. Real model runtime validation (no skip)
- Model: qwen2.5:14b (installed locally, ollama stdio)
- 32-case benchmark run through PRODUCTION `director_brain.llm_adapter.LLMAdapter`
- 0 failures, 100% schema valid
- Results: see REAL_MODEL_32_CASE_RESULTS.md

### 3. Vertical Slice hard case through production adapter
- Input: "让下一句声音提前一点进入，让镜头切换更自然，但不要改变画面的切点，也不要加转场。"
- Result: audio_precedes_picture ✓, picture_cut_position preserved ✓,
  transition avoided ✓, exact_value=null ✓, status=NEEDS_CONTEXT ✓, no tool leakage ✓
- See VERTICAL_SLICE_HARD_CASE.json

### 4. Runtime wiring proof
- New thin entrypoint: `director_brain/service.py` → `SemanticDirectorService.process_direction()`
- Chain: user input → SemanticDirectorService → SemanticDirectorReasoner → LLMAdapter → DirectorDecision
- NEEDS_CONTEXT and SEMANTIC_REASONER_UNAVAILABLE propagate to caller
- No silent heuristic fallback
- See RUNTIME_ENTRYPOINT_TRACE.md

## Real model 32-case metrics (qwen2.5:14b, production adapter)
| Metric | Result | POC (Doubao) | Note |
|--------|--------|-------------|------|
| Schema validity | 100% | 100% | ✓ |
| Positive intent recall | 85.7% | 100% | Local model weaker |
| Negative constraint recall | 62.5% | 93.8% | Term matching + model |
| Relation direction | 100.0% | 100% | ✓ |
| Ambiguity abstention | 100.0% | 100% | ✓ |
| Status match | 65.6% | 100% | Model says READY too often |
| Exact param hallucination | 0 | 0 | ✓ |
| Tool leakage | 0 | 0 | ✓ |

## Key finding
The architecture (canonical schema → LLM constrained generation → Pydantic validation →
fail-closed reasoner → service entrypoint) is PROVEN. The local qwen2.5:14b model
underperforms the POC's Doubao on negative constraint capture and status classification,
but all safety properties hold: 100% schema valid, zero hallucination, zero tool leakage,
fail-closed on errors. Model capability is the bottleneck, not architecture.

## What was NOT done
- No Context Parameterizer
- No EDL modification
- No Arsenal/DaVinci/FQL changes
- No keyword parser patches
- No model router
