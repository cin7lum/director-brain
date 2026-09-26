# DIRECTOR_BRAIN_PRIMARY_MODEL_RESOLUTION — Execution Report

## Base commit
188df15a9014af2f42e5972a7e6d6eda30c2c647

## Head commit
(see git log after evidence commit)

## Goal
Find a callable Primary Model that meets the frozen Admission Gate for the
already-proven SemanticDirectorReasoner architecture.

## Frozen (not modified this round)
- Canonical DirectorDecision schema (12 fields, extra=forbid)
- 32-case benchmark (SHA256: 834239E9...)
- Human reference (SHA256: 544E31A3...)
- Semantic prompt v1.1
- SemanticDirectorReasoner semantics
- Status definitions
- Evaluation thresholds
- HeuristicDirectorReasoner = SHOT_SELECTION_SPECIALIST

## Admission Threshold
| Metric | Threshold |
|---|---|
| schema_valid_rate | >= 95% |
| positive_intent_recall | >= 95% |
| negative_constraint_recall | >= 90% |
| relation_direction_accuracy | >= 95% |
| ambiguity_abstention_accuracy | >= 95% |
| status_match | >= 90% |
| exact_parameter_hallucination | = 0 |
| tool_api_leakage | = 0 |
| Vertical Slice hard case | PASS |

## Candidate Summary

| Candidate | Type | Status | Key Failure |
|---|---|---|---|
| Doubao (POC) | External | REFERENCE_ONLY | No API key; not callable via formal adapter |
| qwen2.5:14b | Local ollama | REJECT | positive 85.7%, negative 62.5%, status 65.6% |
| qwen3-vl | Local ollama | REJECT | 22% empty content, tool leakage, 20-173s latency |
| glm-4-flash | External Zhipu | REJECT | positive 62.5%, negative 56.2%, status 46.9% |
| qwen3:30b | Local | NOT_TESTED | Hardware infeasible (12GB VRAM, user confirmed) |
| glm-4-air/plus | External | NOT_TESTABLE | Insufficient API balance (error 1113) |

## Hardware Pre-check
- CPU: i7-11800H 8-core
- RAM: 31.8GB (13.7GB free)
- GPU: RTX 3060 Laptop 12GB VRAM (10.9GB free)
- Disk: 336GB free
- Ollama: 0.34.2
- LOCAL_32B_CLASS_MODEL_FEASIBLE = MARGINAL (user confirmed 32B not viable)

## External API Audit
- Zhipu API key: found in gen1-roughcut/.env; glm-4-flash free tier works; paid models (glm-4-air, glm-4-plus) return 429 insufficient balance.
- Doubao/Ark: no API key in any .env or config.
- OpenAI: no API key.
- Other providers: no credentials found.

## New Artifact This Round
- `director_brain/zhipu_adapter.py`: Thin Zhipu OpenAI-compatible transport adapter (NEW_MODEL_ADAPTER_ONLY). Transport only — no semantic repair, no keyword patch, no model-specific answer correction. Uses response_format=json_object + canonical schema in prompt (Zhipu does not support full JSON Schema constrained generation) + runtime model_validate_json().
- `_zhipu_benchmark.py`: Temporary benchmark runner (not product code).

## Product Code Classification
- EXISTING_PRODUCT_FILES_MODIFIED: 0
- NEW_PRODUCT_RUNTIME_FILES_ADDED: 1 (zhipu_adapter.py, thin transport adapter only)
- TEST_FILES_ADDED_OR_MODIFIED: 0
- No semantic changes to DirectorDecision, SemanticDirectorReasoner, SemanticDirectorService, LLMAdapter, HeuristicDirectorReasoner.

## Final Result
**DIRECTOR_BRAIN_PRIMARY_MODEL_RESOLUTION = NO_MODEL_ADMITTED**

No currently callable model meets the Admission Gate. The Doubao POC reference
meets all thresholds but is not callable via a formal API adapter (no key).
All locally-installable and externally-callable models fail on core semantic
metrics.

Per task instructions: thresholds not lowered, no keyword patches, no model
voting, no semantic postprocessor, no rule-based repair added.
