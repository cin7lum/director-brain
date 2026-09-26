# Director Brain Phase-1A — Semantic Director Reasoner Integration

**Base commit**: 8ef3b28
**Result**: DIRECTOR_BRAIN_PHASE_1A = IMPLEMENTATION_READY_FOR_REVIEW

## Goal
Integrate the POC-proven "Mature LLM + Structured Output → DirectorDecision"
into 02 AI-Director as a thin, formal capability.

## What was implemented
1. **Canonical DirectorDecision** (`director_brain/models/director_decision.py`)
   - 12 fields, extra="forbid", DecisionStatus enum (5 states)
   - Single source of truth — POC schema not duplicated
   - Tool/NLE/provider-agnostic

2. **LLM Adapter** (`director_brain/llm_adapter.py`)
   - Thin ollama HTTP API wrapper, format=json structured output
   - Fail-closed: any error → SEMANTIC_REASONER_UNAVAILABLE
   - No model router, no retry labyrinth, no agent framework
   - Records model identity, prompt version, latency, raw response

3. **SemanticDirectorReasoner** (`director_brain/semantic_reasoner.py`)
   - Formal interface: user direction + context → DirectorDecision
   - Does NOT do shot ranking, tool selection, execution, or scoring
   - Fail-closed, no keyword fallback

4. **Integration tests** (`tests/unit/test_semantic_reasoner.py`)
   - I1-I12 all passing (32 tests, 1 real-model smoke skipped)

## Integration Guard: Negative Constraint Miss
SD-28 analyzed → ISOLATED_CASE (evaluator representation mismatch, not semantic gap).
No keyword patch added. See NEGATIVE_CONSTRAINT_MISS_ANALYSIS.md.

## Regression
- New tests: 32 passed, 1 skipped (real model requires ollama model)
- Existing tests: 269 passed (5 pre-existing cv2 failures unrelated)
- HeuristicDirectorReasoner: unchanged, still importable, SHOT_SELECTION_SPECIALIST

## What was NOT done
- No DirectorBrief rewrite (boundary documented only)
- No EDL modification (DirectorDecision ≠ EDL)
- No Context Parameterizer (NEEDS_CONTEXT stops here)
- No model router / multi-model
- No Arsenal integration (next phase)
- No DaVinci Vertical Slice re-run
