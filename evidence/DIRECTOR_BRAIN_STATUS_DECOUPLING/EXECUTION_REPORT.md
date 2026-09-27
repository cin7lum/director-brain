# DIRECTOR BRAIN STATUS DECOUPLING — EXECUTION REPORT

## Status: IMPLEMENTATION_READY_FOR_REVIEW

## Problem

Vertical Slice V2 (commit a98e4b9) exposed a status coupling deadlock:
- Semantic Director correctly understood `audio_precedes_picture` + constraints
- LLM returned READY + exact_value=None (correct — user said "一点", not a number)
- Post-validation downgraded READY → NEEDS_CONTEXT because exact_value was None
- Parameterizer with external dialogue onset returned READY exact_value=10f
- Adapter blocked because semantic.status != READY
- Result: full product pipeline could not complete despite all components working

## Root Cause

Three independent concepts were conflated into one `status` field:
1. **Semantic readiness**: Is the user's intent understood?
2. **Parameterization readiness**: Is the exact execution value determined?
3. **Execution readiness**: Can this be handed to 03?

The post-validation rule `READY + exact_value=None → NEEDS_CONTEXT` treated
"parameter missing" as "semantic understanding incomplete", which is incorrect.

## Changes Made

### 1. `director_brain/parameterization/post_validation.py`
- **Removed**: `READY + exact_value=None → NEEDS_CONTEXT` global downgrade
- **Added**: READY + no desired_relation_or_change → NEEDS_CONTEXT (intent unclear)
- **Added**: READY + desired_relation contradicts must_avoid → CONFLICTING_CONSTRAINTS
- **Kept**: CONFLICTING_CONSTRAINTS + exact_value warning
- Post-validator now only checks SEMANTIC consistency, never parameter completeness

### 2. `director_brain/service.py`
- Added `semantic_status` field to `DirectorRequestResult`
- Added `execution_readiness` field to `DirectorRequestResult`
- Renamed `_compute_overall_status` → `_compute_execution_readiness`
- Execution readiness rules:
  - Semantic READY + Param READY → READY
  - Semantic READY + Param NEEDS_CONTEXT → WAITING_FOR_CONTEXT
  - Semantic READY + Param NEEDS_DECISION → WAITING_FOR_DECISION
  - Semantic not READY → propagate semantic status
  - Any CONFLICT/UNSATISFIABLE → BLOCKED

### 3. `director_brain/arsenal_adapter.py`
- Added `from_service_result()` — uses canonical `execution_readiness` from service
- Kept `to_arsenal_parameterization()` — requires both semantic READY + param READY
- No duplicate status logic: adapter delegates to service's readiness

### 4. Tests
- Updated `TestStatusInvariants`: 7 tests for semantic-only validation
- Added `TestStatusDecoupling`: S1-S7 status matrix tests
- Updated `test_phase1a_r1.py`: mock decision now includes desired_relation
- 108 relevant tests pass

## Verification

### Unit Tests
- 57 tests in test_parameterization_phase1.py: PASS
- 108 tests across parameterization + semantic reasoner + service + adapter: PASS

### Fresh Install
- Built wheel: director_brain-0.1.0-py3-none-any.whl (131,980 bytes)
- Clean venv: `.venv_status_decoupling_fresh`
- Module from site-packages: confirmed
- 18 verification assertions: ALL PASS

### Real Boundary Smoke
- Real Ollama qwen2.5:7b semantic reasoning
- V2 controlled fixture (A-B-C, B trimmed src_start=10, handle=10)
- External dialogue onset (frame 10)
- Results:
  - Semantic: READY (exact_value=None, NOT downgraded)
  - Parameterization: READY (10f from dialogue onset)
  - Execution: READY
  - Adapter: emits 03 payload exact_value=10.0
  - NO 03 write performed
- Negative cases:
  - Without dialogue: Semantic READY + WAITING_FOR_DECISION (correct abstention)
  - Semantic incomplete: adapter blocks even with param READY

## Product Code Scope

- **Modified**: 02 post_validation.py, service.py, arsenal_adapter.py, tests
- **NOT modified**: 03 Arsenal, J_CUT, DaVinci Provider, 05 FQL, Render, Safety Kernel

## Files Changed
1. `director_brain/parameterization/post_validation.py`
2. `director_brain/service.py`
3. `director_brain/arsenal_adapter.py`
4. `tests/test_parameterization_phase1.py`
5. `tests/unit/test_phase1a_r1.py`
