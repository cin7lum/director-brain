# Regression Results

## New tests (Phase-1A)
- tests/unit/test_semantic_reasoner.py: 32 passed, 1 skipped
  - I1 entrypoint: 2 passed
  - I2 schema valid: 3 passed
  - I3 Vertical Slice hard case: 6 passed
  - I4 ambiguous: 4 passed
  - I5 needs context: 2 passed
  - I6 conflict: 2 passed
  - I7 negative audio: 2 passed
  - I8 direction: 2 passed
  - I9 no hallucination: 2 passed
  - I10 no leakage: 2 passed
  - I11 fail closed: 3 passed
  - I12 heuristic regression: 2 passed
  - Real model smoke: 1 skipped (ollama model not pulled)

## Existing tests
- 269 passed
- 5 failed: test_roughcut_cli.py (all ModuleNotFoundError: cv2) — PRE-EXISTING, unrelated
- 6 test files excluded from collection: all cv2-dependent (observation_service)

## Pre-existing failures (NOT caused by Phase-1A)
All 5 failures + 6 collection errors are due to `cv2` (OpenCV) not installed
in the shared venv. These are observation_service / media pipeline tests.
Phase-1A does not touch observation_service or media pipeline.
