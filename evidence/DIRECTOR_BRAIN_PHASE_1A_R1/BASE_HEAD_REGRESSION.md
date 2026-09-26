# Base/Head Regression Comparison

## Base commit: 8ef3b28 (before Phase-1A)
- Tests run: 274 (excluding 6 cv2-dependent collection errors)
- Passed: 269
- Failed: 5 (all test_roughcut_cli.py, all ModuleNotFoundError: cv2)
- cv2-dependent collection errors: 6 (test_asr, test_keyframe, test_observation_service,
  test_pipeline, test_vlm_adapter, test_vlm_observation)

## Head commit: R1 (current)
- Tests run: 324 (274 existing + 32 semantic + 18 R1)
- Passed: 319
- Failed: 5 (same test_roughcut_cli.py, same cv2 error)
- cv2-dependent collection errors: same 6

## Comparison
- NEW_REGRESSIONS = 0
- Same 5 cv2 failures in both base and head
- All new tests (semantic + R1) pass
- Existing test count unchanged (269 passed in both)
- No existing tests modified or skipped

## Root cause of pre-existing failures
`observation_service/deterministic_analysis.py` imports `cv2` (OpenCV),
which is not installed in the shared venv. test_roughcut_cli.py dynamically
loads a module that imports observation_service. This is unrelated to
Phase-1A/R1 changes.
