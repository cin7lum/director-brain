# Vertical Slice Hard Case

## Input
"让下一句声音提前一点进入，让镜头切换更自然，但不要改变画面的切点，也不要加转场。"

## Required Criteria for PASS
1. audio_precedes_picture captured (positive intent)
2. picture_cut_position preserved (must_preserve)
3. transition avoided (must_avoid)
4. creative purpose retained
5. exact_value = null (no hallucination)
6. status = NEEDS_CONTEXT (vague "一点")
7. required_context non-empty
8. no tool/API leakage

## Results by Model

### qwen2.5:14b
- audio_precedes_picture: YES
- picture_cut_position preserved: YES
- transition avoided: YES
- exact_value: null
- status: READY (WRONG — should be NEEDS_CONTEXT)
- required_context: may be empty or incomplete
- Verdict: PARTIAL — core intent captured but status semantics wrong

### qwen3-vl
- Result: empty content / FAIL
- Verdict: FAIL

### glm-4-flash
- audio_precedes_picture: YES
- picture_cut_position preserved: YES (in must_preserve)
- transition avoided: intent present but in wrong field (avoid_transition in desired_relation_or_change instead of must_avoid)
- exact_value: null (correct)
- status: READY (WRONG — should be NEEDS_CONTEXT)
- required_context: [] (WRONG — should list dialogue_onset_timing, current_cut_point, audio_waveform)
- confidence: 0.8
- no tool/API leakage in core fields
- Verdict: PARTIAL — semantic intent captured but status and required_context wrong, field placement error

### Doubao (POC reference)
- All criteria met
- Verdict: PASS

## Conclusion
The Vertical Slice hard case is NOT PASSED by any callable model. The core
semantic intent (audio before picture, preserve cut, no transition) is
captured by qwen2.5:14b and glm-4-flash, but both fail on status semantics
(output READY instead of NEEDS_CONTEXT for vague "一点"). This is a model
capability gap in handling parameterization uncertainty.
