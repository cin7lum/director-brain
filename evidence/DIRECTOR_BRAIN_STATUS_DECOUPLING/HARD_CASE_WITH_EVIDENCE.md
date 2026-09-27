# Hard Case: With External Dialogue Evidence

## Request
"让下一句声音提前一点进入，让镜头切换更自然，但不要改变画面的切点，也不要加转场。"

## Context
- V2 controlled fixture: A-B-C, B trimmed src_start=10, handle=10, 24fps
- External dialogue onset evidence: frame 10 (provenance: human annotation)

## Actual Results (Real Boundary Smoke)

### Semantic
- status: READY
- desired_relation: ['audio_precedes_picture']
- must_preserve: ['picture_cut_position']
- must_avoid: ['transition', 'duration_extension']
- exact_value: None
- post_validation_violations: []

### Parameterization
- status: READY
- exact_value: 10.0
- unit: frames
- evidence: ['external_dialogue_onset_frame_10']

### Execution
- execution_readiness: READY
- status (backward compat): READY
- adapter payload: {exact_value: 10.0, unit: frames, ...}

### Key Confirmation
- Semantic was NOT downgraded despite exact_value=None
- Parameterizer correctly used dialogue onset to determine 10f
- Adapter successfully emitted 03 parameterization
- NO 03 write was performed (boundary smoke)
