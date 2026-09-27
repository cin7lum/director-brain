# Real Boundary Smoke Test

## Purpose
Verify that the status decoupling allows the V2 hard case to reach the 02→03 adapter
with external dialogue evidence, WITHOUT actually calling 03 write.

## Environment
- Python: D:\新建豆包\arsenal\.venv_gate_e_final (has both 02 editable + 03 wheel)
- Ollama: localhost:11434, model qwen2.5:7b
- 02 source: D:\新建豆包\AI-Director (updated with status decoupling)

## Fixture
- User request: "让下一句声音提前一点进入，让镜头切换更自然，但不要改变画面的切点，也不要加转场。"
- Timeline: A-B-C controlled, B trimmed source_start=10, handle=10, 24fps
- External evidence: dialogue onset at frame 10 (provenance: human annotation)

## Results

### Positive Case (with dialogue evidence)
- semantic_status: READY
- desired_relation: ['audio_precedes_picture']
- must_preserve: ['picture_cut_position']
- must_avoid: ['transition', 'duration_extension']
- exact_value (semantic): None
- post_validation_violations: [] (NO downgrade!)
- param_status: READY
- param_exact_value: 10.0
- param_evidence: ['external_dialogue_onset_frame_10']
- execution_readiness: READY
- adapter payload: {exact_value: 10.0, unit: frames, ...}
- 03 write: NOT PERFORMED (boundary smoke stops at adapter)

### Negative Case 1 (without dialogue evidence)
- semantic_status: READY (stays READY — not downgraded!)
- param_status: NEEDS_DECISION
- execution_readiness: WAITING_FOR_DECISION
- adapter: returns None (correct abstention)

### Negative Case 2 (semantic incomplete)
- Request: "自然一点。"
- semantic_status: NEEDS_CONTEXT
- execution_readiness: NEEDS_CONTEXT
- adapter: returns None (param READY cannot rescue semantic incompleteness)

## Conclusion
ALL ASSERTIONS PASSED. The status decoupling correctly:
1. Keeps semantic READY when intent is clear but parameter is vague
2. Allows parameterizer to fill the value from evidence
3. Blocks execution when either semantic or parameterization is not ready
4. Never performs a 03 write in this boundary smoke
