# Vertical Slice Hard Case Results

Input: "让下一句声音提前一点进入，让镜头切换更自然，但不要改变画面的切点，也不要加转场。"

## Required checks
1. audio_precedes_picture in desired_relation_or_change
2. picture_cut_position in must_preserve
3. transition in must_avoid
4. exact_value = null
5. status = NEEDS_CONTEXT
6. no tool/API leakage

## Doubao (POC)
- All 6 checks: PASS
- decision_id: SD-01
- relations: ["audio_precedes_picture"]
- must_preserve: ["picture_cut_position"]
- must_avoid: ["transition"]
- exact_value: null
- status: NEEDS_CONTEXT

## qwen2.5:14b (production adapter)
- All 6 checks: PASS
- relations: ["audio_precedes_picture"]
- must_preserve: ["picture_cut_position"]
- must_avoid: ["transition"]
- exact_value: null
- status: NEEDS_CONTEXT
- Latency: 6.2s

## qwen3-vl (production adapter)
- CHECK 1-6: FAIL
- Result: SEMANTIC_REASONER_UNAVAILABLE (empty content)
- Model returned no content under JSON schema constraint
- Hard gate failure
