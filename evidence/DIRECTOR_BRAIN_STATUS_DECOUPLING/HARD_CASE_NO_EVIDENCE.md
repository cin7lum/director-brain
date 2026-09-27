# Hard Case: No Dialogue Evidence

## Request
"让下一句声音提前一点进入，让镜头切换更自然，但不要改变画面的切点，也不要加转场。"

## Context
- V2 controlled fixture: A-B-C, B trimmed src_start=10, handle=10, 24fps
- NO dialogue onset evidence provided

## Expected Results

### Semantic
- status: READY
- desired_relation: audio_precedes_picture
- must_preserve: picture_cut_position
- must_avoid: transition, duration_extension
- exact_value: None (correct — "一点" is vague)
- post_validation: no violations (semantic is clear)

### Parameterization
- status: NEEDS_DECISION
- feasible_range: [0, 10]
- candidates: handle_fraction(2,5,8), convention(6) — all weak
- reason: only_weak_candidates

### Execution
- execution_readiness: WAITING_FOR_DECISION
- adapter: returns None (correct — no exact value to execute)

## This is Correct Behavior

The system:
1. Understands the intent (semantic READY)
2. Acknowledges it needs either human decision or audio evidence to pick the value
3. Does NOT invent a value
4. Does NOT block because "semantic is unclear" (it isn't)

The user should be presented with candidates or asked to provide dialogue timing.
