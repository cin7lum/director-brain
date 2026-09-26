# Negative Constraint Analysis

## qwen2.5:14b (62.5% recall, 10/16)

### Miss categories
1. **Term matching (5 misses)**: Model expresses same semantic with different
   vocabulary. E.g. reference expects "avoid_audio_lead", model puts
   "audio_precedes_picture" in must_avoid. Semantically correct, evaluator
   does exact string match.
2. **Multi-constraint sentences (2 misses)**: Model captures primary intent
   but misses secondary negative constraints. SD-31: captured extend_visible_duration,
   missed no-transition and no-reorder.
3. **Conflict detection (1 miss)**: SD-28 conflict not detected (model said READY
   instead of CONFLICTING_CONSTRAINTS).

### No systematic structural gap
Negative misses are scattered across audio, transition, duration, and order
constraints. No single negation pattern ("不要", "别", "不能") is systematically
missed. The primary cause is model capability (14B local vs POC's stronger model),
not architecture or prompt design.

## qwen3-vl (partial)
- SD-04 ("声音别抢在画面前面"): relations=[], must_avoid=[] — negative not captured
- SD-08 ("不要让下一句声音提前"): FAILED (empty content)
- SD-10 ("这里别拖这么久"): must_avoid=["extend_visible_duration"] — captured but
  as avoid rather than positive shorten
- Multiple cases with must_avoid=[] when negative constraint present

## POC Doubao (93.8%, 15/16)
- Only miss: SD-28 (evaluator representation difference, ISOLATED_CASE)
- No systematic negative gap
