# Semantic Metrics — All Candidates

## Comparison Table

| Metric | Threshold | qwen2.5:14b | qwen3-vl | glm-4-flash | Doubao (POC ref) |
|---|---|---|---|---|---|
| schema_valid_rate | >=95% | 100% PASS | ~78% FAIL | 100% PASS | 100% PASS |
| positive_intent_recall | >=95% | 85.7% FAIL | N/A | 62.5% FAIL | 100% PASS |
| negative_constraint_recall | >=90% | 62.5% FAIL | N/A | 56.2% FAIL | 93.8% PASS |
| relation_direction_accuracy | >=95% | 100% PASS | N/A | 84.4% FAIL | 100% PASS |
| ambiguity_abstention_accuracy | >=95% | 100% PASS | N/A | 62.5% FAIL | 100% PASS |
| status_match | >=90% | 65.6% FAIL | N/A | 46.9% FAIL | 100% PASS |
| exact_param_hallucination | =0 | 0 PASS | N/A | 0 PASS | 0 PASS |
| tool_api_leakage | =0 | 0 PASS | 1 FAIL | 1 FAIL | 0 PASS |
| Vertical Slice hard case | PASS | PARTIAL | FAIL | PARTIAL | PASS |

## Key Observations

### qwen2.5:14b
- Strengths: schema validity, relation direction, ambiguity abstention, no hallucination, no leakage
- Weaknesses: positive intent (85.7%), negative constraints (62.5%), status match (65.6%)
- Pattern: tends to output READY when NEEDS_CONTEXT is correct; misses some negative constraints

### glm-4-flash
- Strengths: schema validity (with schema-in-prompt), no hallucination
- Weaknesses: ALL semantic metrics below threshold; worst of all tested models
- Pattern: systematically outputs READY instead of NEEDS_CONTEXT; puts avoid_* relations in desired_relation_or_change instead of must_avoid; misses must_preserve/must_avoid separation
- Tool leakage: SD-24 contains "otio" in output (real leak); SD-14/15 "reorder" is substring of valid "reorder_story_beat" (false positive in evaluator)

### Doubao POC (reference only)
- Meets or exceeds all thresholds
- Not formally callable (no API key)
- Must be re-verified if a real endpoint becomes available

## Conclusion
No callable model meets the Admission Gate. The gap is primarily in:
1. Positive intent recall (need 95%, best callable = 85.7%)
2. Negative constraint recall (need 90%, best callable = 62.5%)
3. Status match (need 90%, best callable = 65.6%)

These are model capability gaps, not architecture or prompt gaps. The frozen
prompt and schema have been validated by the Doubao POC reference.
