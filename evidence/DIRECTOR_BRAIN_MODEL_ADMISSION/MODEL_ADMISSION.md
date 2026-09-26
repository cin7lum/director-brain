# Model Admission Decision

## Final result: NO_MODEL_ADMITTED

No candidate model meets all admission thresholds through the formal production path.

## Threshold comparison

| Metric | Threshold | Doubao (POC) | qwen2.5:14b | qwen3-vl |
|--------|-----------|-------------|-------------|----------|
| schema_valid | >=95% | 100% ✓ | 100% ✓ | partial ✗ |
| positive_intent | >=95% | 100% ✓ | 85.7% ✗ | partial ✗ |
| negative_constraint | >=90% | 93.8% ✓ | 62.5% ✗ | partial ✗ |
| direction | >=95% | 100% ✓ | 100% ✓ | partial ✗ |
| ambiguity | >=95% | 100% ✓ | 100% ✓ | partial ✗ |
| status_match | >=90% | 100% ✓ | 65.6% ✗ | partial ✗ |
| hallucination | =0 | 0 ✓ | 0 ✓ | 0 ✓ |
| tool_leakage | =0 | 0 ✓ | 0 ✓ | >0 ✗ |
| Vertical Slice | PASS | PASS ✓ | PASS ✓ | FAIL ✗ |
| Adapter callable | YES | NO ✗ | YES ✓ | YES ✓ |

## Classification
- **Doubao**: POC_REFERENCE_ONLY — meets thresholds but not callable
- **qwen2.5:14b**: REJECT_PRIMARY — retained as LOCAL_BASELINE_CANDIDATE
- **qwen3-vl**: REJECT_PRIMARY — hard gate failure (VS + empty content + leakage)

## Paths forward (for Chief decision)
1. **Provide Doubao/Ark API key** → thin OpenAI-compatible adapter → re-run admission
2. **Install stronger local model** (qwen2.5:32b / qwen3:32b) if hardware permits
3. **Accept qwen2.5:14b as interim** with documented semantic limitations
   (NOT recommended — negative constraint 62.5% is a safety concern)
4. **Chief provides external model endpoint** (Zhipu GLM, etc.) with API key

## NOT done (out of scope)
- No Model Router
- No fallback/voting
- No prompt tuning per model
- No benchmark modification
- No schema modification
