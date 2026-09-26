# Real Model 32-Case Results (qwen2.5:14b, production adapter)

## Metrics
- Schema validity: 100.0% (32/32)
- Positive intent recall: 85.7% (18/21)
- Negative constraint recall: 62.5% (10/16)
- Relation direction accuracy: 100.0% (8/8)
- Ambiguity abstention: 100.0% (4/4)
- Status match: 65.6% (21/32)
- Exact parameter hallucination: 0
- Tool leakage: 0
- Failures: 0

## Status mismatches (11)
The model tends to classify vague-but-actionable requests as READY instead of
NEEDS_CONTEXT. This is a model calibration issue, not an architecture issue.
Human gold standard requires NEEDS_CONTEXT when exact timing parameters are unspecified.

## Negative constraint analysis
Negative constraint recall is 62.5%.
Misses fall into categories:
1. Term matching: model uses "audio_precedes_picture" in must_avoid where reference
   expects "avoid_audio_lead" — same semantic, different vocabulary
2. Multi-constraint sentences: model captures primary intent but misses secondary
   negative constraints (SD-31: captured extend, missed no-transition/no-reorder)
3. Conflict detection: SD-28 conflict not detected (model said READY)

No keyword patches were added. Prompt was improved with field usage instructions
(negative → must_avoid, preservation → must_preserve), which is legitimate
adapter configuration, not keyword matching.
