# Semantic Evaluation Results (32-case benchmark)

Carried over from POC, now formal regression asset.

| Metric | POC Result | Phase-1A |
|--------|-----------|----------|
| Schema validity | 100% | 100% (schema unchanged) |
| Positive intent recall | 100% | Maintained |
| Negative constraint recall | 93.8% | Maintained (SD-28 = ISOLATED_CASE) |
| Relation direction | 100% | Maintained |
| Ambiguity abstention | 100% | Maintained |
| Exact param hallucination | 0 | 0 |
| Tool leakage | 0 | 0 |
| Vertical Slice hard case | PASS | PASS (I3 tests) |

The 32-case benchmark remains at:
experiments/director_semantic_poc/benchmark.jsonl
experiments/director_semantic_poc/human_reference.jsonl

Formal regression uses I1-I12 unit tests covering the same dimensions.
