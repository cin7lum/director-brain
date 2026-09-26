# Semantic Evaluation Results

## Metrics
- schema_valid_rate: 100.0%
- positive_intent_recall: 100.0%
- negative_constraint_recall: 93.8%
- relation_direction_accuracy: 100.0%
- ambiguity_abstention_accuracy: 100.0%
- status_match_accuracy: 100.0%
- exact_parameter_hallucination_count: 0
- tool_leakage_count: 0
- vertical_slice_case_pass: True

## Categories covered
- A: Audio/picture temporal relation (8 samples: J-cut, L-cut, negative)
- B: Shot duration (5 samples: hold, shorten, explicit)
- C: Reorder/narrative (3 samples)
- D: Transition constraints (3 samples)
- E: Identity/preservation (3 samples)
- F: Ambiguous requests (4 samples)
- G: Unsupported (1 sample)
- H: Conflicting constraints (1 sample)
- I: Mixed positive+negative (1 sample)
- Paraphrase coverage: same semantic intent expressed in 5+ different ways
