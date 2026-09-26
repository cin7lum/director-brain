# Negative Constraint Miss Analysis — SD-28

## Sample
Input: "画面切点不要改，但把B镜头提前半秒开始。"

## Reference (human)
- status: CONFLICTING_CONSTRAINTS
- negative_constraints: ["conflicting_constraints"]
- positive_intent: null

## Model output
- status: CONFLICTING_CONSTRAINTS ✓
- must_preserve: ["picture_cut_position"] ✓
- parameterization: exact_value=0.5, unit=seconds (explicit from user) ✓
- required_context: ["clarification_of_conflicting_requirements"] ✓
- desired_relation_or_change: [] (no positive action, correctly) ✓

## Evaluator miss
The deterministic evaluator checked for "conflicting" substring in
`must_avoid` / `desired_relation_or_change` / `must_preserve` text fields.
The model expressed the conflict via `status = CONFLICTING_CONSTRAINTS`
instead of putting "conflicting_constraints" in must_avoid.

## Classification: ISOLATED_CASE

This is an evaluator representation mismatch, NOT a semantic gap:
- The model correctly identified the conflict (status = CONFLICTING_CONSTRAINTS)
- The model correctly preserved picture_cut_position
- The model correctly did not choose one constraint over the other
- The model correctly requested clarification

The evaluator's negative_constraint field was too narrow — it only checked
text fields, not the status field. This is a test-harness issue, not a
model/schema issue.

## Action
No schema change needed. No keyword patch needed. The evaluator in the
formal regression will check status field for conflict cases.
