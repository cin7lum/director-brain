# Negative Test: Semantic Conflict Blocks Execution

## Request
"不要让声音提前，但让下一句声音提前10帧。"

## Expected
- Semantic: CONFLICTING_CONSTRAINTS (desired contradicts must_avoid)
- Even with exact_value=10, adapter MUST block

## Post-validation Rule
READY + desired_relation ∩ must_avoid non-empty → CONFLICTING_CONSTRAINTS

## Test
```python
d = DirectorDecision(
    status="READY",
    desired_relation_or_change=["audio_precedes_picture"],
    must_avoid=["audio_precedes_picture"],
    parameterization=Parameterization(exact_value=10.0),
)
fixed, violations = validate_director_decision(d)
assert fixed.status.value == "CONFLICTING_CONSTRAINTS"
assert len(violations) > 0
```

## Result: PASS
Semantic conflict is detected at post-validation, before parameterization.
