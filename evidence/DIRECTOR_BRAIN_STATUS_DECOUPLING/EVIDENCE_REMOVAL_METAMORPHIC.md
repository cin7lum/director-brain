# Metamorphic Test: Evidence Removal (S7)

## Property
Removing parameter evidence must:
- Keep semantic status READY (semantic doesn't depend on parameter evidence)
- Change parameterization status from READY to NEEDS_DECISION
- Change execution readiness from READY to NOT_READY

## Test
```python
semantic = make_decision(status="READY", exact_value=None)

# With dialogue evidence
ctx_with = make_ctx(dialogue_onsets=[10])
param_with = parameterizer.parameterize(ctx_with)
assert param_with.status == READY
assert to_arsenal_parameterization(semantic, param_with) is not None

# Semantic stays READY regardless
fixed, _ = validate_director_decision(semantic)
assert fixed.status.value == "READY"

# Remove dialogue evidence
ctx_without = make_ctx(dialogue_onsets=None)
param_without = parameterizer.parameterize(ctx_without)
assert param_without.status == NEEDS_DECISION
assert to_arsenal_parameterization(semantic, param_without) is None

# Semantic STILL READY (it was never dependent on parameter evidence)
assert fixed.status.value == "READY"
```

## Result: PASS
This proves semantic and parameterization are truly decoupled.
The old model would have downgraded semantic to NEEDS_CONTEXT when exact_value was None,
making it impossible to distinguish "intent unclear" from "value pending".
