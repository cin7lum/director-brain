# Negative Test: Semantic Underspecified Cannot Be Rescued by Parameterization

## Request
"自然一点。" (Just "make it more natural.")

## Expected
- Semantic: NEEDS_CONTEXT / UNDERSPECIFIED (no desired relation, no target)
- Even if parameterization could produce a value, adapter MUST block

## Test
```python
# Semantic incomplete (no desired_relation)
semantic = DirectorDecision(
    status="NEEDS_CONTEXT",
    desired_relation_or_change=[],
    ...
)
# Parameterization READY (hypothetical)
param = ParameterizationDecision(status="READY", exact_value=10.0)
# Adapter MUST return None
assert to_arsenal_parameterization(semantic, param) is None
```

## Result: PASS
Parameterization READY cannot rescue semantic incompleteness.
The adapter requires BOTH semantic READY AND parameterization READY.
