# Old Status Model (Phase 1 — Now Superseded)

## Single Status Field

`DirectorDecision.status` was used to express ALL of:
- Semantic understanding clarity
- Parameter completeness
- Execution readiness

## The Problematic Rule

```python
# post_validation.py (old)
if decision.status == "READY":
    if needs_param and param.exact_value is None:
        decision.status = NEEDS_CONTEXT  # WRONG
```

This rule conflated:
- "Parameter not yet determined" → "Semantic intent not understood"

## Consequences

1. **False NEEDS_CONTEXT**: A clear semantic decision with vague parameters
   was marked as needing more context, even though the context needed was
   parameterization context, not semantic context.

2. **Adapter deadlock**: The adapter required semantic.status == READY, but
   post-validation always downgraded to NEEDS_CONTEXT when exact_value was
   None. Since vague user requests always have exact_value=None, the adapter
   could never proceed.

3. **Parameterizer couldn't help**: Even when the Parameterizer successfully
   determined the exact value from context, the semantic status was already
   downgraded and the adapter wouldn't look at the parameterization result.

4. **Status meaning was ambiguous**: `NEEDS_CONTEXT` could mean:
   - "We don't understand what the user wants" (semantic)
   - "We don't have the audio waveform data" (parameterization)
   - "We don't know the exact frame count" (parameterization)
   These are very different problems requiring different solutions.

## Historical Context

This rule was added in Phase 1 to fix a real bug: the LLM would return READY
without any executable parameter, and the system would falsely claim readiness.
The fix was correct for its time (before the Parameterizer existed), but it
became incorrect once the Parameterizer was introduced as the component
responsible for determining exact values.
