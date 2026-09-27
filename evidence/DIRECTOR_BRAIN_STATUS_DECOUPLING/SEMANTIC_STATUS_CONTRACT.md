# Semantic Status Contract

## DirectorDecision.status

Describes SEMANTIC READINESS only: is the user's intent understood clearly enough to act on?

## Values

| Status | Meaning |
|---|---|
| READY | Desired relation/change, target, must_preserve, must_avoid are all clear. No semantic conflict. |
| NEEDS_CONTEXT | More semantic context needed to understand the intent. |
| UNDERSPECIFIED | The request is too vague to determine even the desired relation. |
| UNSUPPORTED | The requested operation is not in the supported capability set. |
| CONFLICTING_CONSTRAINTS | The request contains contradictory semantic constraints. |

## READY Criteria

A DirectorDecision is semantically READY when:
1. `desired_relation_or_change` is non-empty
2. No contradiction between `desired_relation_or_change` and `must_avoid`
3. `target` semantics are clear (or inferable from relation)
4. No other semantic contradiction

## What READY Does NOT Require

- exact_value (that's parameterization)
- audio waveform data (that's context acquisition)
- dialogue onset detection (that's context acquisition)
- feasible range computation (that's parameterization)
- execution capability availability (that's 03 admission)

## Post-validation Rules

1. READY + empty desired_relation_or_change → NEEDS_CONTEXT
2. READY + desired ∩ must_avoid non-empty → CONFLICTING_CONSTRAINTS
3. CONFLICTING_CONSTRAINTS + exact_value → warning (value ignored)
