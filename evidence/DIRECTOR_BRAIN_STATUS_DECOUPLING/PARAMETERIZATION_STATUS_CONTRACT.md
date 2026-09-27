# Parameterization Status Contract

## ParameterizationDecision.status

Describes PARAMETERIZATION READINESS only: is the exact execution value determined?

## Values

| Status | Meaning |
|---|---|
| READY | exact_value is determined, feasible, and evidence-backed. |
| NEEDS_CONTEXT | Required context (audio handle, source bounds, etc.) is missing. |
| NEEDS_DECISION | Multiple feasible candidates exist, no clear selection basis. |
| CONFLICT | User-specified value conflicts with feasibility bounds. |
| UNSATISFIABLE | The requested operation is physically impossible (e.g., zero handle). |
| UNAVAILABLE | Parameterization cannot be performed for this capability/context. |

## READY Criteria

1. exact_value is not None
2. exact_value is within feasible range [min, max]
3. exact_value has evidence provenance
4. No conflict with must_preserve/must_avoid constraints

## Independence

Parameterization status is COMPLETELY INDEPENDENT from semantic status.
- Semantic READY + Parameterization NEEDS_DECISION = valid (intent clear, value pending)
- Semantic NEEDS_CONTEXT + Parameterization READY = blocked (intent unclear, value irrelevant)
