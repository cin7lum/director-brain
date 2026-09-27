# Adapter Gate Results

## to_arsenal_parameterization(semantic, parameterization)

| Test | Semantic | Parameterization | Result |
|---|---|---|---|
| Both READY | READY | READY (10f) | dict with exact_value=10.0 |
| Param NEEDS_DECISION | READY | NEEDS_DECISION | None |
| Semantic NEEDS_CONTEXT | NEEDS_CONTEXT | READY (10f) | None |
| Param UNSATISFIABLE | READY | UNSATISFIABLE | None |
| Semantic CONFLICT | CONFLICTING_CONSTRAINTS | READY | None |
| No param, semantic has value | READY (exact=10) | None | dict with exact_value=10.0 |
| No param, semantic no value | READY (exact=None) | None | None |

## from_service_result(result)

| execution_readiness | Result |
|---|---|
| READY | 03 parameterization dict |
| WAITING_FOR_DECISION | None |
| WAITING_FOR_CONTEXT | None |
| NEEDS_CONTEXT | None |
| BLOCKED | None |

## Key Invariant
The adapter NEVER produces a payload when:
- Semantic is not READY (intent unclear → unsafe to execute)
- Parameterization is not READY (no exact value → nothing to execute)

Both conditions are required. Neither alone is sufficient.
