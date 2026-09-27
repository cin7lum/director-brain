# Execution Readiness Contract

## DirectorRequestResult.execution_readiness

Derived from semantic + parameterization status. Answers: "Can this be handed to 03?"

## Computation Rules

| Semantic | Parameterization | execution_readiness |
|---|---|---|
| READY | READY | READY |
| READY | NEEDS_CONTEXT | WAITING_FOR_CONTEXT |
| READY | NEEDS_DECISION | WAITING_FOR_DECISION |
| READY | CONFLICT | BLOCKED |
| READY | UNSATISFIABLE | BLOCKED |
| READY | UNAVAILABLE | NOT_READY |
| READY | (none provided) | READY* |
| NEEDS_CONTEXT | any | NEEDS_CONTEXT |
| UNDERSPECIFIED | any | UNDERSPECIFIED |
| CONFLICTING_CONSTRAINTS | any | BLOCKED |
| UNSUPPORTED | any | BLOCKED |

*When no parameterization context is provided, execution readiness defaults to READY
because the semantic decision is clear. Parameters may be obtained later or passed
through from the semantic decision's own parameterization field.

## Adapter Gate

`from_service_result(result)` returns 03 parameterization payload ONLY when
`result.execution_readiness == "READY"`.

This is the SINGLE canonical gate. No duplicate status logic in the adapter.
