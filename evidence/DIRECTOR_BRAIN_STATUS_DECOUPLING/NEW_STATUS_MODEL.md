# New Status Model (Status Decoupling)

## Three Independent Status Concepts

### 1. Semantic Readiness
- **Owner**: Semantic Director + post-validation
- **Field**: `DirectorDecision.status`
- **Question**: "Do we understand what the user wants?"
- **Values**: READY, NEEDS_CONTEXT, UNDERSPECIFIED, UNSUPPORTED, CONFLICTING_CONSTRAINTS
- **Does NOT require**: exact_value, parameterization context, audio evidence

### 2. Parameterization Readiness
- **Owner**: JCutParameterizer
- **Field**: `ParameterizationDecision.status`
- **Question**: "Do we know the exact execution value?"
- **Values**: READY, NEEDS_CONTEXT, NEEDS_DECISION, CONFLICT, UNSATISFIABLE, UNAVAILABLE
- **Does NOT affect**: semantic status

### 3. Execution Readiness
- **Owner**: SemanticDirectorService
- **Field**: `DirectorRequestResult.execution_readiness` (and `.status` for backward compat)
- **Question**: "Can we hand this to 03?"
- **Derived from**: semantic + parameterization
- **Values**: READY, WAITING_FOR_CONTEXT, WAITING_FOR_DECISION, BLOCKED, NOT_READY, + semantic passthrough

## Post-validation: Semantic Only

The post-validator now only checks SEMANTIC consistency:
1. READY + no desired_relation_or_change → NEEDS_CONTEXT
2. READY + desired_relation contradicts must_avoid → CONFLICTING_CONSTRAINTS
3. CONFLICTING_CONSTRAINTS + exact_value → warning (value ignored)

It NEVER checks parameter completeness. It NEVER creates values. It NEVER clamps.

## Adapter: Both Required

`to_arsenal_parameterization(semantic, parameterization)` returns a payload ONLY when:
- semantic.status == READY
- parameterization.status == READY (or semantic has exact_value and no param decision)

Neither condition alone is sufficient.

## Service: Canonical Readiness

`from_service_result(result)` uses `result.execution_readiness == "READY"` as the
single gate. No duplicate status logic in the adapter.

## Key Invariant

**Parameterization READY cannot rescue Semantic not-READY.**
If the user's intent is unclear, having an exact number doesn't make execution safe.

**Semantic READY does not require exact_value.**
The parameterizer determines exact values from context.
