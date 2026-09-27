# Semantic ↔ Parameterization ↔ Execution Status Matrix

## Three Independent Status Concepts

| Concept | Owner | Field | Values |
|---|---|---|---|
| **Semantic Readiness** | Semantic Director + post-validation | `DirectorDecision.status` | READY, NEEDS_CONTEXT, UNDERSPECIFIED, UNSUPPORTED, CONFLICTING_CONSTRAINTS |
| **Parameterization Readiness** | JCutParameterizer | `ParameterizationDecision.status` | READY, NEEDS_CONTEXT, NEEDS_DECISION, CONFLICT, UNSATISFIABLE, UNAVAILABLE |
| **Execution Readiness** | SemanticDirectorService | `DirectorRequestResult.execution_readiness` / `.status` | READY, WAITING_FOR_CONTEXT, WAITING_FOR_DECISION, BLOCKED, NOT_READY, + semantic status passthrough |

## Matrix

| Semantic | Parameterization | Execution Readiness | Adapter Output |
|---|---|---|---|
| READY | READY | **READY** | 03 parameterization dict |
| READY | NEEDS_CONTEXT | WAITING_FOR_CONTEXT | None |
| READY | NEEDS_DECISION | WAITING_FOR_DECISION | None |
| READY | CONFLICT | BLOCKED | None |
| READY | UNSATISFIABLE | BLOCKED | None |
| READY | UNAVAILABLE | NOT_READY | None |
| READY | (none provided) | READY* | None (unless semantic has exact_value) |
| NEEDS_CONTEXT | any | NEEDS_CONTEXT | None |
| UNDERSPECIFIED | any | UNDERSPECIFIED | None |
| UNSUPPORTED | any | BLOCKED | None |
| CONFLICTING_CONSTRAINTS | any | BLOCKED | None |

*When no parameterization context is provided, execution readiness defaults to READY
because the semantic decision is clear and parameters may be obtained later or passed
through from the semantic decision's own parameterization field.

## Key Invariants

1. **Parameterization READY cannot rescue Semantic not-READY.** If the user's intent is
   unclear, having an exact number doesn't make execution safe.

2. **Semantic READY does not require exact_value.** The parameterizer determines exact
   values from context. exact_value=None on a clear semantic decision is valid.

3. **Post-validation only checks semantic consistency.** It never checks parameter
   completeness, never creates values, never clamps.

4. **Adapter requires BOTH semantic READY and parameterization READY.** Neither alone
   is sufficient.

5. **Execution readiness is derived, never stored.** It is computed from the current
   semantic + parameterization state each time.

## Hard Case Verification

User request: "让下一句声音提前一点进入，让镜头切换更自然，但不要改变画面的切点，也不要加转场。"

- **Semantic**: READY (audio_precedes_picture, preserve picture_cut_position, avoid transition)
- **Without dialogue evidence**: Parameterization = NEEDS_DECISION → Execution = WAITING_FOR_DECISION
- **With external dialogue onset (frame 10)**: Parameterization = READY (10f) → Execution = READY → Adapter emits 03 params

This is the correct behavior: the system understands the intent, acknowledges it needs
either human decision or audio evidence to pick the exact value, and proceeds only when
both intent and parameters are clear.
