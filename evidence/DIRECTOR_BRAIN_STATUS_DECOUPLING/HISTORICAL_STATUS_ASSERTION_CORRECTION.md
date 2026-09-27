# Historical Status Assertion Correction

## Summary

Phase 1 introduced a post-validation rule: `READY + exact_value=None → NEEDS_CONTEXT`.
This was intended to prevent the historical bug where the LLM returns READY without
an executable parameter. However, this rule conflated two independent concepts:

1. **Semantic readiness**: Is the user's intent understood? (desired_relation, must_preserve, must_avoid)
2. **Parameterization readiness**: Is the exact execution value determined? (exact_value)

The rule treated "parameter missing" as "semantic understanding incomplete", which is incorrect.
The parameterizer exists precisely to determine exact values from context.

## Assertions Updated

### 1. `test_ready_with_none_exact_downgraded` → `test_ready_with_none_exact_stays_ready`

**Old assertion**: `READY + exact_value=None → NEEDS_CONTEXT`
**New assertion**: `READY + exact_value=None → READY` (semantic clear, parameterizer fills value)
**Reason**: Vertical Slice V2 proved this rule caused a deadlock: semantic was downgraded to
NEEDS_CONTEXT even though the parameterizer could determine the exact value from dialogue onset
evidence. The adapter then blocked because semantic != READY.

### 2. `TestStatusInvariants` docstring

**Old**: "READY + exact_value=None must be invalid for parameter-required capabilities."
**New**: "Semantic status invariants after decoupling from parameterization."

### 3. `test_non_parameter_capability_ready_kept` — REMOVED

This test was redundant after the decoupling. READY without exact_value is now valid
for ALL capabilities (parameterization is handled separately), not just non-parameter ones.

### 4. `test_r1_5_service_wires_reasoner` mock — UPDATED

**Old mock**: `DirectorDecision(decision_id="t", creative_intent="t", status=READY)`
(no desired_relation_or_change)
**New mock**: Added `desired_relation_or_change=["audio_precedes_picture"]`,
`must_preserve`, `must_avoid`
**Reason**: New post-validation correctly downgrades READY with no desired_relation to
NEEDS_CONTEXT. The mock was incomplete — it tested service wiring with a semantically
invalid decision.

## Assertions Preserved (Not Changed)

- `test_ready_with_exact_kept` — READY with exact_value stays READY ✓
- `test_needs_context_unchanged` — NEEDS_CONTEXT stays NEEDS_CONTEXT ✓
- `test_post_validator_never_fills_value` — post-validator never creates exact_value ✓
- `test_param_not_ready_returns_none` — adapter blocks when parameterization not ready ✓
- `test_semantic_not_ready_returns_none` — adapter blocks when semantic not ready ✓
- `test_is_execution_ready_matches` — is_execution_ready matches adapter ✓
- `test_no_param_decision_semantic_ready_with_value` — pass-through when semantic has value ✓
- All 16-case POC regression — parameterizer behavior unchanged ✓
- All metamorphic tests M1-M5 — parameterizer behavior unchanged ✓
- All evidence removal tests — parameterizer evidence dependency unchanged ✓
- All conflict/no-clamping tests — parameterizer conflict behavior unchanged ✓

## New Assertions Added

### Semantic-only validation
- `test_ready_without_desired_relation_downgraded` — READY + no desired_relation → NEEDS_CONTEXT
- `test_ready_with_semantic_conflict_downgraded` — READY + desired contradicts avoid → CONFLICTING_CONSTRAINTS
- `test_conflicting_with_exact_value_warned` — CONFLICTING + exact_value → warning

### Status decoupling S1-S7
- S1: Semantic complete + no parameter → Semantic READY
- S2: Semantic complete + Parameterizer NEEDS_DECISION → execution not ready
- S3: Semantic complete + Parameterizer READY → execution ready
- S4: Semantic incomplete + Parameterizer READY → blocked
- S5: Semantic conflict + Parameterizer READY → blocked
- S6: Semantic ready + parameter infeasible → blocked
- S7: Removing parameter evidence → semantic stays READY → execution NOT_READY

## Evidence for Correction

Vertical Slice V2 (commit a98e4b9) proved:
1. Semantic Director correctly outputs `audio_precedes_picture`, `preserve picture_cut_position`, `avoid transition`
2. LLM returns READY + exact_value=None (correct — user said "一点", not a specific number)
3. Post-validation downgrades to NEEDS_CONTEXT (incorrect — semantic was clear)
4. Parameterizer with external dialogue onset returns READY exact_value=10 (correct)
5. Adapter blocks because semantic.status=NEEDS_CONTEXT (incorrect block)
6. Result: full product pipeline cannot complete even though all components work correctly

The root cause was the status coupling, not any component failure.
