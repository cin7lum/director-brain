# Root Cause Analysis

## The Deadlock

```
User: "让下一句声音提前一点进入..."
  ↓
Semantic Director (LLM)
  desired_relation = audio_precedes_picture ✓
  must_preserve = picture_cut_position ✓
  must_avoid = transition ✓
  exact_value = None (correct — "一点" is vague)
  status = READY (LLM's assessment)
  ↓
Post-validation (Phase 1 fix)
  Rule: READY + exact_value=None → NEEDS_CONTEXT
  status = NEEDS_CONTEXT  ← WRONG: semantic was clear!
  ↓
Parameterizer (with dialogue onset evidence)
  status = READY
  exact_value = 10f
  ↓
Adapter (to_arsenal_parameterization)
  Check: semantic.status == READY?
  → NO (it's NEEDS_CONTEXT)
  → returns None  ← BLOCKED
  ↓
Result: Pipeline dead. 03 never called.
```

## Why the Rule Existed

Phase 1 introduced this rule to fix a historical bug: the LLM would return
READY without any executable parameter, and the system would claim it was
ready to execute when it wasn't.

The rule was a blunt instrument: it prevented the symptom (READY + no param)
but misdiagnosed the cause (semantic unclear vs. parameter pending).

## Why It's Wrong Now

The Parameterizer exists precisely to determine exact values from context.
When the user says "提前一点" (advance a little), the semantic intent is clear:
- What: audio precedes picture
- Preserve: picture cut position
- Avoid: transition

The exact number of frames is NOT a semantic question — it's a parameterization
question. The Parameterizer should determine it from:
- Available audio handle
- Dialogue onset timing
- Creative selection evidence

Downgrading semantic status because the parameter isn't known yet prevents
the Parameterizer from ever being able to help.

## The Correct Model

```
Semantic Readiness (DirectorDecision.status)
  "Do we understand what the user wants?"
  ↓ independent ↓
Parameterization Readiness (ParameterizationDecision.status)
  "Do we know the exact execution value?"
  ↓ combine ↓
Execution Readiness (DirectorRequestResult.execution_readiness)
  "Can we hand this to 03?"
```

Semantic READY + exact_value=None is valid: it means "we understand the intent,
and we need the parameterizer to determine the value."

## Evidence

Vertical Slice V2 (commit a98e4b9) proved:
1. Semantic output was correct (all relations/constraints identified)
2. Parameterizer could determine 10f from dialogue onset
3. The only blocker was the status coupling
4. No component was actually broken — the state model was wrong
