# Negative Constraint Analysis

## Overview
Negative constraint recall is the weakest metric across all callable models:
- qwen2.5:14b: 62.5% (20/32)
- glm-4-flash: 56.2% (18/32)
- Threshold: >=90%

## Common Failure Patterns

### 1. must_avoid vs desired_relation_or_change confusion
Models frequently put negative constraints (avoid_transition, avoid_audio_lead)
into the positive `desired_relation_or_change` field instead of `must_avoid`.
This causes the evaluator to miss the negative constraint.

Example (glm-4-flash SD-01):
```
desired_relation_or_change: ["audio_precedes_picture", "avoid_transition"]
must_avoid: []
```
Should be:
```
desired_relation_or_change: ["audio_precedes_picture"]
must_avoid: ["transition"]
```

### 2. must_preserve missed
Preservation constraints ("不要改变画面的切点", "保留这个人的反应镜头")
are sometimes not captured in `must_preserve`.

### 3. Negative requests parsed as positive
"不要延长这个反应" should produce must_avoid=[duration_extension] with
empty desired_relation_or_change. Models sometimes output extend_visible_duration.

### 4. Preservation vs positive action conflation
"不要改变画面的切点" is a preservation constraint (must_preserve), not a
positive action. Models sometimes put it in desired_relation_or_change as
"preserve_picture_cut".

## qwen2.5:14b Specific
From Phase-1A-R1 real model run, the only documented negative miss was SD-28
(evaluator representation difference, ISOLATED_CASE). However the 62.5% score
suggests broader misses in the formal evaluation. The POC Doubao reference
scored 93.8% (15/16 on the negative subset).

## glm-4-flash Specific
Systematic field placement errors. The model understands the semantic intent
but cannot reliably map to the correct schema fields (must_avoid vs
desired_relation_or_change vs must_preserve).

## Is This a Systematic Gap?
Yes — across both callable models, negative constraint recall is the weakest
metric. This is a model capability limitation, not a prompt or schema issue:
- The prompt explicitly defines field usage rules (lines 71-83 of SYSTEM_PROMPT)
- The schema clearly separates the three fields
- The Doubao POC reference achieves 93.8% with the same prompt and schema

## SD-28 (Conflict Case)
Input: "画面切点不要改，但把B镜头提前半秒开始。"
Expected: CONFLICTING_CONSTRAINTS
This case tests whether the model can detect conflicting constraints.
Both models tend to output READY and attempt to satisfy both constraints
rather than detecting the conflict.
