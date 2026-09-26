# Director Brain Semantic Decision POC

**Base commit (02)**: cc6b25e
**Result**: DIRECTOR_BRAIN_SEMANTIC_DECISION_POC = PROVEN

## Question
Can a mature LLM with Structured Output + a thin film-specific DirectorDecision
schema reliably convert natural-language director requests into tool-agnostic,
NLE-agnostic, provider-agnostic director decision semantics?

## Answer: YES (PROVEN)

### Evaluation Results (32 benchmark samples)

| Metric | Result | Gate |
|--------|--------|------|
| Schema validity | 100.0% (32/32) | >= 95% ✓ |
| Positive intent recall | 100.0% (21/21) | no systematic loss ✓ |
| Negative constraint recall | 93.8% (15/16) | separation correct ✓ |
| Relation direction (J-cut vs L-cut) | 100.0% (8/8) | accurate ✓ |
| Ambiguity abstention | 100.0% (4/4) | abstains ✓ |
| Status match | 100.0% (32/32) | — |
| Exact parameter hallucination | 0 | zero ✓ |
| Tool/API leakage | 0 | zero ✓ |
| Vertical Slice hard case (SD-01) | PASS | ✓ |

### Vertical Slice Hard Case (the original failing input)
Input: "让下一句声音提前一点进入，让镜头切换更自然，但不要改变画面的切点，也不要加转场。"

Model output captures:
- positive: audio_precedes_picture ✓
- must_preserve: picture_cut_position ✓
- must_avoid: transition ✓
- creative_intent: 通过让下一句声音先于画面进入，使镜头切换在感知上更自然流畅 ✓
- exact_value: null (no fabricated 0.4s) ✓
- status: NEEDS_CONTEXT (explicitly asks for dialogue_onset_timing, current_cut_point, audio_waveform) ✓

### Build-vs-Adopt
Natural language understanding: ADOPT mature LLM + Structured Output.
Film-specific layer: KEEP thin DirectorDecision schema + constraints.
HeuristicDirectorReasoner: reposition as SHOT_SELECTION_SPECIALIST (not modified).

### What was NOT done
- No 02 product code modifications (0)
- No 03 product code modifications (0)
- No keyword table / regex router
- No hardcoded J_CUT → if "声音提前" mapping
- No DaVinci/MCP/OTIO tool fields in schema
- No fabricated numeric parameters from vague language
