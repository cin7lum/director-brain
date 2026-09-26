# Admission Decision

## Final Verdict
**DIRECTOR_BRAIN_PRIMARY_MODEL_RESOLUTION = NO_MODEL_ADMITTED**

## Reasoning

### All callable models fail the Admission Gate

| Model | Failing Metrics |
|---|---|
| qwen2.5:14b | positive_intent (85.7% < 95%), negative_constraint (62.5% < 90%), status_match (65.6% < 90%), VS hard case (PARTIAL) |
| qwen3-vl | schema_valid (~78% < 95%), empty content 22%, tool leakage, VS hard case FAIL |
| glm-4-flash | positive_intent (62.5% < 95%), negative_constraint (56.2% < 90%), direction (84.4% < 95%), ambiguity (62.5% < 95%), status_match (46.9% < 90%), tool leakage, VS hard case (PARTIAL) |

### Models not available
- Doubao: meets thresholds in POC but no API key for formal adapter call
- glm-4-air/plus: insufficient API balance
- qwen3:30b: hardware infeasible (user confirmed)
- Other providers: no credentials found

### What was NOT done (per task constraints)
- Thresholds NOT lowered
- No keyword patches added
- No model router / fallback / voting built
- No semantic postprocessor or rule-based repair written
- No prompt overfitting to benchmark cases
- No Human Gold modified
- No benchmark cases removed or simplified

### Root cause
This is a **model capability gap**, not an architecture or prompt gap:
- The frozen prompt and schema are validated by the Doubao POC reference (all metrics pass)
- The SemanticDirectorReasoner architecture is proven (Phase-1A-R1 = ARCHITECTURE_PASS)
- The gap is that no currently-callable model has sufficient semantic reasoning
  ability to reliably separate positive intent from negative constraints and
  correctly handle parameterization uncertainty

### Recommended next steps (for Chief decision)
1. Obtain a valid Doubao/Ark API endpoint and re-run 32-case benchmark via formal adapter
2. OR obtain API balance for Zhipu glm-4-air/plus (stronger than glm-4-flash)
3. OR evaluate other external providers with valid credentials
4. OR Chief decides whether to revisit the product strategy / admission thresholds
5. Local hardware upgrade (GPU with >=24GB VRAM) could enable 30B+ class models
