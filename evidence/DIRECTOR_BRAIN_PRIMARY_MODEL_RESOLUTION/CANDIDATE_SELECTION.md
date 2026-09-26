# Candidate Selection

## Decision process

1. **Hardware pre-check** → LOCAL_32B_CLASS_MODEL_FEASIBLE = MARGINAL
   - RTX 3060 12GB VRAM + 32GB RAM + 336GB disk
   - 30B Q4 model (~18GB) fits with partial GPU offloading
   - Expected: 5-12 tok/s, acceptable for offline benchmark

2. **External model check** → no viable external endpoint
   - Zhipu GLM: API key exists in gen1-roughcut/.env but **insufficient balance**
     (error code 1113: "余额不足或无可用资源包")
   - Doubao/Ark: no API key or endpoint configured in environment or .env
   - OpenAI: no API key
   - Conclusion: no externally-callable strong model with valid credentials

3. **Candidate selected**: qwen3:30b
   - Rationale: newer architecture than qwen2.5, likely better reasoning
   - Same family as proven qwen2.5:14b baseline
   - Text-only model (not vision), suitable for structured output
   - Fits within MARGINAL hardware capacity

4. **Candidates NOT tested**
   - qwen2.5:32b: same family as 14b, similar patterns; qwen3 chosen for newer arch
   - llama3.3:70b: too large for available hardware
   - mistral-small:24b: not available/preferred over qwen family
   - External models: no valid credentials

## Frozen for all candidates
- DirectorDecision schema (SHA256: B66CD897...)
- 32-case benchmark (SHA256: 834239E9...)
- Human reference (SHA256: 544E31A3...)
- Semantic prompt v1.1
- Evaluation methodology
- Admission thresholds
