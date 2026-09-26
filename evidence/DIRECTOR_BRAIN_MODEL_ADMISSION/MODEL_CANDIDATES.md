# Model Candidates

## Candidate 1: Doubao (POC model)
- **Source**: POC Semantic Decision (commit 8ef3b28)
- **POC metrics**: schema 100%, positive 100%, negative 93.8%, direction 100%,
  ambiguity 100%, status 100%, hallucination 0, tool leakage 0, VS PASS
- **Production adapter callable**: NO
- **Reason**: No API key/endpoint. POC used assistant direct generation.
  Environment has no DOUBAO_API_KEY, ARK_API_KEY, OPENAI_API_KEY.
  .env only contains VLM config (ollama).
- **Classification**: POC_REFERENCE_ONLY
- **Action needed**: Chief provides API credentials → thin OpenAI-compatible adapter

## Candidate 2: qwen2.5:14b
- **Source**: ollama local, Q4_K_M, 8.9GB
- **Production path**: SemanticDirectorService → SemanticDirectorReasoner → LLMAdapter ✓
- **Metrics**: schema 100%, positive 85.7%, negative 62.5%, direction 100%,
  ambiguity 100%, status 65.6%, hallucination 0, tool leakage 0, VS PASS
- **Avg latency**: ~4.8s/case
- **Classification**: REJECT_PRIMARY / LOCAL_BASELINE_CANDIDATE
- **Retained for**: offline development, testing, baseline comparison

## Candidate 3: qwen3-vl:latest
- **Source**: ollama local, vision-language model with thinking capability
- **Production path**: SemanticDirectorService → SemanticDirectorReasoner → LLMAdapter ✓
- **Results (18/32 partial, stopped early)**:
  - SD-01 (Vertical Slice): FAILED — empty content
  - Empty content failures: 4/18 (22%)
  - Tool leakage: "J-cut" in SD-06 creative_intent
  - Positive relation misses: SD-05, SD-07, SD-10, SD-11, SD-13
  - Latency: 20-173s per case
- **Classification**: REJECT_PRIMARY
- **Reason**: Vision model not optimized for text-only structured output;
  thinking mode causes empty responses under JSON schema constraint;
  Vertical Slice hard case fails.

## Candidates NOT tested
- qwen2.5:7b: Briefly tested in R1, got SD-01 relation direction wrong. Not re-tested.
- qwen2.5:3b: Too small for semantic reasoning, not expected to meet threshold.
- External models (Zhipu GLM, etc.): No API key available.
