# Model Identities

## Candidates Tested

### 1. qwen2.5:14b (LOCAL_BASELINE_CANDIDATE)
- Runtime: ollama 0.34.2, local
- Size: 9.0 GB
- Quantization: Q4_K_M (default ollama)
- API: ollama HTTP /api/chat with format=JSON Schema
- Source: official Qwen2.5 series
- Status: REJECT_PRIMARY / LOCAL_BASELINE_CANDIDATE
- Metrics: schema 100%, positive 85.7%, negative 62.5%, direction 100%, ambiguity 100%, status 65.6%, hallucination 0, tool leakage 0

### 2. qwen3-vl:latest (REJECT_PRIMARY_FOR_TEXT_SEMANTIC_REASONING)
- Runtime: ollama 0.34.2, local
- Size: 6.1 GB
- API: ollama HTTP /api/chat
- Source: official Qwen3-VL series (vision-language model)
- Status: REJECT_PRIMARY_FOR_TEXT_SEMANTIC_REASONING
- Issues: 22% empty content rate, Vertical Slice SD-01 FAIL, tool leakage ("J-cut"), latency 20-173s/case
- Note: Vision model not suited for text-only structured semantic reasoning

### 3. glm-4-flash (EXTERNAL_FREE_TIER)
- Runtime: Zhipu OpenAI-compatible API (https://open.bigmodel.cn/api/paas/v4)
- Model identity: glm-4-flash
- API: chat/completions with response_format=json_object
- Transport: director_brain/zhipu_adapter.py (thin adapter)
- Schema constraint: json_object mode only (no full JSON Schema); canonical schema included in system prompt as transport adaptation
- Runtime validation: DirectorDecision.model_validate_json()
- Status: REJECT
- Metrics: schema 100%, positive 62.5%, negative 56.2%, direction 84.4%, ambiguity 62.5%, status 46.9%, hallucination 0, tool leakage 1 real (SD-24 "otio"), avg latency 9820ms

### 4. Doubao (POC_REFERENCE_ONLY)
- Runtime: Not formally callable (no API key/endpoint)
- Original POC used Doubao model via interactive session
- Status: REFERENCE_OUTPUT / FEASIBILITY_REFERENCE (NOT verified model result)
- POC metrics: schema 100%, positive 100%, negative 93.8%, direction 100%, ambiguity 100%, status 100%, hallucination 0, tool leakage 0
- Note: Per task instruction, Doubao POC results must be labeled REFERENCE_OUTPUT, not verified model result. If a real Doubao/Ark endpoint becomes available, 32 cases must be re-run from scratch.

## Candidates NOT Tested

### qwen3:30b (LOCAL)
- Size: ~18 GB (Q4)
- Reason: Hardware infeasible. RTX 3060 12GB VRAM + 32GB RAM = MARGINAL at best; user explicitly confirmed "32B跑不了". Partial GPU offloading would result in extremely slow inference (estimated 2-5 tok/s), making 32-case benchmark impractical (estimated 30-60+ min). Download was initiated then stopped per user instruction.

### glm-4-air / glm-4-plus (EXTERNAL_PAID)
- Reason: Zhipu API returns 429 error code 1113 "余额不足或无可用资源包" for all paid models. Only glm-4-flash free tier is accessible.

### Other external providers (OpenAI, SiliconFlow, Groq, Together, etc.)
- Reason: No API keys found in any .env, config, or environment variable across D:\新建豆包\.
