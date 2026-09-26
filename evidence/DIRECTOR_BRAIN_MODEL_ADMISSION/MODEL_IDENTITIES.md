# Model Identities

## qwen2.5:14b
- Provider: ollama (local, http://localhost:11434)
- Model: qwen2.5:14b
- Family: qwen2, 14.8B params
- Quantization: Q4_K_M (8.9GB)
- Context: 32768
- Capabilities: completion, tools
- Transport: HTTP /api/chat, format=JSON Schema
- Temperature: 0.1

## qwen3-vl:latest
- Provider: ollama (local, http://localhost:11434)
- Model: qwen3-vl:latest
- Type: Vision-Language, thinking-enabled
- Transport: HTTP /api/chat, format=JSON Schema
- Note: JSON schema format causes empty responses for ~22% of cases

## Doubao (POC reference)
- Provider: Assistant (direct generation, not API)
- Model identity: not formally versioned in POC
- POC commit: 8ef3b28
- Not callable through LLMAdapter
