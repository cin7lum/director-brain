# LLM Adapter

## Location
`director_brain/llm_adapter.py`

## Primary model path
- ollama HTTP API (http://localhost:11434/api/chat)
- format=json for structured output
- temperature=0.1 (low variance for semantic parsing)
- Default model: llama3.2:3b (override via constructor)

## Fail-closed behavior
- Connection error → SEMANTIC_REASONER_UNAVAILABLE
- Invalid JSON → SEMANTIC_REASONER_UNAVAILABLE
- Schema validation failure → SEMANTIC_REASONER_UNAVAILABLE
- Timeout → SEMANTIC_REASONER_UNAVAILABLE
- NO silent fallback to keyword parser

## Evidence recorded
- model identity
- prompt version (1.0)
- latency_ms
- raw_response (truncated)
- validation errors

## Not implemented (by design)
- Model router
- Multi-model voting
- Retry logic
- Fallback models
- Streaming
