# Real Model Identity

- Provider: ollama (local, http://localhost:11434)
- Model: qwen2.5:14b (Q4_K_M, 8.9GB)
- Model family: qwen2, 14.8B params
- Context length: 32768
- Capabilities: completion, tools
- Transport: HTTP API (/api/chat), format=JSON Schema
- Temperature: 0.1
- Timeout: 300s
- Prompt version: 1.1 (with field usage rules + status definitions)

## Other available models (not used)
- qwen2.5:7b (tested, underperformed on SD-01 relation direction)
- qwen2.5:3b (not tested for benchmark)
- qwen3-vl:latest (vision model, not suitable for text-only structured output)

## Why qwen2.5:14b
- Strongest text-only model available locally
- 7b got SD-01 relation wrong (outgoing vs incoming audio)
- 14b correctly identifies audio_precedes_picture for Vertical Slice case
