# Latency Results

## qwen2.5:14b
- Average: ~4.8s per case
- Range: 4.3s - 6.2s
- 32 cases total: ~154s
- Suitable for interactive use (sub-10s)

## qwen3-vl
- Average (18 cases): ~70s per case
- Range: 21.8s - 173.3s
- 18 cases processed: ~1260s (21 min)
- Projected 32 cases: ~37 min
- NOT suitable for interactive use
- Thinking mode adds significant latency

## Doubao (POC)
- Not measured through API (assistant direct generation)
- Estimated: 2-5s per case (typical LLM API response)
