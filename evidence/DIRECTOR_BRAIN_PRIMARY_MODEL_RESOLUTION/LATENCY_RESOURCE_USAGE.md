# Latency & Resource Usage

## glm-4-flash (External API)
- Average latency: 9820 ms/case
- Range: 3981ms (SD-15) to 15986ms (SD-18)
- 32-case total: ~314 seconds (~5.2 min)
- Resource: external API, no local GPU/CPU inference cost
- Network: HTTPS to open.bigmodel.cn

## qwen2.5:14b (Local ollama)
- GPU: RTX 3060 Laptop 12GB VRAM (partial offload, model 9GB)
- CPU: i7-11800H 8-core
- RAM: ~10GB additional
- Latency: not formally measured in this round; Phase-1A-R1 reported variable
- 32-case total: estimated 10-20 min (depends on output length)

## qwen3-vl (Local ollama)
- GPU: RTX 3060 Laptop 12GB VRAM
- Latency: 20-173s/case (extremely variable, vision model overhead)
- 22% empty content rate (model returns no content)

## qwen3:30b (NOT TESTED)
- Estimated size: ~18GB Q4
- Estimated VRAM usage: would exceed 12GB, requiring heavy CPU offload
- Estimated inference speed: 2-5 tok/s (CPU-bound)
- Estimated 32-case total: 30-60+ min
- User confirmed: "32B跑不了"

## Hardware Summary
| Resource | Value |
|---|---|
| CPU | i7-11800H @2.3GHz 8C/16T |
| RAM | 31.8GB total, ~13.7GB free |
| GPU | RTX 3060 Laptop 6GB+6GB = 12GB VRAM |
| VRAM free | ~10.9GB |
| Disk free | 336GB |
| Ollama | 0.34.2 |
| OS | Windows |

## Conclusion
- External API (glm-4-flash) is fastest but semantically weakest
- Local 14B is feasible but semantically below threshold
- Local 30B+ is infeasible on this hardware
- No model achieves both adequate speed and semantic quality
