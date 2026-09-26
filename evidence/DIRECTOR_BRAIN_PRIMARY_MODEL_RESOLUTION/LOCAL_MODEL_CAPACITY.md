# Local Model Capacity Assessment

## Hardware
| Component | Value |
|-----------|-------|
| CPU | 11th Gen Intel Core i7-11800H @ 2.30GHz, 8 cores / 8 threads |
| RAM | 31.8 GB total, ~13.7 GB free at assessment time |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 12288 MiB (12 GB), ~10.9 GB free |
| Disk D: | 336 GB free |
| OS | Windows |
| Ollama | 0.34.2 |

## Installed models (pre-gate)
- qwen2.5:3b (1.9 GB)
- qwen2.5:7b (4.7 GB)
- qwen2.5:14b (9.0 GB)
- qwen3-vl:latest (6.1 GB)

## 32B-class feasibility analysis

### VRAM capacity
- RTX 3060 Laptop: 12 GB VRAM
- qwen3:30b Q4_K_M: ~18 GB download, ~18-20 GB runtime
- VRAM can hold ~60-65% of model layers
- Remaining ~35-40% offloaded to system RAM

### RAM capacity
- 31.8 GB total, ~13.7 GB free
- Offloaded layers (~7 GB) + OS + other processes fit within free RAM
- No page file thrashing expected

### Disk capacity
- 336 GB free, model download ~18 GB → sufficient

### Expected inference speed
- Partial GPU offloading: estimated 5-12 tokens/sec
- 32 benchmark cases × ~300-500 tokens = 10000-16000 tokens
- Estimated total benchmark time: 15-45 minutes
- Acceptable for offline benchmark, NOT suitable for interactive production use

### Verdict
**LOCAL_32B_CLASS_MODEL_FEASIBLE = MARGINAL**

- Feasible: model fits in VRAM+RAM, disk sufficient, ollama supports partial offloading
- Marginal: significant CPU offloading will reduce speed; 12GB VRAM is below
  the ~20GB needed for full GPU inference of a 30B Q4 model
- Risk: system memory pressure if other applications run concurrently
- Not feasible for: 70B-class models, full GPU offloading, production interactive latency

## External model options
- Zhipu GLM API: key exists in gen1-roughcut/.env but **insufficient balance**
  (error: "余额不足或无可用资源包")
- Doubao/Ark API: no API key or endpoint configured
- OpenAI API: no API key configured
- Conclusion: no externally-callable strong model available with valid credentials
