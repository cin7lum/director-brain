"""阶段 7 · 冻结基准运行器（本地 Ollama 通道）。

用途：评测器 v1.1 反证测试的强化对照——对曾经 REJECT 的本地模型
（qwen2.5:14b）用生产适配器 LLMAdapter（ollama，schema 约束生成）全量
重跑 32 例冻结基准，证明 v1.1 下仍低于准入线（差分效度）。

红线遵守：不改 prompt / 用例 / 金标 / 阈值；本脚本只做传输，与
run_benchmark.py（Ark 通道）输出同构的 jsonl。

用法：
    python experiments/director_semantic_poc/run_benchmark_local.py \
        --model qwen2.5:14b \
        --outputs model_outputs_qwen25_14b_v11_check.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).parent
sys.path.insert(0, str(POC_DIR.parent.parent))

from director_brain.llm_adapter import LLMAdapter  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结基准运行器（本地 Ollama）")
    parser.add_argument("--model", required=True, help="Ollama 模型 tag")
    parser.add_argument("--outputs", required=True, help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 例（调试）")
    args = parser.parse_args()

    benchmark = load_jsonl(POC_DIR / "benchmark.jsonl")
    cases = benchmark[: args.limit] if args.limit else benchmark
    adapter = LLMAdapter(model=args.model)
    print(f"模型: {args.model} ｜ 用例: {len(cases)} 例", flush=True)

    MAX_ATTEMPTS = 2
    out_rows: list[dict] = []
    ok = fail = 0
    latencies: list[int] = []
    t_start = time.time()
    for i, case in enumerate(cases, 1):
        row = None
        result = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result = adapter.generate_decision(case["input"], decision_id=case["id"])
            latencies.append(result.latency_ms)
            if result.decision is not None:
                row = result.decision.model_dump()
                row["decision_id"] = case["id"]  # 确保与基准 ID 对齐
                break
        if row is not None:
            ok += 1
            mark = "OK"
        else:
            # 失败也如实记录：evaluate 按 schema_invalid / missing 计入
            row = {"decision_id": case["id"], "_error": result.error,
                   "_raw": (result.raw_response or "")}
            fail += 1
            mark = "FAIL"
        out_rows.append(row)
        print(f"  [{i}/{len(cases)}] {case['id']} {mark} {result.latency_ms}ms",
              flush=True)

    with open(args.outputs, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n完成：成功 {ok} / 失败 {fail}；"
          f"延迟中位 {sorted(latencies)[len(latencies)//2]}ms；"
          f"总耗时 {time.time()-t_start:.0f}s")
    print(f"输出: {args.outputs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
