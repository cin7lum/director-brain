"""阶段 7 · 冻结基准运行器（火山方舟/豆包）。

对 32 例冻结基准（benchmark.jsonl）逐例调用 ArkLLMAdapter（冻结
SYSTEM_PROMPT v1.1 + json_object 传输适配，与 zhipu 同族），产出
model_outputs_<model>.jsonl 供 evaluate.py 评分。

红线遵守：不改 prompt / 用例 / 金标 / 阈值；本脚本只做传输。
密钥从 ARK_API_KEY 环境变量或 .env 读取，绝不打印。

用法：
    python experiments/director_semantic_poc/run_benchmark.py \
        --model doubao-seed-2-0-lite-260428 \
        --outputs model_outputs_doubao_seed_2_0_lite.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).parent
sys.path.insert(0, str(POC_DIR.parent.parent))

from director_brain.ark_adapter import ArkLLMAdapter  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_env_file() -> None:
    """读本地 .env（不覆盖已有环境变量）。"""
    env_path = POC_DIR.parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结基准运行器（Ark）")
    parser.add_argument("--model", required=True, help="Ark 模型 ID")
    parser.add_argument("--outputs", required=True, help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 例（调试）")
    args = parser.parse_args()

    load_env_file()
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        print("错误：未设置 ARK_API_KEY", file=sys.stderr)
        return 2

    benchmark = load_jsonl(POC_DIR / "benchmark.jsonl")
    cases = benchmark[: args.limit] if args.limit else benchmark
    adapter = ArkLLMAdapter(api_key=api_key, model=args.model)
    print(f"模型: {args.model} ｜ 用例: {len(cases)} 例")

    #: 传输级重试（空内容为间歇性——疑似限流/竞态；重试不改语义层）
    MAX_ATTEMPTS = 3
    BACKOFF = [2, 6]

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
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)])
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
        time.sleep(1)  # 限流间隔

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
