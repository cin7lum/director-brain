"""L2 素材矩阵批量回归（产品表现补强 · 评估层）。

把"链路通畅"从单素材冒烟升级为跨素材证据：对 (素材 × 意图) 组合批量跑
roughcut → L1 指标，聚合成鲁棒性画像。**失败也是数据**——fail-closed 拒绝
出片的任务按"拒绝"记录并提取原因，不掩盖。

矩阵纪律（防 Goodhart）：
- 素材分 dev / holdout 两半，holdout 只在发版前跑，永不用于调参；
- 每季度轮换新素材。

用法：
    python scripts/l2_matrix.py --matrix matrix.json --outdir out/ --report report.md

matrix.json 条目：
    {"tag": "base15", "video": "a.mp4", "target_seconds": 15, "intent": null}
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.l1_metrics import collect_metrics  # noqa: E402

#: fail-closed 拒绝的原因行（从 roughcut 输出中提取）
_REFUSAL_PATTERNS = [
    r"导演放弃（evidence_too_poor）: (.+)",
    r"修复放弃（repair_requires_director）: (\S+)",
    r"\[fail-closed\] (.+)",
]


def _slug(tag: str) -> str:
    return re.sub(r"[^\w-]", "_", tag)


def run_one(cfg: dict, outdir: Path) -> dict:
    """跑一个矩阵任务，返回结果行（拒绝也是合法结果行）。"""
    tag = _slug(cfg["tag"])
    video = cfg["video"]
    target = cfg.get("target_seconds", 15)
    intent = cfg.get("intent")
    stem = Path(video).stem
    out = str(outdir / f"{stem}__{tag}.mp4")

    if not Path(video).is_file():
        return {"tag": cfg["tag"], "material": stem, "status": "missing_material",
                "reason": f"素材不存在: {video}"}

    cmd = [sys.executable, str(Path(__file__).parent / "roughcut.py"),
           "-i", video, "-o", out, "--target-duration", str(target),
           # 候选③：矩阵配置即批量授权动作——渲染闸门需策略确认；
           # 状态机 hash 绑定与账本留痕逐任务照常执法
           "--confirm-strategy"]
    if intent:
        cmd += ["--intent", intent]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          check=False)
    combined = (proc.stdout or "") + (proc.stderr or "")

    if proc.returncode != 0 or not Path(out).is_file():
        reason = "未知失败"
        for pat in _REFUSAL_PATTERNS:
            m = re.search(pat, combined)
            if m:
                reason = m.group(1).strip()[:200]
                break
        status = "refused" if any(
            re.search(p, combined) for p in _REFUSAL_PATTERNS
        ) else "error"
        return {"tag": cfg["tag"], "material": stem, "status": status,
                "roughcut_exit": proc.returncode, "reason": reason}

    edl_json = Path(f"{out}.edl.json")
    plan_json = Path(f"{out}.plan.json")
    edl = json.loads(edl_json.read_text(encoding="utf-8")) if edl_json.is_file() else None
    plan = json.loads(plan_json.read_text(encoding="utf-8")) if plan_json.is_file() else None
    m = collect_metrics(out, edl, plan, float(target))

    duration = (m["pacing"] or {}).get("total_seconds")
    dev_pct = None
    if duration and target:
        dev_pct = round(abs(duration - target) / target * 100, 1)

    return {
        "tag": cfg["tag"], "material": stem, "status": "produced",
        "video": out, "target_seconds": target,
        "intent": intent,
        "roughcut_exit": proc.returncode,
        "hard_defect": m["hard_defect"],
        "duration_s": duration,
        "duration_deviation_pct": dev_pct,
        "shots": (m["pacing"] or {}).get("shot_count"),
        "asl_s": (m["pacing"] or {}).get("asl_seconds"),
        "black_segments": len(m["black_frames"]),
        "freeze_segments": len(m["freezes"]),
        "has_audio": bool((m["streams"] or {}).get("audio")),
    }


def aggregate(rows: list[dict]) -> tuple[str, dict]:
    """聚合成 markdown 报告 + 结构化摘要。"""
    produced = [r for r in rows if r["status"] == "produced"]
    refused = [r for r in rows if r["status"] == "refused"]
    errored = [r for r in rows if r["status"] not in ("produced", "refused")]
    hard = [r for r in produced if r.get("hard_defect")]

    lines: list[str] = ["# L2 素材矩阵 · 鲁棒性画像", ""]
    lines.append(
        f"任务 {len(rows)}：出片 {len(produced)}｜拒绝 {len(refused)}｜"
        f"错误 {len(errored)}｜出片中带硬伤 {len(hard)}"
    )
    lines.append("")
    lines.append("| 任务 | 素材 | 状态 | 时长/目标 | 偏差% | 镜头 | ASL | 黑帧 | 硬伤 | 备注 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"] == "produced":
            lines.append(
                f"| {r['tag']} | {r['material']} | 出片"
                f"{'⚠硬伤' if r['hard_defect'] else '✅'} "
                f"| {r['duration_s']}s/{r['target_seconds']}s "
                f"| {r['duration_deviation_pct']}% | {r['shots']} | {r['asl_s']}s "
                f"| {r['black_segments']} | {'YES' if r['hard_defect'] else 'no'}"
                f" | {r.get('intent') or '-'} |"
            )
        elif r["status"] == "refused":
            lines.append(
                f"| {r['tag']} | {r['material']} | **拒绝** | - | - | - | - | - | - "
                f"| {r.get('reason', '')[:80]} |"
            )
        else:
            lines.append(
                f"| {r['tag']} | {r['material']} | {r['status']} | - | - | - | - | "
                f"- | - | {r.get('reason', '')[:80]} |"
            )
    lines.append("")
    lines.append("> 拒绝 = fail-closed 生效（素材/约束无法满足目标，系统宁可拒绝"
                 "也不硬凑）。拒绝原因是 L2 的重要产出：它是素材鲁棒性画像的一部分。")

    summary = {
        "total": len(rows), "produced": len(produced), "refused": len(refused),
        "errored": len(errored),
        "produced_with_hard_defect": len(hard),
        "refusal_reasons": [r.get("reason", "") for r in refused],
    }
    return "\n".join(lines), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="L2 素材矩阵批量回归")
    parser.add_argument("--matrix", required=True, help="矩阵 JSON（任务清单）")
    parser.add_argument("--outdir", required=True, help="成片输出目录")
    parser.add_argument("--report", default=None, help="markdown 报告输出路径")
    args = parser.parse_args()

    loaded = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    # 允许两种格式：任务数组，或 {"comment":..., "tasks": [...]}
    matrix = loaded["tasks"] if isinstance(loaded, dict) else loaded
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in matrix:
        print(f"[L2] 运行 {cfg['tag']} ({cfg['video']}) ...", flush=True)
        row = run_one(cfg, outdir)
        row["tag"] = cfg["tag"]
        rows.append(row)
        print(f"[L2] -> {row['status']}"
              + (f": {row.get('reason', '')[:80]}" if row.get("reason") else ""),
              flush=True)

    report, summary = aggregate(rows)
    print("\n" + report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
        json_path = Path(args.report).with_suffix(".json")
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n报告已写入: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
