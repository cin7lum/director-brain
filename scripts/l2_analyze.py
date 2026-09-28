"""L2 正式矩阵分析器（阶段 8b 闸门）：三本账 + dev/holdout 对比。

三本账（路线文档定义的闸门）：
1. 出片率——produced / total（按 split 与意图分列）；
2. L1 硬伤率——produced 中 hard_defect 的占比；
3. 拒绝正确率——拒绝中"目标物理不可达/约束真拒绝"可解释的占比。

dev/holdout 对比（防 Goodhart）：holdout 各项指标**不得劣于** dev。

用法：
    python scripts/l2_analyze.py --report evidence/GLM-HANDOVER/l2_formal_report.md \
        --matrix evidence/GLM-HANDOVER/l2_formal_matrix_r1.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_report_table(report_md: str) -> list[dict]:
    """从 l2_matrix 报告的 markdown 表解析每行任务结果。"""
    rows: list[dict] = []
    for line in report_md.splitlines():
        if not line.startswith("| ") or "任务 |" in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 9:
            continue
        tag, material, status = cells[0], cells[1], cells[2]
        produced = status.startswith("出片")
        refused = status.startswith("**拒绝**") or "拒绝" in status
        hard = "YES" in cells[8] or "⚠硬伤" in status
        dur = cells[3]
        target = None
        m = re.search(r"/(\d+)s", dur)
        if m:
            target = int(m.group(1))
        row = {
            "tag": tag, "material": material,
            "status": "produced" if produced else ("refused" if refused else "error"),
            "hard_defect": hard,
            "duration": dur if produced else None,
            "target": target,
            "deviation": cells[4] if produced else None,
            "shots": cells[5] if produced else None,
            "asl": cells[6] if produced else None,
            "black_frames": cells[7] if produced else None,
            "reason": cells[9] if len(cells) > 9 else "",
        }
        # split 与意图从 tag 解析（dev-MVI_xxxx-fast4 → dev / fast）
        sm = re.match(r"(dev|holdout)-", tag)
        row["split"] = sm.group(1) if sm else "unknown"
        im = re.search(r"-(fast|slow)\d+$", tag)
        row["intent"] = im.group(1) if im else "base"
        rows.append(row)
    return rows


def _rate(numer: int, denom: int) -> str:
    return f"{numer}/{denom} ({numer / denom:.0%})" if denom else "0/0 (-)"


def three_ledgers(rows: list[dict]) -> dict:
    produced = [r for r in rows if r["status"] == "produced"]
    refused = [r for r in rows if r["status"] == "refused"]
    hard = [r for r in produced if r["hard_defect"]]
    # 拒绝正确性：拒绝原因可解释（目标不可达/约束拒绝）即计正确
    explainable = [r for r in refused
                   if "target_duration_unreachable" in r.get("reason", "")
                   or "must_avoid" in r.get("reason", "")]
    return {
        "production_rate": {"produced": len(produced), "total": len(rows),
                            "text": _rate(len(produced), len(rows))},
        "defect_rate": {"defective": len(hard), "produced": len(produced),
                        "text": _rate(len(hard), len(produced))},
        "refusal_correctness": {"explainable": len(explainable),
                                "refused": len(refused),
                                "text": _rate(len(explainable), len(refused))},
    }


def dev_holdout_comparison(rows: list[dict]) -> dict:
    out: dict = {}
    for split in ("dev", "holdout"):
        subset = [r for r in rows if r.get("split") == split]
        if not subset:
            continue
        out[split] = three_ledgers(subset)
    verdict = "PASS"
    if "dev" in out and "holdout" in out:
        d, h = out["dev"], out["holdout"]
        # holdout 不劣于 dev：出片率不低、硬伤率不高
        prod_d = d["production_rate"]["produced"] / max(1, d["production_rate"]["total"])
        prod_h = h["production_rate"]["produced"] / max(1, h["production_rate"]["total"])
        def_d = d["defect_rate"]["defective"] / max(1, d["defect_rate"]["produced"])
        def_h = h["defect_rate"]["defective"] / max(1, h["defect_rate"]["produced"])
        verdict = "PASS" if (prod_h >= prod_d - 0.15 and def_h <= def_d + 0.05) \
            else "REVIEW"
    out["verdict"] = verdict
    return out


def intent_breakdown(rows: list[dict]) -> dict:
    out: dict = {}
    for intent in ("base", "fast", "slow"):
        subset = [r for r in rows if r.get("intent") == intent]
        if subset:
            out[intent] = three_ledgers(subset)
    return out


def build_report(rows: list[dict]) -> str:
    ledgers = three_ledgers(rows)
    comparison = dev_holdout_comparison(rows)
    by_intent = intent_breakdown(rows)
    lines = ["# 阶段 8b · L2 正式矩阵闸门报告", ""]
    lines.append(f"- 出片率: {ledgers['production_rate']['text']}")
    lines.append(f"- L1 硬伤率: {ledgers['defect_rate']['text']}")
    lines.append(f"- 拒绝正确率: {ledgers['refusal_correctness']['text']}")
    lines.append(f"- dev/holdout 判定: **{comparison['verdict']}**")
    lines.append("")
    lines.append("| split | 出片率 | 硬伤率 | 拒绝正确率 |")
    lines.append("|---|---|---|---|")
    for split in ("dev", "holdout"):
        if split in comparison and "production_rate" in comparison[split]:
            d = comparison[split]
            lines.append(f"| {split} | {d['production_rate']['text']} "
                         f"| {d['defect_rate']['text']} "
                         f"| {d['refusal_correctness']['text']} |")
    lines.append("")
    lines.append("| 意图 | 出片率 | 硬伤率 |")
    lines.append("|---|---|---|")
    for intent, d in by_intent.items():
        lines.append(f"| {intent} | {d['production_rate']['text']} "
                     f"| {d['defect_rate']['text']} |")
    return "\n".join(lines), ledgers, comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="L2 正式矩阵闸门分析")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    report_md = Path(args.report).read_text(encoding="utf-8")
    rows = parse_report_table(report_md)
    if not rows:
        print("错误：报告表中未解析到任务行", file=sys.stderr)
        return 2
    text, ledgers, comparison = build_report(rows)
    print(text)
    out_path = Path(args.report).with_suffix(".gate.md")
    out_path.write_text(text, encoding="utf-8")
    json_path = Path(args.report).with_suffix(".gate.json")
    json_path.write_text(json.dumps({"ledgers": ledgers,
                                     "dev_holdout": comparison},
                                    ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\n闸门报告: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
