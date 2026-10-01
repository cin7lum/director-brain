# -*- coding: utf-8 -*-
"""02 导演脑 · 一键实机演示。

薄编排生产入口（不引入任何平行路径——演示路径 = 生产路径）：

1. roughcut 生产链出片（状态机执法：--confirm 确认动作绑定 hash 落账本）
2. L1 客观计分卡（黑帧/冻帧/静音/时长偏差/节奏/意图约束）
3. 权威结构化评审（已准入 doubao-seed-2-1-lite，冻结五维量表）

用法：
    python scripts/demo.py                          # 默认素材 + 默认意图
    python scripts/demo.py -i 素材.mp4 --intent "快剪风格，避免模糊镜头" -t 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.l1_metrics import collect_metrics  # noqa: E402
from scripts.roughcut import run_roughcut  # noqa: E402

_DEFAULT_INPUT = r"E:\100CANON\MVI_4215.MP4"
_DEFAULT_INTENT = "快剪风格，保留关键动作，避免模糊镜头"
_DEFAULT_TARGET_S = 5


def main() -> int:
    parser = argparse.ArgumentParser(description="02 导演脑一键演示")
    parser.add_argument("-i", "--input", default=_DEFAULT_INPUT)
    parser.add_argument("--intent", default=_DEFAULT_INTENT)
    parser.add_argument("-t", "--target", type=int, default=_DEFAULT_TARGET_S,
                        help="目标成片时长（秒）")
    parser.add_argument("-o", "--out-dir",
                        default=str(ROOT / "evidence" / "DEMO"))
    parser.add_argument("--semantic", action="store_true",
                        help="语义驱动选片（vlm_semantic 治理内灰度 ACTIVE，跑完回 SHADOW）")
    parser.add_argument("--variants", type=int, default=1,
                        help="多方案对比：N 变体评分择优")
    args = parser.parse_args()

    if not Path(args.input).is_file():
        print(f"错误：素材不存在 {args.input}")
        return 1

    stamp = time.strftime("%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cut = str(out_dir / f"demo_cut_{stamp}.mp4")

    print("=" * 62)
    print("02 导演脑 · 实机演示")
    print(f"  素材: {Path(args.input).name}")
    print(f"  意图: {args.intent}")
    print(f"  目标: {args.target}s")
    print("=" * 62)

    # ---- 1. 生产链出片（确认动作 = 演示操作者，hash 绑定落账本）----
    t0 = time.time()
    if args.semantic:
        # 治理内灰度：try/finally 翻 ACTIVE 并留痕，跑完恢复 SHADOW
        from director_brain.pathway_protocol import (
            PathwayStatus,
            set_pathway_status,
        )
        set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
        try:
            rc = run_roughcut(
                input_path=args.input,
                output_path=cut,
                target_duration=args.target,
                intent_text=args.intent,
                semantic=True,
                confirm_strategy=True,
                variants=args.variants,
            )
        finally:
            set_pathway_status("vlm_semantic", PathwayStatus.SHADOW)
    else:
        rc = run_roughcut(
            input_path=args.input,
            output_path=cut,
            target_duration=args.target,
            intent_text=args.intent,
            confirm_strategy=True,
            variants=args.variants,
        )
    if rc != 0:
        print(f"\n[演示] 生产链返回 {rc}（fail-closed 拒绝属正确行为——"
              f"素材/意图组合不满足，换素材或放宽意图重试）")
        return rc
    print(f"\n[1/3] 成片完成（{time.time() - t0:.1f}s）-> {cut}")

    # ---- 2. L1 客观计分卡 ----
    edl = json.loads(Path(f"{cut}.edl.json").read_text(encoding="utf-8"))
    plan = json.loads(Path(f"{cut}.plan.json").read_text(encoding="utf-8"))
    m = collect_metrics(cut, edl, plan, float(args.target))
    print("[2/3] L1 客观计分卡")
    print(f"  硬伤: {'有（见上）' if m['hard_defect'] else '无'}")
    print(f"  黑帧 {len(m['black_frames'])} 段 ｜ 冻帧 {len(m['freezes'])} 段")
    for f in m["intent"]:
        print(f"  {f['verdict']}: {f['item']}")

    # ---- 3. 权威评审（doubao-seed-2-1-lite，冻结五维量表）----
    print("[3/3] 结构化评审（已准入模型 · 冻结量表）")
    try:
        from director_brain.narrative_analyzer import load_env
        load_env()
        import os
        from scripts.plan_judge import judge_plan
        if not os.environ.get("ARK_API_KEY"):
            print("  跳过（无 ARK_API_KEY）")
        else:
            scores = judge_plan(edl, args.intent)
            for k in ("intent", "narrative", "pacing", "shots", "technical"):
                print(f"  {k}: {scores.get(k)}/5")
            print(f"  总评: {scores.get('comment')}")
            (Path(f"{cut}.judge.json").write_text(
                json.dumps(scores, ensure_ascii=False, indent=2),
                encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  评审跳过（不阻断演示）: {type(exc).__name__}: {str(exc)[:80]}")

    print("=" * 62)
    print("演示工件（双击成片即可观看）:")
    print(f"  成片:   {cut}")
    print(f"  剪辑单: {cut}.edl.json")
    print(f"  决策单: {cut}.plan.json")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
