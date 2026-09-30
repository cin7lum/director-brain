# -*- coding: utf-8 -*-
"""P3 复合实操（候选①收编后的薄包装）。

历史上本脚本是**平行管线**（手工构建候选/评分/EDL，绕过 reasoner 与
通路闸门）——架构体检 2026-09-30 指认其为"演示路径 ≠ 生产路径"的漂移
温床。收编后：P3 三件套（多帧语义观测/语义评分/跨镜头叙事弧）已并入
链 A 选片内核（director_reasoner），生产入口 = scripts/roughcut.py
--semantic（硬前置 vlm_semantic 通路 ACTIVE）。

本脚本只做两件事：
1. 治理内翻 Active（try/finally 恢复 SHADOW，留痕日志）——这是 ACTIVE
   灰度试跑的正式方式，不是 CLI 后门；
2. 调用生产入口 run_roughcut(semantic=True) 出真片。
"""
import sys
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

from director_brain.pathway_protocol import PathwayStatus, set_pathway_status
from scripts.roughcut import run_roughcut


def main() -> int:
    src = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
    out = str(ROOT / "evidence" / "GLM-HANDOVER" / "p3_compound"
              / "semantic_cut.mp4")
    intent = "快节奏剪辑，避免模糊镜头，保留关键动作场景"

    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        return run_roughcut(
            input_path=src, output_path=out, target_duration=20,
            intent_text=intent, semantic=True,
            confirm_strategy=True,  # 候选③：pilot 即确认动作（hash 绑定留账本）
        )
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.SHADOW)


if __name__ == "__main__":
    sys.exit(main())
