# -*- coding: utf-8 -*-
"""切点自然度权威评估（阶段 8c-v2）：我们的切点 vs 源片专业场景结构。

方法论依据（市调）：
- Cutting 等对 150 部好莱坞片的镜头结构研究：专业剪辑的切点与源素材的
  视觉事件边界高度相关——自动剪的切点落在源片场景边界上，是"切点自然"的
  权威代理指标；
- VSUMM/QVHighlights 协议：机器选段与权威参照的对照是标准评估形态。

本脚本对源视频跑 PySceneDetect（ContentDetector，成熟开源，采用级）得到
权威场景边界，再量我们的切点（EDL 派生）落在边界 ±tolerance 内的比例。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

from director_brain.models.edl import EditorialDecisionList  # noqa: E402


def source_scene_boundaries(src: str, threshold: float = 27.0) -> list[float]:
    """PySceneDetect ContentDetector 权威场景边界（秒）。"""
    from scenedetect import ContentDetector, SceneManager, detect, open_video
    video = open_video(src)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    return [s[0].get_seconds() for s in scenes]  # 每个场景的起点（=切点）


def cut_points_from_edl(edl: EditorialDecisionList) -> list[float]:
    """EDL 派生的成片切点（每个镜头的入点，秒）。"""
    return [e.in_frame / e.timebase for e in edl.ordered_edits]


def naturalness(cut_points: list[float], bounds: list[float],
                tolerance: float = 0.5) -> dict:
    hits = sum(1 for c in cut_points
               if any(abs(c - b) <= tolerance for b in bounds))
    return {
        "cut_points": len(cut_points),
        "scene_boundaries": len(bounds),
        "aligned": hits,
        "ratio": round(hits / len(cut_points), 3) if cut_points else None,
        "tolerance_s": tolerance,
    }


def main() -> int:
    src = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
    sidecar = ROOT / "evidence" / "GLM-HANDOVER" / "compound" / "compound_cut.mp4.edl.json"
    if not sidecar.is_file():
        print(f"错误：找不到 EDL sidecar {sidecar}", file=sys.stderr)
        return 2
    edl = EditorialDecisionList.model_validate_json(
        sidecar.read_text(encoding="utf-8"))

    bounds = source_scene_boundaries(src)
    print(f"源片权威场景边界（PySceneDetect ContentDetector）: {len(bounds)} 处")
    print(f"  边界位置: {[round(b, 2) for b in bounds]}")

    cuts = cut_points_from_edl(edl)
    print(f"我们成片的切点: {len(cuts)} 处")
    print(f"  切点位置: {[round(c, 2) for c in cuts]}")

    result = naturalness(cuts, bounds)
    print(f"\n切点自然度: {result['aligned']}/{result['cut_points']} "
          f"= {result['ratio']}（±{result['tolerance_s']}s 内命中权威边界）")
    Path(ROOT / "evidence" / "GLM-HANDOVER" / "cut_naturalness.json").write_text(
        json.dumps({"source": src, "scene_boundaries": bounds,
                    "cut_points": cuts, **result}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("已写入 evidence/GLM-HANDOVER/cut_naturalness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
