# -*- coding: utf-8 -*-
"""阶段 P3-3 真实实测：sintel_trailer 跨镜头叙事理解。"""
import json
import os
import sys

sys.path.insert(0, ".")
from director_brain.narrative_analyzer import analyze_narrative, load_env

load_env()

# sintel_trailer 全部 15 镜头的语义观测（从 VLM 影子期实验缓存读取）
from observation_service.pipeline import analyze_media
from observation_service.semantic_analyzer import analyze_shot_semantic

src = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
tech_obs = analyze_media(src)
semantics = []
for o in tech_obs:
    try:
        sem = analyze_shot_semantic(
            src, o.start_frame, o.end_frame)
        sem["shot_id"] = o.media_asset_id
        semantics.append(sem)
    except Exception as e:
        print(f"  skip {o.media_asset_id}: {e}")

print(f"\n语义观测: {len(semantics)}/{len(tech_obs)} 镜头")
for i, sem in enumerate(semantics):
    print(f"  [{i}] {sem.get('scene_description', '')[:50]}  "
          f"emotion={sem.get('emotional_tone')} role={sem.get('narrative_role')}")

# 跨镜头叙事分析
result = analyze_narrative(semantics, api_key=os.environ.get("ARK_API_KEY", ""))
print("\n=== 跨镜头叙事理解（doubao-seed-2-1-lite）===")
print(f"  叙事弧: {result.get('story_arc', '')}")
print(f"  情绪轨迹: {result.get('emotional_trajectory', [])}")
print(f"  配对: {json.dumps(result.get('pairings', []), ensure_ascii=False)[:200]}")
print(f"  关键节点: {json.dumps(result.get('key_moments', []), ensure_ascii=False)[:200]}")
print(f"  幕边界: {json.dumps(result.get('act_boundaries', []), ensure_ascii=False)[:200]}")
print(f"  推荐顺序: {result.get('suggested_order', [])}")
print(f"  局限: {result.get('limitations', [])}")

out = "evidence/GLM-HANDOVER/narrative_arc.json"
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps({"source": src, "shot_count": len(semantics), **result},
                       ensure_ascii=False, indent=2))
print(f"\n已写入 {out}")
