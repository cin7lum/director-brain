# -*- coding: utf-8 -*-
"""阶段 P3-3 跨镜头叙事理解（候选①收编后的薄示例）。

narrative_analyzer.analyze_narrative 现为内核消费的生产库函数
（传输统一走 llm_adapter.post_chat_json；shot_ids 传入时返回
*_resolved 字段供内核幕重分配）。本脚本保留为该库函数的最小示例。
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

from director_brain.narrative_analyzer import analyze_narrative, load_env

load_env()

# sintel_trailer 语义观测缓存（vlm_prompt_v3_semantic 口径）
cache_files = sorted(
    (ROOT / "evidence" / "GLM-HANDOVER").glob("**/vlm_cache*.json")
)
if not cache_files:
    print("未找到语义观测缓存——请先经生产入口跑一次 roughcut --semantic")
    sys.exit(1)

rows = json.loads(cache_files[0].read_text(encoding="utf-8"))
observations = rows if isinstance(rows, list) else rows.get("items", [])
sems, ids = [], []
for o in observations:
    claim = json.loads(o["claim"]) if isinstance(o.get("claim"), str) else {}
    if not claim.get("scene_description"):
        continue
    sems.append({
        "scene_description": claim["scene_description"],
        "action_type": claim.get("action_type"),
        "emotional_tone": claim.get("emotional_tone"),
        "narrative_role": claim.get("narrative_role"),
        "importance": claim.get("importance"),
    })
    ids.append(o["media_asset_id"])

arc = analyze_narrative(sems, shot_ids=ids)
print(json.dumps(arc, ensure_ascii=False, indent=2)[:2000])
