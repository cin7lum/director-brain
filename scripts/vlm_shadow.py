# -*- coding: utf-8 -*-
"""阶段 P3-1 · VLM 语义观测影子期实验（内核深化第一步）。

实验设计（影子期晋升证据的产出器）：
1. 链 A 照常出片（不含 VLM）→ plan_base；
2. 本机 qwen3-vl 对全部镜头产出语义观测（真金白银的 VLM 调用，带缓存）；
3. 进程内临时将 vlm_semantic 通路置 ACTIVE（响亮声明：**仅本实验进程**，
   主链 roughcut 不受影响、仍走 SHADOW 不可消费）→ 计算"如果 ACTIVE 会怎样"；
4. 恢复通路状态；产出**决策 delta 报告**（选片变化/角色分配/差异数）。

delta = 通路的影子期核心指标（路线文档"三件套"之二）：接入会改变多少决策。
"""
from __future__ import annotations

import json
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

from director_brain.brief_compiler import compile_brief                     # noqa: E402
from director_brain.director_reasoner import get_director_reasoner          # noqa: E402
from director_brain.pathway_protocol import (                               # noqa: E402
    PathwayStatus, get_pathway_status, set_pathway_status,
)
from director_brain.story_graph_builder import build_story_graph            # noqa: E402
from observation_service.ollama_vlm_adapter import OllamaVLMAdapter        # noqa: E402
from observation_service.pipeline import analyze_media                     # noqa: E402
from observation_service.vlm_observation import batch_vlm_observations    # noqa: E402


def selection(edl) -> list[tuple]:
    return [(e.source_asset_id, e.in_frame, e.out_frame) for e in edl.ordered_edits]


def main() -> int:
    src = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
    target_us = 20_000_000
    intent = "快节奏剪辑，避免模糊镜头"

    # 1. 技术观测（链 A 现状）
    tech_obs = analyze_media(src)
    print(f"[1] 技术观测: {len(tech_obs)} 镜头")

    # 2. VLM 语义观测（本机 qwen3-vl，真实调用，带缓存）
    print("[2] VLM 语义观测（本机 qwen3-vl，逐镜头）...")
    from director_brain.config import load_settings
    s = load_settings()
    shots = [{"shot_id": o.media_asset_id, "source_in_us": o.start_frame,
              "source_out_us": o.end_frame, "source_media_hash": o.media_hash}
             for o in tech_obs]
    adapter = OllamaVLMAdapter(model=s.vlm_model, base_url=s.ollama_base_url)
    t0 = time.time()
    vlm_obs = batch_vlm_observations(src, shots, adapter=adapter)
    print(f"    {len(vlm_obs)} 条语义观测，耗时 {time.time()-t0:.0f}s")
    roles = {}
    for o in vlm_obs:
        claim = json.loads(o.claim) if o.claim and o.claim.startswith("{") else {}
        roles[o.media_asset_id] = (claim.get("proposed_role_v2"),
                                   claim.get("shot_function"),
                                   claim.get("motion_amount"))
    for aid, (role, fn, mo) in sorted(roles.items()):
        print(f"    {aid}: role={role} fn={fn} motion={mo}")

    # 3. 链 A 决策（无 VLM，通路保持 SHADOW）
    brief = compile_brief("vlm_shadow", src, tech_obs, target_duration_us=target_us,
                          intent_text=intent)
    graph = build_story_graph(brief, tech_obs)
    reasoner = get_director_reasoner("heuristic")
    edl_base, plan_base = reasoner.generate_plan(brief, graph, tech_obs)
    base_sel = selection(edl_base)
    print(f"\n[3] 链 A 决策（无 VLM）: {len(base_sel)} 镜头 {base_sel}")

    # 4. 影子实验：进程内临时 ACTIVE（实验声明，主链不受影响）
    prev = get_pathway_status("vlm_semantic")
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        all_obs = tech_obs + vlm_obs
        brief2 = compile_brief("vlm_shadow", src, all_obs, target_duration_us=target_us,
                               intent_text=intent)
        graph2 = build_story_graph(brief2, all_obs)
        edl_vlm, plan_vlm = reasoner.generate_plan(brief2, graph2, all_obs)
        vlm_sel = selection(edl_vlm)
    finally:
        set_pathway_status("vlm_semantic", prev)  # 恢复 SHADOW

    # 5. 决策 delta 报告
    changed_out = [x for x in vlm_sel if x not in base_sel]
    changed_in = [x for x in base_sel if x not in vlm_sel]
    print(f"\n[4] 链 B 影子决策（VLM ACTIVE）: {len(vlm_sel)} 镜头 {vlm_sel}")
    print(f"    决策 delta: 新入 {changed_in and len(changed_in)} 处、"
          f"移出 {len(changed_in)} 处、保留 {len(base_sel) - len(changed_in)} 处")
    if changed_out or changed_in:
        print(f"    选片变化: 移出={changed_in} 新入={changed_out}")
    discards = [aid for aid, (role, _, _) in roles.items() if role == "discard"]
    heroes = [aid for aid, (role, _, _) in roles.items() if role == "hero"]
    print(f"    VLM 角色判定: discard={discards} hero={heroes}")

    report = {
        "material": src, "target_s": target_us / 1e6, "intent": intent,
        "vlm_model": s.vlm_model, "vlm_observations": len(vlm_obs),
        "roles": {k: v[0] for k, v in roles.items()},
        "base_selection": base_sel, "vlm_selection": vlm_sel,
        "delta_added": changed_out, "delta_removed": changed_in,
        "pathway_status_restored": str(get_pathway_status("vlm_semantic")),
    }
    out = ROOT / "evidence" / "GLM-HANDOVER" / "vlm_shadow_delta.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n影子期 delta 报告已写入 {out}")
    print("（通路状态已恢复 SHADOW；本实验为晋升证据产出，主链未受影响）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
