"""P1 VLM 真实验收脚本：analyze_media_full → 关系推断 → 选片 → 缓存。

运行：python scripts/verify_vlm_p1.py
需设置环境变量 VLM_MODEL=qwen3-vl（或在 .env 中配置）。
"""
from __future__ import annotations

import json
import os
import sys
import time

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 强制 VLM 模型为 qwen3-vl（config 默认是 glm-4.6v-flash）
os.environ.setdefault("VLM_MODEL", "qwen3-vl")
os.environ.setdefault("VLM_PROVIDER", "ollama")

VIDEO = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    from observation_service.pipeline import analyze_media, analyze_media_full
    from director_brain.relation_inference import infer_relations
    from director_brain.director_reasoner import HeuristicDirectorReasoner
    from director_brain.brief_compiler import compile_brief
    from director_brain.story_graph_builder import build_story_graph
    from director_brain.models.film_observation import ClaimKind

    # ------------------------------------------------------------------
    # 1. VLM 真实调用
    # ------------------------------------------------------------------
    section("1. VLM 真实调用产出语义标签")
    t0 = time.time()
    obs_full = analyze_media_full(VIDEO, vlm=True, asr=False)
    elapsed_first = time.time() - t0

    vlm_obs = [o for o in obs_full if o.observation_type == "vlm_semantic"]
    det_obs = [o for o in obs_full if o.observation_type == "deterministic_technical"]
    print(f"总观测数: {len(obs_full)} (det={len(det_obs)}, vlm={len(vlm_obs)})")
    print(f"镜头数: {len(det_obs)}")
    print(f"VLM 观测数: {len(vlm_obs)} (应=镜头数)")

    success = [o for o in vlm_obs if o.claim_kind == ClaimKind.MODEL_OBSERVATION]
    failed = [o for o in vlm_obs if o.claim_kind == ClaimKind.NOT_DETERMINED]
    print(f"成功: {len(success)}, 失败: {len(failed)}")
    if vlm_obs:
        print(f"成功率: {len(success)/len(vlm_obs)*100:.1f}% (要求>=80%)")

    print(f"\n前 3 个 VLM 观测完整 claim:")
    for o in vlm_obs[:3]:
        claim = json.loads(o.claim)
        print(f"  {o.media_asset_id}: {json.dumps(claim, ensure_ascii=False)}")

    print(f"\n15 镜头语义标签摘要:")
    for o in vlm_obs:
        claim = json.loads(o.claim)
        print(f"  {o.media_asset_id[:20]:20s} role={claim.get('proposed_role_v2','?'):10s} "
              f"fn={claim.get('shot_function','?'):22s} "
              f"motion={claim.get('motion_amount','?'):7s} "
              f"desc={claim.get('frame_description','')[:40]}")

    # ------------------------------------------------------------------
    # 2. 关系推断对比
    # ------------------------------------------------------------------
    section("2. 关系推断对比（纯确定性 vs 含 VLM）")
    brief = compile_brief("vlm_verify", VIDEO, obs_full)
    graph = build_story_graph(brief, obs_full)

    # 纯确定性
    obs_det_only = analyze_media(VIDEO)
    brief_det = compile_brief("vlm_verify_det", VIDEO, obs_det_only)
    graph_det = build_story_graph(brief_det, obs_det_only)
    edges_det = infer_relations(obs_det_only, graph_det)
    print(f"纯确定性边数: {len(edges_det)}")
    for e in edges_det:
        print(f"  {e.edge_id[:50]:50s} type={e.edge_type.value:18s} conf={e.confidence}")

    # 含 VLM
    edges_full = infer_relations(obs_full, graph)
    print(f"\n含 VLM 边数: {len(edges_full)}")
    semantic_edges = [e for e in edges_full if e.edge_id.startswith("semantic_")]
    print(f"  其中 VLM 语义边: {len(semantic_edges)}")
    for e in edges_full:
        prefix = "VLM " if e.edge_id.startswith("semantic_") else "    "
        print(f"  {prefix}{e.edge_id[:50]:50s} type={e.edge_type.value:18s} conf={e.confidence}")

    print(f"\n边数对比: det={len(edges_det)}, full={len(edges_full)} "
          f"({'PASS' if len(edges_full) >= len(edges_det) else 'FAIL'})")

    # ------------------------------------------------------------------
    # 3. 选片对比
    # ------------------------------------------------------------------
    section("3. 选片对比（纯确定性 vs 含 VLM）")
    reasoner = HeuristicDirectorReasoner()

    # 纯确定性
    edl_det, plan_det = reasoner.generate_plan(brief_det, graph_det, obs_det_only)
    print(f"纯确定性 EDL 片段数: {len(edl_det.ordered_edits)}")
    for e in edl_det.ordered_edits:
        print(f"  {e.source_asset_id[:20]:20s} in={e.in_frame/1e6:.2f}s "
              f"out={e.out_frame/1e6:.2f}s rationale={e.rationale[:60]}")

    # 含 VLM
    edl_full, plan_full = reasoner.generate_plan(brief, graph, obs_full)
    print(f"\n含 VLM EDL 片段数: {len(edl_full.ordered_edits)}")
    vlm_rationales = [d for d in plan_full.decisions if "vlm:" in (d.rationale or "")]
    print(f"  含 vlm: 字样的 decision: {len(vlm_rationales)}")
    for e in edl_full.ordered_edits:
        print(f"  {e.source_asset_id[:20]:20s} in={e.in_frame/1e6:.2f}s "
              f"out={e.out_frame/1e6:.2f}s rationale={e.rationale[:80]}")

    # ------------------------------------------------------------------
    # 4. 缓存验证
    # ------------------------------------------------------------------
    section("4. 缓存验证")
    t1 = time.time()
    obs_full2 = analyze_media_full(VIDEO, vlm=True, asr=False)
    elapsed_second = time.time() - t1
    vlm_obs2 = [o for o in obs_full2 if o.observation_type == "vlm_semantic"]
    print(f"首次耗时: {elapsed_first:.1f}s")
    print(f"二次耗时: {elapsed_second:.1f}s")
    print(f"加速比: {elapsed_first/max(elapsed_second,0.1):.1f}x")
    print(f"二次 VLM 观测数: {len(vlm_obs2)}")
    print(f"缓存命中: {'PASS' if elapsed_second < elapsed_first * 0.5 else 'CHECK'}")

    # ------------------------------------------------------------------
    # 5. 汇总
    # ------------------------------------------------------------------
    section("5. 验收汇总")
    checks = [
        ("VLM 观测数=镜头数", len(vlm_obs) == len(det_obs)),
        ("VLM 成功率>=80%", len(vlm_obs) > 0 and len(success)/len(vlm_obs) >= 0.8),
        ("description 非空", all(json.loads(o.claim).get("frame_description") for o in success)),
        ("关系边数>=确定性", len(edges_full) >= len(edges_det)),
        ("有 VLM 语义边", len(semantic_edges) > 0),
        ("选片 rationale 含 vlm", len(vlm_rationales) > 0),
        ("缓存加速", elapsed_second < elapsed_first * 0.5),
    ]
    all_pass = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            all_pass = False
    print(f"\n整体判定: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
