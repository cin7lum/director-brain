# -*- coding: utf-8 -*-
"""D4 · relation_inference 影子期转正/退役判定材料生成器。

按《导演脑-差距与开发计划.md》附录 D 的证据清单，对正式矩阵素材集
直接重跑关系推断（确定性观测，无渲染无 ASR），产出：

1. 覆盖率：有关系边的素材占比 + 每素材边数分布（门槛 ≥60% 素材有边）；
2. 置信分布：两类影子边的 confidence 分位数；
3. 边界重合度：visual_transition 边的连接点 vs PySceneDetect 权威边界
   （±0.5s 口径；门槛 ≥70%）；
4. 因果抽样：causal_candidate 交给已准入模型按"叙事因果合理性"评审
   （代理抽检，非人工——报告如实标注）。

结论建议：达门槛 → 转正提案（交所有者）；不达标 → 诚实退役建议。
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOLERANCE_S = 0.5
COVERAGE_GATE = 0.60
ALIGNMENT_GATE = 0.70


def source_scene_boundaries(src: str) -> list[float]:
    from scenedetect import ContentDetector, SceneManager, open_video
    video = open_video(src)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27.0))
    sm.detect_scenes(video, show_progress=False)
    return [s[1].get_seconds() for s in sm.get_scene_list()]


def main() -> int:
    from director_brain.relation_inference import infer_relations
    from director_brain.story_graph_builder import build_story_graph
    from director_brain.brief_compiler import compile_brief
    from observation_service.pipeline import analyze_media

    matrix = json.loads((ROOT / "evidence/GLM-HANDOVER/l2_formal_matrix_r1.json")
                        .read_text(encoding="utf-8"))
    materials = sorted({t["video"] for t in matrix["tasks"]})
    print(f"矩阵素材 {len(materials)} 个（去重）")

    per_material = []
    all_confs: dict[str, list[float]] = {"visual_transition": [], "causal_candidate": []}
    vt_aligned = vt_total = 0
    causal_samples: list[dict] = []

    for src in materials:
        try:
            obs = analyze_media(src)
            brief = compile_brief("d4_judgment", src, obs)
            graph = build_story_graph(brief, obs)
            edges = infer_relations(obs, graph)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {Path(src).stem}: {type(exc).__name__}: {str(exc)[:60]}")
            continue

        vis_edges = [e for e in edges if e.edge_type.value == "visual_transition"]
        cau_edges = [e for e in edges if e.edge_type.value == "causal_candidate"]
        all_confs["visual_transition"] += [e.confidence for e in vis_edges]
        all_confs["causal_candidate"] += [e.confidence for e in cau_edges]

        # visual_transition 连接点 = 被连两镜头的交界（前镜头 out 点）
        obs_by_id = {o.media_asset_id: o for o in obs
                     if o.observation_type == "deterministic_technical"}
        bounds = source_scene_boundaries(src)
        for e in vis_edges:
            prev = obs_by_id.get(e.from_node)
            if prev is None:
                continue
            boundary_s = prev.end_frame / 1e6
            vt_total += 1
            if any(abs(boundary_s - b) <= TOLERANCE_S for b in bounds):
                vt_aligned += 1

        for e in cau_edges:
            causal_samples.append({
                "material": Path(src).stem,
                "from": e.from_node, "to": e.to_node,
                "conf": e.confidence,
            })

        per_material.append({
            "material": Path(src).stem, "edges": len(edges),
            "visual_transition": len(vis_edges), "causal_candidate": len(cau_edges),
        })

    n = len(per_material)
    with_edge = sum(1 for m in per_material if m["edges"] > 0)
    coverage = with_edge / n if n else 0.0
    alignment = vt_aligned / vt_total if vt_total else 0.0

    print("=" * 62)
    print("D4 · relation_inference 影子期判定材料")
    print("=" * 62)
    print(f"素材覆盖: {with_edge}/{n} = {coverage:.0%}（门槛 ≥{COVERAGE_GATE:.0%}）")
    print(f"边数分布: 均 {sum(m['edges'] for m in per_material)/max(n,1):.1f} 边/素材"
          f" ｜ 中位 {statistics.median([m['edges'] for m in per_material]) if n else 0:.0f}")
    for k, v in all_confs.items():
        if v:
            v.sort()
            print(f"{k} 置信: n={len(v)} 中位 {v[len(v)//2]:.2f} "
                  f"P25 {v[len(v)//4]:.2f} P75 {v[3*len(v)//4]:.2f}")
    print(f"visual_transition 边界重合度: {vt_aligned}/{vt_total} = {alignment:.0%}"
          f"（门槛 ≥{ALIGNMENT_GATE:.0%}；±{TOLERANCE_S}s）")
    print(f"causal_candidate 抽样池: {len(causal_samples)} 例（LLM 代理评审候选）")

    verdict_coverage = coverage >= COVERAGE_GATE
    verdict_align = alignment >= ALIGNMENT_GATE
    print("-" * 62)
    if verdict_coverage and verdict_align:
        print("建议: 覆盖率与边界重合度双达标 → **转正提案**（交所有者；"
              "转正需同时存在内核消费方——当前无消费方，转正=预备态）")
    else:
        print(f"建议: 未达门槛（coverage={'✓' if verdict_coverage else '✗'} "
              f"alignment={'✓' if verdict_align else '✗'}）→ **诚实退役建议（RETIRED）**"
              f"或维持 SHADOW 等待改进——不硬转正")

    out = ROOT / "evidence" / "GLM-HANDOVER" / "d4_relation_judgment.json"
    out.write_text(json.dumps({
        "gate": {"coverage": COVERAGE_GATE, "alignment": ALIGNMENT_GATE},
        "coverage": {"with_edge": with_edge, "materials": n, "ratio": coverage},
        "confidence": {k: {"n": len(v), "median": (statistics.median(v) if v else None)}
                       for k, v in all_confs.items()},
        "visual_transition_alignment": {"aligned": vt_aligned, "total": vt_total,
                                        "ratio": alignment, "tolerance_s": TOLERANCE_S},
        "causal_pool_size": len(causal_samples),
        "per_material": per_material,
        "verdict": ("promotion_proposal" if (verdict_coverage and verdict_align)
                    else "retire_or_keep_shadow"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
