# -*- coding: utf-8 -*-
"""阶段 P3-2 · VLM 影子期跨素材扩充：在正式矩阵素材上批量累积决策 delta。

对 l2_formal_matrix_r1.json 的全部素材逐段执行影子实验（同 vlm_shadow.py 单段
逻辑），聚合产出通路晋升证据：
- VLM 观测有效率（degraded 占比）
- 角色分布（hero/broll/discard）
- 决策 delta 率（多少素材的选片会因 VLM 改变）与分歧明细
- discard 拦截效果（基础选片是否选中了 VLM 判弃的镜头）
"""
from __future__ import annotations

import json
import os
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


def selection(edl) -> list[str]:
    return [e.source_asset_id for e in edl.ordered_edits]


def shadow_one(src: str, target_us: int, intent: str,
               adapter: OllamaVLMAdapter) -> dict:
    tech_obs = analyze_media(src)
    if not tech_obs:
        return {"status": "no_shots", "shots": 0}
    shots = [{"shot_id": o.media_asset_id, "source_in_us": o.start_frame,
              "source_out_us": o.end_frame, "source_media_hash": o.media_hash}
             for o in tech_obs]

    vlm_obs = batch_vlm_observations(src, shots, adapter=adapter)
    degraded = sum(1 for o in vlm_obs if o.claim_kind.value == "NOT_DETERMINED")
    roles = {}
    for o in vlm_obs:
        try:
            claim = json.loads(o.claim) if o.claim.startswith("{") else {}
        except (json.JSONDecodeError, AttributeError):
            claim = {}
        roles[o.media_asset_id] = claim.get("proposed_role_v2")

    brief = compile_brief("vlm_batch", src, tech_obs, target_duration_us=target_us,
                          intent_text=intent)
    graph = build_story_graph(brief, tech_obs)
    reasoner = get_director_reasoner("heuristic")
    edl_base, plan_base = reasoner.generate_plan(brief, graph, tech_obs)
    base_sel = set(selection(edl_base))

    prev = get_pathway_status("vlm_semantic")
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        all_obs = tech_obs + vlm_obs
        brief2 = compile_brief("vlm_batch", src, all_obs,
                               target_duration_us=target_us, intent_text=intent)
        graph2 = build_story_graph(brief2, all_obs)
        edl_vlm, plan_vlm = reasoner.generate_plan(brief2, graph2, all_obs)
        vlm_sel = set(selection(edl_vlm))
    finally:
        set_pathway_status("vlm_semantic", prev)

    # discard 拦截检查：基础选片是否选中了 VLM 判弃镜头
    discards_in_base = sorted(base_sel & {a for a, r in roles.items() if r == "discard"})
    return {
        "status": "ok", "shots": len(shots),
        "vlm_valid": len(vlm_obs) - degraded, "vlm_degraded": degraded,
        "roles": roles,
        "base_selection": sorted(base_sel), "vlm_selection": sorted(vlm_sel),
        "delta": sorted(base_sel.symmetric_difference(vlm_sel)),
        "delta_changed": bool(base_sel != vlm_sel),
        "discards_in_base": discards_in_base,
    }


def main() -> int:
    matrix_path = ROOT / "evidence" / "GLM-HANDOVER" / "l2_formal_matrix_r1.json"
    cfg = json.loads(matrix_path.read_text(encoding="utf-8"))
    # 去重素材（同一素材 3 个意图共享一次 VLM 影子实验）
    seen: dict[str, dict] = {}
    for t in cfg["tasks"]:
        seen.setdefault(t["video"], {"target_seconds": t["target_seconds"],
                                     "intent": t.get("intent")})
    out_rows: list[dict] = []
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    adapter = OllamaVLMAdapter(
        model=os.environ.get("VLM_MODEL", "qwen3-vl:latest"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    t_all = time.time()
    for i, (src, meta) in enumerate(sorted(seen.items()), 1):
        target_us = meta["target_seconds"] * 1_000_000
        print(f"[{i}/{len(seen)}] {Path(src).name} "
              f"(target={meta['target_seconds']}s) ...", flush=True)
        try:
            row = shadow_one(src, target_us, meta["intent"], adapter)
        except Exception as exc:  # noqa: BLE001
            row = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:150]}
        row.update({"video": src, "target_s": meta["target_seconds"],
                    "intent": meta["intent"]})
        out_rows.append(row)
        print(f"    -> {row['status']}"
              + (f" delta={row.get('delta_changed')} shots={row.get('shots')}"
                 if row["status"] == "ok" else f" {row.get('reason', '')[:60]}"),
              flush=True)

    ok_rows = [r for r in out_rows if r.get("status") == "ok"]
    delta_rows = [r for r in ok_rows if r["delta_changed"]]
    total_shots = sum(r["shots"] for r in ok_rows)
    total_degraded = sum(r["vlm_degraded"] for r in ok_rows)
    discard_hits = [r for r in ok_rows if r["discards_in_base"]]

    report = {
        "clips": len(seen), "ok": len(ok_rows),
        "errors": len(out_rows) - len(ok_rows),
        "total_shots_observed": total_shots,
        "vlm_degraded_shots": total_degraded,
        "vlm_valid_rate": round(1 - total_degraded / max(1, total_shots), 3),
        "clips_with_delta": len(delta_rows),
        "delta_rate": round(len(delta_rows) / max(1, len(ok_rows)), 3),
        "discard_intercepted_clips": len(discard_hits),
        "role_distribution": {},
    }
    role_count: dict[str, int] = {}
    for r in ok_rows:
        for role in r["roles"].values():
            role_count[role] = role_count.get(role, 0) + 1
    report["role_distribution"] = role_count

    out = ROOT / "evidence" / "GLM-HANDOVER" / "vlm_shadow_batch_report.json"
    out.write_text(json.dumps({"summary": report, "rows": out_rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 聚合 ===\n{json.dumps(report, ensure_ascii=False, indent=2)}")
    print(f"\n报告: {out} ｜ 总耗时 {time.time()-t_all:.0f}s")
    return 0


if __name__ == "__main__":
    main()
