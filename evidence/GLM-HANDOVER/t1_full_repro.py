# -*- coding: utf-8 -*-
"""T1 修复验收复现（对真实仓 D:\\新建豆包\\AI-Director 运行）。

复刻主控 _provenance/verify/verify_plan_edl_consistency.py 的流程与判定，
覆盖交接文档 §5-T1 的三条验收判据：
  判据1：新校验下，历史分裂形态从 PASS 变 FAIL（能判红）；
  判据2：修复后同一输入 sequence == EDL ids 为 True（不再分裂）；
  判据3：负样本——人为改坏 plan.sequence → 必须 FAIL。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

from observation_service.pipeline import analyze_media          # noqa: E402
from observation_service.asr import transcribe                  # noqa: E402
from director_brain.brief_compiler import compile_brief         # noqa: E402
from director_brain.story_graph_builder import build_story_graph  # noqa: E402
from director_brain.relation_inference import infer_relations   # noqa: E402
from director_brain.director_reasoner import get_director_reasoner  # noqa: E402
from director_brain.plan_validator import validate_plan         # noqa: E402
from director_brain.plan_repair import repair_plan              # noqa: E402


def snapshot(tag: str, edl, plan) -> dict:
    seq = list(plan.sequence)
    edl_ids = [e.source_asset_id for e in edl.ordered_edits]
    dec_refs = [x for d in plan.decisions for x in d.shot_refs]
    print(f"=== {tag} ===")
    print(f"  plan.sequence       len={len(seq)}")
    print(f"  plan.decisions      len={len(plan.decisions)}")
    print(f"  edl.ordered_edits   len={len(edl_ids)}")
    print(f"  sequence == EDL ids ? {seq == edl_ids}")
    return {"seq": seq, "edl_ids": edl_ids, "dec_refs": dec_refs}


def main() -> int:
    vp = str(ROOT / "sintel_ref_480p.mp4")
    print("素材:", vp)

    tech = analyze_media(vp)
    try:
        sp = transcribe(vp)
    except Exception as exc:  # noqa: BLE001
        print("transcribe 失败（降级为空）:", type(exc).__name__, exc)
        sp = []
    obs = tech + sp

    brief = compile_brief(
        project_id="verify",
        video_path=vp,
        observations=obs,
        target_duration_us=15_000_000,
    )
    g = build_story_graph(brief, obs)
    rel = infer_relations(obs, g)
    print(f"故事图: nodes={len(g.nodes)} edges={len(g.edges)}  relation 产出={len(rel)}")

    rz = get_director_reasoner("heuristic")
    edl, plan = rz.generate_plan(brief, g, obs)

    before = snapshot("BEFORE REPAIR", edl, plan)
    v1, e1 = validate_plan(edl, plan, obs)
    print(f"  validate_plan -> valid={v1} errors={e1}")

    outcome = repair_plan(edl, plan, obs)
    if outcome.requires_director:
        print(f"\n[ABSTAIN] {outcome.reason_code}: {outcome.reason}")
        print(">>> 判据2 无法达成（需导演层重出 plan）——按 fail-closed 视为未修复")
        return 1

    after = snapshot("AFTER REPAIR", outcome.edl, outcome.plan)
    v2, e2 = validate_plan(outcome.edl, outcome.plan, obs)
    print(f"  validate_plan -> valid={v2} errors={e2}")
    print(f"  物理调整: {outcome.adjustments}")

    print("\n########## 验收判定 ##########")
    # 判据2：修复后不再分裂
    c2 = after["seq"] == after["edl_ids"] and v2
    print(f"  [判据2] 修复后 sequence==EDL ids 且 valid: {c2}")

    # 判据1：把旧 repair 的分裂形态（EDL 删一段、plan 不动）喂给新 validator 必须红
    import copy
    split_edl = copy.deepcopy(edl)
    split_edl.ordered_edits = list(split_edl.ordered_edits[:-1])  # 模拟旧 rule4/6 删镜头
    sv, se = validate_plan(split_edl, plan, obs)
    c1 = (not sv) and any("plan/edl split" in x for x in se)
    print(f"  [判据1] 历史分裂形态被新校验判红: {c1}（errors 含 plan/edl split: "
          f"{any('plan/edl split' in x for x in se)}）")

    # 判据3：负样本——人为改坏 plan.sequence 必须 FAIL
    bad_plan = plan.model_copy(deep=True)
    if bad_plan.sequence:
        bad_plan.sequence = list(bad_plan.sequence[:-1])
    bv, be = validate_plan(edl, bad_plan, obs)
    c3 = (not bv) and any("plan/edl split" in x for x in be)
    print(f"  [判据3] 负样本（改坏 plan.sequence）判红: {c3}")

    if c1 and c2 and c3:
        print("\n>>> T1 三判据全部通过。")
        return 0
    print("\n>>> 存在未达成的判据。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
