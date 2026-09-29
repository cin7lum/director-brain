# -*- coding: utf-8 -*-
"""P3 复合实操：内核深化三层 + 全部已有能力一次真实出片。

这是"内核不浅"的终极验证——系统用**看懂内容**的方式剪片子：

1. 多帧语义观测（P3-1）：每镜头 3 帧 → VLM → 内容/情绪/叙事角色
2. 跨镜头叙事弧（P3-3）：全部语义观测 → LLM → 叙事弧/幕边界/推荐顺序
3. 语义感知选片（P3-2）：叙事弧驱动四幕分配 + 语义评分排序
4. 转场 + 字幕 + 配乐 + 账本 + L1（已有能力全叠加）
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from director_brain.audit_trail import log_decision                      # noqa: E402
from director_brain.brief_compiler import compile_brief                  # noqa: E402
from director_brain.models.edl import (                                  # noqa: E402
    EditItem, EditorialDecisionList, TransitionSpec,
)
from director_brain.models.director_plan import Decision, DirectorDecisionPlan  # noqa: E402
from director_brain.narrative_analyzer import analyze_narrative          # noqa: E402
from director_brain.plan_validator import validate_plan                  # noqa: E402
from director_brain.plan_repair import repair_plan                       # noqa: E402
from director_brain.semantic_scorer import (                             # noqa: E402
    _ROLE_TO_ACT, compute_semantic_score,
)
from execution.renderer import render_edl                                # noqa: E402
from observation_service.asr import transcribe                           # noqa: E402
from observation_service.pipeline import analyze_media                   # noqa: E402
from observation_service.semantic_analyzer import analyze_shot_semantic  # noqa: E402
from observation_service.srt import build_srt                            # noqa: E402
from scripts.l1_metrics import collect_metrics                           # noqa: E402
from storage.sqlite_repository import SqliteRepository                   # noqa: E402

TIMEBASE = 1_000_000


def main() -> int:
    src = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
    out_dir = ROOT / "evidence" / "GLM-HANDOVER" / "p3_compound"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / "semantic_cut.mp4")
    target_s = 20
    target_us = target_s * TIMEBASE
    intent = "快节奏剪辑，避免模糊镜头，保留关键动作场景"

    # ===== 第 1 层：多帧语义观测 =====
    print("=" * 60)
    print("第 1 层 · 多帧语义观测（P3-1 眼睛升级）")
    print("=" * 60)
    tech_obs = analyze_media(src)
    speech_obs = transcribe(src)
    print(f"技术观测: {len(tech_obs)} 镜头 ｜ 语音: {len(speech_obs)} 段")

    semantics = {}
    for o in tech_obs:
        try:
            sem = analyze_shot_semantic(src, o.start_frame, o.end_frame)
            semantics[o.media_asset_id] = sem
        except Exception as exc:  # noqa: BLE001
            print(f"  语义分析跳过 {o.media_asset_id[:16]}: {str(exc)[:60]}")
    print(f"语义观测: {len(semantics)}/{len(tech_obs)} 镜头")
    for aid, sem in list(semantics.items())[:5]:
        print(f"  {aid[:16]}: {sem['scene_description'][:35]}  "
              f"emotion={sem['emotional_tone']} role={sem['narrative_role']} "
              f"imp={sem['importance']}")

    # ===== 第 2 层：跨镜头叙事弧 =====
    print(f"\n{'=' * 60}")
    print("第 2 层 · 跨镜头叙事弧（P3-3 灵魂升级）")
    print("=" * 60)
    ordered_ids = [o.media_asset_id for o in tech_obs]
    sem_list = [semantics[aid] for aid in ordered_ids if aid in semantics]
    arc = analyze_narrative(sem_list, api_key=os.environ.get("ARK_API_KEY", ""))
    print(f"叙事弧: {arc['story_arc'][:80]}...")
    print(f"幕边界: {arc.get('act_boundaries', [])}")

    # 用叙事弧的幕边界替代时间比例分配
    act_map = {}  # shot_idx → act_name
    for boundary in arc.get("act_boundaries", []):
        act_name = boundary["act"]
        for idx in range(boundary["start"], boundary["end"] + 1):
            act_map[idx] = act_name
    print(f"幕分配: {len(act_map)} 镜头按内容分入四幕")

    # ===== 第 3 层：语义感知选片 + 叙事弧驱动排序 =====
    print(f"\n{'=' * 60}")
    print("第 3 层 · 语义感知选片 + 叙事弧排序（P3-2 大脑升级）")
    print("=" * 60)
    all_obs = tech_obs + speech_obs
    brief = compile_brief("p3_compound", src, all_obs,
                          target_duration_us=target_us, intent_text=intent)

    # 构建候选（带语义评分）
    scored_shots = []
    seen_descs = []
    for idx, o in enumerate(tech_obs):
        if o.media_asset_id not in semantics:
            continue
        sem = semantics[o.media_asset_id]
        act = act_map.get(idx, "develop")
        cand = {
            "source_shot_id": o.media_asset_id,
            "source_media_hash": o.media_hash,
            "source_in_us": o.start_frame,
            "source_out_us": o.end_frame,
            "duration_us": o.end_frame - o.start_frame,
            "blur_score": json.loads(o.claim).get("blur_score", 0),
            "exposure_ok": json.loads(o.claim).get("exposure_ok", False),
            "technical_usable": True,
        }
        score = compute_semantic_score(cand, sem, brief, act, seen_descs)
        seen_descs.append(sem["scene_description"][:30].lower())
        scored_shots.append((score, cand, sem))

    # 按幕分组 + 语义评分排序 + 贪心填充
    act_quotas = {"hook": 0.15, "develop": 0.35, "peak": 0.30, "resolve": 0.20}
    min_clip, max_clip = 800_000, 6_000_000
    edits, decisions = [], []
    filled_total = 0

    for act_name in ("hook", "develop", "peak", "resolve"):
        quota = max(int(act_quotas[act_name] * target_us), min_clip)
        filled = 0
        bucket = sorted(
            [s for s in scored_shots if s[0].assigned_act == act_name],
            key=lambda pair: pair[0].total, reverse=True)
        for score, cand, sem in bucket:
            if filled >= quota:
                break
            dur = min(cand["duration_us"], max_clip, quota - filled)
            if dur < min_clip:
                continue
            in_us = cand["source_in_us"]
            edit = EditItem(
                source_asset_id=cand["source_shot_id"],
                source_media_hash=cand["source_media_hash"],
                in_frame=in_us,
                out_frame=in_us + dur,
                timebase=TIMEBASE,
                act=act_name,
                evidence_type="vlm",
                shot_function=sem.get("shot_function"),
                rationale=(f"act={act_name}, semantic={score.reason}, "
                           f"desc={sem['scene_description'][:30]}"),
            )
            edits.append(edit)
            decisions.append(Decision(
                decision_id=f"dec_{act_name}_{len(decisions)+1:02d}",
                purpose="select_shot_semantic",
                shot_refs=[cand["source_shot_id"]],
                evidence_refs=[cand["source_shot_id"]],
                rationale=edit.rationale,
                confidence=min(0.95, score.total / 500.0 + 0.3),
                requires_approval=score.total < 100,
            ))
            filled += dur
            filled_total += dur
            print(f"  {act_name}: {cand['source_shot_id'][:16]}  "
                  f"score={score.total:.1f}  {score.reason[:40]}")

    # 转场 policy：相邻镜头幕不同时加 dissolve
    n_trans = 0
    for i, e in enumerate(edits[:-1]):
        if e.act and edits[i + 1].act and e.act != edits[i + 1].act:
            e.transition = TransitionSpec(type="xfade", name="dissolve",
                                          duration_us=400_000)
            n_trans += 1
    print(f"转场: {n_trans} 处 dissolve（幕切换处）")

    edl = EditorialDecisionList(
        schema_version="1.0", project_id="p3_compound",
        created_at=int(time.time()), producer="semantic_director_v1",
        source_ref=src, edl_id="edl_p3_semantic", version="1.0",
        brief_version=brief.version, context_id="ctx_p3",
        source_asset_hashes=list({e.source_media_hash for e in edits}),
        timebase=TIMEBASE, ordered_edits=edits,
        expected_duration=sum(e.out_frame - e.in_frame for e in edits),
        approval_state="draft",
    )
    plan = DirectorDecisionPlan(
        schema_version="1.0", project_id="p3_compound",
        created_at=int(time.time()), producer="semantic_director_v1",
        source_ref=src, plan_id="plan_p3_semantic", version="1.0",
        brief_version=brief.version, film_state_version="1.0",
        sequence=[e.source_asset_id for e in edits],
        decisions=decisions,
        constraints=[f"target_duration_us={target_us}",
                     "must_avoid:blur_score<50.0:模糊"],
        open_questions=[], degraded=False,
        degradation_events=[],
        validation_status="pending", approval_state="draft",
    )

    # ===== 验证 + 修复 =====
    ok, errors = validate_plan(edl, plan, all_obs)
    if not ok:
        outcome = repair_plan(edl, plan, all_obs)
        if outcome.requires_director:
            print(f"修复放弃: {outcome.reason_code}")
            return 1
        edl, plan = outcome.edl, outcome.plan
        print(f"修复: {len(outcome.adjustments)} 处物理调整")

    # ===== 字幕 + 配乐 =====
    if speech_obs:
        srt_text = build_srt(speech_obs, edl)
        srt_path = out_dir / "semantic_cut.srt"
        srt_path.write_text(srt_text, encoding="utf-8")
        if srt_text.strip():
            edl.subtitle_refs.append(f"{srt_path}|chi")

    music = out_dir / "music_bed.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=196:duration=30",
                    "-af", "volume=0.2", str(music)],
                   check=True, capture_output=True, timeout=60)
    edl.audio_refs.append(f"{music}|-16")

    # ===== 渲染 =====
    Path(out + ".edl.json").write_text(edl.model_dump_json(indent=2),
                                       encoding="utf-8")
    Path(out + ".plan.json").write_text(plan.model_dump_json(indent=2),
                                        encoding="utf-8")
    render_edl(edl, src, out)

    # ===== L1 检核 =====
    m = collect_metrics(out, json.loads(edl.model_dump_json()),
                        json.loads(plan.model_dump_json()), float(target_s))
    print(f"\n{'=' * 60}")
    print("L1 检核结果")
    print("=" * 60)
    print(f"  硬伤: {m['hard_defect']}")
    print(f"  黑帧: {len(m['black_frames'])} 段")
    print(f"  冻帧: {len(m['freezes'])} 段")
    for f in m["intent"]:
        print(f"  {f['verdict']}: {f['item']} — {f['value'][:50]}")
    print(f"  节奏: shots={m['pacing']['shot_count']} "
          f"ASL={m['pacing']['asl_seconds']}s")

    # ===== 账本 =====
    repo = SqliteRepository(str(ROOT / "data" / "director_brain.db"))
    log_decision(repo, plan.plan_id, "semantic_cut_completed", {
        "semantic_obs_count": len(semantics),
        "narrative_arc": arc["story_arc"][:80],
        "act_boundaries": arc.get("act_boundaries", []),
        "transitions": n_trans,
        "l1_hard_defect": m["hard_defect"],
        "layer": "P3-1+P3-2+P3-3 compound",
    })
    print(f"\n{'=' * 60}")
    print(f"[完成] 语义驱动成片: {out}")
    print(f"  三层内核深化全部参与：多帧观测({len(semantics)}) + 叙事弧 + 语义选片")
    print(f"  叠加能力：{n_trans} 转场 + 字幕 + 配乐 + 账本 + L1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
