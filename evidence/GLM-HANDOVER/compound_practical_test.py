# -*- coding: utf-8 -*-
"""阶段 8a 收官 · 复合实操：全部能力一次真实出片叠加验证。

真实素材（sintel_trailer，含真实对白）+ 用户意图 + 转场（导演层 policy：
镜头时长 ≥1.5s 的接缝加 0.3s fade）+ 字幕（成片时间线自动跟随转场）+
配乐（amix 混音）+ 决策账本 + L1 计分卡——一镜到底的"能应用"证明。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

from director_brain.audit_trail import log_decision                     # noqa: E402
from director_brain.brief_compiler import compile_brief                 # noqa: E402
from director_brain.director_reasoner import get_director_reasoner      # noqa: E402
from director_brain.models.edl import TransitionSpec                    # noqa: E402
from director_brain.plan_repair import repair_plan                      # noqa: E402
from director_brain.plan_validator import validate_plan                 # noqa: E402
from director_brain.story_graph_builder import build_story_graph        # noqa: E402
from execution.renderer import render_edl                               # noqa: E402
from observation_service.asr import transcribe                          # noqa: E402
from observation_service.media_info import probe_audio_stream           # noqa: E402
from observation_service.pipeline import analyze_media                  # noqa: E402
from observation_service.srt import build_srt                           # noqa: E402
from scripts.l1_metrics import collect_metrics                          # noqa: E402
from storage.sqlite_repository import SqliteRepository                  # noqa: E402


def main() -> int:
    src = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
    out_dir = ROOT / "evidence" / "GLM-HANDOVER" / "compound"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / "compound_cut.mp4")
    intent = "快节奏剪辑，避免模糊镜头"

    # 1. 观测 + 真实转写
    obs = analyze_media(src) + transcribe(src)
    speech = [o for o in obs if o.observation_type == "speech_transcript"]
    print(f"[1] 观测: 技术 {len(obs)-len(speech)} + 语音 {len(speech)}")

    # 2. 意图 Brief + 决策
    brief = compile_brief("compound", src, obs, target_duration_us=20_000_000,
                          intent_text=intent)
    graph = build_story_graph(brief, obs)
    edl, plan = get_director_reasoner("heuristic").generate_plan(brief, graph, obs)
    print(f"[2] EDL: {len(edl.ordered_edits)} 镜头")

    # 3. 导演层转场 policy：镜头时长 ≥1.5s 的接缝加 0.3s fade（审美决策属导演层）
    edits = edl.ordered_edits
    n_trans = 0
    for i, e in enumerate(edits[:-1]):
        d_i = e.out_frame - e.in_frame
        d_next = edits[i + 1].out_frame - edits[i + 1].in_frame
        if d_i >= 1_500_000 and d_next >= 1_500_000:
            e.transition = TransitionSpec(type="xfade", name="fade",
                                          duration_us=300_000)
            n_trans += 1
    print(f"[3] 转场 policy: {n_trans} 处 fade(0.3s)")

    # 4. 字幕（成片时间线，自动跟随转场）
    srt_out = out + ".srt"
    Path(srt_out).write_text(build_srt(speech, edl), encoding="utf-8")
    if speech:
        edl.subtitle_refs.append(f"{srt_out}|chi")
    print(f"[4] 字幕: {len(speech)} 条语音 → SRT")

    # 5. 配乐（程序生成的测试乐床）
    music = out.replace(".mp4", "_music.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=220:duration=30",
                    "-af", "volume=0.3", music],
                   check=True, capture_output=True, timeout=60)
    edl.audio_refs.append(f"{music}|-14")

    # 6. 验证 + 修复 + 渲染
    vr = validate_plan(edl, plan, obs)
    if not vr[0]:
        outcome = repair_plan(edl, plan, obs)
        if outcome.requires_director:
            print(f"[6] 修复放弃: {outcome.reason_code}")
            return 1
        edl, plan = outcome.edl, outcome.plan
    edl.subtitle_refs = edl.subtitle_refs  # 字幕引用保持
    Path(out + '.edl.json').write_text(edl.model_dump_json(indent=2), encoding='utf-8')
    try:
        render_edl(edl, src, out)
    except RuntimeError:
        import execution.renderer as R
        fc = R._build_filter_complex(edl, True, "abase")
        print("=== filter_complex（诊断）===")
        for seg in fc.split(";"):
            print("  ", seg[:120])
        raise

    # 7. L1 检核（真实成片）
    m = collect_metrics(out, json.loads(edl.model_dump_json()),
                        json.loads(plan.model_dump_json()), 20.0)
    print(f"[7] L1: 硬伤={m['hard_defect']} 黑帧={len(m['black_frames'])} "
          f"冻帧={len(m['freezes'])} 时长偏差={[f['value'] for f in m['intent'] if '偏差' in f['item']]}")

    # 8. 账本
    repo = SqliteRepository(str(ROOT / "data" / "director_brain.db"))
    log_decision(repo, plan.plan_id, "plan_generated", {
        "transitions": n_trans, "subtitles": bool(speech),
        "music": True, "l1_hard_defect": m["hard_defect"],
    })

    print(f"\n[完成] 复合成片: {out}")
    print(f"  流构成: 视频 + 音频(原声+配乐) + 字幕(mov_text)")
    print(f"  转场: {n_trans} 处 ｜ 字幕跟随时间线: 是")
    print(f"  L1 硬伤: {m['hard_defect']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
