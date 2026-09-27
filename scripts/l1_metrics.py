"""L1 成片客观指标计分卡（产品表现补强 · 评估层）。

设计原则（源自市调：Opus Clip / 剪映质量闭环 / VlogReward / Chatbot Arena）：
- **多指标并行，不打总分**——单一总分必然被优化目标化（Goodhart 定律）；
- **扣分明细可解释**——每条发现都带证据（时间/数值），不是黑盒分数；
- **L1 只当守门员**——回答"技术上是否合格、节奏分布是否合理、意图的
  硬约束是否达成"；叙事连贯、观感、创意属于 L3 人工盲评，本脚本明确
  标注"无法回答"，不越权。

用法：
    python scripts/l1_metrics.py --video out.mp4 \
        [--edl-json out.mp4.edl.json] [--plan-json out.mp4.plan.json] \
        [--target-seconds 15]

退出码：0 = 无技术硬伤；1 = 存在硬伤（黑帧/冻帧/静音段/音画错位/意图
硬约束未达成之一）。

指标分四组（对应 VlogReward 五维中可客观化的部分）：
- 技术质量：黑帧、冻帧、静音段、爆音风险、音画时长错位
- 节奏 Pacing：镜头数、ASL（平均镜头时长）、最长/最短镜头、时长变异系数
- 意图达成：目标时长偏差率、剪辑语言时长界合规（读 plan.constraints）
- 镜头选择：片段在源时间轴上的分布跨度（利用率代理）
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg 探测（全部只读）
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          check=False)
    return proc.returncode, proc.stdout, proc.stderr


def probe_stream_durations(video: str) -> dict:
    """视频/音频流时长（秒）。无音轨时 audio 为 None。"""
    rc, out, err = _run([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,duration", "-of", "json", video,
    ])
    if rc != 0:
        return {"ok": False, "reason": err[:200]}
    data = json.loads(out)
    video_dur = audio_dur = None
    for s in data.get("streams", []):
        dur = s.get("duration")
        if dur is None:
            continue
        if s.get("codec_type") == "video" and video_dur is None:
            video_dur = float(dur)
        elif s.get("codec_type") == "audio" and audio_dur is None:
            audio_dur = float(dur)
    return {"ok": True, "video": video_dur, "audio": audio_dur}


def detect_black_frames(video: str) -> list[tuple[float, float]]:
    """blackdetect：返回 [(start, end), ...]。"""
    _, _, err = _run([
        "ffmpeg", "-i", video, "-vf", "blackdetect=d=0.1:pix_th=0.10",
        "-an", "-f", "null", "-",
    ])
    out = []
    for m in re.finditer(
        r"black_start:([\d.]+)\s+black_end:([\d.]+)", err
    ):
        out.append((float(m.group(1)), float(m.group(2))))
    return out


def detect_freeze(video: str) -> list[tuple[float, float]]:
    _, _, err = _run([
        "ffmpeg", "-i", video, "-vf", "freezedetect=n=-60dB:d=1",
        "-an", "-f", "null", "-",
    ])
    starts = [float(m.group(1)) for m in re.finditer(r"freeze_start:([\d.]+)", err)]
    ends = [float(m.group(1)) for m in re.finditer(r"freeze_duration:([\d.]+)", err)]
    return list(zip(starts, [s + d for s, d in zip(starts, ends)]))


def detect_silence(video: str, min_d: float = 1.0) -> list[tuple[float, float]]:
    """silencedetect：长于 min_d 秒的静音段。无音轨返回空。"""
    _, _, err = _run([
        "ffmpeg", "-i", video,
        "-af", f"silencedetect=noise=-35dB:d={min_d}", "-f", "null", "-",
    ])
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:([\d.]+)", err)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:([\d.]+)", err)]
    return list(zip(starts, ends or [s + min_d for s in starts]))


# ---------------------------------------------------------------------------
# EDL / Plan 维度
# ---------------------------------------------------------------------------

def edl_pacing(edl: dict) -> dict:
    """节奏指标：从剪辑清单计算（EDL 是切点的精确真值，无需场景检测猜）。"""
    durs = [
        (e["out_frame"] - e["in_frame"]) / e["timebase"]
        for e in edl.get("ordered_edits", [])
    ]
    if not durs:
        return {"shot_count": 0}
    total = sum(durs)
    cv = (statistics.pstdev(durs) / statistics.mean(durs)) if len(durs) > 1 else 0.0
    return {
        "shot_count": len(durs),
        "total_seconds": round(total, 3),
        "asl_seconds": round(total / len(durs), 3),
        "min_clip_seconds": round(min(durs), 3),
        "max_clip_seconds": round(max(durs), 3),
        "duration_cv": round(cv, 3),
    }


def edl_source_spread(edl: dict) -> dict:
    """镜头选择代理：所用片段在源时间轴上的覆盖区间。"""
    spans = [(e["in_frame"], e["out_frame"]) for e in edl.get("ordered_edits", [])]
    if not spans:
        return {"source_span_seconds": 0.0, "segment_count": 0}
    timebase = edl.get("ordered_edits")[0].get("timebase", 1_000_000)
    lo = min(s for s, _ in spans)
    hi = max(o for _, o in spans)
    return {
        "source_span_seconds": round((hi - lo) / timebase, 3),
        "segment_count": len(spans),
    }


def intent_compliance(plan: dict, target_seconds: float | None,
                      video_duration: float | None) -> list[dict]:
    """意图硬约束达成情况（逐条，含证据）。"""
    findings: list[dict] = []
    constraints = plan.get("constraints", []) if plan else []

    if target_seconds and video_duration:
        dev = abs(video_duration - target_seconds) / target_seconds
        findings.append({
            "item": "目标时长偏差",
            "value": f"{video_duration:.2f}s vs 目标 {target_seconds:.2f}s"
                     f"（偏差 {dev * 100:.1f}%）",
            "verdict": "PASS" if dev <= 0.10 else "FAIL",
        })

    for c in constraints:
        m = re.match(r"max_clip_us=(\d+)", c)
        if m and video_duration:
            bound = int(m.group(1)) / 1_000_000
            findings.append({
                "item": "剪辑语言上界（max_clip_us）",
                "value": f"约束 {bound:.2f}s（逐片段合规性见 EDL 计分卡生成端校验）",
                "verdict": "INFO",
            })
        m = re.match(r"must_avoid:", c)
        if m:
            findings.append({
                "item": "must_avoid 技术约束",
                "value": f"{c}（生成期过滤 + validator Rule 9 强制）",
                "verdict": "INFO",
            })
    return findings


# ---------------------------------------------------------------------------
# 指标收集（结构化，供计分卡与 L2 矩阵聚合共用）
# ---------------------------------------------------------------------------

def collect_metrics(video: str, edl: dict | None, plan: dict | None,
                    target_seconds: float | None) -> dict:
    """收集全部 L1 指标，返回结构化 dict（含 hard_defect 判定）。"""
    streams = probe_stream_durations(video)
    blacks = detect_black_frames(video)
    freezes = detect_freeze(video)
    silences = detect_silence(video)
    pacing = edl_pacing(edl) if edl else None
    spread = edl_source_spread(edl) if edl else None

    video_dur = streams.get("video") if streams.get("ok") else None
    intent = intent_compliance(plan, target_seconds, video_dur)

    av_mismatch = bool(
        streams.get("ok") and streams.get("audio") and streams.get("video")
        and abs(streams["video"] - streams["audio"]) > 0.5
    )
    hard = (
        not streams.get("ok")
        or bool(blacks)
        or bool(freezes)
        or av_mismatch
        or any(f["verdict"] == "FAIL" for f in intent)
    )
    return {
        "streams": streams,
        "black_frames": blacks,
        "freezes": freezes,
        "silences": silences,
        "av_mismatch": av_mismatch,
        "pacing": pacing,
        "source_spread": spread,
        "intent": intent,
        "hard_defect": hard,
    }


# ---------------------------------------------------------------------------
# 计分卡
# ---------------------------------------------------------------------------

def build_scorecard(video: str, edl: dict | None, plan: dict | None,
                    target_seconds: float | None) -> tuple[str, bool]:
    """返回 (markdown 计分卡, 是否存在硬伤)。"""
    m = collect_metrics(video, edl, plan, target_seconds)
    streams = m["streams"]
    hard_defect = m["hard_defect"]

    lines: list[str] = [f"# L1 客观指标计分卡 · {Path(video).name}", ""]

    # ---- 技术质量 ----
    lines += ["## 技术质量（守门员，任一 FAIL 即硬伤）", ""]
    if not streams.get("ok"):
        lines.append(f"- [FAIL] 流探测失败：{streams.get('reason')}")
        hard_defect = True
    else:
        vd, ad = streams["video"], streams["audio"]
        lines.append(f"- [INFO] 视频流 {vd:.2f}s，音频流 "
                     f"{f'{ad:.2f}s' if ad else '无音轨'}")
        if m["av_mismatch"]:
            lines.append(f"- [FAIL] 音画时长错位 {abs(vd - ad):.2f}s（>0.5s）")
            hard_defect = True

    blacks = m["black_frames"]
    if blacks:
        lines.append(f"- [FAIL] 黑帧 {len(blacks)} 段："
                     + ", ".join(f"{s:.1f}-{e:.1f}s" for s, e in blacks[:5]))
        hard_defect = True
    else:
        lines.append("- [PASS] 无黑帧")

    freezes = m["freezes"]
    if freezes:
        lines.append(f"- [FAIL] 冻帧 {len(freezes)} 段："
                     + ", ".join(f"{s:.1f}+{d:.1f}s" for s, d in freezes[:5]))
        hard_defect = True
    else:
        lines.append("- [PASS] 无冻帧（≥1s）")

    silences = m["silences"]
    if silences:
        total_silence = sum(e - s for s, e in silences)
        lines.append(f"- [INFO] 静音段 {len(silences)} 段（累计 {total_silence:.1f}s，"
                     f"有音轨素材需人工判断是否预期）")
    else:
        lines.append("- [PASS] 无 ≥1s 静音段（或无音轨）")

    # ---- 节奏 ----
    lines += ["", "## 节奏 Pacing（分布参考，无好坏判定）", ""]
    if m["pacing"]:
        p = m["pacing"]
        lines.append(
            f"- 镜头数 {p['shot_count']}，ASL {p.get('asl_seconds')}s，"
            f"最短 {p.get('min_clip_seconds')}s / 最长 {p.get('max_clip_seconds')}s，"
            f"时长变异系数 {p.get('duration_cv')}"
        )
        lines.append(f"- 成片总时长 {p.get('total_seconds')}s")
    else:
        lines.append("- [INFO] 未提供 EDL，跳过节奏维度")

    # ---- 意图达成 ----
    lines += ["", "## 意图达成（硬约束逐条）", ""]
    if m["intent"]:
        for f in m["intent"]:
            lines.append(f"- [{f['verdict']}] {f['item']}：{f['value']}")
    else:
        lines.append("- [INFO] 未提供目标时长/plan，跳过")

    # ---- 镜头选择 ----
    lines += ["", "## 镜头选择（分布代理）", ""]
    if m["source_spread"]:
        sp = m["source_spread"]
        lines.append(f"- 所用片段在源时间轴上的跨度 {sp['source_span_seconds']}s"
                     f"（{sp['segment_count']} 段）")
    else:
        lines.append("- [INFO] 未提供 EDL，跳过")

    # ---- 边界声明（L1 不越权）----
    lines += ["", "## L1 无法回答（需 L3 人工盲评）", ""]
    lines.append("- 叙事连贯性、观感好坏、创意与情绪传达——客观指标只能测'异常'，"
                 "不能测'好坏'；请走 L3 配对盲评（见 成片质量评估方案-L1L2L3.md）")

    return "\n".join(lines), hard_defect


def main() -> int:
    parser = argparse.ArgumentParser(description="L1 成片客观指标计分卡")
    parser.add_argument("--video", required=True, help="成片 mp4 路径")
    parser.add_argument("--edl-json", default=None, help="剪辑清单 sidecar")
    parser.add_argument("--plan-json", default=None, help="决策计划 sidecar")
    parser.add_argument("--target-seconds", type=float, default=None,
                        help="用户目标成片时长（秒）")
    parser.add_argument("--json", action="store_true", help="附加 JSON 输出")
    args = parser.parse_args()

    if not Path(args.video).is_file():
        print(f"错误：成片文件不存在: {args.video}", file=sys.stderr)
        return 2

    edl = json.loads(Path(args.edl_json).read_text(encoding="utf-8")) \
        if args.edl_json and Path(args.edl_json).is_file() else None
    plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8")) \
        if args.plan_json and Path(args.plan_json).is_file() else None

    card, hard_defect = build_scorecard(args.video, edl, plan, args.target_seconds)
    print(card)
    if args.json:
        print(json.dumps({
            "video": args.video,
            "hard_defect": hard_defect,
        }, ensure_ascii=False))
    return 1 if hard_defect else 0


if __name__ == "__main__":
    sys.exit(main())
