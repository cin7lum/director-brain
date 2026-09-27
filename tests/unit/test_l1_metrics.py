"""scripts/l1_metrics.py 单元测试。

文件级探测（黑帧/冻帧）用 ffmpeg 生成的微型素材做真实验证；
ffmpeg 缺失时整模块 skip。EDL/意图维度纯计算，不依赖 ffmpeg。
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from scripts.l1_metrics import (
    build_scorecard,
    detect_black_frames,
    edl_pacing,
    intent_compliance,
    probe_stream_durations,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe 不可用",
)


def _render(tmp_path, name: str, args: list[str]) -> str:
    out = str(tmp_path / name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args, out],
                   check=True, capture_output=True, timeout=60)
    return out


def _edl(durs_us: list[int]) -> dict:
    return {
        "timebase": 1_000_000,
        "ordered_edits": [
            {"in_frame": i * 4_000_000, "out_frame": i * 4_000_000 + d,
             "timebase": 1_000_000}
            for i, d in enumerate(durs_us)
        ],
    }


# ---------------------------------------------------------------------------
# EDL 纯计算维度
# ---------------------------------------------------------------------------

def test_edl_pacing_metrics():
    p = edl_pacing(_edl([1_000_000, 3_000_000]))
    assert p["shot_count"] == 2
    assert p["asl_seconds"] == 2.0
    assert p["min_clip_seconds"] == 1.0
    assert p["max_clip_seconds"] == 3.0
    assert 0 < p["duration_cv"] < 1


def test_intent_duration_deviation():
    findings = intent_compliance(None, 15.0, 13.5)
    dev = [f for f in findings if f["item"] == "目标时长偏差"]
    assert dev and dev[0]["verdict"] == "PASS"  # 10% 边界内
    findings = intent_compliance(None, 15.0, 10.0)
    assert findings[0]["verdict"] == "FAIL"  # 33% 超差 → 硬伤
    findings = intent_compliance(
        {"constraints": ["must_avoid:blur_score<50.0:模糊"]}, None, None)
    assert findings[0]["verdict"] == "INFO"


# ---------------------------------------------------------------------------
# 文件级探测（ffmpeg 真实验证）
# ---------------------------------------------------------------------------

def test_probe_and_blackdetect_on_synthetic_video(tmp_path):
    """前 0.5s 黑屏 + 后 0.5s 白屏 → 探测到黑帧；流时长可读。"""
    video = _render(tmp_path, "split.mp4", [
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.5",
        "-f", "lavfi", "-i", "color=c=white:s=64x64:d=0.5",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
    ])
    info = probe_stream_durations(video)
    assert info["ok"] is True and info["video"] is not None
    blacks = detect_black_frames(video)
    assert blacks, "0.5s 黑屏段必须被 blackdetect 捕获"


def test_scorecard_flags_hard_defect(tmp_path):
    video = _render(tmp_path, "black.mp4", [
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1.0",
    ])
    card, hard = build_scorecard(video, _edl([2_000_000, 1_000_000]),
                                 None, 15.0)
    assert hard is True
    assert "黑帧" in card
    assert "L1 无法回答" in card  # 边界声明必须在场


def test_scorecard_clean_video_no_hard_defect(tmp_path):
    video = _render(tmp_path, "white.mp4", [
        "-f", "lavfi", "-i", "testsrc=s=64x64:d=1.0",
    ])
    card, hard = build_scorecard(video, _edl([1_000_000]), None, 1.0)
    assert hard is False
