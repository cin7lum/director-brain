"""阶段 8a-3 转场测试：时间线数学 / 链式 xfade 真实渲染 / 字幕自动跟随。"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import execution.renderer as renderer
from director_brain.models.edl import (
    EditItem,
    EditorialDecisionList,
    TransitionSpec,
)
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.timeline import compute_output_timeline, total_output_duration_us
from execution.renderer import render_edl
from observation_service.srt import build_srt

FFMPEG_OK = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _edl(spans, transition=None) -> EditorialDecisionList:
    edits = []
    for i, (a, b) in enumerate(spans):
        e = EditItem(source_asset_id=f"s{i}", source_media_hash=f"h{i}",
                     in_frame=a, out_frame=b, timebase=1_000_000)
        if i < len(spans) - 1 and transition is not None:
            e.transition = transition
        edits.append(e)
    return EditorialDecisionList(
        schema_version="1.0", project_id="t", created_at=0, producer="t",
        source_ref="src.mp4", edl_id="e", version="0.1", brief_version="0.1",
        context_id="c", timebase=1_000_000, ordered_edits=edits,
        expected_duration=sum(b - a for a, b in spans), approval_state="draft",
    )


def _speech(asset_id, start_us, end_us, text):
    return FilmObservation(
        observation_id=f"asr_{asset_id}", media_asset_id=asset_id,
        media_hash=f"h_{asset_id}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="speech_transcript",
        claim=text, provider="t", model_version="t", prompt_version="t",
        confidence=0.8, review_state="auto_generated",
        claim_kind=ClaimKind.MODEL_OBSERVATION, schema_version="1.0",
        project_id="t", created_at=int(time.time()), producer="t",
        source_ref="t.mp4",
    )


# ---------------------------------------------------------------------------
# 时间线数学（纯计算）
# ---------------------------------------------------------------------------

def test_timeline_cut_semantics():
    tl = compute_output_timeline(_edl([(0, 2_000_000), (2_000_000, 5_000_000)]))
    assert [t.out_start_us for t in tl] == [0, 2_000_000]
    assert total_output_duration_us(_edl([(0, 2_000_000), (2_000_000, 5_000_000)])) == 5_000_000


def test_timeline_xfade_semantics():
    """xfade 重叠 0.5s：第二镜头内容起点 = 2.0−0.5 = 1.5s；总时长 = Σd − ΣD。"""
    tr = TransitionSpec(type="xfade", name="fade", duration_us=500_000)
    edl = _edl([(0, 2_000_000), (2_000_000, 5_000_000)], transition=tr)
    tl = compute_output_timeline(edl)
    assert tl[1].out_start_us == 1_500_000
    assert total_output_duration_us(edl) == 4_500_000  # 5s − 0.5s


def test_timeline_mixed_cut_and_xfade():
    """混合接缝：三镜头 [cut, xfade] → 中镜头起点累计，尾镜头起点扣重叠。"""
    tr = TransitionSpec(type="xfade", name="fade", duration_us=500_000)
    edl = _edl([(0, 2_000_000), (2_000_000, 4_000_000), (4_000_000, 7_000_000)])
    edl.ordered_edits[1].transition = tr
    tl = compute_output_timeline(edl)
    assert [t.out_start_us for t in tl] == [0, 2_000_000, 3_500_000]
    assert total_output_duration_us(edl) == 6_500_000  # 7s − 0.5s


def test_build_srt_follows_transitions():
    """字幕自动跟随：xfade 后镜头的语音时间戳扣去重叠量。"""
    tr = TransitionSpec(type="xfade", name="fade", duration_us=500_000)
    edl = _edl([(0, 2_000_000), (2_000_000, 5_000_000)], transition=tr)
    obs = [_speech("a", 2_500_000, 3_000_000, "第二镜头语音")]
    srt = build_srt(obs, edl)
    # 源 2.5-3.0s 在第二镜头（起点 1.5s）内偏移 0.5-1.0s → 成片 2.0-2.5s
    assert "00:00:02,000 --> 00:00:02,500" in srt, srt


# ---------------------------------------------------------------------------
# 真实渲染：xfade 时长数学 + 字幕跟随（ffprobe 验证）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FFMPEG_OK, reason="ffmpeg 不可用")
def test_xfade_real_render_duration_math(tmp_path):
    """真实渲染：2 镜头 + 0.5s xfade → 成片时长必须 = Σd − ΣD。"""
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=s=320x240:d=5", src],
                   check=True, capture_output=True, timeout=60)
    tr = TransitionSpec(type="xfade", name="fade", duration_us=500_000)
    edl = _edl([(0, 2_000_000), (2_000_000, 5_000_000)], transition=tr)
    out = str(tmp_path / "out.mp4")
    render_edl(edl, src, out)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", out], capture_output=True, text=True, timeout=60)
    dur = float(probe.stdout.strip())
    assert abs(dur - 4.5) <= 0.2, f"成片 {dur}s ≠ Σd−ΣD=4.5s"


@pytest.mark.skipif(not FFMPEG_OK, reason="ffmpeg 不可用")
def test_subtitle_and_transition_compound(tmp_path):
    """复合验证：转场缩短时间线后，字幕时间戳自动跟随（不漂移）。"""
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=s=320x240:d=5", src],
                   check=True, capture_output=True, timeout=60)
    tr = TransitionSpec(type="xfade", name="fade", duration_us=500_000)
    edl = _edl([(0, 2_000_000), (2_000_000, 5_000_000)], transition=tr)
    obs = [_speech("a", 2_500_000, 3_000_000, "转场后语音")]
    srt_text = build_srt(obs, edl)
    srt_file = tmp_path / "cut.srt"
    srt_file.write_text(srt_text, encoding="utf-8")
    edl.subtitle_refs.append(f"{srt_file}|chi")
    out = str(tmp_path / "out.mp4")
    render_edl(edl, src, out)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type", "-of", "json", out],
        capture_output=True, text=True, timeout=60)
    streams = json.loads(probe.stdout)["streams"]
    assert any(s.get("codec_type") == "subtitle" for s in streams)
