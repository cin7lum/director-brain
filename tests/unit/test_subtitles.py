"""阶段 8a-1 字幕能力测试：SRT 序列化 / 成片时间线映射 / 软字幕流真实渲染。"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import execution.renderer as renderer
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from execution.renderer import render_edl
from observation_service.srt import _fmt_ts, build_srt, map_observations_to_output_timeline

FFMPEG_OK = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _speech(asset_id: str, start_us: int, end_us: int, text: str) -> FilmObservation:
    return FilmObservation(
        observation_id=f"asr_{asset_id}", media_asset_id=asset_id,
        media_hash=f"h_{asset_id}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="speech_transcript",
        claim=text, provider="faster_whisper", model_version="t",
        prompt_version="n/a", confidence=0.8, review_state="auto_generated",
        claim_kind=ClaimKind.MODEL_OBSERVATION, schema_version="1.0",
        project_id="srt", created_at=int(time.time()), producer="faster_whisper",
        source_ref="t.mp4",
    )


def _edl(spans: list[tuple[int, int]]) -> EditorialDecisionList:
    edits = [
        EditItem(source_asset_id=f"s{i}", source_media_hash=f"h{i}",
                 in_frame=a, out_frame=b, timebase=1_000_000)
        for i, (a, b) in enumerate(spans)
    ]
    return EditorialDecisionList(
        schema_version="1.0", project_id="srt", created_at=0, producer="t",
        source_ref="t.mp4", edl_id="e", version="0.1", brief_version="0.1",
        context_id="c", timebase=1_000_000, ordered_edits=edits,
        expected_duration=sum(b - a for a, b in spans), approval_state="draft",
    )


# ---------------------------------------------------------------------------
# 时间戳与映射（纯计算）
# ---------------------------------------------------------------------------

def test_fmt_ts():
    assert _fmt_ts(0) == "00:00:00,000"
    assert _fmt_ts(1_000_000) == "00:00:01,000"
    assert _fmt_ts(3_723_456_789) == "01:02:03,456"
    assert _fmt_ts(-5) == "00:00:00,000"  # 负值钳制


def test_timeline_mapping_drops_unused_regions():
    """成片时间线映射：弃用区间的语音被丢弃，保留区间按片内偏移平移。"""
    edl = _edl([(5_000_000, 8_000_000)])  # 只取源 5-8s
    obs = [
        _speech("a", 1_000_000, 2_000_000, "弃用区间语音"),
        _speech("b", 5_500_000, 6_500_000, "保留语音一"),
        _speech("c", 7_000_000, 9_000_000, "跨界语音（尾部弃用）"),
    ]
    mapped = map_observations_to_output_timeline(obs, edl)
    assert len(mapped) == 2
    # 5.5-6.5s 源 → 成片 0.5-1.5s
    assert mapped[0]["start_us"] == 500_000 and mapped[0]["end_us"] == 1_500_000
    assert mapped[0]["text"] == "保留语音一"
    # 7-9s 与 5-8s 的交集 7-8s → 成片 2.5-3.0s
    assert mapped[1]["start_us"] == 2_000_000 and mapped[1]["end_us"] == 3_000_000


def test_timeline_mapping_contiguous_edits():
    """连续（无间隙）镜头：跨镜头语音在成片中应首尾相接。"""
    edl = _edl([(0, 2_000_000), (2_000_000, 5_000_000)])
    obs = [_speech("a", 1_000_000, 3_000_000, "跨镜头语音")]
    mapped = map_observations_to_output_timeline(obs, edl)
    assert len(mapped) == 2
    assert (mapped[0]["start_us"], mapped[0]["end_us"]) == (1_000_000, 2_000_000)
    assert (mapped[1]["start_us"], mapped[1]["end_us"]) == (2_000_000, 3_000_000)


def test_timeline_mapping_across_multiple_edits():
    """语音段横跨两个镜头 → 按边界切开，两段都映射到成片时间线。"""
    edl = _edl([(0, 2_000_000), (2_000_000, 5_000_000)])
    obs = [_speech("a", 1_000_000, 3_000_000, "跨镜头语音")]
    mapped = map_observations_to_output_timeline(obs, edl)
    assert len(mapped) == 2
    assert (mapped[0]["start_us"], mapped[0]["end_us"]) == (1_000_000, 2_000_000)
    assert (mapped[1]["start_us"], mapped[1]["end_us"]) == (2_000_000, 3_000_000)


def test_build_srt_format():
    edl = _edl([(0, 3_000_000)])
    obs = [_speech("a", 500_000, 2_000_000, "你好世界")]
    srt = build_srt(obs, edl)
    assert "1\n00:00:00,500 --> 00:00:02,000\n你好世界" in srt


# ---------------------------------------------------------------------------
# 真实渲染：软字幕流进 mp4（ffprobe 验证）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FFMPEG_OK, reason="ffmpeg/ffprobe 不可用")
def test_soft_subtitle_stream_in_real_render(tmp_path):
    """真实渲染验证：EDL.subtitle_refs 携带 SRT → 成片含 mov_text 字幕流。"""
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=s=320x240:d=2", src],
                   check=True, capture_output=True, timeout=60)
    srt_file = tmp_path / "cut.srt"
    srt_file.write_text(
        "1\n00:00:00,200 --> 00:00:01,200\n真实渲染字幕\n", encoding="utf-8"
    )
    edl = _edl([(0, 1_000_000), (1_000_000, 2_000_000)])
    edl.subtitle_refs.append(f"{srt_file}|chi")

    out = str(tmp_path / "out.mp4")
    result = render_edl(edl, src, out)
    assert result == out

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name", "-of", "json", out],
        capture_output=True, text=True, timeout=60)
    streams = json.loads(probe.stdout)["streams"]
    sub_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
    assert sub_streams and sub_streams[0]["codec_name"] == "mov_text"


@pytest.mark.skipif(not FFMPEG_OK, reason="ffmpeg/ffprobe 不可用")
def test_render_without_subtitle_refs_unchanged(tmp_path):
    """无字幕引用 → 输出无字幕流（旧行为不回退）。"""
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=s=320x240:d=1", src],
                   check=True, capture_output=True, timeout=60)
    out = str(tmp_path / "out.mp4")
    render_edl(_edl([(0, 1_000_000)]), src, out)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "json", out], capture_output=True, text=True, timeout=60)
    streams = json.loads(probe.stdout)["streams"]
    assert not [s for s in streams if s.get("codec_type") == "subtitle"]
