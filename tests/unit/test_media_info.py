"""observation_service.media_info 音轨探测测试（T4 ASR 归因）。

用 ffmpeg 现场生成两个微型素材：无音轨（纯色视频）与有音轨（正弦波），
验证 probe_audio_stream 的确定性归因。ffmpeg 缺失时整模块 skip。
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from observation_service.media_info import probe_audio_stream

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe 不可用",
)


def _ffmpeg(args: list[str], tmp_path, name: str) -> str:
    out = str(tmp_path / name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args, out],
                   check=True, capture_output=True, timeout=60)
    return out


def test_video_without_audio_stream(tmp_path):
    video = _ffmpeg(
        ["-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.5"], tmp_path, "no_audio.mp4"
    )
    info = probe_audio_stream(video)
    assert info.ok is True
    assert info.has_audio is False
    assert info.duration_us is not None and info.duration_us > 0


def test_video_with_audio_stream(tmp_path):
    video = _ffmpeg(
        ["-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
         "-c:v", "libx264", "-c:a", "aac", "-shortest"],
        tmp_path, "with_audio.mp4",
    )
    info = probe_audio_stream(video)
    assert info.ok is True
    assert info.has_audio is True
    assert info.codec == "aac"


def test_probe_fails_soft_on_garbage(tmp_path):
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not a video")
    info = probe_audio_stream(str(garbage))
    assert info.ok is False
    assert info.reason
