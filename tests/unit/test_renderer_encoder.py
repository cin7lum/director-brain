"""P2-a 渲染硬件编码测试：NVENC 选择、失败回退、集成冒烟。"""
from __future__ import annotations

import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

import execution.renderer as renderer
from director_brain.models.edl import EditItem, EditorialDecisionList
from execution.renderer import (
    _nvenc_available,
    _video_codec_args,
    render_edl,
)


@pytest.fixture()
def reset_nvenc_cache():
    renderer._NVENC_CACHE = None
    yield
    renderer._NVENC_CACHE = None


def _edl() -> EditorialDecisionList:
    edits = [
        EditItem(source_asset_id="s0", source_media_hash="h0",
                 in_frame=0, out_frame=1_000_000, timebase=1_000_000),
        EditItem(source_asset_id="s1", source_media_hash="h1",
                 in_frame=1_000_000, out_frame=2_000_000, timebase=1_000_000),
    ]
    return EditorialDecisionList(
        schema_version="1.0", project_id="p", created_at=0, producer="t",
        source_ref="src.mp4", edl_id="e", version="0.1", brief_version="0.1",
        context_id="c", timebase=1_000_000, ordered_edits=edits,
        expected_duration=2_000_000, approval_state="draft",
    )


def test_nvenc_detection_from_encoders_list(reset_nvenc_cache, monkeypatch):
    fake = MagicMock(returncode=0,
                     stdout=" V....D h264_nvenc           NVIDIA NVENC H.264 encoder")
    monkeypatch.setattr(renderer.subprocess, "run", lambda *a, **kw: fake)
    assert _nvenc_available() is True


def test_nvenc_absent_when_not_listed(reset_nvenc_cache, monkeypatch):
    fake = MagicMock(returncode=0, stdout=" V....D libx264              libx264 H.264")
    monkeypatch.setattr(renderer.subprocess, "run", lambda *a, **kw: fake)
    assert _nvenc_available() is False


def test_codec_args_mapping():
    assert _video_codec_args("h264_nvenc")[:2] == ["-c:v", "h264_nvenc"]
    assert _video_codec_args("libx264")[:2] == ["-c:v", "libx264"]


def test_render_falls_back_to_libx264_on_nvenc_failure(
    reset_nvenc_cache, monkeypatch, tmp_path
):
    """NVENC 运行期失败（如低于最小分辨率）→ 响亮回退软编并成功出片。"""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"fake")
    monkeypatch.setattr(renderer, "_NVENC_CACHE", True)
    monkeypatch.setattr(renderer, "_has_audio_stream", lambda p: False)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "h264_nvenc" in cmd:
            return MagicMock(returncode=1, stderr="NVENC min resolution error")
        return MagicMock(returncode=0)

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    out = str(tmp_path / "out.mp4")
    result = render_edl(_edl(), str(src), out)
    assert result == out
    assert len(calls) == 2
    assert "h264_nvenc" in calls[0]
    assert "libx264" in calls[1]


def test_render_uses_nvenc_when_available(reset_nvenc_cache, monkeypatch, tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"fake")
    monkeypatch.setattr(renderer, "_NVENC_CACHE", True)
    monkeypatch.setattr(renderer, "_has_audio_stream", lambda p: False)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        renderer, "_run_ffmpeg", lambda cmd: calls.append(cmd) or None
    )
    out = str(tmp_path / "out.mp4")
    render_edl(_edl(), str(src), out)
    assert len(calls) == 1 and "h264_nvenc" in calls[0]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not _nvenc_available(),
    reason="ffmpeg/NVENC 不可用",
)
def test_nvenc_real_render_smoke(tmp_path):
    """真实 NVENC 冒烟：320x240 testsrc 源，双镜头 EDL → 出片可读。"""
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=s=320x240:d=2", src],
                   check=True, capture_output=True, timeout=60)
    out = str(tmp_path / "cut.mp4")
    result = render_edl(_edl(), src, out)
    assert result == out
    import os
    assert os.path.getsize(out) > 0
