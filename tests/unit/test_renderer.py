"""execution.renderer 单元测试。

验证：
- 空 EDL 抛 ValueError
- 源视频不存在抛 FileNotFoundError
- 有音轨时 filter_complex 含 trim/atrim、concat n=N:v=1:a=1、-map "[a]"
- 无音轨时不含 [0:a] 与 -map "[a]"，concat a=0
- ffmpeg 失败抛 RuntimeError 且异常信息含 stderr
- edl_to_ffmpeg_concat 生成临时文件，内容含 file/inpoint/outpoint，行数正确
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

from director_brain.models.edl import EditorialDecisionList, EditItem
from execution import renderer


def _make_edl(edits: list[EditItem]) -> EditorialDecisionList:
    return EditorialDecisionList(
        schema_version="1.0",
        project_id="test",
        created_at=0,
        producer="test",
        source_ref="test.mp4",
        edl_id="edl_test",
        version="0.1",
        brief_version="0.1",
        context_id="ctx_test",
        timebase=1_000_000,
        ordered_edits=edits,
        approval_state="draft",
    )


def _edit(in_f: int, out_f: int, asset_id: str = "shot_001") -> EditItem:
    return EditItem(
        source_asset_id=asset_id,
        source_media_hash="hash1",
        in_frame=in_f,
        out_frame=out_f,
        timebase=1_000_000,
    )


def _touch(path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("dummy")
    return path


def _fake_completed(returncode: int = 0, stderr: str = "") -> mock.MagicMock:
    proc = mock.MagicMock()
    proc.returncode = returncode
    proc.stdout = ""
    proc.stderr = stderr
    return proc


def test_render_edl_empty_edl_raises(tmp_path) -> None:
    edl = _make_edl([])
    with pytest.raises(ValueError, match="empty EDL"):
        renderer.render_edl(edl, str(tmp_path / "src.mp4"), str(tmp_path / "out.mp4"))


def test_render_edl_source_not_found(tmp_path) -> None:
    edl = _make_edl([_edit(0, 1_000_000)])
    missing = str(tmp_path / "does_not_exist.mp4")
    with pytest.raises(FileNotFoundError):
        renderer.render_edl(edl, missing, str(tmp_path / "out.mp4"))


def test_filter_complex_with_audio(tmp_path) -> None:
    src = _touch(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    edl = _make_edl(
        [
            _edit(1_000_000, 3_000_000, asset_id="shot_001"),
            _edit(5_000_000, 7_000_000, asset_id="shot_002"),
        ]
    )

    with mock.patch.object(renderer, "_has_audio_stream", return_value=True), mock.patch.object(
        renderer.subprocess, "run", return_value=_fake_completed(0)
    ) as m_run:
        result = renderer.render_edl(edl, src, out)

    assert result == out
    cmd = m_run.call_args.args[0]
    joined = " ".join(cmd)

    # 视频 trim 时间点
    assert "trim=start=1.000000:end=3.000000" in joined
    assert "trim=start=5.000000:end=7.000000" in joined
    # 音频 atrim
    assert "[0:a]atrim=start=1.000000:end=3.000000" in joined
    # concat 交错拼接 n=2 且带音频
    assert "concat=n=2:v=1:a=1[v][a]" in joined
    # 交错标签 [v0][a0][v1][a1]
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]" in joined
    # map 同时映射视频与音频
    assert "-map" in cmd
    assert "[a]" in cmd
    assert "[v]" in cmd


def test_filter_complex_without_audio(tmp_path) -> None:
    src = _touch(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    edl = _make_edl([_edit(1_000_000, 3_000_000)])

    with mock.patch.object(renderer, "_has_audio_stream", return_value=False), mock.patch.object(
        renderer.subprocess, "run", return_value=_fake_completed(0)
    ) as m_run:
        renderer.render_edl(edl, src, out)

    cmd = m_run.call_args.args[0]
    joined = " ".join(cmd)

    assert "[0:a]" not in joined
    assert "atrim" not in joined
    # concat 无音频
    assert "concat=n=1:v=1:a=0[v]" in joined
    # 不应出现 -map "[a]"
    assert "[a]" not in cmd


def test_ffmpeg_failure_raises_with_stderr(tmp_path) -> None:
    src = _touch(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    edl = _make_edl([_edit(0, 1_000_000)])
    err_tail = "Error reinitializing filters! Invalid trim duration"

    with mock.patch.object(renderer, "_has_audio_stream", return_value=True), mock.patch.object(
        renderer.subprocess, "run", return_value=_fake_completed(1, stderr=err_tail)
    ):
        with pytest.raises(RuntimeError, match="ffmpeg render failed") as exc_info:
            renderer.render_edl(edl, src, out)

    assert err_tail in str(exc_info.value)


def test_edl_to_ffmpeg_concat_generates_file() -> None:
    edl = _make_edl(
        [
            _edit(1_000_000, 3_000_000, asset_id="shot_001"),
            _edit(5_000_000, 7_000_000, asset_id="shot_002"),
        ]
    )
    path = renderer.edl_to_ffmpeg_concat(edl)
    try:
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        lines = [ln for ln in content.splitlines() if ln.strip()]
        # 每个 edit 3 行：file / inpoint / outpoint
        assert len(lines) == 6
        assert "file 'shot_001'" in lines
        assert "inpoint 1.000000" in lines
        assert "outpoint 3.000000" in lines
        assert "file 'shot_002'" in lines
        assert "inpoint 5.000000" in lines
        assert "outpoint 7.000000" in lines
    finally:
        if os.path.exists(path):
            os.unlink(path)
