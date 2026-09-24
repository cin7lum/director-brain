"""P1-6 keyframe 临时文件清理测试。

验证 extract_keyframe 改造为上下文管理器后的清理约定：
1. 成功抽帧返回有效路径，with 块内文件存在
2. with 块退出后临时文件被自动删除
3. 连续调用 10 次 temp 目录无 .jpg 堆积
4. 传入 out_path 时写入指定路径且 with 退出后不清理（调用方负责）
5. 视频不存在时 yield 空字符串 ""（fail-soft）
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from observation_service.keyframe import extract_keyframe


def _make_test_video(path: str, duration_sec: int = 3) -> None:
    """用 ffmpeg testsrc 滤镜生成一段彩色测试视频（H.264 / yuv420p）。"""
    cmd = [
        "ffmpeg", "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=320x240:rate=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path, "-y",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


@pytest.fixture()
def test_video(tmp_path):
    """现场生成的 3s 测试视频路径（str）。"""
    p = tmp_path / "kf_test.mp4"
    _make_test_video(str(p), duration_sec=3)
    return str(p)


def _temp_jpgs() -> set[str]:
    """系统临时目录中当前所有 .jpg 文件名集合。"""
    tmpdir = tempfile.gettempdir()
    return {
        name for name in os.listdir(tmpdir)
        if name.lower().endswith(".jpg")
    }


class TestExtractKeyframeContextManager:
    def test_extract_keyframe_returns_path(self, test_video):
        """with 块内 path 非空且文件存在。"""
        with extract_keyframe(test_video, 0, 3_000_000) as path:
            assert path != "", "抽帧失败不应返回空路径"
            assert os.path.exists(path), f"输出文件不存在: {path}"
            assert os.path.getsize(path) > 0, "输出文件大小为 0"

    def test_temp_file_cleaned_after_with(self, test_video):
        """with 块退出后临时文件被删除。"""
        with extract_keyframe(test_video, 0, 3_000_000) as path:
            assert path != ""
            assert os.path.exists(path), "with 块内文件应存在"
            leaked_path = path
        # with 退出后应已清理
        assert not os.path.exists(leaked_path), \
            f"with 退出后临时文件未清理: {leaked_path}"

    def test_consecutive_calls_no_leak(self, test_video):
        """连续调用 10 次，temp 目录 .jpg 数量不增长。"""
        before = _temp_jpgs()
        for _ in range(10):
            with extract_keyframe(test_video, 0, 3_000_000) as path:
                assert path != ""
                assert os.path.exists(path)
        after = _temp_jpgs()
        # 本次运行产生的临时文件应全部被清理，after 中不应出现 before 没有的新文件
        new_files = after - before
        assert not new_files, f"temp 目录出现未清理的临时文件: {new_files}"

    def test_out_path_not_cleaned(self, test_video, tmp_path):
        """传入 out_path 时写入指定路径，with 退出后不清理（调用方负责）。"""
        out_path = str(tmp_path / "my_keyframe.jpg")
        with extract_keyframe(test_video, 0, 3_000_000, out_path=out_path) as path:
            assert path == out_path, "应 yield 传入的 out_path"
            assert os.path.exists(out_path), "with 块内 out_path 文件应存在"
        # with 退出后 out_path 不应被自动清理
        assert os.path.exists(out_path), \
            "传入 out_path 时不应自动清理（调用方负责）"
        # 测试手动清理
        os.remove(out_path)

    def test_nonexistent_video_returns_empty(self):
        """视频路径不存在时 with 块中 path == ""，不抛异常。"""
        with extract_keyframe("/nonexistent/path.mp4", 0, 1_000_000) as path:
            assert path == "", "视频不存在应 yield 空字符串"
