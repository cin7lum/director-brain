"""关键帧抽取模块（修 B3）。

从镜头中点（而非首帧）用 ffmpeg 单帧截图。
中点时间 = (shot_in_us + shot_out_us) / 2 / 1_000_000 秒。
输出到临时文件，返回临时文件路径；失败时返回空字符串。
"""
from __future__ import annotations

import subprocess
import tempfile
import os


def extract_keyframe(video_path: str, shot_in_us: int, shot_out_us: int) -> str:
    """从镜头中点抽取一帧，输出为 JPEG 临时文件。

    Args:
        video_path: 视频文件路径
        shot_in_us: 镜头入点（微秒）
        shot_out_us: 镜头出点（微秒）

    Returns:
        临时 JPEG 文件的绝对路径；失败时返回空字符串 ""。
    """
    midpoint_sec = (shot_in_us + shot_out_us) / 2 / 1_000_000

    # 创建临时输出文件
    fd, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-ss", f"{midpoint_sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        out_path,
        "-y",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            # 清理临时文件
            if os.path.exists(out_path):
                os.remove(out_path)
            return ""
        # 验证输出文件存在且非空
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            if os.path.exists(out_path):
                os.remove(out_path)
            return ""
        return out_path
    except Exception:
        if os.path.exists(out_path):
            os.remove(out_path)
        return ""
