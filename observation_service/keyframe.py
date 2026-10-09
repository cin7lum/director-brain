"""关键帧抽取模块（修 B3 / P1-6）。

从镜头中点（而非首帧）用 ffmpeg 单帧截图。
中点时间 = (shot_in_us + shot_out_us) / 2 / 1_000_000 秒。

清理约定（P1-6）：
    with extract_keyframe(video_path, shot_in_us, shot_out_us) as path:
        ...使用 path...
    # with 块退出时自动删除临时文件

- 未传 out_path：用 tempfile.mkstemp 创建临时 JPEG，with 块退出时自动删除。
- 传入 out_path：写入指定路径，**不清理**（调用方负责）。
- ffmpeg 执行失败（含异常）时清理临时文件并 yield 空字符串 ""（fail-soft）。
"""
from __future__ import annotations

import contextlib
import subprocess
import tempfile
import os

MULTI_FRAME_SAMPLE_POSITIONS = (0.15, 0.50, 0.85)
MULTI_FRAME_SAMPLING_PROFILE = "shot-relative-v1:0.15,0.50,0.85"


@contextlib.contextmanager
def extract_keyframe(video_path: str, shot_in_us: int, shot_out_us: int,
                     out_path: str | None = None):
    """从镜头中点抽取一帧，作为上下文管理器使用。

    Args:
        video_path: 视频文件路径
        shot_in_us: 镜头入点（微秒）
        shot_out_us: 镜头出点（微秒）
        out_path: 可选输出路径。传入时写入指定路径且不清理（调用方负责）；
                  未传入时使用临时文件，with 块退出时自动删除。

    Yields:
        JPEG 文件路径；失败时 yield 空字符串 ""。
    """
    midpoint_sec = (shot_in_us + shot_out_us) / 2 / 1_000_000

    temp_created = out_path is None
    if temp_created:
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
    else:
        path = out_path

    cmd = [
        "ffmpeg",
        "-ss", f"{midpoint_sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        path,
        "-y",
    ]

    try:
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            ok = (
                result.returncode == 0
                and os.path.exists(path)
                and os.path.getsize(path) > 0
            )
        except Exception:
            ok = False

        if not ok:
            # ffmpeg 失败：清理临时文件并 yield 空串（fail-soft）
            if temp_created and os.path.exists(path):
                os.remove(path)
            yield ""
            return

        yield path
    finally:
        # with 块退出（含异常退出）时，自动清理临时文件
        if temp_created and os.path.exists(path):
            os.remove(path)


@contextlib.contextmanager
def extract_keyframes(video_path: str, shot_in_us: int, shot_out_us: int,
                      positions: tuple[float, ...] = MULTI_FRAME_SAMPLE_POSITIONS,
                      *, local_only: bool = False):
    """按镜头内相对位置抽多帧（P3 语义观测：一次多图 VLM 调用）。

    与 :func:`extract_keyframe` 同一清理约定：临时帧 with 块退出自动删除。
    ``local_only=True`` restricts FFmpeg input protocols to local files for
    private-media preview routes; the default preserves existing callers.
    个别位置抽取失败时该帧**不占位**（成功帧列表可能短于 positions）；
    全部失败 yield 空列表（fail-soft，调用方按无帧降级）。
    """
    dur = shot_out_us - shot_in_us
    paths: list[str] = []
    fds = []
    try:
        for frac in positions:
            fd, path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            fds.append(path)
            pos_sec = (shot_in_us + int(dur * frac)) / 1_000_000
            cmd = ["ffmpeg"]
            if local_only:
                # The source has already passed the allowed-root and hash
                # checks. Restrict FFmpeg itself to local-file inputs so a
                # playlist or container reference cannot initiate a network
                # protocol while making a private-media preview.
                cmd.extend(["-protocol_whitelist", "file"])
            cmd.extend([
                "-ss", f"{pos_sec:.3f}",
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                path,
                "-y",
            ])
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                ok = (
                    result.returncode == 0
                    and os.path.exists(path)
                    and os.path.getsize(path) > 0
                )
            except Exception:
                ok = False
            if ok:
                paths.append(path)
        yield paths
    finally:
        for path in fds:
            if os.path.exists(path):
                os.remove(path)
