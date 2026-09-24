"""将 EDL（EditorialDecisionList）渲染为视频文件。

单源视频场景使用 ffmpeg ``filter_complex`` concat 直接渲染，全程不产生中间文件；
``edl_to_ffmpeg_concat`` 为多源场景预留的 concat demuxer 配置生成接口。
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from director_brain.models.edl import EditorialDecisionList

logger = logging.getLogger(__name__)


def _has_audio_stream(video_path: str) -> bool:
    """探测视频文件是否包含音频流。

    通过 ffprobe 查询 code_type 为 audio 的流；有输出行即视为存在音轨。
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - ffprobe 缺失时按无音轨处理
        logger.warning("ffprobe 调用失败，按无音轨处理: %s", exc)
        return False
    output = (result.stdout or "").strip()
    return bool(output)


def _build_filter_complex(edl: EditorialDecisionList, has_audio: bool) -> str:
    """根据 EDL 剪辑点构建 ffmpeg filter_complex 字符串。"""
    parts: list[str] = []
    v_labels: list[str] = []
    a_labels: list[str] = []
    for i, edit in enumerate(edl.ordered_edits):
        start = edit.in_frame / edit.timebase
        end = edit.out_frame / edit.timebase
        start_s = f"{start:.6f}"
        end_s = f"{end:.6f}"
        parts.append(
            f"[0:v]trim=start={start_s}:end={end_s},setpts=PTS-STARTPTS,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2[v{i}];"
        )
        v_labels.append(f"[v{i}]")
        if has_audio:
            parts.append(
                f"[0:a]atrim=start={start_s}:end={end_s},asetpts=PTS-STARTPTS[a{i}];"
            )
            a_labels.append(f"[a{i}]")

    n = len(edl.ordered_edits)
    if has_audio:
        # 交错排列：[v0][a0][v1][a1]...
        interleaved: list[str] = []
        for v, a in zip(v_labels, a_labels):
            interleaved.extend([v, a])
        concat_input = "".join(interleaved)
        parts.append(f"{concat_input}concat=n={n}:v=1:a=1[v][a]")
    else:
        concat_input = "".join(v_labels)
        parts.append(f"{concat_input}concat=n={n}:v=1:a=0[v]")

    return "".join(parts)


def render_edl(edl: EditorialDecisionList, source_video: str, output_path: str) -> str:
    """将 EDL 渲染为视频文件（单源视频主路径）。

    使用 ffmpeg ``filter_complex`` concat 直接裁剪并拼接镜头，不产生中间文件。

    Args:
        edl: 编辑决策表。
        source_video: 源视频文件路径。
        output_path: 输出视频文件路径。

    Returns:
        输出视频文件路径。

    Raises:
        ValueError: EDL 无任何剪辑。
        FileNotFoundError: 源视频不存在。
        RuntimeError: ffmpeg 渲染失败（异常信息含 stderr 尾部）。
    """
    if not edl.ordered_edits:
        raise ValueError("empty EDL: no edits to render")

    if not os.path.isfile(source_video):
        raise FileNotFoundError(f"source video not found: {source_video}")

    has_audio = _has_audio_stream(source_video)
    filter_complex = _build_filter_complex(edl, has_audio)
    logger.info(
        "render EDL: %d edits, source=%s, has_audio=%s",
        len(edl.ordered_edits),
        source_video,
        has_audio,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source_video,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
    ]
    if has_audio:
        cmd += ["-map", "[a]"]
    cmd += [
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-preset",
        "fast",
        output_path,
    ]

    logger.debug("ffmpeg 命令: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-2000:]
        raise RuntimeError(f"ffmpeg render failed: {stderr_tail}")

    return output_path


def edl_to_ffmpeg_concat(
    edl: EditorialDecisionList,
    output_path: str | None = None,
) -> str:
    """生成 ffmpeg concat demuxer 配置文件（多源场景预留接口）。

    每行写入 ``file '<source_asset_id>'``，紧随其后 ``inpoint`` / ``outpoint``（秒）。

    注意：``EditItem.source_asset_id`` 是镜头 ID 而非实际文件路径。单源视频场景应
    直接使用 :func:`render_edl`；本函数为多源场景预留，写入的 ``file`` 行需由调用方
    将其中的镜头 ID 替换为真实视频文件路径后再交由 ffmpeg concat demuxer 使用。

    Args:
        edl: 编辑决策表。
        output_path: 可选的配置文件目标路径；不传则用 ``tempfile.NamedTemporaryFile``
            生成一个 ``delete=False`` 的临时文件。无论哪种方式，返回生成文件路径，
            临时文件均由调用方负责清理。

    Returns:
        生成的 concat 配置文件路径。
    """
    lines: list[str] = []
    for edit in edl.ordered_edits:
        inpoint = edit.in_frame / edit.timebase
        outpoint = edit.out_frame / edit.timebase
        lines.append(f"file '{edit.source_asset_id}'")
        lines.append(f"inpoint {inpoint:.6f}")
        lines.append(f"outpoint {outpoint:.6f}")

    content = "\n".join(lines)
    if lines:
        content += "\n"

    if output_path is None:
        fh = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        with fh:
            fh.write(content)
        path = fh.name
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        path = output_path

    logger.info(
        "已生成 concat demuxer 配置: %s (%d edits)", path, len(edl.ordered_edits)
    )
    return path
