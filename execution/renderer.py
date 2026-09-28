"""将 EDL（EditorialDecisionList）渲染为视频文件。

单源视频场景使用 ffmpeg ``filter_complex`` concat 直接渲染，全程不产生中间文件；
``edl_to_ffmpeg_concat`` 为多源场景预留的 concat demuxer 配置生成接口。

P2-a 硬件加速：检测到 NVIDIA GPU（ffmpeg 带 h264_nvenc）时自动用硬件编码，
失败自动回退 libx264 软编（响亮留痕）。回退同时覆盖 NVENC 最小帧尺寸限制
（低于 145px 宽的微型视频硬件编码器会拒绝）。
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
import tempfile

from director_brain.models.edl import EditorialDecisionList

logger = logging.getLogger(__name__)

_NVENC_CACHE: bool | None = None


def _nvenc_available() -> bool:
    """探测 ffmpeg 是否带 h264_nvenc 硬件编码器（结果缓存）。"""
    global _NVENC_CACHE
    if _NVENC_CACHE is None:
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, check=False, timeout=30,
            )
            _NVENC_CACHE = "h264_nvenc" in (result.stdout or "")
        except (OSError, subprocess.TimeoutExpired):
            _NVENC_CACHE = False
        logger.info("NVENC 检测: %s", "可用" if _NVENC_CACHE else "不可用")
    return _NVENC_CACHE


def _video_codec_args(encoder: str) -> list[str]:
    """编码参数：NVENC p4+vbr(cq23) 与 x264 preset fast 质量语义对齐。"""
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23"]
    return ["-c:v", "libx264", "-preset", "fast"]


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


def _build_filter_complex(
    edl: EditorialDecisionList,
    has_audio: bool,
    audio_out_label: str = "a",
) -> str:
    """根据 EDL 剪辑点构建 ffmpeg filter_complex 字符串。

    audio_out_label：音频链末端标签（8a-2 配乐混音时传 "abase"，混音图再
    产出最终 [a]；无配乐时保持默认 [a]）。
    转场（8a-3）：type=xfade 的接缝走链式 xfade/acrossfade（总时长 = Σd −
    ΣD，offset 数学来自 :mod:`director_brain.timeline`）；cut 或全缺省走
    concat（原路径，行为不变）。
    """
    from director_brain.timeline import compute_output_timeline

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
            f"settb=AVTB,scale=trunc(iw/2)*2:trunc(ih/2)*2[v{i}];"
        )
        v_labels.append(f"v{i}")
        if has_audio:
            parts.append(
                f"[0:a]atrim=start={start_s}:end={end_s},asetpts=PTS-STARTPTS[a{i}];"
            )
            a_labels.append(f"a{i}")

    timeline = compute_output_timeline(edl)
    n = len(edl.ordered_edits)
    use_xfade = any(
        tl.transition_to_next is not None
        and tl.transition_to_next.type == "xfade"
        for tl in timeline[: n - 1]
    )

    if not use_xfade:
        if has_audio:
            interleaved: list[str] = []
            for v, a in zip(v_labels, a_labels):
                interleaved.extend([f"[{v}][{a}]"])
            parts.append(
                f"{''.join(interleaved)}concat=n={n}:v=1:a=1[v][{audio_out_label}]"
            )
        else:
            concat_input = "".join(f"[{v}]" for v in v_labels)
            parts.append(f"{concat_input}concat=n={n}:v=1:a=0[v]")
        return "".join(parts)

    # ---- 链式转场：逐接缝 xfade（视频）/acrossfade（音频），cut 接缝 concat ----
    acc_v = v_labels[0]
    acc_a = a_labels[0] if has_audio else None
    for k in range(n - 1):
        tl = timeline[k]
        junction = tl.transition_to_next
        next_v = v_labels[k + 1]
        is_xfade = (junction is not None and junction.type == "xfade")
        if is_xfade:
            dur_s = junction.duration_us / 1_000_000
            offset_s = (tl.out_start_us + tl.duration_us - junction.duration_us) / 1_000_000
            if offset_s < 0:
                offset_s = 0.0
            parts.append(
                f"[{acc_v}][{next_v}]xfade=transition={junction.name}:"
                f"duration={dur_s:.6f}:offset={offset_s:.6f}[vx{k}];"
            )
            acc_v = f"vx{k}"
            if has_audio:
                adur_s = ((junction.audio_duration_us or junction.duration_us)
                          / 1_000_000)
                parts.append(
                    f"[{acc_a}][{a_labels[k + 1]}]acrossfade=d={adur_s:.6f}[ax{k}];"
                )
                acc_a = f"ax{k}"
        else:
            if has_audio:
                parts.append(
                    f"[{acc_v}][{next_v}]concat=n=2:v=1:a=0[vc{k}];"
                    f"[{acc_a}][{a_labels[k + 1]}]concat=n=2:v=0:a=1[ac{k}];"
                )
                acc_v, acc_a = f"vc{k}", f"ac{k}"
            else:
                parts.append(f"[{acc_v}][{next_v}]concat=n=2:v=1:a=0[vc{k}];")
                acc_v = f"vc{k}"

    if has_audio:
        parts.append(f"[{acc_v}][{acc_a}]concat=n=1:v=1:a=1[v][{audio_out_label}]")
    else:
        parts.append(f"[{acc_v}]null[v]")
    return "".join(parts)


def _run_ffmpeg(cmd: list[str]) -> None:
    """执行 ffmpeg 命令，失败抛 RuntimeError（含 stderr 尾部）。"""
    logger.debug("ffmpeg 命令: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-2000:]
        raise RuntimeError(f"ffmpeg render failed: {stderr_tail}")


def render_edl(edl: EditorialDecisionList, source_video: str, output_path: str) -> str:
    """将 EDL 渲染为视频文件（单源视频主路径）。

    使用 ffmpeg ``filter_complex`` concat 直接裁剪并拼接镜头，不产生中间文件。
    编码器自动选择：有 NVIDIA GPU 用 h264_nvenc 硬编，失败自动回退 libx264。

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

    # 阶段 8a-1/8a-4：字幕（subtitle_refs 内的 SRT 路径）
    # 约定："path|lang[|hard]"——hard=烧录进画面（libass），缺省软字幕流
    subtitle_inputs: list[tuple[str, str]] = []  # 软字幕 (srt_path, lang)
    hardsub_paths: list[str] = []               # 硬字幕（末端 subtitles 滤镜）
    for ref in edl.subtitle_refs:
        fields = ref.split("|")
        path = fields[0]
        lang = fields[1] if len(fields) > 1 else "chi"
        hard = len(fields) > 2 and fields[2] == "hard"
        if not Path(path).is_file():
            logger.warning("字幕文件不存在，跳过: %s", path)
            continue
        if hard:
            hardsub_paths.append(path)
        else:
            subtitle_inputs.append((path, lang))

    # 阶段 8a-2/9：配乐混音（audio_refs: "path|gain_db[|duck]"）
    # duck=人声时自动压低配乐（sidechaincompress，参数取社区标准值）
    music_inputs: list[tuple[str, float]] = []
    duck_requested = False
    for ref in edl.audio_refs:
        if "|" in ref:
            fields = ref.split("|")
            path = fields[0]
            try:
                gain = float(fields[1]) if len(fields) > 1 else -8.0
            except ValueError:
                gain = -8.0
            duck_requested = duck_requested or (len(fields) > 2 and fields[2] == "duck")
        else:
            path, gain = ref, -8.0
        if Path(path).is_file():
            music_inputs.append((path, gain))
        else:
            logger.warning("配乐文件不存在，跳过: %s", path)

    # 混音图必须在命令组装**之前**完成拼接（filter_complex 是不可变字符串）
    audio_out_label = "abase" if (music_inputs and has_audio) else "a"
    filter_complex = _build_filter_complex(edl, has_audio, audio_out_label)
    if music_inputs:
        base_idx = 1 + len(subtitle_inputs)
        total_us = sum(e.out_frame - e.in_frame for e in edl.ordered_edits)
        total_s = f"{total_us / 1_000_000:.6f}"
        fade_start = f"{max(0.0, total_us / 1_000_000 - 0.5):.6f}"
        for j, (_mpath, gain) in enumerate(music_inputs):
            in_idx = base_idx + j
            filter_complex += (
                f";[{in_idx}:a]aloop=loop=-1:size=2000000000,"
                f"atrim=0:{total_s},volume={gain}dB,"
                f"afade=t=in:d=0.5,afade=t=out:st={fade_start}:d=0.5[bg{j}]"
            )
        music_labels = "".join(f"[bg{j}]" for j in range(len(music_inputs)))
        if has_audio:
            if duck_requested:
                # 侧链 ducking：原声（abase）触发压缩配乐（bg 总线）
                filter_complex += (
                    f";[abase]{music_labels}"
                    f"amix=inputs={1 + len(music_inputs)}:duration=first:normalize=0[voicebus];"
                    f"[bg0][voicebus]sidechaincompress=threshold=0.03:ratio=8:"
                    f"attack=50:release=500[ducked]"
                ) if len(music_inputs) == 1 else None
                if len(music_inputs) == 1:
                    filter_complex += (
                        f";[ducked][voicebus]amix=inputs=2:duration=first:"
                        f"normalize=0[a]"
                    )
                else:
                    # 多配乐+duck：MVP 回退普通混音（响亮提示）
                    logger.warning("多配乐 + duck 组合暂不支持，回退普通混音")
                    filter_complex += (
                        f";[abase]{music_labels}"
                        f"amix=inputs={n}:duration=first:normalize=0[a]"
                    )
            else:
                n = 1 + len(music_inputs)
                filter_complex += (
                    f";[abase]{music_labels}"
                    f"amix=inputs={n}:duration=first:normalize=0[a]"
                )
        else:
            n = len(music_inputs)
            filter_complex += (
                f";{music_labels}amix=inputs={n}:duration=first:normalize=0[a]"
            )

    logger.info(
        "render EDL: %d edits, source=%s, has_audio=%s, music=%d, subs=%d",
        len(edl.ordered_edits), source_video, has_audio,
        len(music_inputs), len(subtitle_inputs),
    )

    # 硬字幕（libass 烧录）：Windows 路径转义（反斜杠→正斜杠，冒号→\:）
    # 在命令组装前完成 filter_complex 最终形态（_build_cmd 只读不写）
    if hardsub_paths:
        esc = hardsub_paths[0].replace("\\", "/").replace(":", "\\:")
        filter_complex += f";[v]subtitles=filename='{esc}'[vsub]"

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _build_cmd(encoder: str) -> list[str]:
        video_label = "vsub" if hardsub_paths else "v"
        cmd = ["ffmpeg", "-y", "-i", source_video]
        for srt_path, _lang in subtitle_inputs:
            cmd += ["-i", srt_path]
        for music_path, _gain in music_inputs:
            cmd += ["-i", music_path]
        cmd += ["-filter_complex", filter_complex, "-map", f"[{video_label}]"]
        # 混音/原声都从 filter_complex 的 [a] 出；无音轨且无配乐则无音频流
        if has_audio or music_inputs:
            cmd += ["-map", "[a]"]
        # 字幕流：软字幕的 SRT 是输入 i+1（源视频为输入 0），各取其 0 号流；
        # 硬字幕已烧录，不加流
        for idx, (_srt_path, lang) in enumerate(subtitle_inputs):
            cmd += [
                "-map", f"{idx + 1}:0",
                "-c:s", "mov_text",
                "-metadata:s:s:{0}".format(idx), f"language={lang}",
            ]
        cmd += _video_codec_args(encoder) + ["-c:a", "aac", output_path]
        return cmd

    if _nvenc_available():
        try:
            _run_ffmpeg(_build_cmd("h264_nvenc"))
            return output_path
        except RuntimeError as exc:
            # 响亮回退：硬件编码失败（驱动/最小分辨率/会话数限制）不掩盖
            logger.warning(
                "NVENC 硬件编码失败，回退 libx264 软编：%s",
                str(exc)[-300:],
            )
    _run_ffmpeg(_build_cmd("libx264"))
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
