"""媒体探测（ffprobe 只读，不打分不转码）——全仓 ffprobe 单一入口。

架构体检候选⑤收编：此前 4 处各自手写 ffprobe（context_gateway /
renderer / evaluate_roughcut / 本模块），失败语义不一（静默吞失败记成
fps=0"事实" vs 带归因 fail-soft）。本模块是唯一权威实现：

- :func:`probe_media_meta`：深度探测（时长/帧率/音轨/编码 + 归因），
  ``ok=False`` 时 ``reason`` 说明探测失败原因（ffprobe 缺失/超时/解析）；
- :func:`probe_audio_stream`：音轨归因（T4 ASR 用），深度探测的薄包装。

fail-soft：探测失败返回 ``ok=False`` + 原因，不抛异常（归因本身不能
成为新的故障源）；调用方据此**显式降级**，禁止把失败伪装成"事实为零"。
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

logger = logging.getLogger(__name__)


class AudioTrackInfo:
    """音轨探测结果。``ok=False`` 表示探测本身失败（reason 给出原因）。"""

    def __init__(
        self,
        ok: bool,
        has_audio: bool = False,
        codec: str | None = None,
        duration_us: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.ok = ok
        self.has_audio = has_audio
        self.codec = codec
        self.duration_us = duration_us
        self.reason = reason

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "has_audio": self.has_audio,
            "codec": self.codec,
            "duration_us": self.duration_us,
            "reason": self.reason,
        }


def probe_audio_stream(video_path: str) -> AudioTrackInfo:
    """用 ffprobe 探测视频的音轨情况（深度探测的薄包装，T4 ASR 归因）。

    Returns:
        :class:`AudioTrackInfo`：``ok=True`` 时 ``has_audio`` 为确定性事实；
        ``ok=False`` 时 ``reason`` 说明探测失败原因（ffprobe 缺失/超时/无法解析）。
    """
    meta = probe_media_meta(video_path)
    if not meta.ok:
        return AudioTrackInfo(ok=False, reason=meta.reason)
    return AudioTrackInfo(
        ok=True,
        has_audio=meta.has_audio,
        codec=meta.codec,
        duration_us=meta.duration_us,
    )


@dataclass
class MediaMeta:
    """深度媒体探测结果。``ok=False`` 表示探测本身失败（reason 给出原因）。"""

    ok: bool
    duration_us: int = 0
    fps: float = 0.0
    # Keep ffprobe's native rational clock fields alongside the legacy float.
    # They are metadata, not a claim that VFR frame indexes can be mapped by
    # one constant frame rate.
    r_frame_rate: str | None = None
    avg_frame_rate: str | None = None
    stream_time_base: str | None = None
    has_audio: bool = False
    codec: str | None = None
    time_map: dict | None = None
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "duration_us": self.duration_us,
            "fps": self.fps,
            "r_frame_rate": self.r_frame_rate,
            "avg_frame_rate": self.avg_frame_rate,
            "stream_time_base": self.stream_time_base,
            "has_audio": self.has_audio,
            "codec": self.codec,
            "time_map": self.time_map,
            "reason": self.reason,
        }


def probe_media_meta(video_path: str, *, local_only: bool = False) -> MediaMeta:
    """ffprobe 深度探测：时长 / 帧率 / 音轨 / 编码，失败带归因（候选⑤）。

    ``local_only=True`` restricts FFprobe input protocols to local files for
    project-manifest media supplied through the local-processing API.
    """
    cmd = [
        "ffprobe",
    ]
    if local_only:
        cmd.extend(["-protocol_whitelist", "file"])
    cmd.extend([
        "-v", "error",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,r_frame_rate,avg_frame_rate,"
            "time_base,start_pts,start_time,duration_ts,duration,sample_rate,"
            "channels,channel_layout:stream_tags=language:"
            "stream_disposition=default:format=start_time,duration"
        ),
        "-of", "json", video_path,
    ])
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return MediaMeta(ok=False, reason="ffprobe 不可用")
    except subprocess.TimeoutExpired:
        return MediaMeta(ok=False, reason="ffprobe 超时（30s）")

    if proc.returncode != 0:
        return MediaMeta(
            ok=False,
            reason=f"ffprobe 退出码 {proc.returncode}: {proc.stderr[:200]}",
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return MediaMeta(ok=False, reason=f"ffprobe 输出解析失败: {exc}")

    streams = data.get("streams", []) or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None)

    format_meta = data.get("format") or {}
    duration_raw = format_meta.get("duration")
    duration_us = 0
    if duration_raw is not None:
        try:
            duration_us = int(float(duration_raw) * 1_000_000)
        except (TypeError, ValueError):
            duration_us = 0

    fps = 0.0
    r_frame_rate = None
    avg_frame_rate = None
    stream_time_base = None
    if video_stream:
        rate = video_stream.get("r_frame_rate", "0/0")
        try:
            num, den = rate.split("/")
            num_value, den_value = int(num), int(den)
            if num_value > 0 and den_value > 0:
                fps = round(num_value / den_value, 3)
                r_frame_rate = f"{num_value}/{den_value}"
        except (AttributeError, ValueError, ZeroDivisionError):
            fps = 0.0
        try:
            num, den = video_stream.get("avg_frame_rate", "0/0").split("/")
            num_value, den_value = int(num), int(den)
            if num_value > 0 and den_value > 0:
                avg_frame_rate = f"{num_value}/{den_value}"
        except (AttributeError, ValueError, ZeroDivisionError):
            pass
        try:
            num, den = video_stream.get("time_base", "0/0").split("/")
            num_value, den_value = int(num), int(den)
            if num_value > 0 and den_value > 0:
                stream_time_base = f"{num_value}/{den_value}"
        except (AttributeError, ValueError, ZeroDivisionError):
            pass

    def integer_or_none(value: object) -> int | None:
        try:
            if value is None or isinstance(value, bool):
                return None
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    def decimal_text_or_none(value: object) -> str | None:
        if value is None:
            return None
        raw = str(value)
        try:
            parsed = Decimal(raw)
        except (InvalidOperation, ValueError):
            return None
        return raw if parsed.is_finite() else None

    def positive_rational_or_none(value: object) -> str | None:
        if value is None:
            return None
        raw = str(value)
        try:
            rational = Fraction(raw)
        except (ValueError, ZeroDivisionError):
            return None
        if rational <= 0:
            return None
        return f"{rational.numerator}/{rational.denominator}"

    container_start = decimal_text_or_none(format_meta.get("start_time"))

    def stream_timing(stream: dict, codec_type: str) -> dict:
        time_base = positive_rational_or_none(stream.get("time_base"))
        start_pts = integer_or_none(stream.get("start_pts"))
        start_time = decimal_text_or_none(stream.get("start_time"))
        offset = None
        offset_state = "unavailable"
        if container_start is not None and start_pts is not None and time_base:
            try:
                offset = (
                    Fraction(start_pts) * Fraction(time_base)
                    - Fraction(Decimal(container_start))
                )
                offset_state = "mapped_from_pts"
            except (InvalidOperation, ValueError, ZeroDivisionError):
                offset = None
        if offset is None and container_start is not None and start_time is not None:
            try:
                offset = (
                    Fraction(Decimal(start_time))
                    - Fraction(Decimal(container_start))
                )
                offset_state = "mapped_from_start_time"
            except (InvalidOperation, ValueError, ZeroDivisionError):
                offset = None
        if offset is None:
            offset_numerator = None
            offset_denominator = None
            offset_state = "unavailable"
        else:
            offset_numerator = offset.numerator
            offset_denominator = offset.denominator

        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        default_value = disposition.get("default")
        is_default = (
            bool(default_value)
            if isinstance(default_value, (bool, int))
            else None
        )
        return {
            "stream_index": integer_or_none(stream.get("index")),
            "codec_type": codec_type,
            "codec_name": stream.get("codec_name"),
            "time_base": time_base,
            "start_pts": start_pts,
            "start_time_seconds": start_time,
            "duration_ts": integer_or_none(stream.get("duration_ts")),
            "duration_seconds": decimal_text_or_none(stream.get("duration")),
            "sample_rate": integer_or_none(stream.get("sample_rate")),
            "channels": integer_or_none(stream.get("channels")),
            "channel_layout": stream.get("channel_layout"),
            "language": tags.get("language"),
            "is_default": is_default,
            "source_start_offset_numerator": offset_numerator,
            "source_start_offset_denominator": offset_denominator,
            "source_start_offset_state": offset_state,
        }

    selected_streams = []
    if video_stream is not None:
        selected_streams.append((video_stream, "video"))
    selected_streams.extend((stream, "audio") for stream in audio_streams)
    # ffprobe always emits stream.index for real files. If a provider omits it,
    # decline to invent source stream identities in a persisted time map.
    time_map = None
    if all(integer_or_none(stream.get("index")) is not None
           for stream, _codec_type in selected_streams):
        video_timing = (
            stream_timing(video_stream, "video")
            if video_stream is not None else None
        )
        audio_timing = [stream_timing(stream, "audio")
                        for stream in audio_streams]
        timings = ([video_timing] if video_timing is not None else []) + audio_timing
        mapped_count = sum(
            item["source_start_offset_state"] != "unavailable"
            for item in timings
        )
        video_mapped = (
            video_timing is not None
            and video_timing["source_start_offset_state"] != "unavailable"
        )
        complete = video_mapped and all(
            item["source_start_offset_state"] != "unavailable"
            for item in audio_timing
        )
        mapping_state = (
            "complete" if complete
            else "partial" if mapped_count
            else "unavailable"
        )
        time_map = {
            "container_start_time_seconds": container_start,
            "video_stream": video_timing,
            "audio_streams": audio_timing,
            "mapping_state": mapping_state,
        }

    return MediaMeta(
        ok=True,
        duration_us=duration_us,
        fps=fps,
        r_frame_rate=r_frame_rate,
        avg_frame_rate=avg_frame_rate,
        stream_time_base=stream_time_base,
        has_audio=bool(audio_streams),
        codec=audio_streams[0].get("codec_name") if audio_streams else None,
        time_map=time_map,
    )
