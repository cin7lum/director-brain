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
    has_audio: bool = False
    codec: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "duration_us": self.duration_us,
            "fps": self.fps,
            "has_audio": self.has_audio,
            "codec": self.codec,
            "reason": self.reason,
        }


def probe_media_meta(video_path: str) -> MediaMeta:
    """ffprobe 深度探测：时长 / 帧率 / 音轨 / 编码，失败带归因（候选⑤）。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=codec_type,codec_name,r_frame_rate:format=duration",
        "-of", "json", video_path,
    ]
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

    duration_raw = (data.get("format") or {}).get("duration")
    duration_us = 0
    if duration_raw is not None:
        try:
            duration_us = int(float(duration_raw) * 1_000_000)
        except (TypeError, ValueError):
            duration_us = 0

    fps = 0.0
    if video_stream:
        rate = video_stream.get("r_frame_rate", "0/0")
        try:
            num, den = rate.split("/")
            fps = round(int(num) / int(den), 3) if int(den) else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0

    return MediaMeta(
        ok=True,
        duration_us=duration_us,
        fps=fps,
        has_audio=bool(audio_streams),
        codec=audio_streams[0].get("codec_name") if audio_streams else None,
    )
