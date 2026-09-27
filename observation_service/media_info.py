"""媒体音轨探测（T4 ASR 归因用）：ffprobe 只读探测，不打分不转码。

ASR 通路"0 条转写"必须可归因（真无语音 vs 通路故障 vs 无音轨）。
:func:`probe_audio_stream` 回答"这个视频到底有没有音轨"——它是
ffprobe 的确定性事实（MEASURED），不是推断。

fail-soft：ffprobe 不可用或探测失败时返回 ``ok=False`` 的结果并带原因，
不抛异常（归因本身不能成为新的故障源）。
"""
from __future__ import annotations

import json
import logging
import subprocess

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
    """用 ffprobe 探测视频的音轨情况。

    Returns:
        :class:`AudioTrackInfo`：``ok=True`` 时 ``has_audio`` 为确定性事实；
        ``ok=False`` 时 ``reason`` 说明探测失败原因（ffprobe 缺失/超时/无法解析）。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=codec_type,codec_name:format=duration",
        "-of", "json", video_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False,
        )
    except FileNotFoundError:
        return AudioTrackInfo(ok=False, reason="ffprobe 不可用")
    except subprocess.TimeoutExpired:
        return AudioTrackInfo(ok=False, reason="ffprobe 超时（30s）")

    if proc.returncode != 0:
        return AudioTrackInfo(
            ok=False, reason=f"ffprobe 退出码 {proc.returncode}: {proc.stderr[:200]}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return AudioTrackInfo(ok=False, reason=f"ffprobe 输出解析失败: {exc}")

    streams = data.get("streams", []) or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    duration_raw = (data.get("format") or {}).get("duration")
    duration_us: int | None = None
    if duration_raw is not None:
        try:
            duration_us = int(float(duration_raw) * 1_000_000)
        except (TypeError, ValueError):
            duration_us = None

    if audio_streams:
        return AudioTrackInfo(
            ok=True,
            has_audio=True,
            codec=audio_streams[0].get("codec_name"),
            duration_us=duration_us,
        )
    return AudioTrackInfo(ok=True, has_audio=False, duration_us=duration_us)
