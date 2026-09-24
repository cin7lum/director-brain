"""Film Context Gateway —— ASSET 层构建与读取。

ASSET 层是 :class:`~director_brain.models.film_context.FilmContextSnapshot` 的最底层，
描述单个媒体资产的物理属性与分析索引。本模块从 observation_service 的产出
（``list[FilmObservation]``）叠加出资产级快照，供下游 SCENE / EVIDENCE 层引用。

仅依赖 Python 标准库；视频元数据通过 ``ffprobe`` CLI 获取，失败时 degrade 为零值
而不抛异常。
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from director_brain._utils import file_sha256, short_hash
from director_brain.models.film_context import ContextLayer, FilmContextSnapshot
from director_brain.models.film_observation import FilmObservation
from storage.repository import BrainRepository

_TIMEBASE_US = 1_000_000  # 微秒时基，与 observation_service 一致


def _probe_video_meta(video_path: str) -> dict[str, object]:
    """用 ffprobe 读取时长 / 帧率 / 音轨信息。

    任何失败（命令不存在、非零退出、JSON 解析失败）均 degrade 为零值，不抛异常。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,r_frame_rate",
        "-of", "json", video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {"duration_us": 0, "fps": 0.0, "has_audio": False}
    if proc.returncode != 0:
        return {"duration_us": 0, "fps": 0.0, "has_audio": False}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"duration_us": 0, "fps": 0.0, "has_audio": False}

    duration_us = 0
    fmt = data.get("format") or {}
    raw_duration = fmt.get("duration")
    if raw_duration is not None:
        try:
            duration_us = int(round(float(raw_duration) * _TIMEBASE_US))
        except (TypeError, ValueError):
            duration_us = 0

    fps = 0.0
    has_audio = False
    for stream in data.get("streams") or []:
        codec_type = stream.get("codec_type")
        if codec_type == "audio":
            has_audio = True
        if codec_type == "video" and fps == 0.0:
            rate = stream.get("r_frame_rate", "0/0")
            try:
                num_s, den_s = rate.split("/")
                num, den = int(num_s), int(den_s)
                fps = round(num / den, 3) if den != 0 else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0

    return {"duration_us": duration_us, "fps": fps, "has_audio": has_audio}


def build_asset_index(video_path: str, observations: list[FilmObservation]) -> FilmContextSnapshot:
    """从 observation_service 产出构建 ASSET 层 FilmContextSnapshot。

    Args:
        video_path: 视频文件路径。
        observations: observation_service 产出的观测列表。

    Returns:
        layer=ASSET 的上下文快照；ffprobe / 文件读取失败时 degrade 为零值。
    """
    obs_ids_sorted = sorted(obs.observation_id for obs in observations)

    context_id = (
        "ctx_asset_"
        + short_hash(video_path + "|" + "|".join(obs_ids_sorted))
    )
    analysis_fingerprint = short_hash(video_path + "".join(obs_ids_sorted))

    # file_sha256 不可读时抛 OSError；这里 degrade 为空串，与原 _file_sha256 语义一致
    try:
        file_hash = file_sha256(video_path)
    except OSError:
        file_hash = ""
    meta = _probe_video_meta(video_path)

    shot_count = len({obs.media_asset_id for obs in observations})
    observed_frame_span = sum(obs.end_frame - obs.start_frame for obs in observations)

    sampling_config: dict[str, object] = {
        "source": "observation_service",
        "observation_count": len(observations),
        "video_duration_us": meta["duration_us"],
        "fps": meta["fps"],
        "has_audio": meta["has_audio"],
        "shot_count": shot_count,
        "observed_frame_span_sum": observed_frame_span,
    }

    return FilmContextSnapshot(
        context_id=context_id,
        asset_refs=[video_path],
        source_content_hashes=[file_hash] if file_hash else [],
        layers=[ContextLayer.ASSET],
        analysis_fingerprint=analysis_fingerprint,
        provider="context_gateway",
        model="asset_index_v1",
        prompt_version="n/a",
        sampling_config=sampling_config,
        timebase=_TIMEBASE_US,
        coverage="asset_level",
        rights_scope="internal",
        evidence_refs=[obs.observation_id for obs in observations],
        cache_state="fresh",
        invalidated_at=None,
        schema_version="1.0",
        project_id="unknown",
        created_at=int(time.time()),
        producer="context_gateway",
        source_ref=video_path,
    )


def get_asset_context(context_id: str, repository: BrainRepository) -> FilmContextSnapshot | None:
    """从存储层读取指定 context_id 的 ASSET 快照；不存在返回 None。

    薄封装，便于未来叠加缓存 / 失效逻辑。
    """
    return repository.get(FilmContextSnapshot, context_id)
