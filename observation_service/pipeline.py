"""M1.1 端到端 pipeline：discover_shots -> analyze_shot。"""
from __future__ import annotations

from director_brain.models.film_observation import FilmObservation
from observation_service.deterministic_analysis import analyze_shot
from observation_service.shot_discovery import discover_shots


def analyze_media(video_path: str) -> list[FilmObservation]:
    """对整个媒体切分镜头并逐镜头做确定性技术分析。

    discover_shots 失败返回空列表；analyze_shot 自身不抛异常（失败时返回
    NOT_DETERMINED 观测）。空结果不抛异常。
    """
    return [analyze_shot(video_path, shot) for shot in discover_shots(video_path)]
