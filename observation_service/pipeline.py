"""M1.1 端到端 pipeline：discover_shots -> analyze_shot。"""
from __future__ import annotations

from director_brain.models.film_observation import FilmObservation
from observation_service.asr import transcribe
from observation_service.deterministic_analysis import analyze_shot
from observation_service.shot_discovery import discover_shots
from observation_service.vlm_observation import batch_vlm_observations


def analyze_media(video_path: str) -> list[FilmObservation]:
    """对整个媒体切分镜头并逐镜头做确定性技术分析。

    discover_shots 失败返回空列表；analyze_shot 自身不抛异常（失败时返回
    NOT_DETERMINED 观测）。空结果不抛异常。
    """
    return [analyze_shot(video_path, shot) for shot in discover_shots(video_path)]


def analyze_media_full(
    video_path: str,
    vlm: bool = True,
    asr: bool = True,
    vlm_adapter=None,
) -> list[FilmObservation]:
    """对整个媒体做完整观测：确定性技术分析 + VLM 语义分析 + ASR 语音转写。

    在 :func:`analyze_media` 的快速确定性路径之上，按需追加 VLM 语义观测与
    ASR 转写观测，返回 ``deterministic_obs + vlm_obs + asr_obs`` 的合并列表。

    Args:
        video_path: 视频文件路径。
        vlm: 是否调用 VLM 做逐镜头语义分析。首次调用较慢（约 2-5 分钟，需
            抽取关键帧并请求本地模型），建议配合 analysis_cache 复用结果；
            传 ``False`` 可跳过该部分。
        asr: 是否用 faster-whisper 做语音转写；传 ``False`` 可跳过该部分。
        vlm_adapter: 可选的 VLM 适配器实例，透传给
            :func:`observation_service.vlm_observation.batch_vlm_observations`；
            为 None 时由 batch 函数内部从 config 创建默认 OllamaVLMAdapter。

    Returns:
        合并后的 FilmObservation 列表。vlm=False / asr=False 时对应部分为空。
    """
    shots = discover_shots(video_path)
    deterministic_obs = [analyze_shot(video_path, shot) for shot in shots]

    vlm_obs: list[FilmObservation] = []
    if vlm:
        vlm_obs = batch_vlm_observations(video_path, shots, adapter=vlm_adapter)

    asr_obs: list[FilmObservation] = []
    if asr:
        asr_obs = transcribe(video_path)

    return deterministic_obs + vlm_obs + asr_obs
