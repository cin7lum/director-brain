"""M1.1 端到端 pipeline：discover_shots -> analyze_shot。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from director_brain.analysis_cache import AnalysisCacheStore, compute_fingerprint
from director_brain.models.film_observation import (
    FILM_OBSERVATION_SCHEMA_VERSION,
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)
from director_brain.models.project import ProjectAssetTimeMap
from observation_service.analysis_control import AnalysisCancelledError
from observation_service.asr import transcribe, transcribe_with_status
from observation_service.deterministic_analysis import (
    DETERMINISTIC_ANALYSIS_VERSION,
    DETERMINISTIC_SAMPLE_FRACTIONS,
    DETERMINISTIC_SHAKE_LOOKBACK_US,
    OPENCV_RUNTIME_VERSION,
    analyze_shot,
)
from observation_service.shot_discovery import discover_shots
from observation_service.vlm_observation import batch_vlm_observations


@dataclass(frozen=True)
class ProjectASRAnalysisResult:
    """Project ASR outcome plus usable local deterministic observations."""

    observations: list[FilmObservation]
    failure_code: str | None = None


def _deterministic_shot_fingerprint(shot: dict) -> str:
    """Bind a checkpoint to exact source bytes, segment, algorithm and schema."""
    return compute_fingerprint(
        source_content_hash=str(shot["source_media_hash"]).lower(),
        provider="deterministic_opencv",
        model_version=(
            f"opencv_5.0|analysis={DETERMINISTIC_ANALYSIS_VERSION}|"
            f"runtime={OPENCV_RUNTIME_VERSION}"
        ),
        prompt_version="n/a",
        sampling_config={
            "shot_id": str(shot["shot_id"]),
            "source_in_us": int(shot["source_in_us"]),
            "source_out_us": int(shot["source_out_us"]),
            "sample_fractions": list(DETERMINISTIC_SAMPLE_FRACTIONS),
            "shake_lookback_us": DETERMINISTIC_SHAKE_LOOKBACK_US,
        },
        timebase=1_000_000,
        schema_version=f"FilmObservation/{FILM_OBSERVATION_SCHEMA_VERSION}",
    )


def _validate_deterministic_checkpoint(
    observations: list[FilmObservation], shot: dict
) -> FilmObservation:
    """Fail closed if a stored result does not match its exact source segment."""
    if len(observations) != 1:
        raise ValueError("deterministic shot cache must contain exactly one observation")
    observation = observations[0]
    expected = {
        "observation_id": f"det_{shot['shot_id']}",
        "media_asset_id": str(shot["shot_id"]),
        "media_hash": str(shot["source_media_hash"]).lower(),
        "start_frame": int(shot["source_in_us"]),
        "end_frame": int(shot["source_out_us"]),
        "timebase": 1_000_000,
        "timebase_unit": TimebaseUnit.MICROSECONDS,
        "observation_type": "deterministic_technical",
        "provider": "deterministic_opencv",
        "model_version": "opencv_5.0",
    }
    if any(getattr(observation, field) != value
           for field, value in expected.items()):
        raise ValueError("deterministic shot cache identity does not match source segment")
    if observation.claim_kind not in {ClaimKind.MEASURED, ClaimKind.NOT_DETERMINED}:
        raise ValueError("deterministic shot cache has an invalid claim kind")
    return observation


def _analyze_shots(
    video_path: str,
    shots: list[dict],
    cache: AnalysisCacheStore | None = None,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> list[FilmObservation]:
    observations: list[FilmObservation] = []
    total = len(shots)
    for index, shot in enumerate(shots):
        if cancellation_check is not None and cancellation_check():
            raise AnalysisCancelledError("analysis cancellation requested")
        fingerprint = _deterministic_shot_fingerprint(shot)
        cached = cache.get(fingerprint) if cache is not None else None
        if cached is not None:
            observation = _validate_deterministic_checkpoint(cached, shot)
            observations.append(observation.model_copy(update={"source_ref": video_path}))
            cancelled_after_unit = False
        else:
            observation = analyze_shot(video_path, shot)
            if cache is not None:
                _validate_deterministic_checkpoint([observation], shot)
            observations.append(observation)
            cancelled_after_unit = (
                cancellation_check()
                if cancellation_check is not None else False
            )
            # Repository-backed stores commit each shot immediately. A later
            # process interruption therefore leaves completed shots resumable.
            if cache is not None:
                cache.put(fingerprint, [observation])
        if progress_callback is not None:
            progress_callback("deterministic_analysis", index + 1, total)
        if cancelled_after_unit:
            raise AnalysisCancelledError("analysis cancellation requested")
    if cache is not None:
        cache.flush()
    return observations


def analyze_media(
    video_path: str,
    cache: AnalysisCacheStore | None = None,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> list[FilmObservation]:
    """对整个媒体切分镜头并逐镜头做确定性技术分析。

    discover_shots 失败返回空列表；analyze_shot 自身不抛异常（失败时返回
    NOT_DETERMINED 观测）。空结果不抛异常。
    """
    return _analyze_shots(
        video_path,
        discover_shots(video_path),
        cache=cache,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
    )


def analyze_media_full(
    video_path: str,
    vlm: bool = True,
    asr: bool = True,
    vlm_adapter=None,
    vlm_cache: AnalysisCacheStore | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
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
        vlm_cache: 可选的缓存实现；项目分析可使用 repository-backed
            cache，避免在源素材目录旁写入缓存文件。

    Returns:
        合并后的 FilmObservation 列表。vlm=False / asr=False 时对应部分为空。
    """
    shots = discover_shots(video_path)
    deterministic_obs = _analyze_shots(
        video_path,
        shots,
        cache=vlm_cache,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
    )

    vlm_obs: list[FilmObservation] = []
    if vlm:
        vlm_kwargs = {"adapter": vlm_adapter, "cache": vlm_cache}
        if progress_callback is not None:
            vlm_kwargs["progress_callback"] = progress_callback
        if cancellation_check is not None:
            vlm_kwargs["cancellation_check"] = cancellation_check
        vlm_obs = batch_vlm_observations(
            video_path, shots, **vlm_kwargs)

    asr_obs: list[FilmObservation] = []
    if asr:
        asr_obs = transcribe(video_path)

    return deterministic_obs + vlm_obs + asr_obs


def analyze_media_with_project_asr(
    video_path: str,
    *,
    cache: AnalysisCacheStore | None,
    model_path: str,
    model_digest: str,
    model_version: str,
    source_has_audio: bool | None,
    time_map: ProjectAssetTimeMap | None,
) -> ProjectASRAnalysisResult:
    """Run local deterministic analysis and source-clock-bound project ASR.

    ASR uses faster-whisper's first audio stream. Its timestamp origin is mapped
    to the video's source clock using the manifest's exact rational stream
    offsets. Missing stream timing fails closed but preserves deterministic
    observations so Context can report partial coverage.
    """
    deterministic = analyze_media(video_path, cache=cache)
    if source_has_audio is False:
        return ProjectASRAnalysisResult(deterministic)
    if source_has_audio is not True or time_map is None:
        failure = (
            "speech_analysis_partial"
            if deterministic else "speech_analysis_failed"
        )
        return ProjectASRAnalysisResult(deterministic, failure)
    try:
        source_stream_index, source_time_offset = (
            time_map.first_audio_stream_clock_from_video())
    except ValueError:
        failure = (
            "speech_analysis_partial"
            if deterministic else "speech_analysis_failed"
        )
        return ProjectASRAnalysisResult(deterministic, failure)
    if source_stream_index is None:
        failure = (
            "speech_analysis_partial"
            if deterministic else "speech_analysis_failed"
        )
        return ProjectASRAnalysisResult(deterministic, failure)

    result = transcribe_with_status(
        video_path,
        model_name=model_path,
        model_version=model_version,
        expected_model_digest=model_digest,
        source_has_audio=True,
        source_time_offset_seconds=source_time_offset,
        source_stream_index=source_stream_index,
    )
    observations = deterministic + result.observations
    if result.status == "failed":
        failure = (
            "speech_analysis_partial"
            if observations else "speech_analysis_failed"
        )
        return ProjectASRAnalysisResult(observations, failure)
    return ProjectASRAnalysisResult(observations)
