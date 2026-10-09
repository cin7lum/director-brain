"""pipeline 端到端编排的单元测试（全部用 mock，不触发真实 VLM/ASR）。

覆盖：
- analyze_media_full 默认开启 VLM + ASR 时三段观测拼接正确
- vlm=False / asr=False 时对应部分被跳过
- analyze_media() 快速路径行为完全不变（仅 deterministic_technical）
"""
from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from director_brain.analysis_cache import RepositoryAnalysisCache
from director_brain.models.film_observation import (
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)
from director_brain.models.project import ProjectAssetTimeMap
from observation_service.asr import ASRTranscriptionResult
from observation_service import pipeline
from observation_service.pipeline import (
    AnalysisCancelledError,
    analyze_media,
    analyze_media_full,
    analyze_media_with_project_asr,
)


def _shot(index: int) -> dict:
    return {
        "shot_id": f"shot_test_{index:04d}",
        "source_in_us": index * 2_000_000,
        "source_out_us": (index + 1) * 2_000_000,
        "duration_us": 2_000_000,
        "keyframes": [],
        "source_media_hash": f"hash_test_{index}",
    }


def _obs(observation_type: str):
    return SimpleNamespace(observation_type=observation_type)


def _two_shots():
    return [_shot(0), _shot(1)]


@patch("observation_service.pipeline.transcribe")
@patch("observation_service.pipeline.batch_vlm_observations")
@patch("observation_service.pipeline.analyze_shot")
@patch("observation_service.pipeline.discover_shots")
def test_analyze_media_full_with_vlm_returns_vlm_observations(
    mock_discover, mock_analyze_shot, mock_batch_vlm, mock_transcribe
):
    shots = _two_shots()
    mock_discover.return_value = shots
    mock_analyze_shot.side_effect = lambda vp, shot: _obs("deterministic_technical")
    mock_batch_vlm.return_value = [_obs("vlm_semantic"), _obs("vlm_semantic")]
    mock_transcribe.return_value = [_obs("speech_transcript")]

    vlm_adapter = object()  # mock adapter，不做真实调用
    result = analyze_media_full("fake_video.mp4", vlm=True, asr=True, vlm_adapter=vlm_adapter)

    # VLM 观测数量 == 镜头数
    vlm_obs = [o for o in result if o.observation_type == "vlm_semantic"]
    assert len(vlm_obs) == len(shots)

    # adapter 透传给 batch_vlm_observations
    mock_batch_vlm.assert_called_once_with(
        "fake_video.mp4", shots, adapter=vlm_adapter, cache=None)
    # 三段都在
    assert any(o.observation_type == "deterministic_technical" for o in result)
    assert any(o.observation_type == "speech_transcript" for o in result)
    # 顺序：deterministic + vlm + asr
    types = [o.observation_type for o in result]
    assert types == [
        "deterministic_technical",
        "deterministic_technical",
        "vlm_semantic",
        "vlm_semantic",
        "speech_transcript",
    ]


@patch("observation_service.pipeline.transcribe")
@patch("observation_service.pipeline.batch_vlm_observations")
@patch("observation_service.pipeline.analyze_shot")
@patch("observation_service.pipeline.discover_shots")
def test_analyze_media_full_vlm_false_no_vlm_observations(
    mock_discover, mock_analyze_shot, mock_batch_vlm, mock_transcribe
):
    mock_discover.return_value = _two_shots()
    mock_analyze_shot.side_effect = lambda vp, shot: _obs("deterministic_technical")
    mock_transcribe.return_value = [_obs("speech_transcript")]

    result = analyze_media_full("fake_video.mp4", vlm=False, asr=True)

    mock_batch_vlm.assert_not_called()
    assert not any(o.observation_type == "vlm_semantic" for o in result)
    # deterministic + asr 仍在
    assert len([o for o in result if o.observation_type == "deterministic_technical"]) == 2
    assert any(o.observation_type == "speech_transcript" for o in result)


@patch("observation_service.pipeline.analyze_shot")
@patch("observation_service.pipeline.discover_shots")
def test_analyze_media_unchanged(mock_discover, mock_analyze_shot):
    mock_discover.return_value = _two_shots()
    mock_analyze_shot.side_effect = lambda vp, shot: _obs("deterministic_technical")

    result = analyze_media("fake_video.mp4")

    assert len(result) == 2
    assert all(o.observation_type == "deterministic_technical" for o in result)
    mock_discover.assert_called_once_with("fake_video.mp4")
    assert mock_analyze_shot.call_count == 2


@patch("observation_service.pipeline.analyze_shot")
@patch("observation_service.pipeline.discover_shots")
def test_analyze_media_reports_unit_progress(mock_discover, mock_analyze_shot):
    mock_discover.return_value = _two_shots()
    mock_analyze_shot.side_effect = lambda vp, shot: _obs(
        "deterministic_technical")
    progress = []

    analyze_media(
        "fake_video.mp4",
        progress_callback=lambda phase, completed, total: progress.append(
            (phase, completed, total)),
    )

    assert progress == [
        ("deterministic_analysis", 1, 2),
        ("deterministic_analysis", 2, 2),
    ]


@patch("observation_service.pipeline.analyze_shot")
@patch("observation_service.pipeline.discover_shots")
def test_analyze_media_honors_cancellation_between_units(
    mock_discover, mock_analyze_shot,
):
    mock_discover.return_value = _two_shots()
    mock_analyze_shot.side_effect = lambda vp, shot: _obs(
        "deterministic_technical")
    checks = iter([False, True])

    with pytest.raises(AnalysisCancelledError):
        analyze_media(
            "fake_video.mp4",
            cancellation_check=lambda: next(checks),
        )

    assert mock_analyze_shot.call_count == 1


def test_deterministic_shot_fingerprint_binds_exact_segment():
    from observation_service.pipeline import _deterministic_shot_fingerprint

    shot = {**_shot(0), "source_media_hash": "a" * 64}
    changed_bounds = {**shot, "source_out_us": shot["source_out_us"] + 1}
    changed_content = {**shot, "source_media_hash": "b" * 64}

    assert _deterministic_shot_fingerprint(shot) != (
        _deterministic_shot_fingerprint(changed_bounds)
    )
    assert _deterministic_shot_fingerprint(shot) != (
        _deterministic_shot_fingerprint(changed_content)
    )


def test_deterministic_shot_cache_resumes_after_repository_reopen(
    tmp_path, monkeypatch
):
    """A committed shot survives a failed run and is skipped after reopen."""
    from storage.sqlite_repository import SqliteRepository
    import observation_service.pipeline as pipeline

    source_hash = "a" * 64
    shots = [
        {**_shot(index), "source_media_hash": source_hash}
        for index in range(2)
    ]
    monkeypatch.setattr(pipeline, "discover_shots", lambda _path: shots)

    def observation(path: str, shot: dict) -> FilmObservation:
        return FilmObservation(
            observation_id=f"det_{shot['shot_id']}",
            media_asset_id=shot["shot_id"],
            media_hash=source_hash,
            start_frame=shot["source_in_us"],
            end_frame=shot["source_out_us"],
            timebase=1_000_000,
            timebase_unit=TimebaseUnit.MICROSECONDS,
            observation_type="deterministic_technical",
            claim="{}",
            provider="deterministic_opencv",
            model_version="opencv_5.0",
            prompt_version="n/a",
            confidence=1.0,
            review_state="auto_verified",
            claim_kind=ClaimKind.MEASURED,
            project_id="unknown",
            created_at=1_700_000_000,
            producer="deterministic_opencv",
            source_ref=path,
        )

    db_path = tmp_path / "shot-cache-restart.sqlite"
    first_repository = SqliteRepository(str(db_path))
    first_cache = RepositoryAnalysisCache(
        first_repository, source_hash, "deterministic-test-profile-v1")

    def stop_after_first(path: str, shot: dict) -> FilmObservation:
        if shot["shot_id"] == shots[1]["shot_id"]:
            raise RuntimeError("simulated interruption after first checkpoint")
        return observation(path, shot)

    monkeypatch.setattr(pipeline, "analyze_shot", stop_after_first)
    try:
        try:
            analyze_media("same-content.mp4", cache=first_cache)
            raise AssertionError("the simulated interruption did not occur")
        except RuntimeError as exc:
            assert "simulated interruption" in str(exc)
    finally:
        first_repository.close()

    reopened_repository = SqliteRepository(str(db_path))
    reopened_cache = RepositoryAnalysisCache(
        reopened_repository, source_hash, "deterministic-test-profile-v1")
    recomputed: list[str] = []

    def resume_remaining(path: str, shot: dict) -> FilmObservation:
        recomputed.append(shot["shot_id"])
        return observation(path, shot)

    monkeypatch.setattr(pipeline, "analyze_shot", resume_remaining)
    try:
        resumed = analyze_media("same-content-second-path.mp4", cache=reopened_cache)
        assert [item.observation_id for item in resumed] == [
            f"det_{shot['shot_id']}" for shot in shots
        ]
        assert recomputed == [shots[1]["shot_id"]]
        assert resumed[0].source_ref == "same-content-second-path.mp4"
        assert reopened_cache.get("not-a-shot-fingerprint") is None
    finally:
        reopened_repository.close()


@patch("observation_service.pipeline.transcribe")
@patch("observation_service.pipeline.batch_vlm_observations")
@patch("observation_service.pipeline.analyze_shot")
@patch("observation_service.pipeline.discover_shots")
def test_analyze_media_full_asr_false_no_asr(
    mock_discover, mock_analyze_shot, mock_batch_vlm, mock_transcribe
):
    mock_discover.return_value = _two_shots()
    mock_analyze_shot.side_effect = lambda vp, shot: _obs("deterministic_technical")
    mock_batch_vlm.return_value = [_obs("vlm_semantic")]

    result = analyze_media_full("fake_video.mp4", vlm=True, asr=False)

    mock_transcribe.assert_not_called()
    assert not any(o.observation_type == "speech_transcript" for o in result)
    assert len([o for o in result if o.observation_type == "deterministic_technical"]) == 2
    assert any(o.observation_type == "vlm_semantic" for o in result)


def _project_audio_time_map() -> ProjectAssetTimeMap:
    return ProjectAssetTimeMap.model_validate({
        "container_start_time_seconds": "0",
        "video_stream": {
            "stream_index": 0,
            "codec_type": "video",
            "source_start_offset_numerator": 0,
            "source_start_offset_denominator": 1,
            "source_start_offset_state": "mapped_from_start_time",
            "start_time_seconds": "0",
        },
        "audio_streams": [{
            "stream_index": 1,
            "codec_type": "audio",
            "source_start_offset_numerator": 1,
            "source_start_offset_denominator": 2,
            "source_start_offset_state": "mapped_from_start_time",
            "start_time_seconds": "0.5",
        }],
        "mapping_state": "complete",
    })


def test_project_asr_requires_audio_to_video_clock_mapping(monkeypatch):
    technical = _obs("deterministic_technical")
    monkeypatch.setattr(pipeline, "analyze_media", lambda path, cache=None: [technical])
    transcribe = patch.object(pipeline, "transcribe_with_status")
    with transcribe as mock_transcribe:
        result = analyze_media_with_project_asr(
            "asset.mp4",
            cache=None,
            model_path="local-model",
            model_digest="a" * 64,
            model_version="local-model@sha256:" + "a" * 64,
            source_has_audio=True,
            time_map=None,
        )

    assert result.observations == [technical]
    assert result.failure_code == "speech_analysis_partial"
    mock_transcribe.assert_not_called()


def test_project_asr_failure_preserves_technical_evidence_as_partial(monkeypatch):
    technical = _obs("deterministic_technical")
    monkeypatch.setattr(pipeline, "analyze_media", lambda path, cache=None: [technical])
    failure = ASRTranscriptionResult("failed", [], "asr_transcription_failed")
    with patch.object(pipeline, "transcribe_with_status", return_value=failure) as call:
        result = analyze_media_with_project_asr(
            "asset.mp4",
            cache=None,
            model_path="local-model",
            model_digest="a" * 64,
            model_version="local-model@sha256:" + "a" * 64,
            source_has_audio=True,
            time_map=_project_audio_time_map(),
        )

    assert result.observations == [technical]
    assert result.failure_code == "speech_analysis_partial"
    assert call.call_args.kwargs["source_time_offset_seconds"] == Fraction(1, 2)
    assert call.call_args.kwargs["source_stream_index"] == 1
