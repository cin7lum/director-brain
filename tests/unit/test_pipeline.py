"""pipeline 端到端编排的单元测试（全部用 mock，不触发真实 VLM/ASR）。

覆盖：
- analyze_media_full 默认开启 VLM + ASR 时三段观测拼接正确
- vlm=False / asr=False 时对应部分被跳过
- analyze_media() 快速路径行为完全不变（仅 deterministic_technical）
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from observation_service.pipeline import analyze_media, analyze_media_full


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
    mock_batch_vlm.assert_called_once_with("fake_video.mp4", shots, adapter=vlm_adapter)
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
    # deterministic + vlm 仍在
    assert len([o for o in result if o.observation_type == "deterministic_technical"]) == 2
    assert any(o.observation_type == "vlm_semantic" for o in result)
