"""P2-a ASR 设备策略测试：auto=GPU 优先响亮回退 CPU。"""
from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from director_brain.models.film_observation import TimebaseUnit
from observation_service import asr as asr_mod


class _FakeWhisper:
    """按 device 记录加载尝试；cuda 可编程失败。"""

    cuda_fail = True
    attempts: list[str] = []

    def __init__(self, model_name, device="cpu", compute_type="int8"):
        type(self).attempts.append(device)
        if device == "cuda" and type(self).cuda_fail:
            raise RuntimeError("Library cublas64_12.dll is not found")
        self.device = device

    def transcribe(self, video_path, **kwargs):
        return [], SimpleNamespace(language="en")


@pytest.fixture()
def fake_whisper():
    _FakeWhisper.attempts = []
    with patch.object(asr_mod, "_whisper_cls", lambda: _FakeWhisper):
        yield _FakeWhisper


def test_auto_falls_back_to_cpu_when_cuda_broken(fake_whisper, monkeypatch):
    fake_whisper.cuda_fail = True
    monkeypatch.setenv("ASR_DEVICE", "auto")
    obs = asr_mod.transcribe("fake.mp4", model_name="fake-model")
    assert fake_whisper.attempts == ["cuda", "cpu"]
    assert obs == []


def test_explicit_cpu_single_attempt(fake_whisper, monkeypatch):
    fake_whisper.cuda_fail = True
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    asr_mod.transcribe("fake.mp4", model_name="fake-model")
    assert fake_whisper.attempts == ["cpu"]


def test_cuda_used_when_available(fake_whisper, monkeypatch):
    fake_whisper.cuda_fail = False
    monkeypatch.setenv("ASR_DEVICE", "auto")
    asr_mod.transcribe("fake.mp4", model_name="fake-model")
    assert fake_whisper.attempts == ["cuda"]


def test_total_load_failure_returns_empty(fake_whisper, monkeypatch):
    """所有设备尝试都失败 → fail-soft 返回空列表（不抛异常）。"""
    def _boom(model_name, device="cpu", compute_type="int8"):
        raise RuntimeError("weights missing")

    with patch.object(asr_mod, "_whisper_cls", lambda: _boom):
        monkeypatch.setenv("ASR_DEVICE", "cpu")
        assert asr_mod.transcribe("fake.mp4", model_name="m") == []


def test_transcribe_declares_its_microsecond_timebase(monkeypatch):
    class _SegmentWhisper:
        def __init__(self, model_name, device="cpu", compute_type="int8"):
            pass

        def transcribe(self, video_path, **kwargs):
            return [SimpleNamespace(start=0.25, end=0.75, text="spoken words")], None

    monkeypatch.setenv("ASR_DEVICE", "cpu")
    with patch.object(asr_mod, "_whisper_cls", lambda: _SegmentWhisper):
        observations = asr_mod.transcribe("fake.mp4", model_name="fake-model")

    assert len(observations) == 1
    assert observations[0].start_frame == 250_000
    assert observations[0].end_frame == 750_000
    assert observations[0].timebase == 1_000_000
    assert observations[0].timebase_unit == TimebaseUnit.MICROSECONDS


def test_project_asr_translates_audio_clock_to_source_video_clock(monkeypatch):
    class _SegmentWhisper:
        def __init__(self, model_name, device="cpu", compute_type="int8"):
            pass

        def transcribe(self, video_path, **kwargs):
            return [SimpleNamespace(start=0.25, end=0.75, text="spoken words")], None

    monkeypatch.setenv("ASR_DEVICE", "cpu")
    with patch.object(asr_mod, "_whisper_cls", lambda: _SegmentWhisper):
        result = asr_mod.transcribe_with_status(
            "fake.mp4",
            model_name="local-model",
            model_version="local-model@sha256:" + "a" * 64,
            source_has_audio=True,
            source_time_offset_seconds=Fraction(1, 2),
            source_stream_index=2,
        )

    assert result.status == "observed"
    observation = result.observations[0]
    assert (observation.start_frame, observation.end_frame) == (750_000, 1_250_000)
    assert observation.timebase_unit == TimebaseUnit.MICROSECONDS
    assert observation.source_stream_index == 2
    assert observation.model_version == "local-model@sha256:" + "a" * 64


def test_project_asr_does_not_call_provider_for_video_without_audio(
    fake_whisper, monkeypatch
):
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    result = asr_mod.transcribe_with_status(
        "video-only.mp4", model_name="local-model", source_has_audio=False)

    assert result.status == "completed_empty"
    assert result.failure_code is None
    assert fake_whisper.attempts == []


def test_local_model_tree_digest_changes_when_artifact_changes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.bin"
    weights.write_bytes(b"weights-v1")
    first = asr_mod.local_model_tree_sha256(model)

    weights.write_bytes(b"weights-v2")
    second = asr_mod.local_model_tree_sha256(model)

    assert first != second
