"""P2-a ASR 设备策略测试：auto=GPU 优先响亮回退 CPU。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
