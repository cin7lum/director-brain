"""P1-c 配置诚实化测试：vlm_provider 配置不再是假的。"""
from __future__ import annotations

from dataclasses import dataclass

import observation_service.vlm_observation as vlm_mod


@dataclass
class _Settings:
    vlm_provider: str
    vlm_model: str
    ollama_base_url: str
    zhipu_api_key: str | None
    zhipu_base_url: str


def test_provider_zhipu_constructs_zhipu_adapter(monkeypatch):
    calls = {}

    class _FakeZhipu:
        def __init__(self, api_key=None, model=None):
            calls["zhipu"] = {"api_key": api_key, "model": model}

    class _FakeOllama:
        def __init__(self, **kwargs):
            calls["ollama"] = kwargs

    monkeypatch.setattr(
        vlm_mod, "load_settings",
        lambda: _Settings("zhipu", "glm-4.6v-flash", "http://x", "k", "https://z"),
    )
    monkeypatch.setattr(
        "observation_service.zhipu_vlm_adapter.ZhipuVLMAdapter", _FakeZhipu
    )
    vlm_mod.batch_vlm_observations("dummy.mp4", [], adapter=None)
    assert "zhipu" in calls and "ollama" not in calls
    assert calls["zhipu"]["api_key"] == "k"


def test_provider_ollama_constructs_ollama_adapter(monkeypatch):
    calls = {}

    class _FakeOllama:
        def __init__(self, **kwargs):
            calls["ollama"] = kwargs

    monkeypatch.setattr(
        vlm_mod, "load_settings",
        lambda: _Settings("ollama", "qwen3-vl", "http://localhost:11434", None, ""),
    )
    monkeypatch.setattr(vlm_mod, "OllamaVLMAdapter", _FakeOllama)
    vlm_mod.batch_vlm_observations("dummy.mp4", [], adapter=None)
    assert "ollama" in calls
    assert calls["ollama"]["model"] == "qwen3-vl"
