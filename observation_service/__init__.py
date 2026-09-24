"""observation_service：对影片素材的各类自动观测（ASR / VLM / 关键帧抽帧 / …）。"""
from __future__ import annotations

from observation_service.asr import transcribe
from observation_service.vlm_adapter import VLMAdapter
from observation_service.ollama_vlm_adapter import OllamaVLMAdapter
from observation_service.zhipu_vlm_adapter import ZhipuVLMAdapter
from observation_service.vlm_factory import get_vlm_adapter
from observation_service.keyframe import extract_keyframe
from observation_service.deterministic_analysis import analyze_shot
from observation_service.pipeline import analyze_media
from observation_service.shot_discovery import discover_shots

__all__ = [
    "transcribe",
    "VLMAdapter",
    "OllamaVLMAdapter",
    "ZhipuVLMAdapter",
    "get_vlm_adapter",
    "extract_keyframe",
    "discover_shots",
    "analyze_shot",
    "analyze_media",
]
