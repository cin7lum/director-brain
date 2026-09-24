"""VLM 适配器工厂。

按 provider 字符串创建对应的 VLMAdapter 实例。
"""
from __future__ import annotations

from observation_service.vlm_adapter import VLMAdapter
from observation_service.ollama_vlm_adapter import OllamaVLMAdapter
from observation_service.zhipu_vlm_adapter import ZhipuVLMAdapter


def get_vlm_adapter(provider: str, **config) -> VLMAdapter:
    """按 provider 创建 VLM 适配器。

    Args:
        provider: "ollama" 或 "zhipu"
        **config: 透传给适配器构造函数的关键字参数

    Raises:
        ValueError: 不支持的 provider
    """
    if provider == "ollama":
        return OllamaVLMAdapter(**config)
    if provider == "zhipu":
        return ZhipuVLMAdapter(**config)
    raise ValueError(f"unsupported VLM provider: {provider!r}")
