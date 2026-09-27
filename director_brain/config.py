"""Director Brain 配置管理。从环境变量和 .env 读取，不做臆造默认值。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(key: str, default: str | None = None) -> str | None:
    val = os.environ.get(key)
    if val is not None:
        return val
    # 尝试从项目根 .env 读取
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    return default


@dataclass(frozen=True)
class Settings:
    # 文本 LLM
    text_llm_provider: str
    text_llm_model: str
    # VLM
    vlm_provider: str
    vlm_model: str
    ollama_base_url: str
    zhipu_api_key: str | None
    zhipu_base_url: str
    # 存储
    storage_backend: str
    sqlite_path: str
    object_store_path: str
    # ASR（faster-whisper 本地权重目录或模型名）
    asr_model_path: str
    # 日志
    log_level: str


def load_settings() -> Settings:
    return Settings(
        text_llm_provider=_env("TEXT_LLM_PROVIDER", "zhipu"),
        text_llm_model=_env("TEXT_LLM_MODEL", "glm-4-flash"),
        vlm_provider=_env("VLM_PROVIDER", "ollama"),
        vlm_model=_env("VLM_MODEL", "glm-4.6v-flash"),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        zhipu_api_key=_env("ZHIPU_API_KEY"),
        zhipu_base_url=_env("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        storage_backend=_env("STORAGE_BACKEND", "sqlite"),
        sqlite_path=_env("SQLITE_PATH", "./data/director_brain.db"),
        object_store_path=_env("OBJECT_STORE_PATH", "./data/objects"),
        asr_model_path=_env(
            "ASR_MODEL_PATH",
            r"D:\新建豆包\gen1-roughcut\assets\asr\large-v3-turbo",
        ),
        log_level=_env("LOG_LEVEL", "INFO"),
    )
