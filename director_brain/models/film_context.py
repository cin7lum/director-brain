"""影片上下文快照 FilmContextSnapshot 模型。

一次分析调用产生的、可被下游引用与缓存的影片状态快照。
"""
from __future__ import annotations

import enum
from typing import Any

from pydantic import Field

from director_brain.models.base import BaseRecord


class ContextLayer(str, enum.Enum):
    """上下文覆盖层级。"""

    PROJECT = "project"
    ASSET = "asset"
    SCENE = "scene"
    EVIDENCE = "evidence"


class FilmContextSnapshot(BaseRecord):
    """某次分析调用产生的影片上下文快照（project_id / created_at 由基类提供）。"""

    context_id: str
    asset_refs: list[str] = Field(default_factory=list)
    source_content_hashes: list[str] = Field(default_factory=list)
    layers: list[ContextLayer] = Field(default_factory=list)
    analysis_fingerprint: str
    provider: str
    model: str
    prompt_version: str
    sampling_config: dict[str, Any] = Field(default_factory=dict)
    timebase: int
    coverage: str
    rights_scope: str
    evidence_refs: list[str] = Field(default_factory=list)
    cache_state: str
    invalidated_at: int | None = None
