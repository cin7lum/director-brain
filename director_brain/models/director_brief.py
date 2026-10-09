"""导演简报 DirectorBrief 模型。

创作者意图的权威文本载体，下游所有决策都可追溯回它的某个版本。
"""
from __future__ import annotations

from pydantic import Field

from director_brain.models.base import BaseRecord


class DirectorBrief(BaseRecord):
    """创作者对成片的意图说明。"""

    brief_id: str
    version: str
    source_text: str
    creator_direction: str = Field(
        default="",
        max_length=2000,
        description=(
            "Sanitized, bounded creator-authored direction; distinct from "
            "asset-derived transcript text in source_text."
        ),
    )
    language: str
    intent: str
    audience: str
    target_duration: int
    source_duration_us: int | None = None
    delivery_profile: str
    themes: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    emotional_arc: str
    visual_language: str
    editing_language: str
    sound_language: str
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    privacy_constraints: list[str] = Field(default_factory=list)
    approval_state: str
    approved_by: str | None = None
    approved_at: int | None = None
