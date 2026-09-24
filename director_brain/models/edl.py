"""编辑决策表 EditorialDecisionList (EDL) 与 EditItem 模型。

时间统一以整数帧表示，并由 ``timebase`` 说明帧率。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from director_brain.models.base import BaseRecord


class EditItem(BaseModel):
    """单条剪辑决策（嵌入式组件，非顶层记录）。"""

    model_config = ConfigDict(extra="forbid")

    source_asset_id: str
    source_media_hash: str
    in_frame: int
    out_frame: int
    timebase: int
    transition: str | None = None
    effect_refs: list[str] = Field(default_factory=list)
    shot_function: str | None = None
    rationale: str | None = None


class EditorialDecisionList(BaseRecord):
    """一份完整的编辑决策表。"""

    edl_id: str
    version: str
    brief_version: str
    context_id: str
    source_asset_hashes: list[str] = Field(default_factory=list)
    timebase: int
    ordered_edits: list[EditItem] = Field(default_factory=list)
    decision_refs: list[str] = Field(default_factory=list)
    audio_refs: list[str] = Field(default_factory=list)
    overlay_refs: list[str] = Field(default_factory=list)
    subtitle_refs: list[str] = Field(default_factory=list)
    hard_rule_refs: list[str] = Field(default_factory=list)
    artistic_choices: list[str] = Field(default_factory=list)
    expected_duration: int | None = None
    plan_hash: str | None = None
    approval_state: str
    supersedes_edl_id: str | None = None
