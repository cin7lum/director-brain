"""编辑决策表 EditorialDecisionList (EDL) 与 EditItem 模型。

``in_frame`` / ``out_frame`` are integer source coordinates; ``timebase``
defines ticks per second and ``timebase_unit`` identifies those coordinates.
The current HeuristicDirectorReasoner emits canonical microseconds.
"""
from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from director_brain.models.base import BaseRecord
from director_brain.models.film_observation import TimebaseUnit


class TransitionSpec(BaseModel):
    """转场规格（挂在出点侧：本镜头与下一镜头之间；P2-b 证据包裁定）。

    type="cut" 为硬切（渲染器 concat 路径，缺省行为）；
    type="xfade" 走链式 xfade/acrossfade（总时长 = Σd − ΣD，时间线由
    :mod:`director_brain.timeline` 统一计算，字幕自动跟随）。
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "cut"  # cut | xfade
    name: str = "fade"  # xfade 转场名（ffmpeg xfade 滤镜的 transition 值）
    duration_us: int = Field(default=500_000, gt=0)  # 转场重叠时长（微秒）
    audio_duration_us: int | None = None  # J/L-cut 预留：音频转场时长可与视频解耦


class EditItem(BaseModel):
    """单条剪辑决策（嵌入式组件，非顶层记录）。"""

    model_config = ConfigDict(extra="forbid")

    source_asset_id: str
    source_media_hash: str
    in_frame: int
    out_frame: int
    timebase: int
    timebase_unit: TimebaseUnit = TimebaseUnit.UNKNOWN
    project_asset_id: str | None = None
    source_observation_refs: list[str] = Field(default_factory=list)
    audio_evidence_refs: list[str] = Field(default_factory=list)
    project_story_link_refs: list[str] = Field(default_factory=list)
    project_story_relation_refs: list[str] = Field(default_factory=list)
    source_observation_start: int | None = None
    source_observation_end: int | None = None
    source_timebase: int | None = Field(default=None, gt=0)
    source_timebase_unit: TimebaseUnit | None = None
    transition: TransitionSpec | None = None
    #: D3-b：J/L-cut 音频偏移（导演 artistic choice）。
    #: audio_lead_us>0 = 声音先入（本镜头音频从源时间轴提前 lead 开始，
    #: 越过切点压在前镜头画面尾部）；audio_tail_us>0 = 声音延续
    #: （本镜头音频延出 tail，压在后镜头画面头部）。
    #: 渲染器检测到任一非零偏移即切换音频时间轴图（adelay+amix）。
    audio_lead_us: int = 0
    audio_tail_us: int = 0
    effect_refs: list[str] = Field(default_factory=list)
    shot_function: str | None = None
    rationale: str | None = None
    #: P1-b 结构化字段（替代 rationale 字符串协议）：所属幕
    act: str | None = None
    #: P1-b 结构化字段：证据类型 "heuristic" | "vlm"（修复器优先级判据，
    #: 旧数据为 None 时回退 rationale 字符串嗅探）
    evidence_type: str | None = None

    @model_validator(mode="after")
    def validate_source_provenance(self) -> Self:
        if (self.source_observation_start is None) != (
            self.source_observation_end is None
        ):
            raise ValueError(
                "source observation start/end must be provided together")
        if (self.source_observation_start is not None
                and self.source_observation_end is not None
                and self.source_observation_start >= self.source_observation_end):
            raise ValueError("source interval must be increasing")
        if len(self.source_observation_refs) != len(set(self.source_observation_refs)):
            raise ValueError("source observation references must be unique")
        if len(self.audio_evidence_refs) != len(set(self.audio_evidence_refs)):
            raise ValueError("audio evidence references must be unique")
        if len(self.project_story_link_refs) != len(set(self.project_story_link_refs)):
            raise ValueError("project story link references must be unique")
        if len(self.project_story_relation_refs) != len(set(self.project_story_relation_refs)):
            raise ValueError("project story relation references must be unique")
        if self.project_asset_id is None and (
            self.project_story_link_refs or self.project_story_relation_refs
        ):
            raise ValueError(
                "project story references require a project-bound edit")
        if self.timebase_unit == TimebaseUnit.MICROSECONDS and self.timebase != 1_000_000:
            raise ValueError("microsecond EDL coordinates require timebase=1000000")
        if self.project_asset_id is not None:
            if not self.project_asset_id.strip():
                raise ValueError("project_asset_id must be non-empty")
            if (not self.source_observation_refs
                    or self.source_observation_start is None
                    or self.source_timebase is None
                    or self.source_timebase_unit is None
                    or self.source_timebase_unit == TimebaseUnit.UNKNOWN):
                raise ValueError(
                    "project-bound edits require source evidence and an explicit source interval/timebase"
                )
            if self.source_timebase_unit == TimebaseUnit.MICROSECONDS:
                if self.source_timebase != 1_000_000:
                    raise ValueError(
                        "microsecond source coordinates require timebase=1000000")
                if (self.in_frame < self.source_observation_start
                        or self.out_frame > self.source_observation_end):
                    raise ValueError(
                        "microsecond edit interval must stay within its cited source interval")
        return self

    @field_validator("audio_lead_us", "audio_tail_us")
    @classmethod
    def _audio_offsets_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("audio offsets must be >= 0")
        return v

    @field_validator("transition", mode="before")
    @classmethod
    def _coerce_legacy_transition(cls, v):
        """历史数据兼容：transition 曾是 str（全仓从未生产过，仅防御旧序列化）。

        "cut" → 硬切规格；其余字符串按 xfade 转场名解释。
        """
        if isinstance(v, str):
            if v == "cut":
                return TransitionSpec(type="cut")
            return TransitionSpec(type="xfade", name=v)
        return v


def ordered_unique_source_hashes(edits: list[EditItem]) -> list[str]:
    """Return distinct edit source hashes in first-use order.

    Hash comparison is case-insensitive, while the first spelling is retained
    for backward-compatible serialization.
    """
    result: list[str] = []
    seen: set[str] = set()
    for edit in edits:
        key = edit.source_media_hash.lower()
        if key not in seen:
            seen.add(key)
            result.append(edit.source_media_hash)
    return result


class EditorialDecisionList(BaseRecord):
    """一份完整的编辑决策表。"""

    edl_id: str
    version: str
    brief_version: str
    context_id: str
    source_asset_hashes: list[str] = Field(default_factory=list)
    timebase: int
    timebase_unit: TimebaseUnit = TimebaseUnit.UNKNOWN
    ordered_edits: list[EditItem] = Field(default_factory=list)
    decision_refs: list[str] = Field(default_factory=list)
    audio_refs: list[str] = Field(default_factory=list)
    audio_evidence_refs: list[str] = Field(default_factory=list)
    overlay_refs: list[str] = Field(default_factory=list)
    subtitle_refs: list[str] = Field(default_factory=list)
    hard_rule_refs: list[str] = Field(default_factory=list)
    artistic_choices: list[str] = Field(default_factory=list)
    expected_duration: int | None = None
    plan_hash: str | None = None
    approval_state: str
    supersedes_edl_id: str | None = None

    @model_validator(mode="after")
    def validate_timeline_timebase(self) -> Self:
        if self.timebase_unit == TimebaseUnit.MICROSECONDS and self.timebase != 1_000_000:
            raise ValueError("microsecond EDL timelines require timebase=1000000")
        if self.timebase_unit == TimebaseUnit.MICROSECONDS and any(
            edit.timebase_unit != TimebaseUnit.MICROSECONDS
            for edit in self.ordered_edits
        ):
            raise ValueError("microsecond EDL timelines require microsecond edit coordinates")
        if len(self.audio_evidence_refs) != len(set(self.audio_evidence_refs)):
            raise ValueError("EDL audio evidence references must be unique")
        edit_audio_refs = {
            ref for edit in self.ordered_edits for ref in edit.audio_evidence_refs
        }
        if not edit_audio_refs.issubset(set(self.audio_evidence_refs)):
            raise ValueError(
                "EDL audio evidence references must include each edit's audio evidence")
        return self
