"""编辑决策表 EditorialDecisionList (EDL) 与 EditItem 模型。

时间统一以整数帧表示，并由 ``timebase`` 说明帧率。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from director_brain.models.base import BaseRecord


class TransitionSpec(BaseModel):
    """转场规格（挂在出点侧：本镜头与下一镜头之间；P2-b 证据包裁定）。

    type="cut" 为硬切（渲染器 concat 路径，缺省行为）；
    type="xfade" 走链式 xfade/acrossfade（总时长 = Σd − ΣD，时间线由
    :mod:`director_brain.timeline` 统一计算，字幕自动跟随）。
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "cut"  # cut | xfade
    name: str = "fade"  # xfade 转场名（ffmpeg xfade 滤镜的 transition 值）
    duration_us: int = 500_000  # 转场重叠时长（微秒）
    audio_duration_us: int | None = None  # J/L-cut 预留：音频转场时长可与视频解耦


class EditItem(BaseModel):
    """单条剪辑决策（嵌入式组件，非顶层记录）。"""

    model_config = ConfigDict(extra="forbid")

    source_asset_id: str
    source_media_hash: str
    in_frame: int
    out_frame: int
    timebase: int
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
