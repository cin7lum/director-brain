"""叙事弧模型（阶段 P3-3 · 跨镜头叙事理解）。

VLM 逐镜头观测产出的是"这个镜头里有什么"；本模块产出的是"这些镜头
**连在一起**讲了什么故事"——叙事弧、情绪轨迹、镜头配对、幕边界。

产出方式：一次 LLM 综合调用（doubao-seed-2-1-lite 已准入），输入全部
镜头的语义观测序列，输出叙事理解。不捏造：LLM 未覆盖的镜头保留
原语义数据，不填充编造的叙事关系。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ShotPairing(BaseModel):
    """两个镜头之间的叙事关系。"""

    model_config = {"extra": "forbid"}
    a: int  # 镜头索引（0-based，对应 EDL ordered_edits）
    b: int
    relation: str  # action_reaction | continuity | contrast | cause_effect
    reason: str = ""


class KeyMoment(BaseModel):
    """叙事关键节点。"""

    model_config = {"extra": "forbid"}
    shot_idx: int
    why: str


class ActBoundary(BaseModel):
    """幕边界（由内容驱动，非时间比例）。"""

    model_config = {"extra": "forbid"}
    act: str  # hook | develop | peak | resolve
    start_idx: int
    end_idx: int  # inclusive


class NarrativeArc(BaseModel):
    """跨镜头叙事理解结果。"""

    model_config = {"extra": "forbid"}
    story_arc: str = ""  # 全片叙事弧一句话
    emotional_trajectory: list[str] = Field(default_factory=list)  # 每镜头情绪
    pairings: list[ShotPairing] = Field(default_factory=list)
    key_moments: list[KeyMoment] = Field(default_factory=list)
    act_boundaries: list[ActBoundary] = Field(default_factory=list)
    suggested_order: list[int] = Field(default_factory=list)  # 推荐镜头顺序
    limitations: list[str] = Field(default_factory=list)
