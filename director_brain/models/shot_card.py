"""镜头卡 ShotCard 模型（D1 · 美学策略载体 v1）。

一张卡 = 一套可审计的导演创作选择包：目的/能量/节奏边界/情绪弧适配/
转场策略/已知陷阱。brief 说了什么风格，导演脑按卡剪；卡的选型落
plan/EDL 留痕（账本可审计"本片用了哪张卡"）。

边界裁定（2026-09-30）：卡的定义与选择归 02（01 总纲 §1 自有 IP=
导演判断/意图映射/电影语法）；卡引用的资源走 03 Catalog；参数执行
归 04 + 02 预览渲染器；效果验证归 05 + 开发侧评估栈。

Schema 蓝本致谢：卡片六要素（目的/能量/建议时长/参数/实现说明/已知
陷阱）参考 github.com/Vincentwei1021/video-shotcraft（Apache-2.0）；
实现自建——其卡驱动代码生成合成素材，本卡驱动真素材剪辑决策。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: 合法能量词表（参考 shotcraft 能量维度）
VALID_ENERGY = {"low", "calm", "medium", "high", "escalating", "explosive"}
#: 合法转场策略
VALID_TRANSITION_POLICY = {"none", "dissolve_act_boundary"}


class ShotCard(BaseModel):
    """镜头配方卡：导演创作选择的可复用、可版本化、可审计载体。"""

    model_config = ConfigDict(extra="forbid")

    card_id: str  # 机器名（如 "fast_cut"），库内唯一
    version: str = "1.0"
    name: str  # 展示名
    category: str  # 分类：节奏/情绪/叙事/转场/用途
    purpose: str  # 目的：这张卡把素材剪成什么感觉
    energy: str  # 能量：low|calm|medium|high|escalating|explosive

    #: 节奏边界（覆盖 editing_language_bounds；微秒）
    pacing_override: dict[str, int] = Field(
        default_factory=dict,
        description="{'min_clip_us': int, 'max_clip_us': int}，可只给其一",
    )
    #: 适配的情绪弧（brief.emotional_arc 命中其一即加分）
    emotion_arc_fit: list[str] = Field(default_factory=list)
    #: 适配的剪辑语言（brief.editing_language 命中即加分）
    editing_language_fit: list[str] = Field(default_factory=list)
    #: 转场策略（none | dissolve_act_boundary）
    transition_policy: str = "none"
    #: 节拍对齐提示（D5 落地前仅作策略声明与账本留痕）
    beat_align_hint: bool = False

    #: 适用素材条件
    min_shots: int = 3  # 少于此镜头数时选择器跳过本卡

    #: 已知陷阱（shotcraft 六要素之一——诚实记录卡的失效场景）
    known_pitfalls: list[str] = Field(default_factory=list)

    attribution: str = (
        "builtin v1 (schema inspired by "
        "github.com/Vincentwei1021/video-shotcraft, Apache-2.0)")
