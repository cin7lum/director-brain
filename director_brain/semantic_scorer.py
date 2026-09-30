# -*- coding: utf-8 -*-
"""语义评分库（内核深化第二层 · 候选①收编后为内核专用评分数学）。

历史（评审指认）：本模块曾自带一套完整选片循环（semantic_aware_select /
fused_selection + 私有四幕配额表），与 director_reasoner 生产内核并行——
"第二颗未接线且绕过闸门的脑"。收编后本模块**只保留评分数学**：

- ``compute_semantic_score``：单候选语义评分（importance 基础加权、
  情绪弧匹配、剪辑语言匹配、内容多样性降权）
- 幕映射表来自 :mod:`director_brain.acts`（单一事实源）

唯一调用方 = ``director_reasoner.HeuristicDirectorReasoner``（选片垄断权
在内核；VLM 信号消费由通路闸门 fail-closed 执法）。独立选片循环已删除。
"""
from __future__ import annotations

from dataclasses import dataclass

from director_brain.acts import ROLE_TO_ACT as _ROLE_TO_ACT  # noqa: F401
from director_brain.models.director_brief import DirectorBrief

__all__ = ["SemanticScore", "compute_semantic_score"]


@dataclass
class SemanticScore:
    """单个候选镜头的语义评分明细。"""

    shot_id: str
    base_score: float  # blur × importance_mult
    narrative_bonus: float  # 四幕匹配加分
    emotion_match: float  # 情绪弧匹配加分
    action_match: float  # 剪辑语言匹配加分
    diversity_penalty: float  # 内容重复降分
    total: float  # 最终得分
    assigned_act: str  # 语义驱动的幕分配
    reason: str  # 评分依据


#: emotional_tone ↔ emotional_arc 匹配矩阵
_EMOTION_MATCH: dict[str, set[str]] = {
    "upbeat": {"energetic", "joyful", "neutral"},
    "heartwarming": {"calm", "joyful", "neutral"},
    "suspenseful": {"tense", "dark", "neutral"},
    "calm": {"calm", "neutral"},
    "melancholic": {"dark", "calm", "neutral"},
}

#: editing_language → action_type 偏好
_ACTION_PREF: dict[str, set[str]] = {
    "fast_cut": {"action", "transition"},
    "slow_paced": {"establishing", "emotional", "sensory"},
    "montage": {"action", "transition", "sensory"},
    "jump_cut": {"action", "dialogue"},
}


def _norm_description(desc: str) -> str:
    """归一化场景描述用于相似度比较（取前 30 字符）。"""
    return (desc or "")[:30].lower().strip()


def compute_semantic_score(
    candidate: dict,
    semantic: dict,
    brief: DirectorBrief,
    act: str,
    seen_descriptions: list[str],
) -> SemanticScore:
    """计算单个候选镜头的语义感知评分。

    Args:
        candidate: 内核候选 dict（含 blur_score；select_score 由内核再乘
            结构性 VLM 权重）。
        semantic: VLM 深度语义 claim 字段（emotional_tone / narrative_role /
            action_type / importance / scene_description）。
        brief: 导演简报（提供 emotional_arc / editing_language）。
        act: 当前幕名（hook/develop/peak/resolve）。
        seen_descriptions: 已评分镜头的归一化描述列表（多样性检查）。
    """
    blur = candidate.get("blur_score") or 0
    base = blur

    # -- importance 加权（P3-2：1-5 → 0.4~2.0x）
    imp = semantic.get("importance")
    imp_mult = (0.4 + 0.4 * imp) if isinstance(imp, (int, float)) and 1 <= imp <= 5 else 1.0
    base *= imp_mult

    # -- narrative_role 四幕匹配加分
    role = semantic.get("narrative_role", "")
    assigned = _ROLE_TO_ACT.get(role, act)  # 语义驱动幕分配
    narrative_bonus = 0.3 if assigned == act else 0.0

    # -- emotional_tone 匹配
    tone = semantic.get("emotional_tone", "neutral")
    arc = brief.emotional_arc or "neutral"
    matched_tones = _EMOTION_MATCH.get(arc, {"neutral"})
    emotion_match = 0.2 if tone in matched_tones else 0.0

    # -- action_type 匹配剪辑语言
    action = semantic.get("action_type", "sensory")
    lang = brief.editing_language or "not_determined"
    preferred = _ACTION_PREF.get(lang, set())
    action_match = 0.1 if action in preferred else 0.0

    # -- content diversity：描述相似 → 降分
    desc = _norm_description(semantic.get("scene_description", ""))
    diversity_penalty = 0.0
    for seen in seen_descriptions:
        if desc and seen and (desc in seen or seen in desc):
            diversity_penalty = -0.5
            break

    total = base + narrative_bonus + emotion_match + action_match + diversity_penalty
    reasons = []
    if imp_mult != 1.0:
        reasons.append(f"imp={imp}({imp_mult:.1f}x)")
    if narrative_bonus:
        reasons.append(f"narrative={role}→{assigned}")
    if emotion_match:
        reasons.append(f"emotion={tone}↔{arc}")
    if action_match:
        reasons.append(f"action={action}↔{lang}")
    if diversity_penalty:
        reasons.append("duplicate_content")

    return SemanticScore(
        shot_id=candidate.get("source_shot_id", ""),
        base_score=round(base, 2),
        narrative_bonus=narrative_bonus,
        emotion_match=emotion_match,
        action_match=action_match,
        diversity_penalty=diversity_penalty,
        total=round(total, 2),
        assigned_act=assigned,
        reason="; ".join(reasons) if reasons else "base_score",
    )
