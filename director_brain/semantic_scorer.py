# -*- coding: utf-8 -*-
"""语义感知选片器（内核深化第二层 · 大脑升级）。

当 VLM 语义观测存在时，选片从 "blur × 曝光贪心" 升级为 "语义感知多维选片"：

- **narrative_role 驱动四幕分配**：setup→hook, development→develop,
  climax→peak, resolution→resolve（替代时间比例切分）
- **emotional_tone 匹配目标情绪弧**：emotional_arc 声明的情绪优先入选
- **action_type 匹配剪辑语言**：fast_cut 偏好 action, slow_paced 偏好 calm
- **content diversity**：scene_description 相似的镜头降权（避免重复感）
- **importance 作为基础分**：替代纯 blur 贪心

无 VLM 时回退旧路径（响亮标记），VLM 观测经 pathway gate（必须 ACTIVE）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from director_brain.models.director_brief import DirectorBrief
from director_brain.models.film_observation import FilmObservation


@dataclass
class SemanticScore:
    """单个候选镜头的语义评分明细。"""

    shot_id: str
    base_score: float  # blur × vlm_mult × importance_mult（已有）
    narrative_bonus: float  # 四幕匹配加分
    emotion_match: float  # 情绪弧匹配加分
    action_match: float  # 剪辑语言匹配加分
    diversity_penalty: float  # 内容重复降分
    total: float  # 最终得分
    assigned_act: str  # 语义驱动的幕分配
    reason: str  # 评分依据


#: narrative_role → 四幕映射（语义驱动，非时间比例）
_ROLE_TO_ACT: dict[str, str] = {
    "setup": "hook",
    "development": "develop",
    "climax": "peak",
    "resolution": "resolve",
    "transition": "develop",  # 过渡镜头归 develop
}

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
        candidate: _build_candidates 产物（含 blur_score/exposure_ok）。
        semantic: analyze_shot_semantic 产物（含 emotional_tone/narrative_role 等）。
        brief: 导演简报（提供 emotional_arc / editing_language）。
        act: 当前幕名（hook/develop/peak/resolve）。
        seen_descriptions: 已选镜头的归一化描述列表（多样性检查）。
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


def semantic_aware_select(
    candidates: list[dict],
    semantics: dict[str, dict],
    brief: DirectorBrief,
    target_duration_us: int,
    *,
    min_clip_us: int = 800_000,
    max_clip_us: int = 6_000_000,
) -> tuple[list[dict], list[SemanticScore]]:
    """语义感知选片：按四幕分配 + 语义评分排序 + 贪心填充。

    Returns:
        (selected_clips, all_scores)——selected_clits 为候选字典列表
        （含 assigned_act 字段），all_scores 为全部评分明细。
    """
    scored: list[SemanticScore] = []
    seen_descs: list[str] = []

    # 第一遍：按四幕分组评分
    act_buckets: dict[str, list[tuple[SemanticScore, dict]]] = {
        "hook": [], "develop": [], "peak": [], "resolve": [],
    }
    for c in candidates:
        sem = semantics.get(c.get("source_shot_id", ""), {})
        if not sem:
            continue
        act = _default_act_for_candidate(c)
        score = compute_semantic_score(c, sem, brief, act, seen_descs)
        seen_descs.append(_norm_description(sem.get("scene_description", "")))
        scored.append(score)
        act_buckets[score.assigned_act].append((score, c))

    # 第二遍：逐幕贪心填充（幕配额 = ratio × target，语义角色决定幕分配）
    ratios = {"hook": 0.15, "develop": 0.35, "peak": 0.30, "resolve": 0.20}
    selected: list[dict] = []
    all_scores: list[SemanticScore] = []

    for act_name in ("hook", "develop", "peak", "resolve"):
        quota = max(int(ratios[act_name] * target_duration_us), min_clip_us)
        filled = 0
        bucket = sorted(act_buckets[act_name],
                        key=lambda pair: pair[0].total, reverse=True)
        for score, cand in bucket:
            if filled >= quota:
                break
            dur = cand.get("duration_us", 0)
            clip = min(dur, max_clip_us, quota - filled)
            if clip < min_clip_us:
                continue
            selected.append(cand)
            filled += clip
            all_scores.append(score)

    return selected, all_scores


def _default_act_for_candidate(c: dict) -> str:
    """无语义数据时的幕分配（基于源时间位置，保持向后兼容）。"""
    return "develop"  # 占位：调用方按时间窗口预分组


# ---------------------------------------------------------------------------
# 融合选片：语义优先 + 技术兜底
# ---------------------------------------------------------------------------

def fused_selection(
    candidates: list[dict],
    semantics: dict[str, dict],
    brief: DirectorBrief,
    target_duration_us: int,
    *,
    min_clip_us: int = 800_000,
    max_clip_us: int = 6_000_000,
) -> tuple[list[dict], list[SemanticScore], bool]:
    """融合选片：有语义数据的候选走语义评分，无语义的走技术评分。

    Returns:
        (selected, all_scores, used_semantic)
    """
    has_semantics = bool(semantics)
    if not has_semantics:
        return [], [], False

    # 分离有/无语义数据的候选
    sem_cands = [c for c in candidates if c.get("source_shot_id") in semantics]
    tech_cands = [c for c in candidates if c.get("source_shot_id") not in semantics]

    # 语义选片
    selected, scores = semantic_aware_select(
        sem_cands, semantics, brief, target_duration_us,
        min_clip_us=min_clip_us, max_clip_us=max_clip_us,
    )
    used_semantic = True

    # 技术兜底：如语义选片不足以填满目标时长，从无语义候选中补
    total_selected = sum(c.get("duration_us", 0) for c in selected)
    if total_selected < target_duration_us * 0.9 and tech_cands:
        remaining = target_duration_us - total_selected
        for c in sorted(tech_cands, key=lambda x: x.get("blur_score", 0), reverse=True):
            dur = c.get("duration_us", 0)
            if dur < min_clip_us or dur > max_clip_us:
                continue
            if total_selected + dur <= target_duration_us * 1.1:
                selected.append(c)
                total_selected += dur
            if total_selected >= target_duration_us * 0.9:
                break

    return selected, scores, used_semantic
