# -*- coding: utf-8 -*-
"""跨镜头叙事理解器（阶段 P3-3 · 内核融合后为生产库函数）。

输入全部镜头的语义观测序列（scene_description / emotional_tone /
action_type / importance），一次 LLM 综合调用产出：

- **叙事弧**：这些镜头连在一起讲了什么故事
- **情绪轨迹**：每镜头的情绪标签（calm→tense→climax→resolution）
- **镜头配对**：action_reaction / continuity / contrast 关系
- **幕边界**：内容驱动的 hook/develop/peak/resolve 分割点
- **推荐顺序**：叙事流最优的镜头排列

架构体检候选①收编后：本模块是**内核消费的库函数**（唯一生产入口 =
``scripts/roughcut.py --semantic``，通路 vlm_semantic 须 ACTIVE），不再
是演示旁路。传输统一走 :func:`llm_adapter.post_chat_json`（候选④收编，
私有 urllib 已删）；场景描述为视频派生不可信文本，按 S5 消毒。传入
``shot_ids`` 时结果附 ``*_resolved`` 字段（LLM 返回的是序列下标，内核
需要镜头 id）。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from director_brain.input_sanitizer import sanitize_untrusted
from director_brain.llm_adapter import (
    LLMTransportError,
    extract_json_object,
    post_chat_json,
)

ROOT = Path(__file__).resolve().parent.parent

_NARRATIVE_PROMPT = """You are a professional film editor analyzing the narrative structure of a sequence of shots.

Below are the semantic observations for each shot (in source order). Analyze them as a SEQUENCE - how do they connect narratively, what story do they tell, where are the emotional shifts?

Return ONLY JSON:
{
  "story_arc": "one sentence describing the narrative arc of this sequence",
  "emotional_trajectory": ["calm", "tense", "climax", "resolution", ...],
  "pairings": [{"a": 0, "b": 1, "relation": "action_reaction|continuity|contrast|cause_effect", "reason": "..."}],
  "key_moments": [{"shot_idx": 0, "why": "why this shot is a turning point"}],
  "act_boundaries": [{"act": "hook", "start": 0, "end": 1}, ...],
  "suggested_order": [2, 0, 1, 3],
  "limitations": ["what you cannot determine from metadata alone"]
}

Rules:
- shot indices are 0-based, matching the input order
- suggested_order must contain ALL shot indices exactly once
- pairings use source order indices (a < b)
- act_boundaries: act is hook/develop/peak/resolve; start/end are shot indices (inclusive)
- limitations: list what cannot be determined from semantic metadata alone
- emotional_trajectory: one emotion label per shot, in input order
- If the shots have no clear narrative, say so honestly in story_arc"""


def _build_sequence_prompt(semantics: list[dict]) -> str:
    """把逐镜头语义观测序列格式化为 LLM 可读文本。

    scene_description 为视频内容派生的不可信文本（画面中的标语/字幕可能
    携带注入），按 S5 规范消毒；枚举字段来自受限词表无需处理。
    """
    lines = []
    for i, sem in enumerate(semantics):
        parts = [f"Shot {i}:"]
        if sem.get("scene_description"):
            parts.append(f"  content: {sanitize_untrusted(str(sem['scene_description']))}")
        if sem.get("action_type"):
            parts.append(f"  action: {sem['action_type']}")
        if sem.get("emotional_tone"):
            parts.append(f"  emotion: {sem['emotional_tone']}")
        if sem.get("narrative_role"):
            parts.append(f"  narrative_role: {sem['narrative_role']}")
        if sem.get("importance"):
            parts.append(f"  importance: {sem['importance']}/5")
        if sem.get("motion_amount"):
            parts.append(f"  motion: {sem['motion_amount']}")
        lines.append("\n".join(parts))
    return "\n---\n".join(lines)


def _resolve_ids(result: dict, shot_ids: list[str]) -> dict:
    """把 LLM 返回的序列下标解析为镜头 id（内核消费所需）。"""
    resolved = {"shot_ids": list(shot_ids)}

    boundaries_resolved = []
    for b in result.get("act_boundaries", []):
        lo, hi = int(b.get("start", -1)), int(b.get("end", -1))
        if 0 <= lo <= hi < len(shot_ids):
            boundaries_resolved.append({
                "act": b.get("act"),
                "shot_ids": shot_ids[lo:hi + 1],
            })
    resolved["act_boundaries_resolved"] = boundaries_resolved

    order = result.get("suggested_order")
    if (
        isinstance(order, list)
        and sorted(x for x in order if isinstance(x, int)) == list(range(len(shot_ids)))
    ):
        resolved["suggested_order_resolved"] = [shot_ids[i] for i in order]
    return resolved


def analyze_narrative(
    semantics: list[dict],
    *,
    shot_ids: list[str] | None = None,
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    model: str = "doubao-seed-2-1-lite-260915",
    api_key: str | None = None,
    timeout: int = 300,
) -> dict:
    """跨镜头叙事综合：一次 LLM 调用理解全部镜头的叙事流。

    Args:
        semantics: 逐镜头语义观测列表（vlm_prompt_v3_semantic claim 口径），
            **按源顺序**排列——act_boundaries/suggested_order 的下标基于此序。
        shot_ids: 与 semantics 等长的镜头 id 列表；传入时结果附
            act_boundaries_resolved / suggested_order_resolved。
        base_url: LLM API base URL。
        model: LLM 模型 ID（须为已准入模型）。
        api_key: API key。

    Returns:
        叙事理解 dict（含 story_arc/emotional_trajectory/pairings/...）。
        传输失败抛 LLMTransportError，解析失败抛 ValueError（调用方决定
        降级策略——生产入口为响亮跳过，不阻断主链）。
    """
    if not api_key:
        api_key = os.environ.get("ARK_API_KEY", "")
    if not api_key:
        raise LLMTransportError("ARK_API_KEY 未设置")

    seq_text = _build_sequence_prompt(semantics)
    content = post_chat_json(
        base_url, api_key, model, _NARRATIVE_PROMPT, seq_text,
        timeout=timeout, max_tokens=4096,
    )
    result = extract_json_object(content)
    if shot_ids is not None:
        if len(shot_ids) != len(semantics):
            raise ValueError("shot_ids 与 semantics 长度不一致")
        result.update(_resolve_ids(result, shot_ids))
    return result


def load_env() -> None:
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
