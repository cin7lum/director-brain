# -*- coding: utf-8 -*-
"""跨镜头叙事理解器（阶段 P3-3 · 灵魂升级）。

输入全部镜头的语义观测序列（scene_description / emotional_tone /
action_type / importance），一次 LLM 综合调用产出：

- **叙事弧**：这些镜头连在一起讲了什么故事
- **情绪轨迹**：每镜头的情绪标签（calm→tense→climax→resolution）
- **镜头配对**：action_reaction / continuity / contrast 关系
- **幕边界**：内容驱动的 hook/develop/peak/resolve 分割点
- **推荐顺序**：叙事流最优的镜头排列

输出驱动选片排序与转场选择——系统从"逐镜头评分"升级为"理解叙事流"。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from director_brain.input_sanitizer import sanitize_untrusted

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


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    lo, hi = text.find("{"), text.rfind("}")
    if lo < 0 or hi <= lo:
        raise ValueError(f"无 JSON: {text[:100]}")
    return json.loads(text[lo:hi + 1])


def analyze_narrative(
    semantics: list[dict],
    *,
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    model: str = "doubao-seed-2-1-lite-260915",
    api_key: str | None = None,
    timeout: int = 300,
) -> dict:
    """跨镜头叙事综合：一次 LLM 调用理解全部镜头的叙事流。

    Args:
        semantics: 逐镜头语义观测列表（semantic_analyzer 产物）。
        base_url: LLM API base URL。
        model: LLM 模型 ID（须为已准入模型）。
        api_key: API key。

    Returns:
        叙事理解 dict（含 story_arc/emotional_trajectory/pairings/...）。
    """
    if not api_key:
        api_key = os.environ.get("ARK_API_KEY", "")
    if not api_key:
        raise RuntimeError("ARK_API_KEY 未设置")

    seq_text = _build_sequence_prompt(semantics)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _NARRATIVE_PROMPT},
            {"role": "user", "content": seq_text},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())

    content = resp["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
        content = re.sub(r"```\s*$", "", content).strip()
    lo, hi = content.find("{"), content.rfind("}")
    if lo < 0:
        raise ValueError(f"叙事分析无 JSON: {content[:100]}")
    return json.loads(content[lo:hi + 1])


def load_env() -> None:
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
