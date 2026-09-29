# -*- coding: utf-8 -*-
"""多帧语义观测器（内核深化第一层 · 眼睛升级）。

旧：每镜头取中点 1 帧 → 单帧 VLM → {shot_function, role, motion}
新：每镜头抽 3 帧（前/中/尾）→ 一次多图 VLM 调用 → 深度语义分析：
    场景描述、主体列表、动作类型、情绪基调、叙事角色、时间变化、importance

设计要点：
- 单次多图调用（本机 qwen3-vl 已验证 2 帧可用；3 帧保守测试通过）
- 带 think:false 协商（qwen3-vl 思考模式吞输出）
- JSON 提取加固（首{末}+剥围栏）
- 缓存键含 prompt 版本（vlm_prompt_v3_semantic）
- 降级：VLM 不可用 → 降级为单帧旧路径（响亮标记）
"""
from __future__ import annotations

import base64
import json
import re
import urllib.request

_SEMANTIC_PROMPT = """Analyze these 3 frames from one video shot. Return ONLY JSON:
{"desc": "中文一句话场景描述",
 "subjects": ["主体列表"],
 "action": "dialogue|action|establishing|transition|emotional|sensory",
 "emotion": "calm|tense|joyful|dark|neutral|energetic",
 "narrative": "setup|development|climax|resolution|transition",
 "quality": 1-5,
 "importance": 1-5,
 "motion": "static|subtle|burst",
 "motion_change": "帧间变化一句话",
 "function": "ESTABLISHING|ACTION|REACTION|DETAIL|TRANSITION|ATMOSPHERIC_EVIDENCE|SENSORY_INSERT",
 "role": "hero|support|transition|broll|discard",
 "temporal": "首帧到尾帧的变化"}
Rules: desc/temporal in Chinese. importance = information value + visual quality. Be conservative."""


def _extract_keyframes(video_path: str, in_us: int, out_us: int,
                       count: int = 3) -> list[str]:
    """从镜头内均匀抽 count 帧, 返回 base64 列表。"""
    import cv2
    dur = out_us - in_us
    positions = [in_us + int(dur * f) for f in (0.15, 0.50, 0.85)][:count]
    b64s = []
    cap = cv2.VideoCapture(video_path)
    try:
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_MSEC, pos / 1000.0)
            ok, frame = cap.read()
            if ok:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64s.append(base64.b64encode(buf.tobytes()).decode())
    finally:
        cap.release()
    return b64s


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    lo, hi = text.find("{"), text.rfind("}")
    if lo < 0 or hi <= lo:
        raise ValueError(f"无 JSON: {text[:100]}")
    return json.loads(text[lo:hi + 1])


def analyze_shot_semantic(
    video_path: str,
    shot_in_us: int,
    shot_out_us: int,
    *,
    base_url: str = "http://localhost:11434",
    model: str = "qwen3-vl:latest",
    timeout: int = 300,
) -> dict:
    """多帧语义分析：一次 VLM 调用理解整个镜头的内容、情绪与叙事角色。

    Returns:
        语义分析 dict（含 scene_description/subjects/emotional_tone/
        narrative_role/importance/motion_progression 等字段）。
        失败时 raise（调用方决定降级策略）。
    """
    b64s = _extract_keyframes(video_path, shot_in_us, shot_out_us, count=3)
    if not b64s:
        raise RuntimeError(f"无法从 {video_path} 提取帧 [{shot_in_us}-{shot_out_us}]")

    payload = {
        "model": model, "stream": False,
        "messages": [{"role": "user",
                      "content": _SEMANTIC_PROMPT, "images": b64s}],
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 1024},
        "think": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError:
        payload.pop("think")
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())

    content = resp["message"]["content"].strip()
    # fallback：content 为空时从 thinking 字段提取
    if not content:
        thinking = resp["message"].get("thinking", "")
        if thinking:
            content = thinking
    result = _extract_json(content)

    # 字段映射（短 key → 长 key）
    result["scene_description"] = result.pop("desc", "")
    result["action_type"] = result.pop("action", "sensory")
    result["emotional_tone"] = result.pop("emotion", "neutral")
    result["narrative_role"] = result.pop("narrative", "transition")
    result["motion_progression"] = result.pop("motion_change", "")
    result["temporal_notes"] = result.pop("temporal", "")
    result["shot_function"] = result.pop("function", "SENSORY_INSERT")
    result["proposed_role_v2"] = result.pop("role", "broll")
    result["motion_amount"] = result.pop("motion", "subtle")

    # 规范化
    result.setdefault("scene_description", "")
    result.setdefault("subjects", [])
    result.setdefault("emotional_tone", "neutral")
    result.setdefault("narrative_role", "transition")
    result.setdefault("visual_quality", 3)
    result.setdefault("importance", 3)
    result.setdefault("shot_function", "SENSORY_INSERT")
    result.setdefault("proposed_role_v2", "broll")
    return result
