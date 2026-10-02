"""Ollama qwen3-vl VLM 适配器。

基于 stdlib urllib 调用 Ollama /api/chat 接口。
无需 API key。Fail-closed：任何错误（网络、超时、解析失败）
返回 degraded dict，status=FAILED，不抛异常。
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from observation_service.vlm_adapter import (
    VLMAdapter,
    OBSERVED,
    FAILED,
    FT_NETWORK,
    FT_RATE_LIMIT,
    FT_DECODE,
    FT_PARSE,
    FT_EMPTY,
    FT_UNKNOWN,
)


_VALID_FUNCTIONS = {
    "ESTABLISHING", "ACTION", "REACTION", "DETAIL",
    "TRANSITION", "ATMOSPHERIC_EVIDENCE", "SENSORY_INSERT",
}
_VALID_MOTION = {"static", "subtle", "burst"}
_VALID_ROLES = {"hero", "support", "transition", "broll", "discard"}
#: P3-1 深度语义词表（analyze_frames 多帧模式）
_VALID_NARRATIVE = {"setup", "development", "climax", "resolution", "transition"}
_VALID_EMOTION = {"calm", "tense", "joyful", "dark", "neutral", "energetic"}
_VALID_ACTION = {"dialogue", "action", "establishing", "transition", "emotional", "sensory"}

_PROMPT = """You are looking at one frame extracted from a short-video shot.

Step 1: Describe the frame in one Chinese sentence (what do you actually see?
Setting, subjects, action, lighting).

Step 2: Return ONLY a JSON object (no markdown fences, no prose) with these 6 fields:
{
  "shot_function": one of ESTABLISHING/ACTION/REACTION/DETAIL/TRANSITION/ATMOSPHERIC_EVIDENCE/SENSORY_INSERT,
  "sensory_wet_heat": number 0-1 or null,
  "sensory_mood_intensity": number 0-1 or null,
  "motion_amount": one of static/subtle/burst,
  "proposed_role_v2": one of hero/support/transition/broll/discard,
  "importance": integer 1-5 (how essential this shot is to a cut of this
    material: 5=must keep, 4=valuable, 3=fine but replaceable, 2=weak,
    1=nearly useless; judge by information value, visual quality and
    narrative contribution)
}

Rules: pick the fallback value if you cannot tell; never invent numbers you
cannot support. Keep step 1 and step 2 on separate lines.
"""

#: P3-1 多帧深度语义 prompt（原 observation_service/semantic_analyzer 收编：
#: 私有 cv2 抽帧与 urllib 传输并入适配器层；实测 qwen3-vl 长 prompt +
#: 多图会全输出进 thinking，故精简 + format:"json" + think:false）。
_SEMANTIC_PROMPT = """Analyze these 3 frames from one video shot. Return ONLY JSON:
{"desc": "中文一句话场景描述",
 "subjects": ["主体列表"],
 "people": ["可见人物的外观描述（颜色+衣物+发型，如'红衣短发女孩'）"],
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


def _extract_json(text: str) -> dict | None:
    """从模型回复中提取 JSON：优先 ```json 围栏，否则取第一个 { 到最后一个 }。"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
    else:
        lo = text.find("{")
        hi = text.rfind("}")
        if lo == -1 or hi == -1 or hi <= lo:
            return None
        candidate = text[lo:hi + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _norm_float(v) -> float | None:
    """将值归一化到 0-1 浮点数；非法值返回 None。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f < 0.0 or f > 1.0:
        return None
    return f


class OllamaVLMAdapter(VLMAdapter):
    """本地 Ollama qwen3-vl 适配器。"""

    def __init__(
        self,
        model: str = "qwen3-vl",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def analyze_frame(self, image_path: str) -> dict:
        """分析单帧图片。Fail-closed：任何错误返回 degraded dict。"""
        # 读取并 base64 编码图片
        try:
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        except Exception as e:
            return self._degraded(f"image read failed: {e}", FT_DECODE)

        body = {
            "model": self.model,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": _PROMPT,
                "images": [b64],
            }],
        }

        # 调用 Ollama /api/chat
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            raw_text = (data.get("message") or {}).get("content", "")
        except urllib.error.HTTPError as e:
            ft = FT_RATE_LIMIT if e.code == 429 else FT_NETWORK
            return self._degraded(f"http {e.code}: {e.reason}", ft)
        except Exception as e:
            return self._degraded(f"{type(e).__name__}: {e}", FT_NETWORK)

        if not raw_text:
            return self._degraded("empty reply", FT_EMPTY)

        # 分离描述与 JSON
        parsed = _extract_json(raw_text)
        brace = raw_text.find("{")
        desc = raw_text[:brace].strip() if brace != -1 else raw_text.strip()
        desc = desc.rstrip("。.\n ")

        if parsed is None:
            return self._degraded(
                f"json parse failed; raw={raw_text[:120]}", FT_PARSE,
            )

        warnings: list[str] = []

        sf_raw = parsed.get("shot_function")
        if sf_raw in _VALID_FUNCTIONS:
            shot_function = sf_raw
        else:
            shot_function = "SENSORY_INSERT"
            warnings.append(
                f"shot_function: got {sf_raw!r}, fallback to 'SENSORY_INSERT'",
            )

        mo_raw = parsed.get("motion_amount")
        if mo_raw in _VALID_MOTION:
            motion_amount = mo_raw
        else:
            motion_amount = "subtle"
            warnings.append(
                f"motion_amount: got {mo_raw!r}, fallback to 'subtle'",
            )

        role_raw = parsed.get("proposed_role_v2")
        if role_raw in _VALID_ROLES:
            proposed_role_v2 = role_raw
        else:
            proposed_role_v2 = "broll"
            warnings.append(
                f"proposed_role_v2: got {role_raw!r}, fallback to 'broll'",
            )

        # P3-2：TVSum 同构 importance（1-5 整数；缺/非法 → None=未标，
        # 下游记 0 并按未标注处理——不捏造）
        imp_raw = parsed.get("importance")
        importance = imp_raw if isinstance(imp_raw, int) and 1 <= imp_raw <= 5 else None

        return {
            "shot_function": shot_function,
            "sensory_wet_heat": _norm_float(parsed.get("sensory_wet_heat")),
            "sensory_mood_intensity": _norm_float(
                parsed.get("sensory_mood_intensity"),
            ),
            "motion_amount": motion_amount,
            "proposed_role_v2": proposed_role_v2,
            "importance": importance,
            "frame_description": desc,
            "status": OBSERVED,
            "degraded": bool(warnings),
            "confidence_type": "SELF_REPORTED",
            "_warnings": warnings,
        }

    def _degraded(self, reason: str, failure_type: str = FT_UNKNOWN) -> dict:
        """生成 degraded dict（所有字段扁平，fallback 值）。"""
        return {
            "shot_function": "SENSORY_INSERT",
            "sensory_wet_heat": None,
            "sensory_mood_intensity": None,
            "motion_amount": "subtle",
            "proposed_role_v2": "broll",
            "frame_description": "",
            "status": FAILED,
            "degraded": True,
            "confidence_type": "UNAVAILABLE",
            "failure_type": failure_type,
            "degrade_reason": reason,
            "_warnings": [],
        }

    # ------------------------------------------------------------------
    # P3-1 多帧深度语义（observation_service/semantic_analyzer 收编）
    # ------------------------------------------------------------------

    def analyze_frames(self, image_paths: list[str]) -> dict:
        """镜头内多帧一次 VLM 调用 → 深度语义扁平 dict。

        返回 analyze_frame 的全部字段，另加：scene_description / subjects /
        action_type / emotional_tone / narrative_role / visual_quality /
        motion_progression / temporal_notes。fail-closed：任何错误返回
        degraded dict，不抛异常。
        """
        if not image_paths:
            return self._degraded("no frames provided", FT_DECODE)

        try:
            b64s = [
                base64.b64encode(Path(p).read_bytes()).decode()
                for p in image_paths
            ]
        except Exception as e:
            return self._degraded(f"image read failed: {e}", FT_DECODE)

        body = {
            "model": self.model,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": _SEMANTIC_PROMPT,
                "images": b64s,
            }],
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 1024},
            "think": False,  # qwen3-vl 思考模式吞可见输出
        }
        raw_text, err = self._chat(body)
        if err is not None and "format" in str(err).lower():
            # 旧版 ollama 不认 format/think → 去掉重试一次
            body.pop("think", None)
            body.pop("format", None)
            raw_text, err = self._chat(body)
        if err is not None:
            return self._degraded(str(err), FT_NETWORK)
        if not raw_text:
            # thinking 字段兜底（P3-1 实测：多图长 prompt 全输出进 thinking）
            return self._degraded("empty reply", FT_EMPTY)

        parsed = _extract_json(raw_text)
        if parsed is None:
            return self._degraded(
                f"json parse failed; raw={raw_text[:120]}", FT_PARSE)

        warnings: list[str] = []

        def _vocab(value, valid, fallback, label):
            if value in valid:
                return value
            warnings.append(f"{label}: got {value!r}, fallback to {fallback!r}")
            return fallback

        def _int5(value):
            return value if isinstance(value, int) and 1 <= value <= 5 else None

        shot_function = _vocab(
            parsed.get("function"), _VALID_FUNCTIONS,
            "SENSORY_INSERT", "shot_function")
        role = _vocab(
            parsed.get("role"), _VALID_ROLES, "broll", "proposed_role_v2")
        motion = _vocab(parsed.get("motion"), _VALID_MOTION, "subtle", "motion_amount")
        narrative = _vocab(
            parsed.get("narrative"), _VALID_NARRATIVE,
            "transition", "narrative_role")
        emotion = _vocab(
            parsed.get("emotion"), _VALID_EMOTION, "neutral", "emotional_tone")
        action = _vocab(
            parsed.get("action"), _VALID_ACTION, "sensory", "action_type")

        return {
            # 与 analyze_frame 同形的基础字段
            "shot_function": shot_function,
            "sensory_wet_heat": _norm_float(parsed.get("sensory_wet_heat")),
            "sensory_mood_intensity": _norm_float(
                parsed.get("sensory_mood_intensity")),
            "motion_amount": motion,
            "proposed_role_v2": role,
            "importance": _int5(parsed.get("importance")),
            "frame_description": str(parsed.get("desc", "")).strip(),
            "status": OBSERVED,
            "degraded": bool(warnings),
            "confidence_type": "SELF_REPORTED",
            "_warnings": warnings,
            # P3-1 深度语义字段
            "scene_description": str(parsed.get("desc", "")).strip(),
            "subjects": parsed.get("subjects") or [],
            "people": [s for s in (parsed.get("people") or [])
                       if isinstance(s, str) and s.strip()],
            "action_type": action,
            "emotional_tone": emotion,
            "narrative_role": narrative,
            "visual_quality": _int5(parsed.get("quality")),
            "motion_progression": str(parsed.get("motion_change", "")).strip(),
            "temporal_notes": str(parsed.get("temporal", "")).strip(),
        }

    def _chat(self, body: dict) -> tuple[str, Exception | None]:
        """POST /api/chat，返回 (content, error)；二者只会有一个非空。"""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            content = (data.get("message") or {}).get("content", "")
            if not content:
                thinking = (data.get("message") or {}).get("thinking", "")
                content = thinking or ""
            return content, None
        except urllib.error.HTTPError as e:
            return "", RuntimeError(f"http {e.code}: {e.reason}")
        except Exception as e:
            return "", e
