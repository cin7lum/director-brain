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

_PROMPT = """You are looking at one frame extracted from a short-video shot.

Step 1: Describe the frame in one Chinese sentence (what do you actually see?
Setting, subjects, action, lighting).

Step 2: Return ONLY a JSON object (no markdown fences, no prose) with these 5 fields:
{
  "shot_function": one of ESTABLISHING/ACTION/REACTION/DETAIL/TRANSITION/ATMOSPHERIC_EVIDENCE/SENSORY_INSERT,
  "sensory_wet_heat": number 0-1 or null,
  "sensory_mood_intensity": number 0-1 or null,
  "motion_amount": one of static/subtle/burst,
  "proposed_role_v2": one of hero/support/transition/broll/discard
}

Rules: pick the fallback value if you cannot tell; never invent numbers you
cannot support. Keep step 1 and step 2 on separate lines.
"""


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

        return {
            "shot_function": shot_function,
            "sensory_wet_heat": _norm_float(parsed.get("sensory_wet_heat")),
            "sensory_mood_intensity": _norm_float(
                parsed.get("sensory_mood_intensity"),
            ),
            "motion_amount": motion_amount,
            "proposed_role_v2": proposed_role_v2,
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
