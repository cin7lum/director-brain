"""不可信输入消毒（阶段 S5 · 方案 §8 提示注入防护）。"""
from __future__ import annotations
import re

_MAX_UNTRUSTED_LEN = 2000

_ROLE_MARKERS_RE = re.compile(
    r"<\|[^|]*\|>"                    # <|im_start|> / <|im_end|>
    r"|</?\s*(?:system|assistant|user)\s*>"
    r"|\b(?:SYSTEM|ASSISTANT|USER)\b\s*[:：]",
    re.IGNORECASE,
)
_OVERRIDE_RE = re.compile(
    r"(?:ignore|disregard|forget|override)\s+(?:all\s+|the\s+)?"
    r"(?:above|previous|prior|earlier|system|initial)\s+"
    r"(?:instructions?|prompts?|rules?|constraints?)",
    re.IGNORECASE,
)
_TOOL_RE = re.compile(
    r"(?:call|invoke|execute|run|use)\s+(?:the\s+)?"
    r"(?:ffmpeg|ffprobe|davinci|resolve|shell|bash|python|script|tool|api|command)\b",
    re.IGNORECASE,
)
_FORMAT_RE = re.compile(
    r"(?:output|respond|reply|answer)\s+(?:in|with|as)\s+"
    r"(?:plain\s+text|markdown|html|code|raw)\b",
    re.IGNORECASE,
)


def sanitize_untrusted(text: str, max_len: int = _MAX_UNTRUSTED_LEN) -> str:
    if not text:
        return text
    c = text
    c = _ROLE_MARKERS_RE.sub("", c)
    c = _OVERRIDE_RE.sub("", c)
    c = _TOOL_RE.sub("", c)
    c = _FORMAT_RE.sub("", c)
    return c[:max_len].strip()


def wrap_untrusted(text: str, label: str = "user input") -> str:
    c = sanitize_untrusted(text)
    return f"[BEGIN {label} (untrusted)]\n{c}\n[END {label}]"
