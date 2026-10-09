"""Source checks for explicit numeric parameter claims.

These checks establish literal value/unit provenance only. They do not prove
that a parameter is semantically appropriate or feasible on a timeline.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


_NUMBER_TOKEN = re.compile(
    r"(?<![0-9.])[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?![0-9.])"
)

_FRAME_UNITS = frozenset({"frame", "frames", "帧", "帧数"})


def is_frame_unit(unit: str | None) -> bool:
    """Whether ``unit`` is an explicit spelling accepted by the J-cut path."""
    return isinstance(unit, str) and unit.strip().casefold() in _FRAME_UNITS


def _unit_pattern(unit: str) -> re.Pattern[str]:
    normalized = unit.strip().casefold()
    if normalized in _FRAME_UNITS:
        expression = r"(?:frames?|帧(?:数)?)(?![a-z])"
    elif normalized in {"second", "seconds", "sec", "secs", "s", "秒", "秒钟"}:
        expression = r"(?:seconds?|secs?|s(?![a-z])|秒(?:钟)?)"
    elif normalized in {"millisecond", "milliseconds", "ms", "毫秒"}:
        expression = r"(?:milliseconds?|ms|毫秒)(?![a-z])"
    elif normalized in {"percent", "percentage", "%", "百分比"}:
        expression = r"(?:percent(?:age)?|%|百分比)(?![a-z])"
    else:
        expression = re.escape(unit.strip()) + r"(?![a-z0-9_])"
    return re.compile(expression, re.IGNORECASE)


def has_explicit_numeric_unit_quote(
    exact_value: float,
    unit: str | None,
    evidence_quotes: list[str],
) -> bool:
    """Require one quoted excerpt to contain the exact numeric value and unit.

    Callers must separately verify that each quote is an exact substring of
    the original request. Arabic-digit forms are intentionally required here;
    unsupported number words fail closed rather than becoming executable values.
    """
    if not isinstance(unit, str) or not unit.strip():
        return False
    try:
        expected = Decimal(str(exact_value))
    except (InvalidOperation, ValueError):
        return False
    if not expected.is_finite():
        return False

    unit_pattern = _unit_pattern(unit)
    for quote in evidence_quotes:
        if not isinstance(quote, str):
            continue
        for number_match in _NUMBER_TOKEN.finditer(quote):
            try:
                observed = Decimal(number_match.group())
            except InvalidOperation:
                continue
            if observed != expected:
                continue

            suffix = quote[number_match.end():].lstrip()
            if suffix[:1] in {"-", "‐", "‑", "–", "—"}:
                suffix = suffix[1:].lstrip()
            if unit_pattern.match(suffix):
                return True
    return False
