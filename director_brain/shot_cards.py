"""镜头卡库：加载、校验与 brief→卡选择（D1 · 美学策略载体 v1）。

卡库为版本化数据文件（``cards/v1.json``），过 :class:`ShotCard` schema
校验即可入库——第三方供卡的插拔点与 register_reasoner 同哲学。
选择器是确定性规则匹配（非 LLM）：editing_language / emotional_arc /
素材镜头数条件打分，平分时按卡库声明顺序。
"""
from __future__ import annotations

import json
from pathlib import Path

from director_brain.models.director_brief import DirectorBrief
from director_brain.models.shot_card import ShotCard, VALID_ENERGY, VALID_TRANSITION_POLICY

_CARDS_DIR = Path(__file__).resolve().parent / "cards"
_SCHEMA_VERSION = "1.0"
_CARD_FILE_VERSION = "1"  # 文件名版本（v1.json）；卡内 version 字段为 1.0


def load_cards() -> list[ShotCard]:
    """加载全部内置卡（schema 校验失败即抛错——坏卡不能静默入库）。"""
    path = _CARDS_DIR / f"v{_CARD_FILE_VERSION}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    cards = [ShotCard(**item) for item in raw]
    ids = [c.card_id for c in cards]
    if len(ids) != len(set(ids)):
        raise ValueError(f"卡库 card_id 重复: {ids}")
    for c in cards:
        if c.energy not in VALID_ENERGY:
            raise ValueError(f"卡 {c.card_id} 非法 energy: {c.energy}")
        if c.transition_policy not in VALID_TRANSITION_POLICY:
            raise ValueError(f"卡 {c.card_id} 非法 transition_policy: {c.transition_policy}")
    return cards


def get_card(card_id: str) -> ShotCard | None:
    """按 card_id 取卡；不存在返回 None。"""
    for c in load_cards():
        if c.card_id == card_id:
            return c
    return None


def select_card(
    brief: DirectorBrief,
    shot_count: int,
    *,
    card_id: str | None = None,
) -> ShotCard | None:
    """brief→卡确定性匹配。

    card_id 显式指定时直接取（不存在抛 ValueError——响亮失败）；
    未指定时按规则打分选最优：剪辑语言命中(+2) > 情绪弧命中(+1) >
    能量近似(+0.5)，素材镜头数不满足 min_shots 的卡直接跳过。
    无卡可配（含素材条件全不满足）返回 None——调用方按无卡路径工作。
    """
    cards = load_cards()
    if card_id:
        for c in cards:
            if c.card_id == card_id:
                _require_conditions(c, shot_count)
                return c
        raise ValueError(
            f"未知镜头卡: {card_id!r}；可用: {[c.card_id for c in cards]}")

    best: tuple[float, ShotCard] | None = None
    for c in cards:
        if shot_count < c.min_shots:
            continue
        score = 0.0
        if brief.editing_language in c.editing_language_fit:
            score += 2.0
        arc = (brief.emotional_arc or "neutral").strip().lower()
        if arc in c.emotion_arc_fit:
            score += 1.0
        energy_near = {
            "low": {"low", "calm"}, "calm": {"calm", "low", "medium"},
            "medium": {"medium", "calm", "high"},
            "high": {"high", "medium", "escalating"},
            "escalating": {"escalating", "high"},
        }.get(arc, set())
        if c.energy in energy_near:
            score += 0.5
        if best is None or score > best[0]:
            best = (score, c)
    if best is None or best[0] <= 0.0:
        return None
    return best[1]


def _require_conditions(card: ShotCard, shot_count: int) -> None:
    if shot_count < card.min_shots:
        raise ValueError(
            f"镜头卡 {card.card_id} 需要至少 {card.min_shots} 个镜头，"
            f"当前素材 {shot_count} 个——条件不满足（响亮失败）")
