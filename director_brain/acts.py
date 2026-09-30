"""四幕结构单一事实源（架构体检候选⑥收编）。

"四幕"此前在四处各自定义（story_graph_builder._ACTS 区间形状 /
director_reasoner._ACT_RATIO 配额形状 / strategy_selector._ACTS /
semantic_scorer 私有配额表），一致性靠"与 _ACTS 保持一致"式注释——
注释对齐就是漂移邀请函。本模块是唯一权威定义：

- 配额表（ACT_RATIO）与区间表（ACT_INTERVALS）由同一份数据**代码级派生**
  （区间 = 配额累积和），幕语义修改只动 ACT_RATIO 一处；
- 排序（ACT_ORDER）、镜头功能（ACT_FUNCTION）、VLM 叙事角色映射
  （ROLE_TO_ACT）同源收编。

约束：ACT_RATIO 四值之和必须为 1.0（import 时强制校验）。
"""
from __future__ import annotations

ACT_NAMES: tuple[str, ...] = ("hook", "develop", "peak", "resolve")

#: 四幕时长配额（占目标成片比例）——幕时长语义的唯一修改点。
ACT_RATIO: dict[str, float] = {
    "hook": 0.15,
    "develop": 0.35,
    "peak": 0.30,
    "resolve": 0.20,
}

assert abs(sum(ACT_RATIO.values()) - 1.0) < 1e-9, "四幕配额之和必须为 1.0"

#: EDL 排序顺序（hook → develop → peak → resolve）。
ACT_ORDER: dict[str, int] = {name: i for i, name in enumerate(ACT_NAMES)}

#: 幕 → EditItem.shot_function。
ACT_FUNCTION: dict[str, str] = {
    "hook": "opening",
    "develop": "pacing",
    "peak": "peak",
    "resolve": "closing",
}

#: 四幕时间区间（起点比例, 终点比例, 中文标签）——由 ACT_RATIO 累积派生，
#: 与配额表代码级一致（此前是注释级一致）。
_ACT_LABELS: dict[str, str] = {
    "hook": "开场",
    "develop": "发展",
    "peak": "高潮",
    "resolve": "收尾",
}
ACT_INTERVALS: list[tuple[str, float, float, str]] = []
_lo = 0.0
for _name in ACT_NAMES:
    _hi = _lo + ACT_RATIO[_name]
    ACT_INTERVALS.append(
        (_name, round(_lo, 10), round(_hi, 10), _ACT_LABELS[_name])
    )
    _lo = _hi

#: VLM narrative_role → 幕（P3-2 语义驱动幕分配；内核消费须过通路闸门）。
ROLE_TO_ACT: dict[str, str] = {
    "setup": "hook",
    "development": "develop",
    "climax": "peak",
    "resolution": "resolve",
    "transition": "develop",  # 过渡镜头归 develop
}
