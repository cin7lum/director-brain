"""用户意图约束解释器（产品表现补强 P1-a）。

审计发现：Brief 的 must_include/must_avoid 从未被下游消费——用户说
"避免模糊镜头"对成片零影响（意图入口是安慰剂）。本模块把规则抽取出的
约束**解释为当前证据能力下可确定性执行的指令**：

- must_avoid 命中技术词表 → 生成 :class:`TechnicalAvoidRule`（对候选
  过滤、写入 plan.constraints、validator 判红三处生效）；
- 未命中词表的 must_avoid / 全部 must_include → 属**语义约束**，当前
  链 A 只有技术观测、无内容语义证据，无法验证——诚实记为不可验证
  （等待 VLM 语义通路 ACTIVE 后升级），绝不静默假装执行。

阈值标定说明：各技术阈值是启发式初值（blur/亮度分布依据 deterministic
观测的典型量级），应由 L1 指标矩阵数据修订，不拍脑袋固化。
"""
from __future__ import annotations

from dataclasses import dataclass

from director_brain.models.director_brief import DirectorBrief


@dataclass(frozen=True)
class TechnicalAvoidRule:
    """一条可确定性执行的排除性约束。

    metric 取观测 claim 中的数值指标名；predicate 为 ``below``/``above``，
    表示"该指标低于/高于 threshold 的镜头违反本约束"。
    """

    term: str          # 触发的用户词（如"模糊"）
    metric: str        # claim 指标名：blur_score / brightness_mean / shake_score
    predicate: str     # "below" | "above"
    threshold: float

    def violated(self, metrics: dict) -> bool:
        value = metrics.get(self.metric)
        if not isinstance(value, (int, float)):
            return False
        if self.predicate == "below":
            return float(value) < self.threshold
        return float(value) > self.threshold

    def encode(self) -> str:
        """编码为 plan.constraints 条目（validator 据此判红）。

        格式：``must_avoid:<metric><<或>>:<threshold>:<term>``。
        与既有 ``target_duration_us=...`` 字符串约束同族；结构化约束
        字段列入后续演进。
        """
        op = "<" if self.predicate == "below" else ">"
        return f"must_avoid:{self.metric}{op}{self.threshold}:{self.term}"


#: 技术词表：用户词（子串匹配）→ 技术排除规则。
#: 阈值初值依据 deterministic 观测典型量级标定，由 L1 矩阵数据修订。
_TECHNICAL_AVOID_VOCABULARY: list[tuple[tuple[str, ...], TechnicalAvoidRule]] = [
    (("模糊", "虚焦", "失焦"), TechnicalAvoidRule("模糊", "blur_score", "below", 50.0)),
    (("黑屏", "黑帧", "过暗", "太暗"), TechnicalAvoidRule("过暗", "brightness_mean", "below", 40.0)),
    (("过曝", "过亮", "太亮", "白屏"), TechnicalAvoidRule("过曝", "brightness_mean", "above", 220.0)),
    (("抖动", "晃动", "抖得"), TechnicalAvoidRule("抖动", "shake_score", "above", 0.5)),
]


@dataclass
class ConstraintDirectives:
    """对 Brief 约束的解释结果。"""

    #: (brief 条目原文, 命中的技术规则)；同一词条目可命中多条
    avoid_rules: list[tuple[str, TechnicalAvoidRule]]
    #: 当前证据能力无法验证的约束 (kind, 原文)
    unverifiable: list[tuple[str, str]]


def interpret_constraints(brief: DirectorBrief) -> ConstraintDirectives:
    """把 Brief 的 must_avoid/must_include 解释为可执行指令与不可验证项。"""
    avoid_rules: list[tuple[str, TechnicalAvoidRule]] = []
    unverifiable: list[tuple[str, str]] = []

    for item in brief.must_avoid:
        matched = False
        for keywords, rule in _TECHNICAL_AVOID_VOCABULARY:
            if any(k in item for k in keywords):
                avoid_rules.append((item, rule))
                matched = True
        if not matched:
            unverifiable.append(("must_avoid", item))

    for item in brief.must_include:
        # must_include 需要"画面里有 X"的正向语义证据，技术观测无法验证
        unverifiable.append(("must_include", item))

    return ConstraintDirectives(avoid_rules=avoid_rules, unverifiable=unverifiable)


def candidate_violated_rules(candidate: dict, rules: list[TechnicalAvoidRule]) -> list[TechnicalAvoidRule]:
    """返回候选镜头违反的技术排除规则列表。

    candidate 为 ``_build_candidates`` 产物；claim 数值已在构建期展开的
    只有 blur_score/exposure_ok，其余指标按需从 ``_claim_metrics`` 读取。
    """
    metrics = candidate.get("_claim_metrics") or {}
    return [r for r in rules if r.violated(metrics)]
