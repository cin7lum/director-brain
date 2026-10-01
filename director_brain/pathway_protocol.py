"""信号通路接线协议（T4）。

02 的三条感知信号通路（VLM 语义观测 / 镜头关系推断 / ASR 语音转写）
历史上"算了但不进决策"或被静默降级消费。本协议把它们的状态显式化：

- EXPERIMENTAL：已实现，**不可达主链决策函数**（默认态，止血）；
- SHADOW：影子运行——随主链计算并全量上报（逐条、含归因），不驱动决策；
- ACTIVE：已接入决策。前置条件：影子期证据（覆盖率/质量/决策 delta）
  达标并经主控批准；切换必须走 :func:`set_pathway_status` 留痕；
- RETIRED：永久废弃（如语义错误被证实无法修复）。

铁律：
1. 决策函数消费某通路前必须通过 :func:`ensure_decision_use_allowed`——
   非 ACTIVE 一律抛 :class:`PathwayNotActiveError`（fail-closed）。
2. 禁止半消费：信号要么全量上报（影子），要么不计算；"算了只打印条数"
   视为缺陷。
3. 技术信号（视觉突变等）不得直推语义标签（情绪/身份/因果）——语义
   结论必须由语义证据（VLM 情绪观测、ASR 内容、用户意图）支撑。
"""
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class PathwayStatus(str, Enum):
    """信号通路状态。"""

    EXPERIMENTAL = "EXPERIMENTAL"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


class PathwayNotActiveError(RuntimeError):
    """通路未 ACTIVE 时试图在决策中使用其信号（协议违规，fail-closed）。"""


class UnknownPathwayError(ValueError):
    """未登记的通路名。"""


#: 通路登记表：通路名 → 当前状态（默认 EXPERIMENTAL，止血态）。
_STATUS: dict[str, PathwayStatus] = {
    # vlm_semantic 置 ACTIVE（2026-09-30，灰度验证通过后按批准计划转正）：
    # 证据链 = 多轮真机 ACTIVE 出片（sintel 语义 pilot 20.00s 整/PASS、
    # plan_judge narrative 4-5、demo 素材 L1 零硬伤）+ 影子对账无吊销事件
    # + P0 暗镜头排除不变量补漏后黑尾消失。生产入口仍需 --semantic 显式
    # 开启（默认 CLI 行为不变）；EXPERIMENTAL/SHADOW 回退 =
    # set_pathway_status("vlm_semantic", PathwayStatus.SHADOW)。
    "vlm_semantic": PathwayStatus.ACTIVE,
    "relation_inference": PathwayStatus.EXPERIMENTAL,
    #: ASR 已随主链计算并全量归因上报（T4），其 Brief 消费不驱动选片——
    #: 现实角色即影子，故默认 SHADOW 而非 EXPERIMENTAL。
    "asr_transcript": PathwayStatus.SHADOW,
    #: 链 B 语义决策（阶段 7.5）：已准入模型（doubao-seed-2-1-lite-260915，
    #: 见 evidence/DIRECTOR_BRAIN_MODEL_ADMISSION/MODEL_ADMISSION.md）经
    #: ark_adapter 并行产出 DirectorDecision，全量对账上报，不驱动成片。
    #: 与 asr_transcript 同理：现实角色即影子，默认 SHADOW；置 ACTIVE 属
    #: 治理决策，须用户确认且以 L2/L3 影子期证据为前提。
    "semantic_reasoner": PathwayStatus.SHADOW,
}

#: 各通路的一句话职责（上报用）。
_PATHWAY_PURPOSE: dict[str, str] = {
    "vlm_semantic": "VLM 逐镜头语义观测（shot_function/role/motion）",
    "relation_inference": "镜头间关系边（技术信号 + 可选 VLM 语义）",
    "asr_transcript": "ASR 语音转写观测",
    "semantic_reasoner": "链 B 语义决策（LLM→DirectorDecision，影子对账）",
}


def get_pathway_status(pathway: str) -> PathwayStatus:
    """读取通路当前状态；未登记通路抛 :class:`UnknownPathwayError`。"""
    try:
        return _STATUS[pathway]
    except KeyError:
        raise UnknownPathwayError(f"未登记的信号通路: {pathway!r}") from None


def set_pathway_status(pathway: str, status: PathwayStatus | str) -> PathwayStatus:
    """切换通路状态（feature flag），留痕日志。

    任何切换都会记录 旧状态 → 新状态；把通路置为 ACTIVE 属治理决策，
    调用方须能出示影子期证据（本函数只负责留痕，不做证据校验）。
    """
    if isinstance(status, str):
        status = PathwayStatus(status)
    if pathway not in _STATUS:
        raise UnknownPathwayError(f"未登记的信号通路: {pathway!r}")
    old = _STATUS[pathway]
    _STATUS[pathway] = status
    logger.info("pathway status: %s %s -> %s", pathway, old.value, status.value)
    return status


def ensure_decision_use_allowed(pathway: str) -> None:
    """决策函数消费通路信号前的强制闸门：非 ACTIVE 抛错（fail-closed）。"""
    status = get_pathway_status(pathway)
    if status is not PathwayStatus.ACTIVE:
        raise PathwayNotActiveError(
            f"信号通路 {pathway!r} 状态为 {status.value}，不可进入决策"
            f"（需 ACTIVE；影子期信号只记录不驱动）"
        )


def describe() -> str:
    """渲染当前通路状态表（供运行摘要/证据展示）。"""
    lines = ["通路状态（T4 shadow 协议）:"]
    for pathway, status in _STATUS.items():
        lines.append(f"  - {pathway}: {status.value} —— {_PATHWAY_PURPOSE[pathway]}")
    return "\n".join(lines)
