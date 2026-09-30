"""链 B 语义决策影子运行器（阶段 7.5）。

已准入模型（doubao-seed-2-1-lite-260915，见
``evidence/DIRECTOR_BRAIN_MODEL_ADMISSION/MODEL_ADMISSION.md``）经
:class:`~director_brain.ark_adapter.ArkLLMAdapter` 对用户意图文本并行产出
DirectorDecision，与链 A（确定性启发式）的执行结果做全量对账上报。

铁律（与 :mod:`director_brain.pathway_protocol` 一致）：

- 通路 ``semantic_reasoner`` 处于 SHADOW：影子输出只上报（账本 + 对账
  报告），绝不进入选片/修复/渲染决策；消费侧由
  :func:`~director_brain.pathway_protocol.ensure_decision_use_allowed`
  fail-closed 把关（SHADOW 下消费即抛错）。
- fail-soft 响亮降级：无 key / 配置缺失 / API 失败 / 校验失败都记录原因
  并放行主链，绝不静默、绝不阻断、绝不做语义补救（无关键词兜底——红线）。
- 传输级重试（限流/竞态类间歇失败）与 run_benchmark 同参数；重试不改
  语义层。
- 账本记录 model / prompt / schema 三版本，形成可复算链。

对账口径（诚实边界）：
- 链 B 负向承诺（must_avoid + must_preserve）逐条对照链 A 已编码的
  ``must_avoid:<metric><op>:<threshold>:<term>`` 确定性规则；命中为
  "已覆盖"（链 A 有执法），未命中为"未覆盖"——即治理文件登记的
  "语义类 must_avoid 无独立确定性执法"风险的具体实例，逐条可见。
- 状态对账：链 A 通过验证继续出片而链 B 判 NEEDS_CONTEXT/CONFLICTING/
  UNSUPPORTED（或反向）都记为状态分歧，只上报不处置。
- 正向意图（desired_relation_or_change）当前纯信息项：链 A 尚无同层
  消费者，不参与对账。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from director_brain.ark_adapter import ArkLLMAdapter
from director_brain.audit_trail import log_decision
from director_brain.semantic_reasoner import SemanticDirectorReasoner, SemanticReasonerResult
from director_brain.pathway_protocol import (
    PathwayStatus,
    get_pathway_status,
)

logger = logging.getLogger(__name__)

#: 影子通路名（pathway_protocol 注册表键）。
SHADOW_PATHWAY = "semantic_reasoner"

#: 环境变量：key 缺失 → 跳过；model 缺失 → 拒绝默认（防回退到被拒模型）。
_ENV_API_KEY = "ARK_API_KEY"
_ENV_MODEL = "ARK_MODEL"

#: 传输级重试参数（与 experiments/director_semantic_poc/run_benchmark.py 一致）。
_MAX_ATTEMPTS = 3
_BACKOFF_S = (2.0, 6.0)


def _read_env(key: str) -> str | None:
    """读环境变量，回落项目根 .env（与 config.py 同纪律，不入库）。"""
    import os

    val = os.environ.get(key)
    if val:
        return val
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key and v.strip():
            return v.strip()
    return None


def build_chain_b_adapter() -> tuple[object | None, str | None]:
    """按 TEXT_LLM_PROVIDER 构建链 B 传输适配器（成品级扫荡：接通死配置）。

    provider：
    - ``ark``（默认）：ARK_API_KEY + ARK_MODEL。模型不设默认：ARK_MODEL
      未钉扎时拒绝构造——默认值曾是被拒模型（doubao-seed-2.0-lite 别名），
      静默回退等于绕过准入。
    - ``zhipu``：ZHIPU_API_KEY + TEXT_LLM_MODEL（此前 text_llm_* 配置
      零消费；影子对账不驱动成片，model 全量记录进对账报告）。
    - ``ollama``：TEXT_LLM_MODEL + OLLAMA_BASE_URL（本地传输）。

    返回 (adapter, None) 或 (None, 原因)。
    """
    from director_brain.config import load_settings

    settings = load_settings()
    provider = (settings.text_llm_provider or "ark").strip().lower()

    if provider == "zhipu":
        from director_brain.zhipu_adapter import ZhipuLLMAdapter

        api_key = _read_env("ZHIPU_API_KEY") or settings.zhipu_api_key
        if not api_key:
            return None, "未设置 ZHIPU_API_KEY——链 B 影子跳过（不阻断主链）"
        return ZhipuLLMAdapter(
            api_key=api_key,
            model=_read_env("TEXT_LLM_MODEL") or settings.text_llm_model,
            base_url=settings.zhipu_base_url,
        ), None

    if provider == "ollama":
        from director_brain.llm_adapter import LLMAdapter

        return LLMAdapter(
            model=_read_env("TEXT_LLM_MODEL") or settings.text_llm_model,
            base_url=settings.ollama_base_url,
        ), None

    # ark（默认）
    api_key = _read_env(_ENV_API_KEY)
    if not api_key:
        return None, f"未设置 {_ENV_API_KEY}——链 B 影子跳过（不阻断主链）"
    model = _read_env(_ENV_MODEL)
    if not model:
        return None, (
            f"未设置 {_ENV_MODEL}——须钉扎准入快照 ID（如 "
            f"doubao-seed-2-1-lite-260915），拒绝默认到未准入模型"
        )
    return ArkLLMAdapter(api_key=api_key, model=model), None


def build_ark_adapter() -> tuple[ArkLLMAdapter | None, str | None]:
    """向后兼容别名（ark 分支语义不变）；新代码用 :func:`build_chain_b_adapter`。"""
    adapter, reason = build_chain_b_adapter()
    if adapter is not None and not isinstance(adapter, ArkLLMAdapter):
        return None, f"TEXT_LLM_PROVIDER 非 ark（当前 {type(adapter).__name__}）"
    return adapter, reason


@dataclass
class ShadowReport:
    """一次影子运行的全量上报。status: completed/failed/skipped。"""

    status: str
    reason: str | None = None
    pathway_status: str = ""
    model: str | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    latency_ms: int | None = None
    attempts: int = 0
    chain_b: dict | None = None
    #: 链 B 负向承诺中被链 A 确定性规则覆盖的项
    covered_negatives: list[str] = field(default_factory=list)
    #: 链 B 负向承诺中链 A 无确定性执法的项（人工抽检清单，逐条可见）
    uncovered_negatives: list[str] = field(default_factory=list)
    #: 状态分歧描述（无分歧为 None）
    status_divergence: str | None = None
    positive_intents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pathway": SHADOW_PATHWAY,
            "pathway_status": self.pathway_status,
            "status": self.status,
            "reason": self.reason,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "chain_b": self.chain_b,
            "covered_negatives": self.covered_negatives,
            "uncovered_negatives": self.uncovered_negatives,
            "status_divergence": self.status_divergence,
            "positive_intents": self.positive_intents,
        }


def extract_chain_a_avoid_terms(constraints: list[str] | None) -> list[str]:
    """从 plan.constraints 提取链 A 已编码的 must_avoid 术语。

    格式（intent_constraints.TechnicalAvoidRule.encode）：
    ``must_avoid:<metric><op><threshold>:<term>``。
    """
    terms: list[str] = []
    for c in constraints or []:
        if not isinstance(c, str) or not c.startswith("must_avoid:"):
            continue
        rest = c.split(":", 1)[1]
        term = rest.rsplit(":", 1)[-1].strip()
        if term:
            terms.append(term)
    return terms


#: 中英对账词表（P0 修复：链 B 英文枚举 ↔ 链 A 中文术语，双向映射）。
#: 同义/近义对放同一组——对账时跨语言命中视为已覆盖。
_BILINGUAL_GROUPS: list[set[str]] = [
    {"模糊", "blurry", "blur", "out_of_focus", "虚焦", "失焦"},
    {"黑屏", "黑帧", "black_frame", "black", "过暗", "too_dark"},
    {"抖动", "shake", "shaky", "晃动"},
    {"过曝", "过亮", "overexposed", "too_bright", "白屏"},
    {"transition", "转场"},
    {"slow_motion", "慢动作"},
    {"reorder", "重排", "重排序"},
    {"reorder_story_beat", "叙事重排"},
    {"action_scene", "动作场景", "关键动作", "key_action"},
    {"audio_lead", "声音提前"},
    {"extend", "延长"},
    {"shorten", "缩短"},
]


def _is_covered(token: str, chain_a_terms: list[str]) -> bool:
    """链 B 负向令牌是否被链 A 术语覆盖。

    两层匹配：
    1. 双向子串（原有逻辑）；
    2. 中英词表：token 与 chain_a_term 属同一同义组即视为覆盖
       （P0 修复：链 B 英文枚举 blurry_shots ↔ 链 A 中文 模糊 不再漏配）。
    """
    t = token.lower().strip().replace("_", " ").strip()
    if not t:
        return True
    for term in chain_a_terms:
        st = term.lower().strip().replace("_", " ").strip()
        if not st:
            continue
        # 直接子串
        if t in st or st in t:
            return True
        # 中英词表：token 与 term 是否属同义组
        for group in _BILINGUAL_GROUPS:
            g = {w.lower() for w in group}
            t_words = set(t.split()) | {t}
            s_words = set(st.split()) | {st}
            if t_words & g and s_words & g:
                return True
            # 单词级：token 或 term 包含组内词汇
            for w in g:
                if (w in t or t in w) and any(
                    sw in st or st in sw for sw in g if sw != w
                ):
                    return True
                if w in t and any(sw in st or st in sw for sw in g):
                    return True
    return False


def _status_divergence(chain_b_status: str, chain_a_valid: bool | None) -> str | None:
    """状态对账：两个方向都算分歧（信息项，不处置）。"""
    if chain_a_valid is None:
        return None
    if chain_a_valid and chain_b_status != "READY":
        return f"chain_a_proceeded_but_chain_b_status={chain_b_status}"
    if not chain_a_valid and chain_b_status == "READY":
        return "chain_a_rejected_but_chain_b_ready"
    return None


def _log(ledger, decision_id: str, action: str, detail: dict) -> None:
    """账本 fail-soft：与 roughcut._safe_log 同纪律，失败响亮不阻断。"""
    if ledger is None:
        return
    try:
        log_decision(ledger, decision_id, action, detail)
    except Exception as exc:  # noqa: BLE001
        logger.warning("账本写入失败（%s）: %s: %s", action, type(exc).__name__, exc)


def _write_sidecar(report: ShadowReport, output_path: str | None) -> None:
    """对账报告落盘（<output>.shadow.json，与 .edl.json/.plan.json 同族）。"""
    if not output_path:
        return
    try:
        sidecar = f"{output_path}.shadow.json"
        Path(sidecar).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("影子对账报告写入失败（不影响主链）: %s: %s",
                       type(exc).__name__, exc)


def run_shadow_semantic(
    intent_text: str,
    *,
    chain_a_constraints: list[str] | None = None,
    chain_a_valid: bool | None = None,
    decision_id: str | None = None,
    ledger=None,
    output_path: str | None = None,
    adapter_factory=None,
    max_attempts: int = _MAX_ATTEMPTS,
    backoff_s: tuple[float, ...] = _BACKOFF_S,
) -> ShadowReport:
    """运行一次链 B 影子语义决策并全量对账。

    adapter_factory（测试注入点）：``factory() -> (adapter | None, reason)``,
    adapter 须暴露 ``generate_decision(user_input, decision_id=...) -> LLMResult``
    与 ``model`` 属性。默认用 :func:`build_ark_adapter` 从环境构建。

    任何失败路径都返回 ShadowReport（status=skipped/failed + 原因），绝不
    抛异常阻断主链——影子信号的故障也必须可见，但无权拦住成片。
    """
    pathway_status = get_pathway_status(SHADOW_PATHWAY)
    base = ShadowReport(status="skipped", pathway_status=pathway_status.value)

    if pathway_status is PathwayStatus.EXPERIMENTAL:
        base.reason = "通路处于 EXPERIMENTAL（止血态）：不计算影子信号"
        return base
    if not intent_text or not intent_text.strip():
        base.reason = "无用户意图文本：链 B 无输入可解析"
        return base

    if adapter_factory is None:
        adapter_factory = build_chain_b_adapter
    adapter, reason = adapter_factory()
    if adapter is None:
        base.reason = reason or "适配器不可用"
        _log(ledger, decision_id or "shadow_unknown", "semantic_shadow_skipped",
             base.to_dict())
        return base

    # 候选②合并：影子消费必须经 SemanticDirectorReasoner（链 B 唯一的
    # "一句话→决策"路径），不再直调传输适配器——此前两条不相交实现
    # （reasoner 路径 vs 影子直调）各自维护重试/解析/fail-closed。
    reasoner = SemanticDirectorReasoner(llm_adapter=adapter)

    result = None
    attempts = 0
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            result = reasoner.reason(
                intent_text, decision_id=decision_id or "shadow_decision"
            )
        except Exception as exc:  # noqa: BLE001
            # 影子故障不得阻断主链：reasoner/适配器逃逸的意外异常按传输
            # 失败处理（响亮降级）。防扩散层——任何从影子通路逃逸的异常
            # 都等于阻断成片。
            result = SemanticReasonerResult(
                decision=None,
                error=f"shadow adapter raised: {type(exc).__name__}: {exc}",
            )
        if result.decision is not None:
            break
        if attempt < max_attempts:
            time.sleep(backoff_s[min(attempt - 1, len(backoff_s) - 1)])

    if result is None or result.decision is None:
        error = (result.error if result else "reasoner returned no result") or "unknown"
        base.status = "failed"
        base.attempts = attempts
        base.reason = f"链 B 调用失败（已重试 {attempts} 次）: {error}"
        base.latency_ms = result.latency_ms if result else None
        _log(ledger, decision_id or "shadow_unknown", "semantic_shadow_failed",
             base.to_dict())
        return base

    decision = result.decision
    chain_a_terms = extract_chain_a_avoid_terms(chain_a_constraints)
    negatives = [t for t in (list(decision.must_avoid) + list(decision.must_preserve)) if t]
    covered = [t for t in negatives if _is_covered(t, chain_a_terms)]
    uncovered = [t for t in negatives if not _is_covered(t, chain_a_terms)]

    report = ShadowReport(
        status="completed",
        pathway_status=pathway_status.value,
        model=result.model or getattr(adapter, "model", None),
        prompt_version=result.prompt_version,
        schema_version=result.schema_version,
        latency_ms=result.latency_ms,
        attempts=attempts,
        chain_b=decision.model_dump(mode="json"),
        covered_negatives=covered,
        uncovered_negatives=uncovered,
        status_divergence=_status_divergence(decision.status.value, chain_a_valid),
        positive_intents=list(decision.desired_relation_or_change),
    )
    _log(ledger, decision_id or f"shadow_{decision.decision_id}",
         "semantic_shadow_completed", report.to_dict())
    _write_sidecar(report, output_path)
    return report
