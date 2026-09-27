"""阶段 7.5 链 B 影子语义决策单元测试。

覆盖：通路注册默认态、fail-closed 消费闸门、无 key/无模型钉扎的响亮跳过、
负向对账口径、状态对账双方向、传输重试、账本动作、sidecar 落盘。
"""
from __future__ import annotations

import json

import pytest

from director_brain import pathway_protocol as pp
from director_brain import semantic_shadow as ss
from director_brain.llm_adapter import LLMResult
from director_brain.models.director_decision import DecisionStatus, DirectorDecision


@pytest.fixture()
def restore_semantic_pathway():
    """每个用例后恢复 semantic_reasoner 默认态（SHADOW）。"""
    yield
    pp.set_pathway_status("semantic_reasoner", pp.PathwayStatus.SHADOW)


def _decision(**overrides) -> DirectorDecision:
    base = dict(
        decision_id="shadow_test",
        creative_intent="测试意图",
        target=["current_picture_cut"],
        desired_relation_or_change=["extend_visible_duration"],
        must_preserve=["picture_cut_position"],
        must_avoid=["transition", "模糊"],
        status=DecisionStatus.READY,
        confidence=0.9,
    )
    base.update(overrides)
    return DirectorDecision(**base)


class FakeAdapter:
    model = "fake-model"

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def generate_decision(self, user_input, context=None, decision_id=None):
        self.calls += 1
        return self._results.pop(0)


def _ok(result: DirectorDecision) -> LLMResult:
    return LLMResult(decision=result, model="fake-model",
                     prompt_version="test_prompt_v1", schema_version="pydantic_x",
                     latency_ms=42, schema_constrained=True)


def _fail(error: str) -> LLMResult:
    return LLMResult(decision=None, error=error, model="fake-model", latency_ms=5)


def _factory(adapter):
    """适配器注入工厂：契约与 build_ark_adapter 一致，返回 (adapter, reason)。"""
    return lambda: (adapter, None)


# ── 通路协议 ──

def test_semantic_reasoner_registered_as_shadow():
    """链 B 语义决策默认 SHADOW（现实角色即影子，同 asr_transcript 先例）。"""
    assert pp.get_pathway_status("semantic_reasoner") is pp.PathwayStatus.SHADOW


def test_shadow_output_cannot_drive_decisions():
    """消费闸门 fail-closed：SHADOW 状态的链 B 输出进入决策即抛错。"""
    with pytest.raises(pp.PathwayNotActiveError):
        pp.ensure_decision_use_allowed("semantic_reasoner")
    pp.set_pathway_status("semantic_reasoner", pp.PathwayStatus.ACTIVE)
    pp.ensure_decision_use_allowed("semantic_reasoner")  # ACTIVE 不抛


# ── 配置缺省的响亮跳过 ──

def test_no_api_key_skips_with_reason(monkeypatch):
    monkeypatch.setattr(ss, "_read_env", lambda key: None)
    adapter, reason = ss.build_ark_adapter()
    assert adapter is None
    assert "ARK_API_KEY" in reason
    report = ss.run_shadow_semantic("保留反应镜头，不要转场")
    assert report.status == "skipped"
    assert "ARK_API_KEY" in report.reason


def test_missing_model_pin_refuses_default(monkeypatch):
    """ARK_MODEL 未钉扎时拒绝构造——默认别名是被拒模型，静默回退=绕过准入。"""
    monkeypatch.setattr(
        ss, "_read_env", lambda key: "fake-key" if key == "ARK_API_KEY" else None)
    adapter, reason = ss.build_ark_adapter()
    assert adapter is None
    assert "ARK_MODEL" in reason


def test_empty_intent_skips():
    report = ss.run_shadow_semantic("   ")
    assert report.status == "skipped"
    assert "无用户意图文本" in report.reason


def test_experimental_pathway_does_not_compute(restore_semantic_pathway):
    """EXPERIMENTAL（止血态）下不计算影子信号（禁止半消费）。"""
    pp.set_pathway_status("semantic_reasoner", pp.PathwayStatus.EXPERIMENTAL)
    called = []
    report = ss.run_shadow_semantic(
        "测试", adapter_factory=lambda: (called.append(1), "unused"))
    assert report.status == "skipped"
    assert "EXPERIMENTAL" in report.reason
    assert called == []


# ── 对账口径 ──

_CHAIN_A = [
    "target_duration_us=15000000",
    "must_avoid:blur_score<10.0:模糊",
    "min_clip_us=400000",
]


def test_completed_report_full_accounting():
    adapter = FakeAdapter([_ok(_decision())])
    report = ss.run_shadow_semantic(
        "保留反应镜头，不要转场，画面别模糊",
        chain_a_constraints=_CHAIN_A,
        chain_a_valid=True,
        decision_id="shadow_plan_x",
        adapter_factory=_factory(adapter),
    )
    assert report.status == "completed"
    assert report.model == "fake-model"
    assert report.prompt_version == "test_prompt_v1"
    assert report.latency_ms == 42
    assert report.covered_negatives == ["模糊"]
    assert report.uncovered_negatives == ["transition", "picture_cut_position"]
    assert report.status_divergence is None  # 链 A 有效 + 链 B READY：一致
    assert report.positive_intents == ["extend_visible_duration"]
    assert report.chain_b["decision_id"] == "shadow_test"
    assert adapter.calls == 1


def test_status_divergence_both_directions():
    # 链 A 继续出片但链 B 说缺上下文
    adapter = FakeAdapter([_ok(_decision(status=DecisionStatus.NEEDS_CONTEXT))])
    report = ss.run_shadow_semantic(
        "停顿再完整一点", chain_a_constraints=_CHAIN_A, chain_a_valid=True,
        adapter_factory=_factory(adapter))
    assert report.status_divergence == "chain_a_proceeded_but_chain_b_status=NEEDS_CONTEXT"

    # 链 A 拒绝但链 B 认为可执行
    adapter = FakeAdapter([_ok(_decision())])
    report = ss.run_shadow_semantic(
        "测试", chain_a_constraints=_CHAIN_A, chain_a_valid=False,
        adapter_factory=_factory(adapter))
    assert report.status_divergence == "chain_a_rejected_but_chain_b_ready"


def test_empty_negatives_no_crash():
    adapter = FakeAdapter([_ok(_decision(must_avoid=[], must_preserve=[]))])
    report = ss.run_shadow_semantic(
        "测试", chain_a_constraints=_CHAIN_A, adapter_factory=_factory(adapter))
    assert report.status == "completed"
    assert report.covered_negatives == []
    assert report.uncovered_negatives == []


# ── 传输重试与 fail-soft ──

def test_adapter_failure_is_fail_soft():
    adapter = FakeAdapter([
        _fail("SEMANTIC_REASONER_UNAVAILABLE: boom"),
        _fail("SEMANTIC_REASONER_UNAVAILABLE: boom"),
    ])
    report = ss.run_shadow_semantic(
        "测试", adapter_factory=_factory(adapter),
        max_attempts=2, backoff_s=(0.0, 0.0))
    assert report.status == "failed"
    assert "boom" in report.reason
    assert report.attempts == 2


def test_adapter_unexpected_exception_never_blocks_chain():
    """适配器抛意外异常时按传输失败处理——影子故障绝不阻断主链。"""

    class ExplodingAdapter:
        model = "fake-model"

        def generate_decision(self, user_input, context=None, decision_id=None):
            raise RuntimeError("unexpected internal bug")

    report = ss.run_shadow_semantic(
        "测试", adapter_factory=_factory(ExplodingAdapter()),
        max_attempts=2, backoff_s=(0.0, 0.0))
    assert report.status == "failed"
    assert "unexpected internal bug" in report.reason
    assert report.attempts == 2


def test_transport_retry_then_success():
    adapter = FakeAdapter([
        _fail("SEMANTIC_REASONER_UNAVAILABLE: transient"),
        _fail("SEMANTIC_REASONER_UNAVAILABLE: transient"),
        _ok(_decision()),
    ])
    report = ss.run_shadow_semantic(
        "测试", adapter_factory=_factory(adapter),
        max_attempts=3, backoff_s=(0.0, 0.0))
    assert adapter.calls == 3
    assert report.status == "completed"


# ── 账本与 sidecar ──

def test_ledger_actions_logged(monkeypatch):
    logged = []

    def fake_log(ledger, decision_id, action, detail):
        logged.append((decision_id, action, detail))

    monkeypatch.setattr(ss, "log_decision", fake_log)
    adapter = FakeAdapter([_ok(_decision())])
    ss.run_shadow_semantic(
        "测试", chain_a_constraints=_CHAIN_A, decision_id="shadow_plan_x",
        ledger=object(), adapter_factory=_factory(adapter))
    assert logged[-1][1] == "semantic_shadow_completed"
    assert logged[-1][0] == "shadow_plan_x"
    assert logged[-1][2]["pathway"] == "semantic_reasoner"
    assert logged[-1][2]["pathway_status"] == "SHADOW"
    assert "model" in logged[-1][2]

    # 失败路径同样留痕
    adapter = FakeAdapter([_fail("SEMANTIC_REASONER_UNAVAILABLE: x")])
    ss.run_shadow_semantic(
        "测试", ledger=object(), adapter_factory=_factory(adapter),
        max_attempts=1, backoff_s=(0.0,))
    assert logged[-1][1] == "semantic_shadow_failed"


def test_sidecar_written_next_to_output(tmp_path):
    adapter = FakeAdapter([_ok(_decision())])
    out = str(tmp_path / "out.mp4")
    ss.run_shadow_semantic(
        "测试", chain_a_constraints=_CHAIN_A, output_path=out,
        adapter_factory=_factory(adapter))
    sidecar = tmp_path / "out.mp4.shadow.json"
    assert sidecar.is_file()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["pathway"] == "semantic_reasoner"
    assert data["status"] == "completed"
    assert data["uncovered_negatives"] == ["transition", "picture_cut_position"]


# ── 链 A 约束解析 ──

def test_extract_chain_a_avoid_terms():
    terms = ss.extract_chain_a_avoid_terms(_CHAIN_A)
    assert terms == ["模糊"]
    assert ss.extract_chain_a_avoid_terms(None) == []
    # 宽泛解析：畸形条目把剩余部分整体当术语（对覆盖率记账安全——
    # 只会多报覆盖候选，不会漏报未覆盖项）
    assert ss.extract_chain_a_avoid_terms(["must_avoid:broken"]) == ["broken"]
