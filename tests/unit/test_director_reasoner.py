"""M2.2 Director Reasoner 单元测试。

验证：
- HeuristicDirectorReasoner.generate_plan 返回 (edl, plan) 元组且字段正确
- edl.timebase == 1_000_000，ordered_edits 非空，expected_duration 正确
- plan.decisions 数量 == edl.ordered_edits 数量，validation_status 有值
- LLMDirectorReasoner 可实例化但 generate_plan 抛 NotImplementedError
- get_director_reasoner 工厂映射正确，未知 strategy 抛 ValueError
"""
from __future__ import annotations

import json
import time

import pytest

from director_brain.brief_compiler import compile_brief
from director_brain.director_reasoner import (
    DirectorReasoner,
    get_director_reasoner,
    HeuristicDirectorReasoner,
    LLMDirectorReasoner,
)
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.story_graph_builder import build_story_graph


def _make_tech_obs(
    index: int,
    start_us: int,
    end_us: int,
    *,
    blur_score: float = 150.0,
    exposure_ok: bool = True,
) -> FilmObservation:
    claim = json.dumps(
        {
            "blur_score": blur_score,
            "brightness_mean": 120.0,
            "exposure_ok": exposure_ok,
            "shake_score": 5.0,
        }
    )
    return FilmObservation(
        observation_id=f"det_shot_{index:08d}",
        media_asset_id=f"shot_{index:08d}",
        media_hash=f"hash_{index}",
        start_frame=start_us,
        end_frame=end_us,
        timebase=1_000_000,
        observation_type="deterministic_technical",
        claim=claim,
        provider="deterministic_opencv",
        model_version="opencv_5.0",
        prompt_version="n/a",
        confidence=1.0,
        review_state="auto_verified",
        claim_kind=ClaimKind.MEASURED,
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="deterministic_opencv",
        source_ref="dummy.mp4",
    )


def _build_context():
    obs = [
        _make_tech_obs(0, 0, 3_000_000, blur_score=200.0),
        _make_tech_obs(1, 3_000_000, 6_000_000, blur_score=50.0),
        _make_tech_obs(2, 6_000_000, 10_000_000, blur_score=300.0),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    return brief, graph, obs


def test_generate_plan_returns_tuple_of_edl_and_plan():
    brief, graph, obs = _build_context()
    reasoner = HeuristicDirectorReasoner()
    edl, plan = reasoner.generate_plan(brief, graph, obs)
    assert isinstance(edl, EditorialDecisionList)
    assert isinstance(plan, DirectorDecisionPlan)


def test_edl_fields():
    brief, graph, obs = _build_context()
    reasoner = HeuristicDirectorReasoner()
    edl, _ = reasoner.generate_plan(brief, graph, obs)
    assert edl.timebase == 1_000_000
    assert len(edl.ordered_edits) > 0
    assert edl.brief_version == brief.version
    assert edl.context_id == graph.graph_id
    assert edl.approval_state == "draft"
    # expected_duration == 所有片段时长之和
    expected = sum(e.out_frame - e.in_frame for e in edl.ordered_edits)
    assert edl.expected_duration == expected
    # source_asset_hashes 去重且覆盖所有片段
    assert set(edl.source_asset_hashes) == {e.source_media_hash for e in edl.ordered_edits}


def test_plan_decisions_match_editems():
    brief, graph, obs = _build_context()
    reasoner = HeuristicDirectorReasoner()
    edl, plan = reasoner.generate_plan(brief, graph, obs)
    assert plan.validation_status
    assert len(plan.decisions) == len(edl.ordered_edits)
    assert plan.sequence == [e.source_asset_id for e in edl.ordered_edits]
    for dec in plan.decisions:
        assert dec.purpose == "select_shot"
        assert len(dec.shot_refs) == 1
        assert len(dec.evidence_refs) == 1


def test_llm_reasoner_raises_not_implemented():
    brief, graph, obs = _build_context()
    reasoner = LLMDirectorReasoner(provider="ollama")
    assert reasoner.provider == "ollama"
    with pytest.raises(NotImplementedError):
        reasoner.generate_plan(brief, graph, obs)


def test_factory_heuristic():
    reasoner = get_director_reasoner("heuristic")
    assert isinstance(reasoner, HeuristicDirectorReasoner)
    assert isinstance(reasoner, DirectorReasoner)


def test_factory_llm():
    reasoner = get_director_reasoner("llm", provider="ollama", config={"model": "x"})
    assert isinstance(reasoner, LLMDirectorReasoner)
    assert reasoner.provider == "ollama"


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        get_director_reasoner("unknown_strategy")


def test_four_act_selection_each_act_has_decision_and_act_order():
    """四幕 graph + mock 观测：每幕至少 1 个片段，按幕顺序排列。"""
    # 100s 源素材，每幕放一个镜头
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=200.0),       # hook
        _make_tech_obs(1, 20_000_000, 22_000_000, blur_score=180.0),  # develop
        _make_tech_obs(2, 60_000_000, 62_000_000, blur_score=300.0),  # peak
        _make_tech_obs(3, 90_000_000, 92_000_000, blur_score=150.0),  # resolve
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    reasoner = HeuristicDirectorReasoner()
    edl, plan = reasoner.generate_plan(brief, graph, obs)

    # 每幕至少 1 个 decision 带 act= 标记
    act_decisions = [d for d in plan.decisions if "act=" in (d.rationale or "")]
    assert len(act_decisions) >= 4, f"only {len(act_decisions)} act-labeled decisions"

    # 按幕顺序：hook 在前，然后 develop, peak, resolve
    act_order_seen = []
    for d in plan.decisions:
        rationale = d.rationale or ""
        for act in ("hook", "develop", "peak", "resolve"):
            if f"act={act}" in rationale and act not in act_order_seen:
                act_order_seen.append(act)
                break
    assert act_order_seen == ["hook", "develop", "peak", "resolve"], act_order_seen

    # shot_function 基于幕赋值
    funcs = {e.shot_function for e in edl.ordered_edits}
    assert "opening" in funcs
    assert "closing" in funcs
