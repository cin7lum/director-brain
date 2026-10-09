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
    EvidenceTooPoorError,
    derive_candidate_edl_source_asset_alignment,
    _build_candidates,
    get_director_reasoner,
    HeuristicDirectorReasoner,
    LLMDirectorReasoner,
)
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.director_plan import ProjectNarrativeEvidenceRef
from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_observation import (
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)
from director_brain.models.story_graph import StoryGraph, StoryNode, StoryNodeType
from director_brain.pathway_protocol import (
    PathwayNotActiveError,
    PathwayStatus,
    set_pathway_status,
)
from director_brain.story_graph_builder import build_story_graph


@pytest.mark.parametrize(
    ("source_indexes", "selected_indexes", "expected"),
    [
        ([], [], "no_cited_sources"),
        ([0], [0], "all_cited_assets_selected"),
        ([0, 1], [0], "some_cited_assets_selected"),
        ([0], [1], "no_cited_assets_selected"),
    ],
)
def test_candidate_edl_source_asset_alignment_is_mechanical_and_bounded(
    source_indexes, selected_indexes, expected,
):
    refs = [
        ProjectNarrativeEvidenceRef(
            project_asset_id=f"project-asset-{index}",
            source_media_hash=(str(index + 1) * 64),
            source_asset_id=f"source-{index}",
            observation_id=f"observation-{index}",
        )
        for index in range(2)
    ]
    selected = {
        (
            refs[index].project_asset_id,
            refs[index].source_media_hash.lower(),
            refs[index].source_asset_id,
        )
        for index in selected_indexes
    }

    assert derive_candidate_edl_source_asset_alignment(
        [refs[index] for index in source_indexes], selected,
    ) == expected


@pytest.fixture()
def vlm_pathway_active():
    """把 vlm_semantic 通路临时置为 ACTIVE（2026-09-30 灰度转正后即默认态；闸门本体由显式设 EXPERIMENTAL 的用例验证）。"""
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    yield
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)  # 恢复默认（灰度转正后）


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


def _make_vlm_obs(index, start_us, end_us, *, role="hero", shot_function="ACTION", motion="subtle"):
    claim = json.dumps({"shot_function": shot_function, "proposed_role_v2": role, "motion_amount": motion, "frame_description": "test"})
    return FilmObservation(
        observation_id=f"vlm_shot_{index:08d}",
        media_asset_id=f"shot_{index:08d}",
        media_hash=f"hash_{index}",
        start_frame=start_us, end_frame=end_us, timebase=1_000_000,
        observation_type="vlm_semantic", claim=claim,
        provider="ollama_qwen3_vl", model_version="qwen3-vl:latest",
        prompt_version="vlm_prompt_v1", confidence=0.7,
        review_state="auto_generated", claim_kind=ClaimKind.MODEL_OBSERVATION,
        schema_version="1.0", project_id="test_proj",
        created_at=int(time.time()), producer="ollama_qwen3_vl", source_ref="dummy.mp4",
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


@pytest.mark.parametrize(
    ("audio_style", "offset_field"),
    [("j_cut", "audio_lead_us"), ("l_cut", "audio_tail_us")],
)
def test_audio_bridge_offsets_only_cover_real_adjacent_boundaries(
    audio_style, offset_field,
):
    brief, graph, obs = _build_context()
    narrative = {
        "suggested_order_resolved": [
            "shot_00000002", "shot_00000001", "shot_00000000",
        ],
        "act_boundaries_resolved": [],
    }

    edl, plan = HeuristicDirectorReasoner().generate_plan(
        brief, graph, obs, narrative=narrative,
        narrative_order_priority=True, audio_style=audio_style,
    )

    edits = edl.ordered_edits
    assert len(edits) >= 2
    assert edits[0].source_asset_id == "shot_00000002"
    assert all(getattr(edit, offset_field) == 0 for edit in edits)
    assert f"audio_bridge_style={audio_style}" in plan.constraints
    assert "audio_bridge_fallback=fixed_offset_disabled" in plan.constraints
    assert any(
        item.startswith(f"audio_bridge_no_supported_boundary={audio_style}:")
        for item in plan.open_questions
    )


@pytest.mark.parametrize(
    ("audio_style", "offset_field", "target_index"),
    [("j_cut", "audio_lead_us", 1), ("l_cut", "audio_tail_us", 0)],
)
def test_audio_bridge_uses_exact_crossing_asr_span_and_binds_evidence(
    audio_style, offset_field, target_index,
):
    brief, graph, observations = _build_context()
    narrative = {
        "suggested_order_resolved": [
            "shot_00000002", "shot_00000001", "shot_00000000",
        ],
        "act_boundaries_resolved": [],
    }
    reasoner = HeuristicDirectorReasoner()
    baseline_edl, _ = reasoner.generate_plan(
        brief, graph, observations, narrative=narrative,
        narrative_order_priority=True,
    )
    target = baseline_edl.ordered_edits[target_index]
    boundary = target.in_frame if audio_style == "j_cut" else target.out_frame
    source = next(
        item for item in observations
        if item.observation_type == "deterministic_technical"
        and item.media_asset_id == target.source_asset_id
        and item.media_hash == target.source_media_hash
    )
    speech = source.model_copy(update={
        "observation_id": f"asr_bridge_{audio_style}",
        "source_observation_id": source.observation_id,
        "start_frame": boundary - 200_000,
        "end_frame": boundary + 350_000,
        "observation_type": "speech_transcript",
        "claim": "synthetic fixture content",
        "provider": "local_asr_fixture",
        "model_version": "fixture-1",
        "prompt_version": "fixture-1",
        "claim_kind": ClaimKind.MODEL_OBSERVATION,
        "timebase_unit": TimebaseUnit.MICROSECONDS,
    })

    edl, plan = reasoner.generate_plan(
        brief, graph, [*observations, speech], narrative=narrative,
        narrative_order_priority=True, audio_style=audio_style,
    )

    selected = next(
        item for item in edl.ordered_edits
        if item.source_asset_id == target.source_asset_id
    )
    expected_offset = 200_000 if audio_style == "j_cut" else 350_000
    assert getattr(selected, offset_field) == expected_offset
    assert selected.audio_evidence_refs == [speech.observation_id]
    assert edl.audio_evidence_refs == [speech.observation_id]
    assert edl.audio_refs == []
    decision = next(
        item for item in plan.decisions
        if item.shot_refs == [selected.source_asset_id]
    )
    assert speech.observation_id in decision.evidence_refs
    assert decision.requires_approval is True
    assert "audio_bridge_asr_timing_unverified" in plan.open_questions
    assert f"audio_bridge_style={audio_style}" in plan.constraints


@pytest.mark.parametrize("audio_style", ["j_cut", "l_cut"])
@pytest.mark.parametrize("mismatch", ["hash", "boundary"])
def test_audio_bridge_rejects_unbound_or_non_crossing_asr(
    audio_style, mismatch,
):
    brief, graph, observations = _build_context()
    narrative = {
        "suggested_order_resolved": [
            "shot_00000002", "shot_00000001", "shot_00000000",
        ],
        "act_boundaries_resolved": [],
    }
    reasoner = HeuristicDirectorReasoner()
    baseline_edl, _ = reasoner.generate_plan(
        brief, graph, observations, narrative=narrative,
        narrative_order_priority=True,
    )
    target_index = 1 if audio_style == "j_cut" else 0
    target = baseline_edl.ordered_edits[target_index]
    boundary = target.in_frame if audio_style == "j_cut" else target.out_frame
    source = next(
        item for item in observations
        if item.observation_type == "deterministic_technical"
        and item.media_asset_id == target.source_asset_id
        and item.media_hash == target.source_media_hash
    )
    speech = source.model_copy(update={
        "observation_id": f"asr_mismatch_{audio_style}_{mismatch}",
        "source_observation_id": source.observation_id,
        "start_frame": boundary if mismatch == "boundary" else boundary - 200_000,
        "end_frame": boundary + 350_000,
        "observation_type": "speech_transcript",
        "claim": "synthetic fixture content",
        "provider": "local_asr_fixture",
        "model_version": "fixture-1",
        "prompt_version": "fixture-1",
        "claim_kind": ClaimKind.MODEL_OBSERVATION,
        "timebase_unit": TimebaseUnit.MICROSECONDS,
        **({"media_hash": "wrong-source-hash"} if mismatch == "hash" else {}),
    })

    edl, plan = reasoner.generate_plan(
        brief, graph, [*observations, speech], narrative=narrative,
        narrative_order_priority=True, audio_style=audio_style,
    )

    selected = next(
        item for item in edl.ordered_edits
        if item.source_asset_id == target.source_asset_id
    )
    offset_field = "audio_lead_us" if audio_style == "j_cut" else "audio_tail_us"
    assert getattr(selected, offset_field) == 0
    assert selected.audio_evidence_refs == []
    assert edl.audio_evidence_refs == []
    assert any(
        item.startswith(f"audio_bridge_no_supported_boundary={audio_style}:")
        for item in plan.open_questions
    )


def test_audio_bridge_respects_asr_decision_use_gate():
    brief, graph, observations = _build_context()
    narrative = {
        "suggested_order_resolved": [
            "shot_00000002", "shot_00000001", "shot_00000000",
        ],
        "act_boundaries_resolved": [],
    }
    reasoner = HeuristicDirectorReasoner()
    base_edl, _ = reasoner.generate_plan(
        brief, graph, observations, narrative=narrative,
        narrative_order_priority=True,
    )
    target = base_edl.ordered_edits[1]
    source = next(
        item for item in observations
        if item.observation_type == "deterministic_technical"
        and item.media_asset_id == target.source_asset_id
        and item.media_hash == target.source_media_hash
    )
    speech = source.model_copy(update={
        "observation_id": "asr_bridge_gated",
        "source_observation_id": source.observation_id,
        "start_frame": target.in_frame - 100_000,
        "end_frame": target.in_frame + 100_000,
        "observation_type": "speech_transcript",
        "claim": "synthetic fixture content",
        "provider": "local_asr_fixture",
        "model_version": "fixture-1",
        "prompt_version": "fixture-1",
        "claim_kind": ClaimKind.MODEL_OBSERVATION,
        "timebase_unit": TimebaseUnit.MICROSECONDS,
    })

    set_pathway_status("asr_transcript", PathwayStatus.SHADOW)
    try:
        with pytest.raises(PathwayNotActiveError, match="asr_transcript"):
            reasoner.generate_plan(
                brief, graph, [*observations, speech], narrative=narrative,
                narrative_order_priority=True, audio_style="j_cut",
            )
    finally:
        set_pathway_status("asr_transcript", PathwayStatus.ACTIVE)


@pytest.mark.parametrize("audio_style", ["j_cut", "l_cut"])
def test_audio_bridge_does_not_offset_a_single_edit(audio_style):
    observations = [_make_tech_obs(0, 0, 3_000_000)]
    brief = compile_brief(
        "test_proj", "dummy.mp4", observations,
        target_duration_us=3_000_000,
    )
    graph = build_story_graph(brief, observations)

    edl, _plan = HeuristicDirectorReasoner().generate_plan(
        brief, graph, observations, audio_style=audio_style,
    )

    assert len(edl.ordered_edits) == 1
    assert edl.ordered_edits[0].audio_lead_us == 0
    assert edl.ordered_edits[0].audio_tail_us == 0


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


def test_project_bound_reasoner_rejects_observations_from_another_asset():
    observations = [
        _make_tech_obs(0, 0, 3_000_000).model_copy(update={
            "project_asset_id": "asset-a",
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        }),
        _make_tech_obs(1, 3_000_000, 6_000_000).model_copy(update={
            "project_asset_id": "asset-b",
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        }),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", observations[:1])
    graph = build_story_graph(brief, observations[:1])

    with pytest.raises(ValueError, match="match the StoryGraph project asset"):
        HeuristicDirectorReasoner().generate_plan(brief, graph, observations)


def test_project_bound_reasoner_rejects_non_microsecond_source_timebase():
    observations = [
        _make_tech_obs(0, 0, 25).model_copy(update={
            "project_asset_id": "asset-a",
            "timebase": 25,
            "timebase_unit": TimebaseUnit.FRAMES,
        }),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", observations)
    graph = build_story_graph(brief, observations)

    with pytest.raises(ValueError, match="microsecond source timebase"):
        HeuristicDirectorReasoner().generate_plan(brief, graph, observations)


def test_project_bound_edits_and_decisions_keep_exact_asset_and_evidence_refs():
    observations = [
        _make_tech_obs(index, start, end).model_copy(update={
            "project_asset_id": "asset-a",
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        })
        for index, (start, end) in enumerate((
            (0, 3_000_000),
            (3_000_000, 6_000_000),
            (6_000_000, 10_000_000),
        ))
    ]
    brief = compile_brief("test_proj", "dummy.mp4", observations)
    graph = build_story_graph(brief, observations)

    edl, plan = HeuristicDirectorReasoner().generate_plan(
        brief, graph, observations)
    from director_brain.plan_validator import validate_plan

    _is_valid, errors = validate_plan(edl, plan, observations)
    assert not any("source observation" in error for error in errors)
    assert not any("same project asset and evidence ref" in error for error in errors)

    assert edl.timebase_unit == TimebaseUnit.MICROSECONDS
    assert edl.ordered_edits
    for edit, decision in zip(edl.ordered_edits, plan.decisions):
        source = next(
            observation for observation in observations
            if observation.media_asset_id == edit.source_asset_id
        )
        assert edit.timebase_unit == TimebaseUnit.MICROSECONDS
        assert edit.project_asset_id == "asset-a"
        assert edit.source_observation_refs == [source.observation_id]
        assert edit.source_timebase == 1_000_000
        assert edit.source_timebase_unit == TimebaseUnit.MICROSECONDS
        assert (
            edit.source_observation_start <= edit.in_frame < edit.out_frame
            <= edit.source_observation_end
        )
        assert decision.project_asset_id == "asset-a"
        assert decision.evidence_refs == [source.observation_id]

    invalid_edl = edl.model_copy(deep=True)
    invalid_edit = invalid_edl.ordered_edits[0]
    invalid_edit.out_frame = invalid_edit.source_observation_end + 1
    is_valid, errors = validate_plan(invalid_edl, plan, observations)
    assert not is_valid
    assert any("outside its cited source observation" in error for error in errors)


def test_project_bound_reasoner_rejects_noncanonical_microsecond_timebase():
    observations = [
        _make_tech_obs(0, 0, 3_000_000).model_copy(update={
            "project_asset_id": "asset-a",
            "timebase": 2_000_000,
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        }),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", observations)
    graph = build_story_graph(brief, observations)

    with pytest.raises(ValueError, match="canonical microsecond timebase"):
        HeuristicDirectorReasoner().generate_plan(brief, graph, observations)


def test_llm_reasoner_is_gated_while_semantic_pathway_is_shadow(monkeypatch):
    brief, graph, obs = _build_context()
    reasoner = LLMDirectorReasoner(provider="ollama")
    assert reasoner.provider == "ollama"
    set_pathway_status("director_strategy_reasoning", PathwayStatus.SHADOW)
    monkeypatch.setattr(reasoner, "_analyze", lambda _observations: pytest.fail(
        "provider must not be called while the pathway is SHADOW"))

    with pytest.raises(PathwayNotActiveError, match="director_strategy_reasoning"):
        reasoner.generate_plan(brief, graph, obs)


def test_llm_reasoner_passes_creator_direction_with_brief(monkeypatch):
    from director_brain import narrative_analyzer

    direction = "Open with the quiet arrival, then build toward the reunion."
    source_hash = "a" * 64
    observations = []
    for index in range(2):
        tech = _make_tech_obs(index, index * 3_000_000,
                              (index + 1) * 3_000_000).model_copy(update={
            "media_hash": source_hash,
            "project_asset_id": "asset-a",
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        })
        semantic = _make_vlm_obs(index, tech.start_frame,
                                 tech.end_frame).model_copy(update={
            "media_hash": source_hash,
            "project_asset_id": "asset-a",
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        })
        observations.extend((tech, semantic))
    brief = compile_brief(
        "test_proj", "dummy.mp4", observations, intent_text=direction)
    captured = {}

    def fake_analysis(semantics, **kwargs):
        captured.update(kwargs)
        return {"result": "test"}

    monkeypatch.setattr(narrative_analyzer, "analyze_narrative", fake_analysis)
    result = LLMDirectorReasoner(provider="ollama")._analyze(
        brief, observations)

    assert result == {"result": "test"}
    assert json.loads(captured["director_brief"])["creator_direction"] == direction


def test_llm_reasoner_shadow_plan_is_non_confirmable(monkeypatch):
    from director_brain.plan_state import confirm_strategy

    source_hash = "a" * 64
    observations = [
        _make_tech_obs(index, 0, 4_000_000)
        for index in range(3)
    ]
    observations = [item.model_copy(update={
        "media_hash": source_hash,
        "timebase_unit": TimebaseUnit.MICROSECONDS,
    }) for item in observations]
    observations.extend([
        _make_vlm_obs(index, item.start_frame, item.end_frame,
                      role="development").model_copy(update={
            "media_hash": source_hash,
            "timebase_unit": TimebaseUnit.MICROSECONDS,
        })
        for index, item in enumerate(observations)
    ])
    brief = compile_brief("test_proj", "dummy.mp4", observations)
    graph = build_story_graph(brief, observations)
    narrative = {
        "story_arc": "A small moment grows into a shared celebration.",
        "emotional_trajectory": ["quiet", "warm", "joyful"],
        "pairings": [],
        "key_moments": [],
        "act_boundaries": [],
        "suggested_order": [2, 1, 0],
        "act_boundaries_resolved": [],
        "suggested_order_resolved": [
            "shot_00000002", "shot_00000001", "shot_00000000",
        ],
        "limitations": ["Synthetic observations; no source video was assessed."],
    }
    reasoner = LLMDirectorReasoner(provider="ollama")
    monkeypatch.setattr(reasoner, "_analyze", lambda _brief, _observations: narrative)
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    set_pathway_status("director_strategy_reasoning", PathwayStatus.SHADOW)

    edl, plan = reasoner.generate_shadow_plan(brief, graph, observations)

    assert plan.sequence == [item.source_asset_id for item in edl.ordered_edits]
    assert any(
        item.startswith("narrative_reorder:applied:planner_sort_changed=false:")
        for item in plan.open_questions
    )
    assert "director_reasoner_shadow_candidate=not_confirmable" in plan.constraints
    assert "director_reasoner_prompt_version=2.3" in plan.constraints
    assert "director_reasoner_temperature=0" in plan.constraints
    assert "director_reasoner_quality_not_proven" in plan.open_questions
    assert edl.producer == plan.producer == "llm_director_reasoner_shadow_v0.1"
    with pytest.raises(ValueError, match="shadow-only"):
        confirm_strategy(plan, edl)


def test_llm_reasoner_rejects_remote_endpoints_by_default():
    from director_brain.llm_adapter import LLMTransportError

    with pytest.raises(LLMTransportError, match="loopback") as rejected:
        LLMDirectorReasoner(config={"base_url": "https://example.com/v1"})
    assert rejected.value.failure_code == "provider_configuration_error"


def test_project_reasoner_uses_existing_configured_ollama_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
    monkeypatch.setenv("PROJECT_LOCAL_REASONER_MODEL_DIGEST", "")
    monkeypatch.setenv("PROJECT_LOCAL_REASONER_RUNTIME_VERSION", "")

    reasoner = LLMDirectorReasoner()

    assert reasoner.base_url == "http://127.0.0.1:11435/v1"
    assert reasoner._runtime_binding is None

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11436/v1/")
    normalized = LLMDirectorReasoner()
    assert normalized.base_url == "http://127.0.0.1:11436/v1"

    explicit = LLMDirectorReasoner(config={
        "base_url": "http://127.0.0.1:11437/v1",
    })
    assert explicit.base_url == "http://127.0.0.1:11437/v1"


def test_project_reasoner_reports_missing_loopback_endpoint_as_provider_config(
    monkeypatch,
):
    from director_brain.llm_adapter import LLMTransportError

    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    with pytest.raises(LLMTransportError) as missing_endpoint:
        LLMDirectorReasoner()
    assert missing_endpoint.value.failure_code == "provider_configuration_error"


@pytest.mark.parametrize(
    ("provider", "config"),
    [
        ("openai", {}),
        ("ollama", {"timeout": "not-an-integer"}),
        ("ollama", {"temperature": 2.1}),
        ("ollama", {"model_digest": "a" * 64}),
        ("ollama", {
            "model_digest": "not-a-digest",
            "runtime_version": "0.32.14",
        }),
        ("ollama", {
            "model_digest": "a" * 64,
            "runtime_version": "invalid version",
        }),
        ("ollama", {
            "base_url": "http://127.0.0.1:11434/custom",
            "model_digest": "a" * 64,
            "runtime_version": "0.32.14",
        }),
    ],
)
def test_project_reasoner_classifies_invalid_provider_options(
    monkeypatch, provider, config,
):
    from director_brain.llm_adapter import LLMTransportError

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("PROJECT_LOCAL_REASONER_MODEL_DIGEST", "")
    monkeypatch.setenv("PROJECT_LOCAL_REASONER_RUNTIME_VERSION", "")
    with pytest.raises(LLMTransportError) as invalid_configuration:
        LLMDirectorReasoner(provider=provider, config=config)
    assert invalid_configuration.value.failure_code == "provider_configuration_error"


def test_project_reasoner_requires_complete_runtime_pin_for_formal_use(monkeypatch):
    from director_brain.llm_adapter import LLMTransportError
    from observation_service.ollama_vlm_adapter import (
        LocalVLMRuntimeBindingError,
        OllamaVLMAdapter,
    )

    # Explicit empty inherited values keep this test independent of a local .env.
    monkeypatch.setenv("PROJECT_LOCAL_REASONER_MODEL_DIGEST", "")
    monkeypatch.setenv("PROJECT_LOCAL_REASONER_RUNTIME_VERSION", "")
    unpinned = LLMDirectorReasoner()
    with pytest.raises(LLMTransportError) as missing_pin:
        unpinned._verify_model_binding(required=True)
    assert missing_pin.value.failure_code == "provider_configuration_error"

    with pytest.raises(LLMTransportError, match="must be configured together") as (
        invalid_pin
    ):
        LLMDirectorReasoner(config={"model_digest": "a" * 64})
    assert invalid_pin.value.failure_code == "provider_configuration_error"

    monkeypatch.setenv("PROJECT_LOCAL_REASONER_MODEL_DIGEST", "b" * 64)
    monkeypatch.setenv("PROJECT_LOCAL_REASONER_RUNTIME_VERSION", "0.32.14")
    verified = []

    def verify(adapter):
        verified.append((adapter.model, adapter.model_digest, adapter.runtime_version))

    monkeypatch.setattr(OllamaVLMAdapter, "verify_runtime_binding", verify)
    pinned = LLMDirectorReasoner()
    binding = pinned._verify_model_binding(required=True)
    assert binding == {"model_digest": "b" * 64, "runtime_version": "0.32.14"}
    assert verified == [("qwen2.5:7b", "b" * 64, "0.32.14")]

    def reject_binding(_adapter):
        raise LocalVLMRuntimeBindingError("provider detail must stay private")

    monkeypatch.setattr(OllamaVLMAdapter, "verify_runtime_binding", reject_binding)
    with pytest.raises(LLMTransportError) as mismatch:
        pinned._verify_model_binding(required=True)
    assert mismatch.value.failure_code == "provider_model_binding_error"
    assert "provider detail" not in str(mismatch.value)


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


@pytest.mark.usefixtures("vlm_pathway_active")
def test_candidates_include_vlm_fields():
    """直接调用 _build_candidates，验证 VLM 语义字段写入 candidate。"""
    tech_obs = [
        _make_tech_obs(0, 0, 3_000_000, blur_score=200.0),
        _make_tech_obs(1, 3_000_000, 6_000_000, blur_score=50.0),
    ]
    vlm_obs = [
        _make_vlm_obs(0, 0, 3_000_000, role="hero", shot_function="ACTION", motion="subtle"),
        _make_vlm_obs(1, 3_000_000, 6_000_000, role="broll",
                      shot_function="ATMOSPHERIC_EVIDENCE", motion="static"),
    ]
    candidates = _build_candidates(tech_obs, vlm_obs)
    by_id = {c["source_shot_id"]: c for c in candidates}

    c0 = by_id["shot_00000000"]
    assert c0["vlm_shot_function"] == "ACTION"
    assert c0["vlm_role"] == "hero"
    assert c0["vlm_motion"] == "subtle"

    c1 = by_id["shot_00000001"]
    assert c1["vlm_shot_function"] == "ATMOSPHERIC_EVIDENCE"
    assert c1["vlm_role"] == "broll"
    assert c1["vlm_motion"] == "static"


def test_no_vlm_observations_fields_none():
    """不传 vlm_obs 或传空列表时，VLM 字段全为 None（向后兼容）。"""
    tech_obs = [_make_tech_obs(0, 0, 3_000_000, blur_score=200.0)]

    for cands in (_build_candidates(tech_obs), _build_candidates(tech_obs, [])):
        c = cands[0]
        assert c["vlm_shot_function"] is None
        assert c["vlm_role"] is None
        assert c["vlm_motion"] is None


def _vlm_competing_context():
    """hook 幕内：discard(blur=300) 与 hero(blur=200) 竞争；尾部放一个 filler 镜头撑起总时长。"""
    tech_obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=300.0),            # discard
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=200.0),     # hero
        _make_tech_obs(2, 90_000_000, 92_000_000, blur_score=150.0),   # filler(resolve)
    ]
    vlm_obs = [
        _make_vlm_obs(0, 0, 2_000_000, role="discard",
                      shot_function="ACTION", motion="subtle"),
        _make_vlm_obs(1, 2_000_000, 4_000_000, role="hero",
                      shot_function="ACTION", motion="subtle"),
        _make_vlm_obs(2, 90_000_000, 92_000_000, role="support",
                      shot_function="opening", motion="subtle"),
    ]
    observations = tech_obs + vlm_obs
    brief = compile_brief("test_proj", "dummy.mp4", observations)
    graph = build_story_graph(brief, observations)
    return brief, graph, observations


@pytest.mark.usefixtures("vlm_pathway_active")
def test_discard_role_gets_downgraded_in_edl():
    """blur 更高的 discard 镜头因 VLM 降权(x0.2)而落选，hero(x1.1) 被选中。"""
    brief, graph, observations = _vlm_competing_context()
    reasoner = HeuristicDirectorReasoner()
    edl, _ = reasoner.generate_plan(brief, graph, observations)

    selected_ids = [e.source_asset_id for e in edl.ordered_edits]
    # discard（原始 blur 更高）被降权后不应入选；hero 应入选
    assert "shot_00000000" not in selected_ids, selected_ids
    assert "shot_00000001" in selected_ids, selected_ids


@pytest.mark.usefixtures("vlm_pathway_active")
def test_vlm_rationale_propagates_to_decision():
    """VLM 权重命中时，decision.rationale 含 'vlm:' 字样。"""
    brief, graph, observations = _vlm_competing_context()
    reasoner = HeuristicDirectorReasoner()
    _, plan = reasoner.generate_plan(brief, graph, observations)

    vlm_rationales = [
        d.rationale for d in plan.decisions if "vlm:" in (d.rationale or "")
    ]
    assert vlm_rationales, "expected at least one decision rationale to contain vlm:"


def test_no_shot_selected_twice():
    """T1 导演层防线：一份 plan 中同一镜头只允许出现一次。

    历史缺陷：空幕借片不排除已选镜头 → 同一镜头被选入两幕 → EDL 出现
    同 asset 嵌套区间 → validator 判 overlap → 旧 repair 越权删镜头兜底。
    """
    # 素材时间轴上只有 2 个镜头，但有 4 幂 → 必然触发空幕借片
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=200.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=180.0),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    reasoner = HeuristicDirectorReasoner()
    edl, plan = reasoner.generate_plan(brief, graph, obs)

    edl_ids = [e.source_asset_id for e in edl.ordered_edits]
    assert len(edl_ids) == len(set(edl_ids)), (
        f"同一镜头被选中多次: {edl_ids}"
    )
    assert plan.sequence == edl_ids
    # 每个镜头的时间区间互不嵌套/重叠（同源嵌套是历史 overlap 的根源）
    sorted_by_in = sorted(edl.ordered_edits, key=lambda e: e.in_frame)
    for k in range(len(sorted_by_in) - 1):
        assert sorted_by_in[k].out_frame <= sorted_by_in[k + 1].in_frame, (
            f"镜头 {sorted_by_in[k].source_asset_id} 与 "
            f"{sorted_by_in[k + 1].source_asset_id} 区间重叠"
        )


# ---------------------------------------------------------------------------
# T2：静默降级显式化
# ---------------------------------------------------------------------------

def test_clean_input_not_degraded():
    """证据充足的正常输入：degraded=False、无降级事件（防"永远标降级"假通过）。"""
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=200.0),
        _make_tech_obs(1, 20_000_000, 22_000_000, blur_score=180.0),
        _make_tech_obs(2, 60_000_000, 62_000_000, blur_score=300.0),
        _make_tech_obs(3, 90_000_000, 92_000_000, blur_score=150.0),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    edl, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)
    assert plan.degraded is False
    assert plan.degradation_events == []


def test_global_relaxation_recorded():
    """blur 全部低于阈值但曝光合格 → level=1 放宽（仅曝光）→ 必须留痕。"""
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=5.0),
        _make_tech_obs(1, 20_000_000, 22_000_000, blur_score=5.0),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    edl, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)

    assert plan.degraded is True
    assert any(
        e.startswith("relax_technical_usable:scope=global:level=1")
        for e in plan.degradation_events
    ), plan.degradation_events
    # 放宽后仍应产出可用选片
    assert len(edl.ordered_edits) >= 1


def test_evidence_too_poor_raises():
    """候选 >=2 却没有 2 个曝光合格 → 抛 EvidenceTooPoorError（fail-closed）。

    历史行为是把全部镜头强制 usable 静默注水——已废除。
    """
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=200.0, exposure_ok=False),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=200.0, exposure_ok=False),
        _make_tech_obs(2, 4_000_000, 6_000_000, blur_score=200.0, exposure_ok=False),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    with pytest.raises(EvidenceTooPoorError):
        HeuristicDirectorReasoner().generate_plan(brief, graph, obs)


def test_single_unusable_shot_raises():
    """唯一的镜头也不曝光合格 → 零证据锚点 → 抛 EvidenceTooPoorError。"""
    obs = [_make_tech_obs(0, 0, 2_000_000, blur_score=200.0, exposure_ok=False)]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    with pytest.raises(EvidenceTooPoorError):
        HeuristicDirectorReasoner().generate_plan(brief, graph, obs)


def test_level2_force_all_recorded_when_one_anchor():
    """有 1 个曝光锚点但原始可用不足 2 个 → level=2 强制可用必须留痕。

    构造：shot0 曝光合格但 blur 低（锚点，blur 判据不过）、shot1 曝光不合格。
    原始可用 0 个 → level1（仅曝光）剩 1 个 → level2 强制全部 → 留痕。
    """
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=5.0, exposure_ok=True),
        _make_tech_obs(1, 20_000_000, 22_000_000, blur_score=200.0, exposure_ok=False),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    edl, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)
    assert plan.degraded is True
    assert any(
        e.startswith("relax_technical_usable:scope=global:level=2")
        for e in plan.degradation_events
    ), plan.degradation_events
    assert len(edl.ordered_edits) >= 1


def test_degraded_selection_forces_approval_and_low_confidence():
    """放宽入选的镜头：confidence 按原始可用率计算 → 低分且 requires_approval。"""
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=5.0, exposure_ok=True),
        _make_tech_obs(1, 20_000_000, 22_000_000, blur_score=200.0, exposure_ok=False),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    _, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)
    assert plan.degraded is True
    # 原始可用 0/2 → confidence 上限 0.4*0 + blur 项 < 0.6 → 全部要求审批
    assert all(d.requires_approval for d in plan.decisions), [
        (d.decision_id, d.confidence, d.requires_approval) for d in plan.decisions
    ]
    assert any(d.confidence is not None and d.confidence < 0.6 for d in plan.decisions)


def test_borrowed_shot_recorded():
    """空幕借片（从未用池借用）→ borrowed_shot 事件留痕。"""
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=200.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=180.0),
        _make_tech_obs(2, 90_000_000, 92_000_000, blur_score=150.0),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    edl, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)
    assert plan.degraded is True
    assert any(e.startswith("borrowed_shot:act=") for e in plan.degradation_events), (
        plan.degradation_events
    )


def _act_graph(act_shots: dict[str, list[str]]) -> StoryGraph:
    """手工构造四幕图（只含 ACT 节点），用于精确控制每幕的候选集合。"""
    nodes = [
        StoryNode(
            node_id=f"act_{name}",
            node_type=StoryNodeType.ACT,
            ref_id=f"act_{name}",
            attributes={"act": name, "shot_ids": ids},
        )
        for name, ids in act_shots.items()
    ]
    return StoryGraph(
        schema_version="1.0", project_id="test_proj", created_at=0,
        producer="test", source_ref="dummy.mp4",
        graph_id="graph_test", version="0.1", nodes=nodes, edges=[],
    )


def test_act_duration_relaxation_recorded():
    """本幕选中时长 < 目标 50% → 放宽本幕判据重选并采纳 → 必须留痕。

    构造：hook 幕含 A（1s，blur=200 可用）与 C（3s，blur=5 不可用），
    全局池另有 B（可用）保证不触发全局放宽。首选只有 A（1s < 2.25s 的
    50%），放宽后 C 入选补足时长 → 采纳 → act 级 relax 事件。
    """
    obs = [
        _make_tech_obs(0, 0, 1_000_000, blur_score=200.0),          # A：可用、短
        _make_tech_obs(2, 6_000_000, 9_000_000, blur_score=5.0),    # C：不可用、长
        _make_tech_obs(1, 10_000_000, 13_000_000, blur_score=200.0),  # B：可用
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs, target_duration_us=15_000_000)
    graph = _act_graph({
        "hook": ["shot_00000000", "shot_00000002"],
        "develop": ["shot_00000001"],
    })
    edl, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)

    assert plan.degraded is True
    act_events = [
        e for e in plan.degradation_events
        if e.startswith("relax_technical_usable:scope=act:hook:")
    ]
    assert act_events, plan.degradation_events
    # 放宽后 hook 幕时长应被补足（> 1s）
    hook_edits = [e for e in edl.ordered_edits if "act=hook" in (e.rationale or "")]
    hook_dur = sum(e.out_frame - e.in_frame for e in hook_edits)
    assert hook_dur > 1_000_000, f"hook_dur={hook_dur}"


def test_degraded_plan_still_consistent_and_valid():
    """降级选片的 plan↔EDL 一致性不受影响（T1 校验对降级计划同样成立）。"""
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=5.0),
        _make_tech_obs(1, 20_000_000, 22_000_000, blur_score=5.0),
    ]
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    graph = build_story_graph(brief, obs)
    edl, plan = HeuristicDirectorReasoner().generate_plan(brief, graph, obs)
    assert plan.degraded is True
    assert plan.sequence == [e.source_asset_id for e in edl.ordered_edits]
    assert len(plan.decisions) == len(edl.ordered_edits)
