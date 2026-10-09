"""Director Brain 数据模型验收测试。

严格覆盖 W1（Director Brain M0 里程碑）的 7 项验收标准：
1. 每个模型可实例化，必填字段缺失时报 ValidationError
2. claim_kind 只接受 6 种枚举值，非法值报错
3. EditItem 结构完整，in_frame/out_frame 为整数
4. extra="forbid"：传入未定义字段报错
5. 模型间引用关系正确（EDL.ordered_edits 是 List[EditItem]，Plan.decisions 是 List[Decision]）
6. 所有模型带 schema_version 字段
7. 本文件随实现完成后被 pytest 全部跑通
"""
from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_context import ContextLayer, FilmContextSnapshot
from director_brain.models.film_entity import (
    EntityType,
    FilmEntity,
    RelationType,
    StoryRelation,
)
from director_brain.models.film_observation import (
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)
from director_brain.models.project import ProjectAssetTimeMap
from director_brain.models.revision import RevisionProposal
from director_brain.models.story_graph import (
    StoryEdge,
    StoryEdgeType,
    StoryGraph,
    StoryNode,
    StoryNodeType,
)


# --- 顶层记录共享的公共字段最小填充 ---
COMMON = dict(
    project_id="proj-001",
    created_at=1700000000,
    producer="pytest",
    source_ref="brief://v1",
)

# --- 每个顶层记录模型的最小合法字段（不含公共字段） ---
MINIMAL = {
    DirectorBrief: dict(
        brief_id="brief-001",
        version="1.0",
        source_text="开场镜头...",
        language="zh",
        intent="情感短片",
        audience="大众",
        target_duration=120,
        delivery_profile="web",
        emotional_arc="rising",
        visual_language="cinematic",
        editing_language="smooth",
        sound_language="ambient",
        approval_state="draft",
    ),
    FilmContextSnapshot: dict(
        context_id="ctx-001",
        analysis_fingerprint="fp-abc",
        provider="zhipu",
        model="glm-4.6v",
        prompt_version="1",
        timebase=25,
        coverage="full",
        rights_scope="licensed",
        cache_state="fresh",
    ),
    FilmObservation: dict(
        observation_id="obs-001",
        media_asset_id="asset-001",
        media_hash="hash-123",
        start_frame=0,
        end_frame=25,
        timebase=25,
        observation_type="action",
        claim="人物走入画面",
        provider="zhipu",
        model_version="glm-4.6v",
        prompt_version="1",
        confidence=0.9,
        review_state="pending",
        claim_kind=ClaimKind.MODEL_OBSERVATION,
    ),
    FilmEntity: dict(
        entity_id="ent-001",
        entity_type=EntityType.PERSON,
        display_name="主角",
        identity_confidence=0.95,
    ),
    StoryRelation: dict(
        relation_id="rel-001",
        from_entity="ent-001",
        to_entity="ent-002",
        relation_type="friend",
        confirmation_state="inferred",
        privacy_class="public",
    ),
    StoryGraph: dict(
        graph_id="graph-001",
        version="1.0",
    ),
    EditorialDecisionList: dict(
        edl_id="edl-001",
        version="1.0",
        brief_version="1.0",
        context_id="ctx-001",
        timebase=25,
        approval_state="draft",
    ),
    DirectorDecisionPlan: dict(
        plan_id="plan-001",
        version="1.0",
        brief_version="1.0",
        film_state_version="1",
        validation_status="unverified",
        approval_state="draft",
    ),
    RevisionProposal: dict(
        proposal_id="rev-001",
        change_summary="调整节奏",
        approval_state="draft",
    ),
}

RECORD_MODELS = list(MINIMAL.keys())


def _make(model_cls, **overrides):
    """用公共字段 + 最小字段构造一个合法实例。"""
    kwargs = dict(COMMON)
    kwargs.update(MINIMAL[model_cls])
    kwargs.update(overrides)
    return model_cls(**kwargs)


# 嵌入式组件的合法最小字段（非顶层记录，无公共字段）
EMBEDDED_CASES = [
    (
        EditItem,
        dict(
            source_asset_id="asset-001",
            source_media_hash="hash-123",
            in_frame=0,
            out_frame=25,
            timebase=25,
        ),
    ),
    (Decision, dict(decision_id="dec-001", purpose="开场建立空间")),
    (
        StoryNode,
        dict(
            node_id="n1",
            node_type=StoryNodeType.PERSON,
            ref_id="ent-001",
        ),
    ),
    (
        StoryEdge,
        dict(
            edge_id="e1",
            from_node="n1",
            to_node="n2",
            edge_type=StoryEdgeType.TEMPORAL,
            inference_status="observed",
            confidence=0.8,
        ),
    ),
]


# ---------------------------------------------------------------------------
# 标准 1：可实例化 + 必填字段缺失报 ValidationError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_cls", RECORD_MODELS)
def test_record_instantiable(model_cls):
    obj = _make(model_cls)
    assert obj is not None


@pytest.mark.parametrize("model_cls", RECORD_MODELS)
def test_missing_required_raises(model_cls):
    with pytest.raises(ValidationError):
        model_cls()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 标准 2：claim_kind 只接受 6 种枚举值
# ---------------------------------------------------------------------------
def test_claim_kind_has_exactly_six_values():
    assert len(list(ClaimKind)) == 6


@pytest.mark.parametrize("kind", list(ClaimKind))
def test_claim_kind_accepted(kind):
    obs = _make(FilmObservation, claim_kind=kind)
    assert obs.claim_kind is kind


def test_claim_kind_invalid_string_raises():
    with pytest.raises(ValidationError):
        _make(FilmObservation, claim_kind="NOT_A_REAL_KIND")


# ---------------------------------------------------------------------------
# 标准 3：EditItem 结构完整，in_frame/out_frame 为整数
# ---------------------------------------------------------------------------
def test_edititem_frames_are_int():
    item = EditItem(
        source_asset_id="asset-001",
        source_media_hash="hash-123",
        in_frame=10,
        out_frame=35,
        timebase=25,
    )
    assert isinstance(item.in_frame, int)
    assert isinstance(item.out_frame, int)
    assert item.in_frame == 10
    assert item.out_frame == 35


def test_edititem_has_required_clip_fields():
    item = EditItem(
        source_asset_id="asset-001",
        source_media_hash="hash-123",
        in_frame=10,
        out_frame=35,
        timebase=25,
    )
    for fname in ("source_asset_id", "source_media_hash", "in_frame",
                  "out_frame", "timebase"):
        assert fname in EditItem.model_fields


def test_edititem_missing_required_raises():
    with pytest.raises(ValidationError):
        EditItem(in_frame=0, out_frame=25, timebase=25)


def test_project_asset_time_map_requires_all_declared_streams_for_complete():
    video = {
        "stream_index": 0,
        "codec_type": "video",
        "time_base": "1/90000",
        "start_pts": 9000,
        "start_time_seconds": "0.100000",
        "source_start_offset_numerator": 0,
        "source_start_offset_denominator": 1,
        "source_start_offset_state": "mapped_from_pts",
    }
    audio = {
        "stream_index": 1,
        "codec_type": "audio",
        "time_base": "1/48000",
        "start_pts": 16800,
        "start_time_seconds": "0.350000",
        "sample_rate": 48000,
        "source_start_offset_numerator": 1,
        "source_start_offset_denominator": 4,
        "source_start_offset_state": "mapped_from_pts",
    }
    mapping = ProjectAssetTimeMap.model_validate({
        "container_start_time_seconds": "0.100000",
        "video_stream": video,
        "audio_streams": [audio],
        "mapping_state": "complete",
    })
    assert mapping.audio_streams[0].source_start_offset_numerator == 1
    assert mapping.first_audio_stream_clock_from_video() == (
        1, Fraction(1, 4))

    audio["source_start_offset_numerator"] = None
    audio["source_start_offset_denominator"] = None
    audio["source_start_offset_state"] = "unavailable"
    with pytest.raises(ValidationError, match="mapping_state must be partial"):
        ProjectAssetTimeMap.model_validate({
            "container_start_time_seconds": "0.100000",
            "video_stream": video,
            "audio_streams": [audio],
            "mapping_state": "complete",
        })


def test_project_audio_clock_requires_video_and_audio_offsets():
    mapping = ProjectAssetTimeMap.model_validate({
        "container_start_time_seconds": "0",
        "video_stream": {
            "stream_index": 0,
            "codec_type": "video",
            "source_start_offset_state": "unavailable",
        },
        "audio_streams": [{
            "stream_index": 1,
            "codec_type": "audio",
            "source_start_offset_state": "unavailable",
        }],
        "mapping_state": "unavailable",
    })
    with pytest.raises(ValueError, match="audio-to-video source clock mapping"):
        mapping.first_audio_stream_clock_from_video()


def test_project_bound_edit_requires_explicit_observation_and_source_clock():
    with pytest.raises(ValidationError, match="project-bound edits require"):
        EditItem(
            source_asset_id="shot-1",
            source_media_hash="hash-1",
            in_frame=100,
            out_frame=900,
            timebase=1_000_000,
            timebase_unit=TimebaseUnit.MICROSECONDS,
            project_asset_id="asset-1",
        )

    item = EditItem(
        source_asset_id="shot-1",
        source_media_hash="hash-1",
        in_frame=200,
        out_frame=800,
        timebase=1_000_000,
        timebase_unit=TimebaseUnit.MICROSECONDS,
        project_asset_id="asset-1",
        source_observation_refs=["obs-1"],
        source_observation_start=100,
        source_observation_end=900,
        source_timebase=1_000_000,
        source_timebase_unit=TimebaseUnit.MICROSECONDS,
    )
    assert item.project_asset_id == "asset-1"


# ---------------------------------------------------------------------------
# 标准 4：extra="forbid"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_cls", RECORD_MODELS)
def test_record_extra_field_forbidden(model_cls):
    with pytest.raises(ValidationError):
        _make(model_cls, bogus_undefined_field="nope")


@pytest.mark.parametrize("model_cls,kwargs", EMBEDDED_CASES)
def test_embedded_extra_field_forbidden(model_cls, kwargs):
    with pytest.raises(ValidationError):
        model_cls(**kwargs, bogus_undefined_field="nope")


# ---------------------------------------------------------------------------
# 标准 5：模型间引用关系正确
# ---------------------------------------------------------------------------
def test_edl_ordered_edits_is_list_of_edititem():
    item = EditItem(
        source_asset_id="asset-001",
        source_media_hash="hash-123",
        in_frame=0,
        out_frame=25,
        timebase=25,
    )
    edl = _make(EditorialDecisionList, ordered_edits=[item])
    assert isinstance(edl.ordered_edits, list)
    assert len(edl.ordered_edits) == 1
    assert isinstance(edl.ordered_edits[0], EditItem)


def test_plan_decisions_is_list_of_decision():
    dec = Decision(decision_id="dec-001", purpose="开场建立空间")
    plan = _make(DirectorDecisionPlan, decisions=[dec])
    assert isinstance(plan.decisions, list)
    assert isinstance(plan.decisions[0], Decision)
    assert plan.decisions[0].requires_approval is False


def test_storygraph_holds_nodes_and_edges():
    node = StoryNode(
        node_id="n1", node_type=StoryNodeType.EVENT, ref_id="obs-001"
    )
    edge = StoryEdge(
        edge_id="e1",
        from_node="n1",
        to_node="n2",
        edge_type=StoryEdgeType.EMOTIONAL_TURN,
        inference_status="inferred",
        confidence=0.7,
    )
    g = _make(StoryGraph, nodes=[node], edges=[edge])
    assert isinstance(g.nodes[0], StoryNode)
    assert isinstance(g.edges[0], StoryEdge)


# ---------------------------------------------------------------------------
# 标准 6：所有顶层记录模型带 schema_version 字段
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_cls", RECORD_MODELS)
def test_schema_version_present_on_records(model_cls):
    assert "schema_version" in model_cls.model_fields
    obj = _make(model_cls)
    assert obj.schema_version


# ---------------------------------------------------------------------------
# 枚举冒烟：确保 ContextLayer / EntityType 可取
# ---------------------------------------------------------------------------
def test_context_layer_enum_values():
    assert ContextLayer.PROJECT
    assert ContextLayer.ASSET
    assert ContextLayer.SCENE
    assert ContextLayer.EVIDENCE


def test_entity_type_enum_values():
    assert EntityType.PERSON
    assert EntityType.LOCATION
    assert EntityType.EVENT
    assert EntityType.OBJECT


# ---------------------------------------------------------------------------
# RelationType 枚举：合法字符串自动转换，非法值报 ValidationError
# ---------------------------------------------------------------------------
def test_relation_type_accepts_enum_member():
    rel = _make(StoryRelation, relation_type=RelationType.FRIEND)
    assert rel.relation_type is RelationType.FRIEND


def test_relation_type_accepts_legal_string():
    rel = _make(StoryRelation, relation_type="friend")
    assert rel.relation_type is RelationType.FRIEND


def test_relation_type_invalid_string_raises():
    with pytest.raises(ValidationError):
        _make(StoryRelation, relation_type="invalid")
