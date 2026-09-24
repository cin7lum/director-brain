"""M2.2 Relation Inference 单元测试。

用 mock FilmObservation 列表验证 infer_relations 的三条 Tier-A 规则：
- 空观测 / <2 镜头 → 空列表，不抛异常
- REACTION：相邻 blur 差异小（|Δblur|<50）且时长在 0.5s-8s
- CONTRAST：相邻 brightness 突变（|Δbrightness|>60）
- MONTAGE：连续 3+ 短镜头（每个 <1s）
- 所有边 inference_status="inferred" 且 evidence_refs 指向两条观测
"""
from __future__ import annotations

import json
import time

from director_brain.brief_compiler import compile_brief
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.models.story_graph import StoryEdgeType
from director_brain.relation_inference import infer_relations
from director_brain.story_graph_builder import build_story_graph


def _make_tech_obs(
    index: int,
    start_us: int,
    end_us: int,
    *,
    blur_score: float = 100.0,
    brightness_mean: float = 100.0,
    shake_score: float = 5.0,
    exposure_ok: bool = True,
) -> FilmObservation:
    """构造一条 deterministic_technical 观测（指标量纲与真实 OpenCV 输出一致）。"""
    claim = json.dumps(
        {
            "blur_score": blur_score,          # Laplacian 方差，量级 0~数千
            "brightness_mean": brightness_mean,  # 灰度均值 0~255
            "exposure_ok": exposure_ok,
            "shake_score": shake_score,         # absdiff 均值 0~数十
            "dominant_hue": 30.0,
            "saturation_mean": 80.0,
            "center_weight": 1.0,
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


def _graph(obs):
    brief = compile_brief("test_proj", "dummy.mp4", obs)
    return build_story_graph(brief, obs)


def _make_vlm_obs(index, start_us, end_us, *, role="hero", shot_function="ACTION"):
    """构造一条成功的 vlm_semantic 观测（claim_kind=MODEL_OBSERVATION）。"""
    claim = json.dumps({
        "shot_function": shot_function,
        "proposed_role_v2": role,
        "motion_amount": "subtle",
        "frame_description": "test",
    })
    return FilmObservation(
        observation_id=f"vlm_shot_{index:08d}",
        media_asset_id=f"shot_{index:08d}",
        media_hash=f"hash_{index}",
        start_frame=start_us,
        end_frame=end_us,
        timebase=1_000_000,
        observation_type="vlm_semantic",
        claim=claim,
        provider="ollama_qwen3_vl",
        model_version="qwen3-vl:latest",
        prompt_version="vlm_prompt_v1",
        confidence=0.7,
        review_state="auto_generated",
        claim_kind=ClaimKind.MODEL_OBSERVATION,
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="ollama_qwen3_vl",
        source_ref="dummy.mp4",
    )


def test_empty_observations_returns_empty():
    assert infer_relations([], _graph([])) == []


def test_less_than_two_shots_returns_empty():
    obs = [_make_tech_obs(0, 0, 2_000_000)]
    assert infer_relations(obs, _graph(obs)) == []


def test_reaction_edge_on_small_blur_diff():
    # 两镜头均 2s（落在 0.5s-8s），blur 差 20（<50）→ REACTION
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=100.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=120.0, brightness_mean=100.0),
    ]
    edges = infer_relations(obs, _graph(obs))
    reaction = [e for e in edges if e.edge_id.startswith("reaction_")]
    assert len(reaction) == 1
    assert reaction[0].edge_type == StoryEdgeType.CAUSAL_CANDIDATE
    assert reaction[0].from_node == "shot_00000000"
    assert reaction[0].to_node == "shot_00000001"
    assert 0.6 <= reaction[0].confidence <= 0.8


def test_contrast_edge_on_brightness_jump():
    # brightness 差 160（>60）→ CONTRAST
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=30.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=105.0, brightness_mean=190.0),
    ]
    edges = infer_relations(obs, _graph(obs))
    contrast = [e for e in edges if e.edge_id.startswith("contrast_")]
    assert len(contrast) >= 1
    assert contrast[0].edge_type == StoryEdgeType.EMOTIONAL_TURN
    assert 0.5 <= contrast[0].confidence <= 0.9


def test_montage_edges_on_consecutive_short_shots():
    # 连续 3 个镜头，每个 <1s → 2 条相邻 MONTAGE 边
    obs = [
        _make_tech_obs(0, 0, 500_000, blur_score=100.0, brightness_mean=100.0),
        _make_tech_obs(1, 500_000, 900_000, blur_score=100.0, brightness_mean=100.0),
        _make_tech_obs(2, 900_000, 1_300_000, blur_score=100.0, brightness_mean=100.0),
    ]
    edges = infer_relations(obs, _graph(obs))
    montage = [e for e in edges if e.edge_id.startswith("montage_")]
    assert len(montage) == 2
    assert all(e.edge_type == StoryEdgeType.TEMPORAL for e in montage)
    # 跨幕边置信度下调 0.05（0.7 -> 0.65）
    assert all(abs(e.confidence - 0.65) < 0.01 for e in montage)


def test_all_edges_have_inferred_status_and_evidence():
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=30.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=120.0, brightness_mean=200.0),
        _make_tech_obs(2, 4_000_000, 6_000_000, blur_score=130.0, brightness_mean=205.0),
    ]
    edges = infer_relations(obs, _graph(obs))
    assert len(edges) > 0
    for e in edges:
        assert e.inference_status == "inferred"
        # 前两条证据是观测 id；后续可能追加幕节点 id（graph 消费）
        assert len(e.evidence_refs) >= 2
        assert e.evidence_refs[0].startswith("det_shot_")
        assert e.evidence_refs[1].startswith("det_shot_")


def test_no_duplicate_edges_for_same_pair_and_type():
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=30.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=120.0, brightness_mean=200.0),
    ]
    edges = infer_relations(obs, _graph(obs))
    keys = [(e.from_node, e.to_node, e.edge_type) for e in edges]
    assert len(keys) == len(set(keys))


def test_same_act_edge_gets_confidence_boost_and_act_evidence():
    """同幕内镜头边：confidence +0.1，evidence_refs 含幕节点 id。"""
    # total_us=100M；shot0/shot1 都落在 hook 幕（ratio < 0.15）
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=100.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=120.0, brightness_mean=100.0),
        _make_tech_obs(2, 90_000_000, 100_000_000, blur_score=200.0, brightness_mean=100.0),
    ]
    graph = _graph(obs)
    edges = infer_relations(obs, graph)
    reaction = [e for e in edges if e.edge_id.startswith("reaction_shot_00000000")]
    assert len(reaction) == 1
    e = reaction[0]
    # 基础 confidence = 0.8 - (20/50)*0.2 = 0.72；同幕 +0.1 = 0.82
    assert abs(e.confidence - 0.82) < 0.01, f"unexpected confidence: {e.confidence}"
    # evidence_refs 必须包含幕节点 id
    assert "act_hook" in e.evidence_refs


def test_semantic_continuity_same_role():
    """相邻镜头 proposed_role_v2 相同 → SEMANTIC_CONTINUITY 边。"""
    # blur 差 100（>=50）→ 不触发确定性 reaction，避免与 VLM 边去重冲突
    tech = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=100.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=200.0, brightness_mean=100.0),
    ]
    vlm = [
        _make_vlm_obs(0, 0, 2_000_000, role="hero", shot_function="ACTION"),
        _make_vlm_obs(1, 2_000_000, 4_000_000, role="hero", shot_function="ACTION"),
    ]
    obs = tech + vlm
    edges = infer_relations(obs, _graph(obs))
    cont = [e for e in edges if e.edge_id.startswith("semantic_continuity_")]
    assert len(cont) == 1
    assert cont[0].edge_type == StoryEdgeType.CAUSAL_CANDIDATE
    assert cont[0].from_node == "shot_00000000"
    assert cont[0].to_node == "shot_00000001"
    # 前两条证据指向 VLM 观测；后续可能追加幕节点 id
    assert cont[0].evidence_refs[:2] == ["vlm_shot_00000000", "vlm_shot_00000001"]


def test_semantic_contrast_function_diff():
    """ESTABLISHING → ACTION（高对比对）→ SEMANTIC_CONTRAST 边。"""
    tech = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=100.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=200.0, brightness_mean=100.0),
    ]
    vlm = [
        _make_vlm_obs(0, 0, 2_000_000, role="hero", shot_function="ESTABLISHING"),
        _make_vlm_obs(1, 2_000_000, 4_000_000, role="support", shot_function="ACTION"),
    ]
    obs = tech + vlm
    edges = infer_relations(obs, _graph(obs))
    contrast = [e for e in edges if e.edge_id.startswith("semantic_contrast_")]
    assert len(contrast) == 1
    assert contrast[0].edge_type == StoryEdgeType.EMOTIONAL_TURN
    assert contrast[0].from_node == "shot_00000000"
    assert contrast[0].to_node == "shot_00000001"


def test_discard_filter_skips_edges():
    """任一端点 role=discard → 不生成以该镜头为端点的 VLM 语义边。"""
    tech = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=100.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=300.0, brightness_mean=100.0),
        _make_tech_obs(2, 4_000_000, 6_000_000, blur_score=100.0, brightness_mean=100.0),
    ]
    vlm = [
        _make_vlm_obs(0, 0, 2_000_000, role="hero", shot_function="ACTION"),
        _make_vlm_obs(1, 2_000_000, 4_000_000, role="discard", shot_function="ACTION"),
        _make_vlm_obs(2, 4_000_000, 6_000_000, role="hero", shot_function="ACTION"),
    ]
    obs = tech + vlm
    edges = infer_relations(obs, _graph(obs))
    semantic = [e for e in edges if e.edge_id.startswith("semantic_")]
    assert semantic == []


def test_no_vlm_observations_fallback():
    """只有 deterministic 观测 → 不产生任何 semantic_ 边（向后兼容）。"""
    obs = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=100.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=200.0, brightness_mean=100.0),
    ]
    edges = infer_relations(obs, _graph(obs))
    assert not any(e.edge_id.startswith("semantic_") for e in edges)


def test_vlm_and_deterministic_edges_coexist():
    """VLM 边与确定性边共存：reaction（CAUSAL_CANDIDATE）+ semantic_contrast（EMOTIONAL_TURN）。"""
    # blur 差 20（<50）、时长 2s → 确定性 reaction 边触发；brightness 相同 → 无确定性 contrast
    tech = [
        _make_tech_obs(0, 0, 2_000_000, blur_score=100.0, brightness_mean=100.0),
        _make_tech_obs(1, 2_000_000, 4_000_000, blur_score=120.0, brightness_mean=100.0),
    ]
    vlm = [
        _make_vlm_obs(0, 0, 2_000_000, role="hero", shot_function="ESTABLISHING"),
        _make_vlm_obs(1, 2_000_000, 4_000_000, role="support", shot_function="ACTION"),
    ]
    obs = tech + vlm
    edges = infer_relations(obs, _graph(obs))

    reaction = [e for e in edges if e.edge_id.startswith("reaction_")]
    vlm_contrast = [e for e in edges if e.edge_id.startswith("semantic_contrast_")]
    assert len(reaction) == 1
    assert reaction[0].edge_type == StoryEdgeType.CAUSAL_CANDIDATE
    assert len(vlm_contrast) == 1
    assert vlm_contrast[0].edge_type == StoryEdgeType.EMOTIONAL_TURN

    # 同一对镜头允许不同类型边共存，但 (from, to, type) 不重复
    keys = [(e.from_node, e.to_node, e.edge_type) for e in edges]
    assert len(keys) == len(set(keys))
