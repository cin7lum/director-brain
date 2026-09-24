"""M2.1 Story Graph Builder 单元测试。

用 mock deterministic_technical 观测验证 build_story_graph：
- 返回 4 个幕节点（hook/develop/peak/resolve）
- 返回 3 条 temporal 顺序边
- 空观测仍返回 4 节点 3 边
- 每幕 shot_count 之和 = 总镜头数
- producer 字段正确
"""
from __future__ import annotations

import json
import time

from director_brain.brief_compiler import compile_brief
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.models.story_graph import StoryEdgeType, StoryNodeType
from director_brain.story_graph_builder import build_story_graph


def _make_tech_obs(index: int, start_us: int, end_us: int) -> FilmObservation:
    claim = json.dumps(
        {
            "blur_score": 0.2,
            "brightness_mean": 0.5,
            "exposure_ok": True,
            "shake_score": 0.1,
            "dominant_hue": 0.3,
            "saturation_mean": 0.4,
            "center_weight": 0.5,
        }
    )
    return FilmObservation(
        observation_id=f"tech_{index:04d}",
        media_asset_id=f"shot_{index:08d}",
        media_hash=f"hash_{index}",
        start_frame=start_us,
        end_frame=end_us,
        timebase=1_000_000,
        observation_type="deterministic_technical",
        claim=claim,
        provider="deterministic",
        model_version="v0.1",
        prompt_version="n/a",
        confidence=1.0,
        review_state="auto_generated",
        claim_kind=ClaimKind.MEASURED,
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="deterministic_analysis",
        source_ref="dummy.mp4",
    )


def _make_brief(obs):
    return compile_brief("proj_graph", "dummy.mp4", obs)


def test_empty_observations_still_returns_four_nodes_and_three_edges():
    brief = _make_brief([])
    graph = build_story_graph(brief, [])
    assert len(graph.nodes) == 4
    acts = [n.attributes.get("act") for n in graph.nodes]
    assert acts == ["hook", "develop", "peak", "resolve"]
    assert len(graph.edges) == 3
    assert graph.producer == "story_graph_v0.1"
    assert graph.project_id == "proj_graph"
    assert graph.graph_id.startswith("graph_")
    assert graph.version == "0.1"
    # 边界：start_us == end_us == 0
    for n in graph.nodes:
        assert n.attributes["start_us"] == 0
        assert n.attributes["end_us"] == 0
        assert n.attributes["shot_count"] == 0


def test_four_act_nodes_present():
    # 总时长 100s = 100_000_000 us；按 10 个镜头均匀分布
    obs = []
    total_us = 100_000_000
    per = total_us // 10
    for i in range(10):
        obs.append(_make_tech_obs(i, i * per, (i + 1) * per))

    brief = _make_brief(obs)
    graph = build_story_graph(brief, obs)

    assert len(graph.nodes) == 4
    acts = [n.attributes["act"] for n in graph.nodes]
    assert acts == ["hook", "develop", "peak", "resolve"]

    # 节点类型必须是 ACT（四幕节点）
    for n in graph.nodes:
        assert n.node_type == StoryNodeType.ACT

    # 每幕 shot_count 之和 = 10
    assert sum(n.attributes["shot_count"] for n in graph.nodes) == 10

    # 时间范围单调递增
    ends = [n.attributes["end_us"] for n in graph.nodes]
    assert ends == sorted(ends)
    assert graph.nodes[-1].attributes["end_us"] == total_us


def test_three_temporal_edges_in_order():
    obs = [_make_tech_obs(i, i * 1_000_000, (i + 1) * 1_000_000) for i in range(5)]
    brief = _make_brief(obs)
    graph = build_story_graph(brief, obs)

    assert len(graph.edges) == 3
    expected_pairs = [
        ("act_hook", "act_develop"),
        ("act_develop", "act_peak"),
        ("act_peak", "act_resolve"),
    ]
    actual_pairs = [(e.from_node, e.to_node) for e in graph.edges]
    assert actual_pairs == expected_pairs
    for e in graph.edges:
        assert e.edge_type == StoryEdgeType.TEMPORAL
        assert e.inference_status == "structural"
        assert e.confidence == 1.0


def test_shot_ids_partition_obs():
    obs = [_make_tech_obs(i, i * 2_000_000, (i + 1) * 2_000_000) for i in range(20)]
    brief = _make_brief(obs)
    graph = build_story_graph(brief, obs)

    all_ids = []
    for n in graph.nodes:
        all_ids.extend(n.attributes["shot_ids"])
    # 不重不漏
    expected_ids = [f"shot_{i:08d}" for i in range(20)]
    assert sorted(all_ids) == sorted(expected_ids)
