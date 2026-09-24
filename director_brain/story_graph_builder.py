"""M2.1 Story Graph Builder：从 Brief + 观测构建四幕故事图。

按时间比例把 deterministic_technical 观测划分到 hook / develop / peak /
resolve 四幕，每幕产出一个 :class:`StoryNode`（node_type=ACT），幕间加三条
TEMPORAL 顺序边。

边界：总时长为 0 或无观测时仍返回 4 个零长节点与 3 条边，不抛异常。
"""
from __future__ import annotations

import time

from director_brain._utils import short_hash
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.film_observation import FilmObservation
from director_brain.models.story_graph import (
    INFERENCE_STATUS_STRUCTURAL,
    StoryEdge,
    StoryEdgeType,
    StoryGraph,
    StoryNode,
    StoryNodeType,
)

PRODUCER = "story_graph_v0.1"

#: 四幕时间比例区间（起点比例，终点比例，中文标签）。
_ACTS: list[tuple[str, float, float, str]] = [
    ("hook", 0.00, 0.15, "开场"),
    ("develop", 0.15, 0.50, "发展"),
    ("peak", 0.50, 0.80, "高潮"),
    ("resolve", 0.80, 1.00, "收尾"),
]


def _assign_acts(tech_obs: list[FilmObservation], total_us: int):
    """把排序后的观测按 start_frame 比例分配到四幕。

    Returns:
        list of length 4，每项为该幕的观测子列表。
    """
    buckets: list[list[FilmObservation]] = [[], [], [], []]
    if total_us <= 0:
        return buckets
    for o in tech_obs:
        ratio = o.start_frame / total_us
        # 最后一帧兜底归入 resolve
        if ratio >= _ACTS[3][1] - 1e-9:
            buckets[3].append(o)
            continue
        for idx, (_name, lo, hi, _label) in enumerate(_ACTS):
            if lo <= ratio < hi:
                buckets[idx].append(o)
                break
    return buckets


def build_story_graph(
    brief: DirectorBrief,
    observations: list[FilmObservation],
) -> StoryGraph:
    """从 Brief 与观测构建四幕故事图。

    Args:
        brief: ``compile_brief`` 的产出，提供 project_id / source_ref。
        observations: deterministic_technical 观测列表（ASR 观测会被忽略）。

    Returns:
        4 个幕节点 + 3 条 temporal 边的 :class:`StoryGraph`。
    """
    tech_obs = sorted(
        [o for o in observations if o.observation_type == "deterministic_technical"],
        key=lambda o: o.start_frame,
    )

    total_us = tech_obs[-1].end_frame if tech_obs else 0
    buckets = _assign_acts(tech_obs, total_us)

    nodes: list[StoryNode] = []
    for idx, (act_name, lo, hi, label) in enumerate(_ACTS):
        act_obs = buckets[idx]
        if total_us > 0:
            start_us = int(total_us * lo)
            end_us = int(total_us * hi)
        else:
            start_us = 0
            end_us = 0
        nodes.append(
            StoryNode(
                node_id=f"act_{act_name}",
                node_type=StoryNodeType.ACT,
                ref_id=f"act_{act_name}",
                attributes={
                    "act": act_name,
                    "label": label,
                    "start_us": start_us,
                    "end_us": end_us,
                    "shot_count": len(act_obs),
                    "shot_ids": [o.media_asset_id for o in act_obs],
                },
            )
        )

    # 幕间三条顺序边
    edges: list[StoryEdge] = []
    for i in range(len(_ACTS) - 1):
        from_act = _ACTS[i][0]
        to_act = _ACTS[i + 1][0]
        edges.append(
            StoryEdge(
                edge_id=f"temporal_{from_act}_to_{to_act}",
                from_node=f"act_{from_act}",
                to_node=f"act_{to_act}",
                edge_type=StoryEdgeType.TEMPORAL,
                inference_status=INFERENCE_STATUS_STRUCTURAL,
                evidence_refs=[o.observation_id for o in buckets[i]],
                confidence=1.0,
            )
        )

    graph_id = f"graph_{brief.project_id}_{short_hash(brief.source_ref)}"

    return StoryGraph(
        schema_version="1.0",
        project_id=brief.project_id,
        created_at=int(time.time()),
        producer=PRODUCER,
        source_ref=brief.source_ref,
        graph_id=graph_id,
        version="0.1",
        nodes=nodes,
        edges=edges,
    )
