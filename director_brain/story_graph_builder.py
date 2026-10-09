"""M2.1 Story Graph Builder：从 Brief + 观测构建四幕故事图。

按时间比例把 deterministic_technical 观测划分到 hook / develop / peak /
resolve 四幕，每幕产出一个 :class:`StoryNode`（node_type=ACT），幕间加三条
TEMPORAL 顺序边。

边界：总时长为 0 或无观测时仍返回 4 个零长节点与 3 条边，不抛异常。
"""
from __future__ import annotations

import json
import time

from director_brain.utils import short_hash
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.film_observation import (
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)
from director_brain.models.story_graph import (
    INFERENCE_STATUS_STRUCTURAL,
    StoryEdge,
    StoryEdgeType,
    StoryGraph,
    StoryNode,
    StoryNodeType,
)

PRODUCER = "story_graph_v0.2"

#: 四幕时间比例区间：单一事实源 acts.ACT_INTERVALS（由 ACT_RATIO 累积派生，
#: 架构体检候选⑥收编——此前本模块自带一份区间表靠注释与 reasoner 对齐）。
from director_brain.acts import ACT_INTERVALS as _ACTS  # noqa: E402


def _assign_acts(
    tech_obs: list[FilmObservation], timeline_start: int, timeline_end: int
):
    """把同一素材时间轴上的观测按相对位置分配到四幕。

    Returns:
        list of length 4，每项为该幕的观测子列表。
    """
    buckets: list[list[FilmObservation]] = [[], [], [], []]
    duration = timeline_end - timeline_start
    if duration <= 0:
        return buckets
    for o in tech_obs:
        ratio = (o.start_frame - timeline_start) / duration
        # 最后一帧兜底归入 resolve
        if ratio >= _ACTS[3][1] - 1e-9:
            buckets[3].append(o)
            continue
        for idx, (_name, lo, hi, _label) in enumerate(_ACTS):
            if lo <= ratio < hi:
                buckets[idx].append(o)
                break
    return buckets


def _semantic_mention_nodes(
    project_asset_id: str | None,
    observations: list[FilmObservation],
) -> list[StoryNode]:
    """Expose model-described people/events as source-local evidence nodes.

    These nodes preserve the exact source observation and time anchor. They are
    mentions from one observation, not persistent identities or inferred event
    links; cross-asset relationships remain a separately reviewed overlay.
    """
    nodes: list[StoryNode] = []
    eligible = sorted(
        (
            observation for observation in observations
            if observation.observation_type == "vlm_semantic"
            and observation.claim_kind == ClaimKind.MODEL_OBSERVATION
        ),
        key=lambda observation: (
            observation.start_frame,
            observation.end_frame,
            observation.observation_id,
        ),
    )
    for observation in eligible:
        try:
            claim = json.loads(observation.claim) if observation.claim else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(claim, dict):
            continue

        anchor = {
            "project_asset_id": project_asset_id,
            "source_content_hash": observation.media_hash.lower(),
            "source_start": observation.start_frame,
            "source_end": observation.end_frame,
            "timebase": observation.timebase,
            "timebase_unit": observation.timebase_unit.value,
        }
        provenance = {
            "evidence_refs": [observation.observation_id],
            "source_anchor": anchor,
            "claim_kind": observation.claim_kind.value,
            "review_state": observation.review_state,
            "provider": observation.provider,
            "model_version": observation.model_version,
            "prompt_version": observation.prompt_version,
            "confidence": observation.confidence,
            "identity_scope": "single_observation",
        }

        people = claim.get("people")
        if isinstance(people, list):
            for index, description in enumerate(people):
                if not isinstance(description, str) or not description.strip():
                    continue
                nodes.append(StoryNode(
                    node_id=(
                        "person_mention_"
                        + short_hash(
                            f"{project_asset_id}|{observation.observation_id}|{index}"
                        )
                    ),
                    node_type=StoryNodeType.PERSON_MENTION,
                    ref_id=observation.observation_id,
                    attributes={
                        **provenance,
                        "mention_index": index,
                        "description": description,
                    },
                ))

        event_fields = {
            key: claim.get(key)
            for key in ("action_type", "scene_description", "temporal_notes")
            if isinstance(claim.get(key), str) and claim[key].strip()
        }
        if event_fields:
            nodes.append(StoryNode(
                node_id=(
                    "event_mention_"
                    + short_hash(
                        f"{project_asset_id}|{observation.observation_id}|event"
                    )
                ),
                node_type=StoryNodeType.EVENT_MENTION,
                ref_id=observation.observation_id,
                attributes={
                    **provenance,
                    "semantic_fields": event_fields,
                },
            ))
    return nodes


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
    return _build_source_story_graph(
        project_id=brief.project_id,
        source_ref=brief.source_ref,
        observations=observations,
    )


def build_asset_story_graph(
    project_id: str,
    asset_id: str,
    observations: list[FilmObservation],
) -> StoryGraph:
    """Build one project-bound asset graph without inventing a DirectorBrief.

    Project aggregation supplies only persisted observations for one manifest
    asset. The opaque source ref prevents local filesystem paths from entering
    the graph contract.
    """
    if not project_id or not asset_id:
        raise ValueError("project_id and asset_id must be non-empty")
    if any(obs.project_id != project_id for obs in observations):
        raise ValueError("asset observations must belong to the requested project")
    if any(obs.project_asset_id != asset_id for obs in observations):
        raise ValueError("asset observations must match the requested project asset")
    return _build_source_story_graph(
        project_id=project_id,
        source_ref=f"asset://{short_hash(asset_id)}",
        observations=observations,
    )


def _build_source_story_graph(
    *,
    project_id: str,
    source_ref: str,
    observations: list[FilmObservation],
) -> StoryGraph:
    """Shared source-local implementation for legacy and manifest-bound input."""
    tech_obs = sorted(
        [o for o in observations if o.observation_type == "deterministic_technical"],
        key=lambda o: o.start_frame,
    )

    project_asset_ids = {o.project_asset_id for o in tech_obs}
    if len(project_asset_ids) > 1:
        if None in project_asset_ids:
            raise ValueError("StoryGraph cannot mix bound and unbound observations")
        raise ValueError(
            "StoryGraph requires one project_asset_id; cross-asset timelines "
            "must be modeled explicitly before graph construction"
        )

    timebases = {(o.timebase, o.timebase_unit) for o in tech_obs}
    if len(timebases) > 1:
        raise ValueError("StoryGraph observations must share one timebase and unit")

    timeline_start = min((o.start_frame for o in tech_obs), default=0)
    timeline_end = max((o.end_frame for o in tech_obs), default=0)
    duration = timeline_end - timeline_start
    buckets = _assign_acts(tech_obs, timeline_start, timeline_end)
    timebase, timebase_unit = next(iter(timebases), (None, TimebaseUnit.UNKNOWN))
    project_asset_id = next(iter(project_asset_ids), None)

    nodes: list[StoryNode] = []
    for idx, (act_name, lo, hi, label) in enumerate(_ACTS):
        act_obs = buckets[idx]
        if duration > 0:
            start_value = timeline_start + int(duration * lo)
            end_value = timeline_start + int(duration * hi)
        else:
            start_value = timeline_start
            end_value = timeline_start
        attributes = {
            "act": act_name,
            "label": label,
            "start_value": start_value,
            "end_value": end_value,
            "timebase": timebase,
            "timebase_unit": timebase_unit.value,
            "shot_count": len(act_obs),
            "shot_ids": [o.media_asset_id for o in act_obs],
        }
        if timebase_unit == TimebaseUnit.MICROSECONDS:
            attributes["start_us"] = start_value
            attributes["end_us"] = end_value
        nodes.append(
            StoryNode(
                node_id=f"act_{act_name}",
                node_type=StoryNodeType.ACT,
                ref_id=f"act_{act_name}",
                attributes=attributes,
            )
        )

    # Preserve people/event descriptions as individually anchored, unreviewed
    # source observations. No nodes are merged across shots or project assets.
    nodes.extend(_semantic_mention_nodes(project_asset_id, observations))

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

    graph_id = f"graph_{project_id}_{short_hash(source_ref)}"

    return StoryGraph(
        schema_version="1.2",
        project_id=project_id,
        created_at=int(time.time()),
        producer=PRODUCER,
        source_ref=source_ref,
        graph_id=graph_id,
        version="0.1",
        timeline_scope=(
            "project_asset" if project_asset_id else "single_request_source_unbound"
        ),
        project_asset_id=project_asset_id,
        timebase=timebase,
        timebase_unit=timebase_unit if tech_obs else None,
        nodes=nodes,
        edges=edges,
    )
