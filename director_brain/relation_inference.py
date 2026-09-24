"""M2.2 Relation Inference：从确定性技术指标推断镜头间叙事关系。

迁移 GEN-1 Tier-A 规则到确定性实现（当前无 VLM 标签，用 OpenCV 技术指标替代）：

- REACTION：相邻镜头 blur 差异小（|Δblur|<50）且两镜头时长均在
  0.5s-8s → 动作连续，edge_type=CAUSAL_CANDIDATE。
- CONTRAST：相邻镜头 brightness 突变（|Δbrightness|>60）或 blur 突变
  （|Δblur|>100）或 shake 相对比值 >=3 → 情绪转折，edge_type=EMOTIONAL_TURN。
- MONTAGE：连续 3+ 个镜头每个时长 <1s → 快速蒙太奇，对序列内每对相邻镜头
  加一条 TEMPORAL 边。

边界：只处理 deterministic_technical 观测；空观测或 <2 个镜头返回空列表，
不抛异常。同一对 (from, to, 关系类型) 只产出一条边。
"""
from __future__ import annotations

import json

from director_brain.models.film_observation import FilmObservation
from director_brain.models.story_graph import (
    INFERENCE_STATUS_INFERRED,
    StoryEdge,
    StoryEdgeType,
    StoryGraph,
)

#: REACTION 时长窗口（微秒）：0.5s ~ 8s。
_REACTION_MIN_US = 500_000
_REACTION_MAX_US = 8_000_000
#: REACTION blur 差异阈值。
_REACTION_BLUR_DIFF = 50.0
#: CONTRAST 触发阈值。
_CONTRAST_BRIGHTNESS_DIFF = 60.0
_CONTRAST_BLUR_DIFF = 100.0
_CONTRAST_SHAKE_RATIO = 3.0
#: MONTAGE 单镜头时长上限（微秒）：<1s。
_MONTAGE_MAX_US = 1_000_000
_MONTAGE_RUN_MIN = 3


def _parse_claim(claim: str) -> dict:
    """解析观测 claim JSON；失败或非 dict 返回空 dict。"""
    try:
        data = json.loads(claim)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _num(data: dict, key: str) -> float | None:
    v = data.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _reaction_edge(a: FilmObservation, b: FilmObservation, ma: dict, mb: dict) -> StoryEdge | None:
    """相邻镜头动作连续（blur 差异小且时长适中）。"""
    dur_a = a.end_frame - a.start_frame
    dur_b = b.end_frame - b.start_frame
    if not (_REACTION_MIN_US <= dur_a <= _REACTION_MAX_US
            and _REACTION_MIN_US <= dur_b <= _REACTION_MAX_US):
        return None
    blur_a = _num(ma, "blur_score")
    blur_b = _num(mb, "blur_score")
    if blur_a is None or blur_b is None:
        return None
    diff = abs(blur_a - blur_b)
    if diff >= _REACTION_BLUR_DIFF:
        return None
    # blur 差异越小置信度越高：0 -> 0.8，50 -> 0.6
    confidence = 0.8 - (diff / _REACTION_BLUR_DIFF) * 0.2
    return StoryEdge(
        edge_id=f"reaction_{a.media_asset_id}__to__{b.media_asset_id}",
        from_node=a.media_asset_id,
        to_node=b.media_asset_id,
        edge_type=StoryEdgeType.CAUSAL_CANDIDATE,
        inference_status=INFERENCE_STATUS_INFERRED,
        evidence_refs=[a.observation_id, b.observation_id],
        confidence=round(confidence, 3),
    )


def _contrast_edge(a: FilmObservation, b: FilmObservation, ma: dict, mb: dict) -> StoryEdge | None:
    """相邻镜头视觉指标突变（亮度/模糊/抖动任一）。"""
    magnitudes: list[float] = []

    brightness_a = _num(ma, "brightness_mean")
    brightness_b = _num(mb, "brightness_mean")
    if brightness_a is not None and brightness_b is not None:
        d = abs(brightness_a - brightness_b)
        if d > _CONTRAST_BRIGHTNESS_DIFF:
            # 阈值 60 -> 0，255 -> 1
            magnitudes.append(min((d - _CONTRAST_BRIGHTNESS_DIFF)
                                  / (255.0 - _CONTRAST_BRIGHTNESS_DIFF), 1.0))

    blur_a = _num(ma, "blur_score")
    blur_b = _num(mb, "blur_score")
    if blur_a is not None and blur_b is not None:
        d = abs(blur_a - blur_b)
        if d > _CONTRAST_BLUR_DIFF:
            magnitudes.append(min((d - _CONTRAST_BLUR_DIFF) / 900.0, 1.0))

    shake_a = _num(ma, "shake_score")
    shake_b = _num(mb, "shake_score")
    if (shake_a is not None and shake_b is not None
            and shake_a > 0 and shake_b > 0):
        ratio = max(shake_a, shake_b) / min(shake_a, shake_b)
        if ratio >= _CONTRAST_SHAKE_RATIO:
            # ratio 3 -> 0，10 -> 1
            magnitudes.append(min((ratio - _CONTRAST_SHAKE_RATIO) / 7.0, 1.0))

    if not magnitudes:
        return None
    confidence = 0.5 + max(magnitudes) * 0.4
    return StoryEdge(
        edge_id=f"contrast_{a.media_asset_id}__to__{b.media_asset_id}",
        from_node=a.media_asset_id,
        to_node=b.media_asset_id,
        edge_type=StoryEdgeType.EMOTIONAL_TURN,
        inference_status=INFERENCE_STATUS_INFERRED,
        evidence_refs=[a.observation_id, b.observation_id],
        confidence=round(confidence, 3),
    )


def _montage_edges(tech_sorted: list[FilmObservation]) -> list[StoryEdge]:
    """连续 3+ 个 <1s 短镜头 → 序列内每对相邻镜头一条 TEMPORAL 边。"""
    short = [(o.end_frame - o.start_frame) < _MONTAGE_MAX_US for o in tech_sorted]
    edges: list[StoryEdge] = []
    n = len(tech_sorted)
    i = 0
    while i < n:
        if not short[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and short[j + 1]:
            j += 1
        if (j - i + 1) >= _MONTAGE_RUN_MIN:
            for k in range(i, j):
                a = tech_sorted[k]
                b = tech_sorted[k + 1]
                edges.append(StoryEdge(
                    edge_id=f"montage_{a.media_asset_id}__to__{b.media_asset_id}",
                    from_node=a.media_asset_id,
                    to_node=b.media_asset_id,
                    edge_type=StoryEdgeType.TEMPORAL,
                    inference_status=INFERENCE_STATUS_INFERRED,
                    evidence_refs=[a.observation_id, b.observation_id],
                    confidence=0.7,
                ))
        i = j + 1
    return edges


def infer_relations(
    observations: list[FilmObservation],
    graph: StoryGraph,
) -> list[StoryEdge]:
    """从 deterministic_technical 观测推断镜头间叙事关系边。

    Args:
        observations: ``analyze_media`` 的输出（ASR 观测会被忽略）。
        graph: M2.1 产出的故事图（本函数不修改它，仅作上下文）。

    Returns:
        推断出的 :class:`StoryEdge` 列表；空观测或 <2 镜头时返回空列表。
    """
    tech = sorted(
        [o for o in observations if o.observation_type == "deterministic_technical"],
        key=lambda o: o.start_frame,
    )
    if len(tech) < 2:
        return []

    claims = [_parse_claim(o.claim) for o in tech]
    dedup: dict[tuple[str, str, StoryEdgeType], StoryEdge] = {}

    def _add(edge: StoryEdge | None) -> None:
        if edge is None:
            return
        key = (edge.from_node, edge.to_node, edge.edge_type)
        dedup.setdefault(key, edge)

    for i in range(len(tech) - 1):
        a, b = tech[i], tech[i + 1]
        ma, mb = claims[i], claims[i + 1]
        _add(_reaction_edge(a, b, ma, mb))
        _add(_contrast_edge(a, b, ma, mb))

    for edge in _montage_edges(tech):
        _add(edge)

    # ---- 消费 graph：四幕节点 attributes["shot_ids"] 建立 shot -> act 映射 ----
    shot_to_act_node: dict[str, str] = {}
    for node in graph.nodes:
        if node.attributes.get("act") is None:
            continue
        for sid in node.attributes.get("shot_ids", []):
            shot_to_act_node[sid] = node.node_id

    result: list[StoryEdge] = []
    for edge in dedup.values():
        act_from = shot_to_act_node.get(edge.from_node)
        act_to = shot_to_act_node.get(edge.to_node)
        confidence = edge.confidence
        evidence = list(edge.evidence_refs)

        if act_from and act_to and act_from == act_to:
            # 同幕内镜头：置信度上调，证据追加该幕节点
            confidence = min(confidence + 0.1, 0.95)
            evidence.append(act_from)
        elif act_from or act_to:
            # 跨幕镜头：置信度下调，证据追加涉及的幕节点
            confidence = max(confidence - 0.05, 0.3)
            for act_node in (act_from, act_to):
                if act_node and act_node not in evidence:
                    evidence.append(act_node)

        result.append(edge.model_copy(update={
            "confidence": round(confidence, 3),
            "evidence_refs": evidence,
        }))

    return result
