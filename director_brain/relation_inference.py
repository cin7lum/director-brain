"""M2.2 Relation Inference：从确定性技术指标推断镜头间关系边。

迁移 GEN-1 Tier-A 规则到确定性实现（当前无 VLM 标签，用 OpenCV 技术指标替代）：

- REACTION：相邻镜头 blur 差异小（|Δblur|<50）且两镜头时长均在
  0.5s-8s → 动作连续，edge_type=CAUSAL_CANDIDATE。
- CONTRAST：相邻镜头 brightness 突变（|Δbrightness|>60）或 blur 突变
  （|Δblur|>100）或 shake 相对比值 >=3 → 视觉突变，edge_type=VISUAL_TRANSITION。
  （T4 正名：技术指标突变在电影语义上不等于情绪转折——亮度突变可能只是
  室内外切换。EMOTIONAL_TURN 保留在枚举中，供未来真正的语义证据
  （VLM 情绪观测/用户意图）使用，技术信号一律不得直推情绪语义。）
- MONTAGE：连续 3+ 个镜头每个时长 <1s → 快速蒙太奇，对序列内每对相邻镜头
  加一条 TEMPORAL 边。

边界：只处理 deterministic_technical 观测；空观测或 <2 个镜头返回空列表，
不抛异常。同一对 (from, to, 关系类型) 只产出一条边。

T4 通路状态：本模块产出为 SHADOW 信号——主链（roughcut）不将其并入
故事图、不传给 reasoner，只做全量上报（逐条边，不只计数）。
"""
from __future__ import annotations

import json

from director_brain.models.film_observation import (
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)
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

#: VLM 语义规则：SEMANTIC_CONTINUITY 基础置信度（同幕/跨幕后续统一调整）。
_VLM_CONTINUITY_CONFIDENCE = 0.75
#: VLM 语义规则：SEMANTIC_CONTRAST 基础置信度。
_VLM_CONTRAST_CONFIDENCE = 0.65
#: VLM 语义规则：高对比 shot_function 相邻对（有序，与相邻镜头方向一致）。
_VLM_CONTRAST_PAIRS = frozenset({
    ("ESTABLISHING", "ACTION"),
    ("ESTABLISHING", "REACTION"),
    ("ACTION", "DETAIL"),
    ("DETAIL", "ACTION"),
    ("ATMOSPHERIC_EVIDENCE", "ACTION"),
    ("ACTION", "ATMOSPHERIC_EVIDENCE"),
})


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
        edge_type=StoryEdgeType.VISUAL_TRANSITION,
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


def _vlm_semantic_edges(
    tech_sorted: list[FilmObservation],
    vlm_by_shot: dict[str, FilmObservation],
) -> list[StoryEdge]:
    """基于 VLM 语义观测推断相邻镜头关系边。

    只对 ``tech_sorted`` 中的相邻镜头对生效，且双方都必须有成功的
    VLM 观测（``claim_kind=MODEL_OBSERVATION``）。

    - SEMANTIC_CONTINUITY：两镜头 ``proposed_role_v2`` 相同 → CAUSAL_CANDIDATE。
    - SEMANTIC_CONTRAST：两镜头 ``shot_function`` 落在高对比集合 → VISUAL_TRANSITION
      （镜头功能对比属风格信号，仍非情绪证据）。
    - DISCARD_FILTER：任一镜头 role=discard，跳过该配对。
    """
    edges: list[StoryEdge] = []
    for i in range(len(tech_sorted) - 1):
        a = tech_sorted[i]
        b = tech_sorted[i + 1]
        va = vlm_by_shot.get(a.media_asset_id)
        vb = vlm_by_shot.get(b.media_asset_id)
        if va is None or vb is None:
            continue

        ca = _parse_claim(va.claim)
        cb = _parse_claim(vb.claim)
        role_a = ca.get("proposed_role_v2")
        role_b = cb.get("proposed_role_v2")

        # DISCARD_FILTER：涉及 discard 镜头的配对不生成任何 VLM 语义边
        if role_a == "discard" or role_b == "discard":
            continue

        # SEMANTIC_CONTINUITY：相邻镜头 role 相同
        if role_a and role_a == role_b:
            edges.append(StoryEdge(
                edge_id=f"semantic_continuity_{a.media_asset_id}__to__{b.media_asset_id}",
                from_node=a.media_asset_id,
                to_node=b.media_asset_id,
                edge_type=StoryEdgeType.CAUSAL_CANDIDATE,
                inference_status=INFERENCE_STATUS_INFERRED,
                evidence_refs=[va.observation_id, vb.observation_id],
                confidence=_VLM_CONTINUITY_CONFIDENCE,
            ))

        # SEMANTIC_CONTRAST：相邻镜头 shot_function 高对比对
        func_pair = (ca.get("shot_function"), cb.get("shot_function"))
        if func_pair in _VLM_CONTRAST_PAIRS:
            edges.append(StoryEdge(
                edge_id=f"semantic_contrast_{a.media_asset_id}__to__{b.media_asset_id}",
                from_node=a.media_asset_id,
                to_node=b.media_asset_id,
                edge_type=StoryEdgeType.VISUAL_TRANSITION,
                inference_status=INFERENCE_STATUS_INFERRED,
                evidence_refs=[va.observation_id, vb.observation_id],
                confidence=_VLM_CONTRAST_CONFIDENCE,
            ))
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

    vlm_obs = [
        o for o in observations
        if o.observation_type == "vlm_semantic"
        and o.claim_kind == ClaimKind.MODEL_OBSERVATION
    ]
    scoped_observations = [*tech, *vlm_obs]
    project_asset_ids = {
        observation.project_asset_id for observation in scoped_observations
    }
    if len(project_asset_ids) > 1:
        if None in project_asset_ids:
            raise ValueError(
                "relation inference cannot mix bound and unbound observations"
            )
        raise ValueError(
            "relation inference requires one project_asset_id; "
            "cross-asset inference is not admitted"
        )
    project_asset_id = next(iter(project_asset_ids), None)
    if graph.project_asset_id != project_asset_id:
        raise ValueError("relation inference observations do not match graph asset scope")

    timebases = {(o.timebase, o.timebase_unit) for o in tech}
    if len(timebases) != 1:
        raise ValueError("relation inference requires one consistent source timebase")
    timebase, timebase_unit = next(iter(timebases))
    if timebase_unit != TimebaseUnit.MICROSECONDS:
        raise ValueError(
            "relation inference duration rules require microsecond observations"
        )
    if graph.timebase != timebase or graph.timebase_unit != timebase_unit:
        raise ValueError("relation inference observations do not match graph timebase")

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

    # ---- VLM 语义规则：仅当存在成功的 vlm_semantic 观测时生效 ----
    # 没有 VLM 观测时该块整体跳过，函数行为与纯确定性规则一致（向后兼容）。
    if vlm_obs:
        vlm_by_shot = {o.media_asset_id: o for o in vlm_obs}
        for edge in _vlm_semantic_edges(tech, vlm_by_shot):
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
