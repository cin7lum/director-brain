"""Film Context Gateway：四层渐进披露（PROJECT / ASSET / SCENE / EVIDENCE）。

方案 §4.2 要求逐层按需展开，不做全量媒体倾倒。本模块实现四层构建与
``ContextGateway`` 管理器：

- **PROJECT**（顶层摘要）：Brief 概要、授权/隐私约束、当前 plan 版本、素材覆盖
- **ASSET**：资产 hash、时长、帧率、音轨、镜头索引与分析来源（已有）
- **SCENE**：时间窗内的相邻镜头、候选池、故事图证据
- **EVIDENCE**：原始关键帧、波形、邻接镜头（仅歧义/高风险决策展开）

渐进披露规则：上层引用下层 evidence_refs；向下展开由调用方按需触发；
EVIDENCE 层展开必须带 ``reason``（审计留痕）。
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from director_brain.models.director_brief import DirectorBrief
from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_context import ContextLayer, FilmContextSnapshot
from director_brain.models.film_observation import FilmObservation
from director_brain.models.story_graph import StoryGraph
from director_brain.utils import file_sha256, short_hash
from director_brain.config import load_settings

_TIMEBASE_US = 1_000_000


def _file_hash_or_empty(video_path: str) -> str:
    """文件 sha256；不可读返回空串（四层 build_* 共用的样板收敛）。"""
    try:
        return file_sha256(video_path)
    except OSError:
        return ""


def _probe_video_meta(video_path: str) -> dict[str, object]:
    """媒体元探测：统一走 media_info（候选⑤收编，带归因）。

    返回含 ``probe_ok`` / ``probe_reason`` 归因字段——探测失败**不得**
    被记成 fps=0 的"事实"（旧实现的归因裂缝，架构体检 ⑤ 指认）。
    """
    from observation_service.media_info import probe_media_meta

    meta = probe_media_meta(video_path)
    result: dict[str, object] = {
        "duration_us": meta.duration_us,
        "fps": meta.fps,
        "has_audio": meta.has_audio,
        "probe_ok": meta.ok,
    }
    if not meta.ok:
        result["probe_reason"] = meta.reason
    return result


def _snapshot(
    layer: ContextLayer,
    context_id: str,
    video_path: str,
    file_hash: str,
    observations: list[FilmObservation],
    sampling_config: dict,
    evidence_refs: list[str],
    coverage: str,
) -> FilmContextSnapshot:
    """统一快照构造（公共字段齐备）。"""
    return FilmContextSnapshot(
        context_id=context_id,
        asset_refs=[video_path],
        source_content_hashes=[file_hash] if file_hash else [],
        layers=[layer],
        analysis_fingerprint=short_hash(
            video_path + "|" + "|".join(sorted(evidence_refs))
        ),
        provider="context_gateway",
        model=f"{layer.value}_context_v1",
        prompt_version="n/a",
        sampling_config=sampling_config,
        timebase=_TIMEBASE_US,
        coverage=coverage,
        rights_scope="internal",
        evidence_refs=evidence_refs,
        cache_state="fresh",
        invalidated_at=None,
        schema_version="1.0",
        project_id="unknown",
        created_at=int(time.time()),
        producer="context_gateway",
        source_ref=video_path,
    )


# ---------------------------------------------------------------------------
# ASSET 层（已有实现，保留）
# ---------------------------------------------------------------------------

def build_asset_context(
    video_path: str,
    observations: list[FilmObservation],
) -> FilmContextSnapshot:
    """ASSET 层：资产物理属性 + 分析索引（原 build_asset_index 重命名）。"""
    obs_ids = sorted(o.observation_id for o in observations)
    file_hash = _file_hash_or_empty(video_path)
    meta = _probe_video_meta(video_path)
    asset_cfg: dict = {
        "observation_count": len(observations),
        "video_duration_us": meta["duration_us"],
        "fps": meta["fps"],
        "has_audio": meta["has_audio"],
        "probe_ok": meta["probe_ok"],
        "shot_count": len({o.media_asset_id for o in observations}),
    }
    if not meta["probe_ok"]:
        # 归因裂缝修补：探测失败必须可见（旧实现静默记 fps=0 当"事实"）
        asset_cfg["probe_reason"] = meta.get("probe_reason")
    return _snapshot(
        ContextLayer.ASSET,
        "ctx_asset_" + short_hash(video_path + "|" + "|".join(obs_ids)),
        video_path, file_hash, observations,
        asset_cfg,
        [o.observation_id for o in observations],
        "asset_level",
    )


# ---------------------------------------------------------------------------
# PROJECT 层（新增）
# ---------------------------------------------------------------------------

def build_project_context(
    brief: DirectorBrief,
    video_path: str,
    observations: list[FilmObservation],
) -> FilmContextSnapshot:
    """PROJECT 层：Brief 概要 + 授权约束 + 素材覆盖概况。最高层摘要，最小披露。"""
    file_hash = _file_hash_or_empty(video_path)
    ctx_id = "ctx_project_" + short_hash(brief.brief_id + video_path)
    return _snapshot(
        ContextLayer.PROJECT,
        ctx_id, video_path, file_hash, observations,
        {
            "brief_id": brief.brief_id,
            "brief_version": brief.version,
            "intent": brief.intent,
            "target_duration_us": brief.target_duration,
            "language": brief.language,
            "delivery_profile": brief.delivery_profile,
            "must_include_count": len(brief.must_include),
            "must_avoid_count": len(brief.must_avoid),
            "privacy_constraints": brief.privacy_constraints,
            "approval_state": brief.approval_state,
        },
        [brief.brief_id],
        "project_summary",
    )


# ---------------------------------------------------------------------------
# SCENE 层（新增）
# ---------------------------------------------------------------------------

def build_scene_context(
    video_path: str,
    observations: list[FilmObservation],
    graph: "StoryGraph",
    edl: "EditorialDecisionList",
    window_us: tuple[int, int] | None = None,
) -> FilmContextSnapshot:
    """SCENE 层：时间窗内的相邻镜头、候选池、故事图证据。

    window_us: (start_us, end_us) 感兴趣的时间窗；None = 全片。
    """
    file_hash = _file_hash_or_empty(video_path)

    in_scope = []
    for o in observations:
        if window_us and (o.end_frame <= window_us[0] or o.start_frame >= window_us[1]):
            continue
        in_scope.append(o)

    scene_edges = [
        e for e in graph.edges
        if e.edge_type.value in ("temporal", "causal_candidate", "visual_transition")
    ]
    acts = [
        n for n in graph.nodes if n.attributes.get("act")
    ]

    ctx_id = "ctx_scene_" + short_hash(
        video_path + "|" + str(window_us) + "|" + graph.graph_id
    )
    return _snapshot(
        ContextLayer.SCENE,
        ctx_id, video_path, file_hash, in_scope,
        {
            "window_us": list(window_us) if window_us else None,
            "act_count": len(acts),
            "act_labels": [n.attributes["act"] for n in acts],
            "graph_edges_in_scope": len(scene_edges),
            "shots_in_scope": len({o.media_asset_id for o in in_scope}),
            "selected_shot_count": len(edl.ordered_edits),
        },
        [o.observation_id for o in in_scope] + [graph.graph_id],
        "scene_context",
    )


# ---------------------------------------------------------------------------
# EVIDENCE 层（新增——仅歧义/高风险决策展开，必须带 reason）
# ---------------------------------------------------------------------------

def build_evidence_context(
    video_path: str,
    observations: list[FilmObservation],
    target_shot_ids: list[str],
    reason: str,
) -> FilmContextSnapshot:
    """EVIDENCE 层：指定镜头的原始证据（关键帧路径、波形引用、邻接镜头）。

    必须传 reason（审计留痕）——方案要求 EVIDENCE 层只在歧义或高风险
    决策时按需展开，且记录展开理由。
    """
    if not reason or not reason.strip():
        raise ValueError("EVIDENCE 层展开必须提供 reason（审计留痕）")

    file_hash = _file_hash_or_empty(video_path)

    target_obs = [o for o in observations if o.media_asset_id in target_shot_ids]
    ctx_id = "ctx_evidence_" + short_hash(
        video_path + "|" + "|".join(sorted(target_shot_ids)) + "|" + reason
    )
    return _snapshot(
        ContextLayer.EVIDENCE,
        ctx_id, video_path, file_hash, target_obs,
        {
            "target_shot_ids": sorted(target_shot_ids),
            "expansion_reason": reason,
            "frame_count": len(target_obs),
        },
        [o.observation_id for o in target_obs],
        "evidence_window",
    )


def get_asset_context(context_id: str, repository) -> FilmContextSnapshot | None:
    """从存储层读取指定 context_id 的快照；不存在返回 None。"""
    from director_brain.models.film_context import FilmContextSnapshot as _FCS
    return repository.get(_FCS, context_id)


# ---------------------------------------------------------------------------
# ContextGateway 管理器（渐进披露入口）
# ---------------------------------------------------------------------------

class ContextGateway:
    """四层 Film Context 渐进披露管理器。

    用法：
        gw = ContextGateway(video_path, observations, brief=brief)
        project_snap = gw.project()             # 顶层摘要
        asset_snap = gw.asset()                  # 展开到资产层
        scene_snap = gw.scene(graph=graph, edl=edl)  # 展开到场景层
        ev_snap = gw.evidence(["shot_001"], reason="歧义镜头复核")  # 按需
    """

    def __init__(
        self,
        video_path: str,
        observations: list[FilmObservation],
        brief: DirectorBrief | None = None,
        graph: "StoryGraph | None" = None,
        edl: "EditorialDecisionList | None" = None,
    ):
        self._video = video_path
        self._observations = observations
        self._brief = brief
        self._graph = graph
        self._edl = edl
        self._cache: dict[str, FilmContextSnapshot] = {}

    def project(self) -> FilmContextSnapshot:
        if "project" not in self._cache and self._brief:
            self._cache["project"] = build_project_context(
                self._brief, self._video, self._observations)
        return self._cache["project"]

    def asset(self) -> FilmContextSnapshot:
        if "asset" not in self._cache:
            self._cache["asset"] = build_asset_context(
                self._video, self._observations)
        return self._cache["asset"]

    def scene(self, window_us: tuple[int, int] | None = None) -> FilmContextSnapshot:
        key = f"scene_{window_us}"
        if key not in self._cache and self._graph and self._edl:
            self._cache[key] = build_scene_context(
                self._video, self._observations, self._graph, self._edl, window_us)
        return self._cache[key]

    def evidence(self, shot_ids: list[str], reason: str) -> FilmContextSnapshot:
        key = f"evidence_{'|'.join(sorted(shot_ids))}_{reason}"
        if key not in self._cache:
            self._cache[key] = build_evidence_context(
                self._video, self._observations, shot_ids, reason)
        return self._cache[key]

    def status(self) -> dict:
        """当前已展开的层与缓存状态。"""
        return {
            "layers_expanded": list(self._cache.keys()),
            "total_snapshots": len(self._cache),
        }
