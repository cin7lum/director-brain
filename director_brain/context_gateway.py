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
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Mapping

from director_brain.models.director_brief import DirectorBrief
from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_context import (
    AssetAnalysisState,
    ContextLayer,
    FilmContextSnapshot,
    ObservationTimebase,
    ProjectAssetCoverage,
)
from director_brain.models.film_observation import (
    FILM_OBSERVATION_SCHEMA_VERSION,
    FilmObservation,
    TimebaseUnit,
)
from director_brain.models.project import (
    FilmProjectManifest,
    ProjectAnalysisProfile,
    ProjectAsset,
    RightsState,
)
from director_brain.models.story_graph import StoryGraph
from director_brain.utils import file_sha256, short_hash
from director_brain.config import load_settings

_TIMEBASE_US = 1_000_000
PROJECT_CONTEXT_PROFILE = "local-deterministic-multi-asset-v1"
PROJECT_CONTEXT_SCHEMA_VERSION = "1.5"


def _file_hash_or_empty(video_path: str) -> str:
    """文件 sha256；不可读返回空串（四层 build_* 共用的样板收敛）。"""
    try:
        return file_sha256(video_path)
    except OSError:
        return ""


def project_local_asr_model_binding() -> tuple[Path, str, str, str]:
    """Resolve a content-addressed, locally installed project ASR runtime.

    The model must be a local directory. A bare model alias is rejected so a
    project request cannot trigger an implicit download from a model registry.
    """
    settings = load_settings()
    model_path = Path(settings.asr_model_path).expanduser()
    if not model_path.is_absolute():
        raise ValueError("project ASR requires an absolute local model directory")
    from observation_service.asr import local_model_tree_sha256

    try:
        digest = local_model_tree_sha256(model_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("project ASR local model artifact is unavailable") from exc

    from importlib.metadata import PackageNotFoundError, version

    try:
        runtime = ";".join(
            f"{distribution}={version(distribution)}"
            for distribution in ("faster-whisper", "ctranslate2", "av")
        )
    except PackageNotFoundError as exc:
        raise ValueError("project ASR runtime dependencies are not installed") from exc
    return model_path.resolve(strict=True), digest, model_path.name, runtime


def _probe_video_meta(video_path: str) -> dict[str, object]:
    """媒体元探测：统一走 media_info（候选⑤收编，带归因）。

    返回含 ``probe_ok`` / ``probe_reason`` 归因字段——探测失败**不得**
    被记成 fps=0 的"事实"（旧实现的归因裂缝，架构体检 ⑤ 指认）。
    """
    from observation_service.media_info import probe_media_meta

    meta = probe_media_meta(video_path, local_only=True)
    result: dict[str, object] = {
        "duration_us": meta.duration_us,
        "fps": meta.fps,
        "r_frame_rate": meta.r_frame_rate,
        "avg_frame_rate": meta.avg_frame_rate,
        "stream_time_base": meta.stream_time_base,
        "has_audio": meta.has_audio,
        "time_map": meta.time_map,
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
    units = {item.timebase_unit for item in observations}
    timebases = {item.timebase for item in observations}
    if not observations:
        timeline_scope = "single_asset"
        timebase = _TIMEBASE_US
        timebase_unit = TimebaseUnit.MICROSECONDS
    elif len(units) == 1 and TimebaseUnit.UNKNOWN not in units and len(timebases) == 1:
        timeline_scope = "single_asset"
        timebase = next(iter(timebases))
        timebase_unit = next(iter(units))
    else:
        timeline_scope = "mixed_or_unknown"
        timebase = None
        timebase_unit = None
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
        timeline_scope=timeline_scope,
        timebase=timebase,
        timebase_unit=timebase_unit,
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
        "r_frame_rate": meta.get("r_frame_rate"),
        "avg_frame_rate": meta.get("avg_frame_rate"),
        "stream_time_base": meta.get("stream_time_base"),
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


def project_analysis_profile(
    analysis_profile: ProjectAnalysisProfile | str = (
        ProjectAnalysisProfile.DETERMINISTIC_V1
    ),
) -> str:
    """Version string for the local analysis components that feed project context."""
    from observation_service.deterministic_analysis import (
        DETERMINISTIC_ANALYSIS_VERSION,
    )
    from observation_service.shot_discovery import (
        DISCOVERY_VERSION,
        SCENEDETECT_VERSION,
    )

    discovery_backend = (
        os.environ.get("SHOT_DISCOVERY_BACKEND") or "scenedetect"
    ).lower()
    base_profile = (
        f"{PROJECT_CONTEXT_PROFILE}|det={DETERMINISTIC_ANALYSIS_VERSION}|"
        f"shot_backend={discovery_backend}|legacy={DISCOVERY_VERSION}|"
        f"scenedetect={SCENEDETECT_VERSION}"
    )
    selected = ProjectAnalysisProfile(analysis_profile)
    if selected == ProjectAnalysisProfile.DETERMINISTIC_V1:
        return base_profile
    if selected == ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1:
        _model_path, digest, model_label, runtime = project_local_asr_model_binding()
        return (
            f"{base_profile}|profile={selected.value}|provider=faster_whisper|"
            f"model={model_label}@sha256:{digest}|runtime={runtime}|"
            "audio_clock=first_stream_offset_from_video_v1|"
            "offset_rounding=nearest_microsecond_ties_to_even|"
            f"observation_schema=FilmObservation/{FILM_OBSERVATION_SCHEMA_VERSION}|"
            "cache_policy=speech_failures_retryable_v1"
        )

    settings = load_settings()
    digest = (settings.project_local_vlm_digest or "").removeprefix(
        "sha256:").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("project local VLM requires a pinned 64-character model digest")
    runtime_version = (settings.project_local_vlm_runtime_version or "").strip()
    if not runtime_version:
        raise ValueError("project local VLM requires a pinned Ollama runtime version")
    model = settings.project_local_vlm_model.strip()
    if not model:
        raise ValueError("project local VLM requires a model tag")

    from observation_service.keyframe import MULTI_FRAME_SAMPLING_PROFILE
    from observation_service.ollama_vlm_adapter import (
        OllamaVLMAdapter,
        SEMANTIC_GENERATION_PROFILE,
        SEMANTIC_OBSERVATION_MAPPER_VERSION,
        SEMANTIC_PROMPT_SHA256,
        SEMANTIC_PROMPT_VERSION,
    )

    # Validate the destination without making a network request. Inference
    # verifies the pinned runtime and model blob immediately before first use.
    try:
        OllamaVLMAdapter(
            model=model,
            base_url=settings.ollama_base_url,
            model_digest=digest,
            runtime_version=runtime_version,
            enforce_loopback=True,
        )
    except RuntimeError as exc:
        raise ValueError("project local VLM must use a loopback Ollama URL") from exc
    return (
        f"{base_profile}|profile={selected.value}|provider=ollama_qwen3_vl|"
        f"model={model}@sha256:{digest}|ollama={runtime_version}|"
        f"prompt={SEMANTIC_PROMPT_VERSION}:{SEMANTIC_PROMPT_SHA256}|"
        f"sampling={MULTI_FRAME_SAMPLING_PROFILE}|"
        f"generation={SEMANTIC_GENERATION_PROFILE}|"
        f"observation_map={SEMANTIC_OBSERVATION_MAPPER_VERSION}|"
        f"observation_schema=FilmObservation/{FILM_OBSERVATION_SCHEMA_VERSION}|"
        "cache_policy=semantic_failures_retryable_v1"
    )


def project_context_id(manifest: FilmProjectManifest) -> str:
    """Stable id scoped to an immutable manifest revision and analysis profile."""
    return "ctx_project_" + short_hash(
        f"{manifest.manifest_id}|{PROJECT_CONTEXT_SCHEMA_VERSION}|"
        f"{project_analysis_profile(manifest.analysis_profile)}"
    )


def project_asset_analysis_fingerprint(
    asset: ProjectAsset,
    analysis_profile: ProjectAnalysisProfile | str = (
        ProjectAnalysisProfile.DETERMINISTIC_V1
    ),
) -> str:
    """Exact local analysis cache key; content/profile/timebase are all bound."""
    from director_brain.analysis_cache import compute_fingerprint

    if asset.source_content_hash is None:
        raise ValueError("inventory-only assets do not have an analysis fingerprint")
    profile_id = project_analysis_profile(analysis_profile)
    selected = ProjectAnalysisProfile(analysis_profile)
    return compute_fingerprint(
        source_content_hash=asset.source_content_hash.lower(),
        provider=(
            "local_deterministic_pipeline"
            if selected == ProjectAnalysisProfile.DETERMINISTIC_V1
            else (
                "local_deterministic_plus_faster_whisper"
                if selected == ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1
                else "local_deterministic_plus_ollama_vlm"
            )
        ),
        model_version=profile_id,
        prompt_version=(
            "n/a"
            if selected in (
                ProjectAnalysisProfile.DETERMINISTIC_V1,
                ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1,
            )
            else "|".join(profile_id.split("|prompt=", 1)[1].split("|", 1)[:1])
        ),
        sampling_config={
            "pipeline": (
                "analyze_media"
                if selected == ProjectAnalysisProfile.DETERMINISTIC_V1
                else (
                    "analyze_media_with_project_asr"
                    if selected == ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1
                    else "analyze_media_full_local_vlm"
                )
            ),
            "shot_discovery_backend": (
                os.environ.get("SHOT_DISCOVERY_BACKEND") or "scenedetect"
            ).lower(),
            "analysis_profile": selected.value,
        },
        timebase=_TIMEBASE_US,
        schema_version=f"FilmObservation/{FILM_OBSERVATION_SCHEMA_VERSION}",
    )


def build_multi_asset_project_context(
    manifest: FilmProjectManifest,
    observations_by_asset: Mapping[str, list[FilmObservation]],
    analysis_cache_state_by_asset: Mapping[str, str] | None = None,
    analysis_state_by_asset: Mapping[str, AssetAnalysisState] | None = None,
    failure_code_by_asset: Mapping[str, str] | None = None,
    blocked_reason_by_asset: Mapping[str, str] | None = None,
) -> FilmContextSnapshot:
    """Build an evidence-linked project summary without merging source timelines."""
    expected_ids = [asset.asset_id for asset in manifest.assets]
    if set(observations_by_asset) != set(expected_ids):
        raise ValueError("observations must be supplied for every manifest asset exactly once")

    ordered_assets = sorted(manifest.assets, key=lambda asset: asset.order)
    asset_coverage: list[ProjectAssetCoverage] = []
    all_observations: list[FilmObservation] = []
    evidence_refs: list[str] = []
    for asset in ordered_assets:
        observations = observations_by_asset[asset.asset_id]
        if observations and asset.source_content_hash is None:
            raise ValueError(
                f"asset {asset.asset_id} has observations without a verified source hash")
        if (asset.source_content_hash is not None and any(
            obs.media_hash.lower() != asset.source_content_hash.lower()
            for obs in observations
        )):
            raise ValueError(
                f"observation source hash does not match asset {asset.asset_id}")
        if any(obs.project_id != manifest.project_id for obs in observations):
            raise ValueError(
                f"observation project_id does not match {manifest.project_id}")
        if any(obs.project_asset_id != asset.asset_id for obs in observations):
            raise ValueError(
                f"observation project_asset_id does not match {asset.asset_id}")
        ids = [obs.observation_id for obs in observations]
        if any(obs.timebase_unit == TimebaseUnit.UNKNOWN for obs in observations):
            raise ValueError(
                f"asset {asset.asset_id} contains observations with unknown timebase units")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate observation_id in asset {asset.asset_id}")
        analysis_state = (
            analysis_state_by_asset.get(asset.asset_id)
            if analysis_state_by_asset else None
        ) or ProjectAssetCoverage.derive_state(
            len(observations),
            provided=(analysis_cache_state_by_asset or {}).get(asset.asset_id) == "provided",
        )
        failure_code = (failure_code_by_asset or {}).get(asset.asset_id)
        blocked_reason = (blocked_reason_by_asset or {}).get(asset.asset_id)
        if failure_code and observations and not (
            analysis_state == AssetAnalysisState.PARTIAL
            and failure_code in (
                "semantic_analysis_partial", "speech_analysis_partial")
        ):
            raise ValueError(
                f"asset {asset.asset_id} cannot report this failure with observations")
        asset_coverage.append(ProjectAssetCoverage(
            asset_id=asset.asset_id,
            order=asset.order,
            source_content_hash=(
                asset.source_content_hash.lower()
                if asset.source_content_hash is not None else None
            ),
            duration_us=asset.duration_us,
            fps=asset.fps,
            r_frame_rate=asset.r_frame_rate,
            avg_frame_rate=asset.avg_frame_rate,
            stream_time_base=asset.stream_time_base,
            has_audio=asset.has_audio,
            time_map=asset.time_map,
            probe_ok=asset.probe_ok,
            observation_count=len(observations),
            evidence_refs=ids,
            observation_timebases=[
                ObservationTimebase(value=value, unit=unit)
                for value, unit in sorted({
                    (obs.timebase, obs.timebase_unit)
                    for obs in observations
                }, key=lambda item: (item[0], item[1].value))
            ],
            analysis_state=analysis_state,
            analysis_cache_state=(
                (analysis_cache_state_by_asset or {}).get(asset.asset_id, "unspecified")
            ),
            failure_code=failure_code,
            blocked_reason=blocked_reason,
            rights_state=asset.rights.state.value,
            rights_evidence_state=asset.rights.evidence_state,
        ))
        all_observations.extend(observations)
        evidence_refs.extend(ids)

    duplicate_evidence = len(evidence_refs) != len(set(evidence_refs))
    if duplicate_evidence:
        raise ValueError("observation_id values must be unique across project assets")
    local_processing_claimed = all(
        asset.rights.state == RightsState.LOCAL_PROCESSING_ALLOWED
        for asset in ordered_assets
    )
    total_observations = len(all_observations)
    incomplete_states = {
        AssetAnalysisState.PARTIAL,
        AssetAnalysisState.FAILED,
        AssetAnalysisState.SOURCE_UNAVAILABLE,
        AssetAnalysisState.SOURCE_CHANGED,
        AssetAnalysisState.NOT_AUTHORIZED,
    }
    has_incomplete_asset = any(
        entry.analysis_state in incomplete_states for entry in asset_coverage
    )
    if all(
        entry.analysis_state == AssetAnalysisState.NOT_AUTHORIZED
        for entry in asset_coverage
    ):
        coverage = (
            "project_single_asset_inventory_only"
            if len(ordered_assets) == 1 else "project_inventory_only"
        )
    elif has_incomplete_asset:
        coverage = (
            "project_single_asset_partial"
            if len(ordered_assets) == 1 else "project_multi_asset_partial"
        )
    elif total_observations == 0:
        coverage = (
            "project_single_asset_analyzed_empty"
            if len(ordered_assets) == 1 else "project_multi_asset_analyzed_empty"
        )
    elif len(ordered_assets) == 1:
        coverage = "project_single_asset_observed"
    else:
        coverage = "project_multi_asset_observed"

    return FilmContextSnapshot(
        context_id=project_context_id(manifest),
        project_manifest_id=manifest.manifest_id,
        project_revision=manifest.revision,
        asset_refs=[asset.asset_id for asset in ordered_assets],
        source_content_hashes=[
            asset.source_content_hash.lower()
            if asset.source_content_hash is not None else None
            for asset in ordered_assets
        ],
        layers=[ContextLayer.PROJECT],
        analysis_fingerprint=short_hash(
            PROJECT_CONTEXT_SCHEMA_VERSION + "|" + manifest.manifest_id + "|"
            + project_analysis_profile(
                manifest.analysis_profile) + "|" + "|".join(
                f"{asset.asset_id}:{asset.source_content_hash or 'unverified'}:"
                f"{(analysis_state_by_asset or {}).get(asset.asset_id, AssetAnalysisState.OBSERVED).value}:"
                f"{(failure_code_by_asset or {}).get(asset.asset_id, '')}:"
                f"{(blocked_reason_by_asset or {}).get(asset.asset_id, '')}:"
                f"{','.join(o.observation_id for o in observations_by_asset[asset.asset_id])}"
                for asset in ordered_assets
            )
        ),
        provider=(
            "none"
            if all(entry.analysis_state == AssetAnalysisState.NOT_AUTHORIZED
                   for entry in asset_coverage)
            else (
                "deterministic_opencv"
                if manifest.analysis_profile == ProjectAnalysisProfile.DETERMINISTIC_V1
                else (
                    "deterministic_opencv+faster_whisper"
                    if manifest.analysis_profile
                    == ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1
                    else "deterministic_opencv+ollama_qwen3_vl"
                )
            )
        ),
        model=(
            "manifest_only"
            if all(entry.analysis_state == AssetAnalysisState.NOT_AUTHORIZED
                   for entry in asset_coverage)
            else project_analysis_profile(manifest.analysis_profile)
        ),
        prompt_version=(
            "n/a"
            if manifest.analysis_profile in (
                ProjectAnalysisProfile.DETERMINISTIC_V1,
                ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1,
            )
            else project_analysis_profile(manifest.analysis_profile).split(
                "|prompt=", 1)[1].split("|", 1)[0]
        ),
        sampling_config={
            "manifest_boundary_state": manifest.boundary_state,
            "boundary_basis": manifest.boundary_basis.value,
            "boundary_source_ref": manifest.boundary_source_ref,
            "boundary_evidence_refs": manifest.boundary_evidence_refs,
            "asset_count": len(ordered_assets),
            "observation_count": total_observations,
            "observation_scope": "per_asset_preserving_explicit_timebase_units",
            "analysis_profile": (
                None
                if all(entry.analysis_state == AssetAnalysisState.NOT_AUTHORIZED
                       for entry in asset_coverage)
                else project_analysis_profile(manifest.analysis_profile)
            ),
        },
        timeline_scope="project_per_asset",
        timebase=None,
        timebase_unit=None,
        asset_coverage=asset_coverage,
        coverage=coverage,
        rights_scope=(
            "local_processing_claimed_by_manifest"
            if local_processing_claimed else "unverified"
        ),
        evidence_refs=evidence_refs,
        cache_state="fresh",
        invalidated_at=None,
        schema_version=PROJECT_CONTEXT_SCHEMA_VERSION,
        project_id=manifest.project_id,
        created_at=int(time.time()),
        producer="context_gateway",
        source_ref=manifest.manifest_id,
    )


# ---------------------------------------------------------------------------
# SCENE 层（新增）
# ---------------------------------------------------------------------------

def validate_window_us(
    observations: list[FilmObservation], window_us: tuple[int, int]
) -> None:
    """Fail closed unless a microsecond window has one verified source clock."""
    if len(window_us) != 2 or window_us[0] >= window_us[1]:
        raise ValueError("window_us must be an increasing [start_us, end_us] pair")
    if not observations:
        raise ValueError("window_us cannot be verified without observations")
    if any(o.timebase_unit != TimebaseUnit.MICROSECONDS for o in observations):
        raise ValueError(
            "window_us requires every observation to declare microseconds"
        )
    if len({o.project_asset_id for o in observations}) > 1:
        raise ValueError("window_us cannot combine multiple project assets")
    if len({(o.timebase, o.timebase_unit) for o in observations}) > 1:
        raise ValueError("window_us requires one consistent source timebase")


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
    if window_us is not None:
        validate_window_us(observations, window_us)
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
