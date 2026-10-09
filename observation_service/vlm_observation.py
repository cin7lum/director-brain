"""VLM 结果 → FilmObservation 转换层与批量分析。

将 OllamaVLMAdapter 返回的扁平 dict 转换为 FilmObservation
（observation_type="vlm_semantic"），并支持批量镜头分析 + analysis_cache 缓存。

失败降级：VLM 调用失败（degraded=True 且 status=FAILED）时仍产出观测，
claim_kind=NOT_DETERMINED，claim 含失败原因，不中断整体批量流程。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from director_brain.utils import file_sha256, short_hash
from director_brain.analysis_cache import AnalysisCache, AnalysisCacheStore
from director_brain.config import load_settings
from observation_service.analysis_control import AnalysisCancelledError
from director_brain.models.film_observation import (
    ClaimKind,
    FILM_OBSERVATION_SCHEMA_VERSION,
    FilmObservation,
    TimebaseUnit,
)
from observation_service.keyframe import (
    MULTI_FRAME_SAMPLING_PROFILE,
    extract_keyframes,
)
from observation_service.ollama_vlm_adapter import (
    LocalVLMRuntimeBindingError,
    OllamaVLMAdapter,
    SEMANTIC_GENERATION_PROFILE,
    SEMANTIC_OBSERVATION_MAPPER_VERSION,
    SEMANTIC_PROMPT_SHA256,
    SEMANTIC_PROMPT_VERSION,
)
from observation_service.vlm_adapter import VLMAdapter

_PROVIDER = "ollama_qwen3_vl"
# v4：D2 人物外观字段（people）——实体聚类（entity_resolver）消费；
# 缓存键随版本失效
_PROMPT_VERSION = SEMANTIC_PROMPT_VERSION
_TIMEBASE_US = 1_000_000


def _claim_payload(vlm_result: dict) -> dict:
    """从 VLM 结果提取需要序列化到 claim 的语义字段。"""
    return {
        "shot_function": vlm_result.get("shot_function"),
        "motion_amount": vlm_result.get("motion_amount"),
        "proposed_role_v2": vlm_result.get("proposed_role_v2"),
        "frame_description": vlm_result.get("frame_description", ""),
        "sensory_wet_heat": vlm_result.get("sensory_wet_heat"),
        "sensory_mood_intensity": vlm_result.get("sensory_mood_intensity"),
        # 阶段 P3-2：TVSum 同构 importance 标注（1-5，打标数据集字段）
        "importance": vlm_result.get("importance"),
        # P3-1 深度语义（内核语义融合消费：narrative_role 驱动幕分配、
        # emotional_tone 匹配情绪弧、scene_description 供多样性降权）
        "scene_description": vlm_result.get("scene_description", ""),
        "subjects": vlm_result.get("subjects") or [],
        # D2：人物外观描述（entity_resolver 聚类消费）
        "people": vlm_result.get("people") or [],
        "action_type": vlm_result.get("action_type"),
        "emotional_tone": vlm_result.get("emotional_tone"),
        "narrative_role": vlm_result.get("narrative_role"),
        "visual_quality": vlm_result.get("visual_quality"),
        "motion_progression": vlm_result.get("motion_progression", ""),
        "temporal_notes": vlm_result.get("temporal_notes", ""),
    }


def _resolve_model_version(adapter) -> str:
    """从适配器获取 model_version（ollama 模型名补 :latest 后缀）。"""
    model = getattr(adapter, "model", "qwen3-vl")
    if ":" not in model:
        model = f"{model}:latest"
    digest = getattr(adapter, "model_digest", None)
    runtime_version = getattr(adapter, "runtime_version", None)
    if digest:
        model += f"@sha256:{digest.removeprefix('sha256:').lower()}"
    if runtime_version:
        model += f"|ollama:{runtime_version}"
    return model


def vlm_result_to_observation(
    vlm_result: dict,
    shot: dict,
    frame_path: str,
    *,
    model_version: str = "qwen3-vl:latest",
) -> FilmObservation:
    """将 VLM 适配器结果转换为 FilmObservation。

    Args:
        vlm_result: OllamaVLMAdapter.analyze_frame 返回的扁平 dict。
        shot: 镜头 dict（含 shot_id / source_in_us / source_out_us / source_media_hash）。
        frame_path: 关键帧路径（用作 source_ref；临时帧已清理时可传视频路径）。
        model_version: 模型版本标识，默认 ``qwen3-vl:latest``。

    Returns:
        FilmObservation，observation_type="vlm_semantic"。
        成功时 claim_kind=MODEL_OBSERVATION，confidence=0.7（SELF_REPORTED）；
        失败时 claim_kind=NOT_DETERMINED，confidence=0.3，claim 含失败原因。
    """
    shot_id = shot["shot_id"]
    is_failure = (
        vlm_result.get("degraded") is True
        and vlm_result.get("status") == "FAILED"
    )

    if is_failure:
        claim_kind = ClaimKind.NOT_DETERMINED
        confidence = 0.3
        payload = _claim_payload(vlm_result)
        payload["error"] = vlm_result.get("degrade_reason", "unknown vlm failure")
        payload["failure_type"] = vlm_result.get("failure_type")
    else:
        claim_kind = ClaimKind.MODEL_OBSERVATION
        if vlm_result.get("confidence_type") == "SELF_REPORTED":
            confidence = 0.7
        else:
            confidence = 0.5
        payload = _claim_payload(vlm_result)

    return FilmObservation(
        observation_id=f"vlm_{shot_id}",
        media_asset_id=shot_id,
        media_hash=shot["source_media_hash"],
        start_frame=int(shot["source_in_us"]),
        end_frame=int(shot["source_out_us"]),
        timebase=_TIMEBASE_US,
        timebase_unit=TimebaseUnit.MICROSECONDS,
        observation_type="vlm_semantic",
        claim=json.dumps(payload, ensure_ascii=False),
        provider=_PROVIDER,
        model_version=model_version,
        prompt_version=_PROMPT_VERSION,
        confidence=confidence,
        review_state="auto_generated",
        claim_kind=claim_kind,
        schema_version=FILM_OBSERVATION_SCHEMA_VERSION,
        project_id="unknown",
        created_at=int(time.time()),
        producer=_PROVIDER,
        source_ref=frame_path or "unknown",
    )


def _cache_fingerprint(
    video_hash: str, model_version: str, shot: dict
) -> str:
    """基于视频哈希 + 模型版本 + 镜头范围生成稳定缓存指纹。"""
    return short_hash(
        f"{video_hash}|{model_version}|{_PROMPT_VERSION}|"
        f"{SEMANTIC_PROMPT_SHA256}|{MULTI_FRAME_SAMPLING_PROFILE}|"
        f"{SEMANTIC_GENERATION_PROFILE}|"
        f"{SEMANTIC_OBSERVATION_MAPPER_VERSION}|"
        f"FilmObservation/{FILM_OBSERVATION_SCHEMA_VERSION}|"
        f"{shot['source_in_us']}|{shot['source_out_us']}|{_PROVIDER}"
    )


def _degraded_result(reason: str, failure_type: str = "UNKNOWN") -> dict:
    """构造与 OllamaVLMAdapter._degraded 兼容的失败 dict。"""
    return {
        "shot_function": "SENSORY_INSERT",
        "sensory_wet_heat": None,
        "sensory_mood_intensity": None,
        "motion_amount": "subtle",
        "proposed_role_v2": "broll",
        "frame_description": "",
        "status": "FAILED",
        "degraded": True,
        "confidence_type": "UNAVAILABLE",
        "failure_type": failure_type,
        "degrade_reason": reason,
        "_warnings": [],
    }


def batch_vlm_observations(
    video_path: str,
    shots: list[dict],
    adapter=None,
    cache: AnalysisCacheStore | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> list[FilmObservation]:
    """对每个镜头抽取中点关键帧并调用 VLM 分析，返回 FilmObservation 列表。

    流程：对每个镜头 ``with extract_keyframe(...)`` → ``adapter.analyze_frame``
    → ``vlm_result_to_observation``。用 analysis_cache 缓存（指纹基于视频 hash +
    模型版本 + 镜头范围），跳过已缓存镜头。失败镜头产出 NOT_DETERMINED 观测，
    不中断整体。批量结束后调用 ``cache.flush()`` 持久化。

    Args:
        video_path: 视频文件路径。
        shots: 镜头列表（discover_shots 的输出）。
        adapter: VLM 适配器；为 None 时从 config 读取创建 OllamaVLMAdapter。
        cache: AnalysisCache 实例；为 None 时创建默认缓存（视频同目录 vlm_cache/）。

    Returns:
        与 shots 等长的 FilmObservation 列表（observation_type="vlm_semantic"）。
    """
    if adapter is None:
        settings = load_settings()
        if settings.vlm_provider == "zhipu":
            # P1-c：配置诚实化——vlm_provider 配置此前是假的（硬编码 Ollama）
            from observation_service.zhipu_vlm_adapter import ZhipuVLMAdapter

            adapter = ZhipuVLMAdapter(
                api_key=settings.zhipu_api_key,
                model=settings.vlm_model,
            )
        else:
            adapter = OllamaVLMAdapter(
                model=settings.vlm_model,
                base_url=settings.ollama_base_url,
            )

    model_version = _resolve_model_version(adapter)

    if cache is None:
        cache_dir = Path(video_path).resolve().parent / "vlm_cache"
        cache = AnalysisCache(
            persist_path=str(cache_dir / "vlm_cache.json")
        )

    try:
        video_hash = file_sha256(video_path)
    except OSError:
        video_hash = "unknown"

    observations: list[FilmObservation] = []
    total = len(shots)

    if total == 0 and progress_callback is not None:
        progress_callback("semantic_analysis", 0, 0)

    for idx, shot in enumerate(shots):
        if cancellation_check is not None and cancellation_check():
            raise AnalysisCancelledError("analysis cancellation requested")
        shot_id = shot["shot_id"]
        fp = _cache_fingerprint(video_hash, model_version, shot)

        cache_key = f"{fp}|{_PROMPT_VERSION}"
        cached = cache.get(cache_key)
        if cached and all(
            item.claim_kind == ClaimKind.MODEL_OBSERVATION
            for item in cached
        ):
            print(f"[VLM {idx + 1}/{total}] cache hit  {shot_id}")
            observations.extend(cached)
            if progress_callback is not None:
                progress_callback("semantic_analysis", idx + 1, total)
            continue

        t0 = time.time()
        in_us = int(shot["source_in_us"])
        out_us = int(shot["source_out_us"])

        try:
            # P3-1（候选①收编）：镜头内 3 帧（15%/50%/85%）一次多图调用，
            # 抽帧统一走 keyframe.extract_keyframes（私有 cv2 已删除）。
            # 适配器未实现 analyze_frames 时退化为首帧单帧（响亮提示）。
            with extract_keyframes(video_path, in_us, out_us) as frame_paths:
                if not frame_paths:
                    vlm_result = _degraded_result(
                        "keyframe extraction failed", "DECODE"
                    )
                else:
                    if type(adapter).analyze_frames is VLMAdapter.analyze_frames:
                        print(
                            f"[VLM {idx + 1}/{total}] note: adapter 无多帧实现，"
                            f"退化为首帧单帧模式"
                        )
                    vlm_result = adapter.analyze_frames(frame_paths)
        except LocalVLMRuntimeBindingError:
            raise
        except Exception as exc:
            vlm_result = _degraded_result(
                f"batch error: {type(exc).__name__}: {exc}", "UNKNOWN"
            )

        cancelled_after_unit = (
            cancellation_check()
            if cancellation_check is not None else False
        )

        elapsed = time.time() - t0
        obs = vlm_result_to_observation(
            vlm_result, shot, video_path, model_version=model_version
        )
        observations.append(obs)
        # Failure observations are evidence that an attempt failed, not a
        # reusable analysis result. Keep them in the project snapshot but let
        # the next request retry this shot.
        if obs.claim_kind == ClaimKind.MODEL_OBSERVATION:
            cache.put(cache_key, [obs])

        status = "OK  " if obs.claim_kind == ClaimKind.MODEL_OBSERVATION else "FAIL"
        print(
            f"[VLM {idx + 1}/{total}] {status} {shot_id} "
            f"({elapsed:.1f}s) role={vlm_result.get('proposed_role_v2')} "
            f"fn={vlm_result.get('shot_function')}"
        )
        if progress_callback is not None:
            progress_callback("semantic_analysis", idx + 1, total)
        if cancelled_after_unit:
            raise AnalysisCancelledError("analysis cancellation requested")

    cache.flush()
    return observations
