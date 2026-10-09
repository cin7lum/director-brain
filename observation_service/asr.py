"""M1.3 离线 ASR 语音转写：faster-whisper → FilmObservation。

用 CTranslate2 后端的 faster-whisper 对视频做语音转写，每个带时间戳的
segment 映射成一条 :class:`~director_brain.models.film_observation.FilmObservation`。

保持 **fail-soft** 的返回约定（任何失败都返回空列表，绝不向上抛异常），但
不再静默吞错：faster_whisper import 失败、模型权重缺失/加载失败、转写运行期
异常都会通过 ``logging.warning`` 输出原因，便于区分「模型不可用」与「视频无
语音」。视频无语音 / 转写结果为空属于正常空结果，不打 warning。时间以微秒为
整数帧存储，``timebase=1_000_000``。

P2-a 设备策略：``ASR_DEVICE`` 环境变量 = ``auto``（默认，先试 GPU
cuda/float16，失败响亮回退 CPU int8）/ ``cuda`` / ``cpu``。GPU 需要
CUDA 运行库（cublas/cudnn，见 .env.example 注释的安装配方）；缺库时
auto 会自动落到 CPU，不阻断转写。
"""
from __future__ import annotations

import logging
import os
import time
import hashlib
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from director_brain.utils import file_sha256
from director_brain.config import load_settings
from director_brain.models.film_observation import (
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)

logger = logging.getLogger(__name__)

#: ASR 时间基：微秒。faster-whisper 的 segment.start/end 为秒，乘此系数得整数微秒。
_TIMEBASE_US = 1_000_000

#: ASR 默认置信度（faster-whisper 不直接给出归一化置信度，这里给一个保守默认值）。
_DEFAULT_CONFIDENCE = 0.8

#: 视频文件不可读（mock 测试 / 文件缺失）时 media_hash 的回退占位符。
_MEDIA_HASH_PLACEHOLDER = "asr_placeholder"

_MODEL_TREE_DIGEST_CACHE: dict[
    str, tuple[tuple[tuple[str, int, int], ...], str]
] = {}
_MODEL_TREE_DIGEST_LOCK = threading.Lock()


@dataclass(frozen=True)
class ASRTranscriptionResult:
    """Outcome that distinguishes valid empty speech from provider failure."""

    status: Literal["observed", "completed_empty", "failed"]
    observations: list[FilmObservation]
    failure_code: str | None = None


def local_model_tree_sha256(model_path: str | Path) -> str:
    """Hash a local ASR model tree, caching only while its file stat set is stable.

    The path must already exist locally. This function never resolves or downloads
    a model name from an external registry.
    """
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        raise ValueError("project ASR model path must be absolute")
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project ASR model path must be a local directory")

    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("project ASR model directory is empty")
    signature = tuple(
        (
            item.relative_to(root).as_posix(),
            item.stat().st_size,
            item.stat().st_mtime_ns,
        )
        for item in files
    )
    cache_key = str(root)
    with _MODEL_TREE_DIGEST_LOCK:
        cached = _MODEL_TREE_DIGEST_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

        digest = hashlib.sha256()
        for item in files:
            relative = item.relative_to(root).as_posix().encode("utf-8")
            digest.update(relative + b"\0")
            with item.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
        result = digest.hexdigest()
        final_signature = tuple(
            (
                item.relative_to(root).as_posix(),
                item.stat().st_size,
                item.stat().st_mtime_ns,
            )
            for item in files
        )
        if final_signature != signature:
            raise RuntimeError("project ASR model changed while hashing")
        _MODEL_TREE_DIGEST_CACHE[cache_key] = (signature, result)
        return result


def _whisper_cls():
    from faster_whisper import WhisperModel

    return WhisperModel


def _media_sha256(video_path: str) -> str:
    """流式计算视频文件的 sha256。

    复用 :func:`director_brain.utils.file_sha256`；文件不存在或不可读时
    返回 :data:`_MEDIA_HASH_PLACEHOLDER`，不阻断转写主流程（mock 测试传入的
    是不存在的假路径）。
    """
    try:
        return file_sha256(video_path)
    except Exception:
        return _MEDIA_HASH_PLACEHOLDER


def transcribe_with_status(
    video_path: str,
    model_name: str | None = None,
    language: str | None = None,
    *,
    model_version: str | None = None,
    expected_model_digest: str | None = None,
    source_has_audio: bool | None = None,
    source_time_offset_seconds: Fraction = Fraction(0),
    source_stream_index: int | None = None,
) -> ASRTranscriptionResult:
    """把视频中的语音转写成时间戳化的 FilmObservation 列表。

    Args:
        video_path: 视频文件路径。
        model_name: faster-whisper 模型名或本地权重目录；为 None 时从
            ``director_brain.config`` 读取 ``ASR_MODEL_PATH``（默认指向 GEN-1
            已下载的 large-v3-turbo）。
        language: 可选语言代码（如 ``"zh"``）；为 None 时由模型自动检测。

    Returns:
        每个转写片段对应一条 ``observation_type="speech_transcript"`` 的观测；
        返回状态和观测列表。已确认无音轨或无转写片段是 completed_empty；
        模型/解码失败保留具体 failure_code，不与合法空结果混淆。
    """
    if source_has_audio is False:
        return ASRTranscriptionResult("completed_empty", [])

    if model_name is None:
        model_name = load_settings().asr_model_path

    # 模型不可用（faster_whisper 未安装 / import 失败）：降级空列表并 warning
    try:
        _whisper_cls()
    except Exception as exc:
        logger.warning("ASR 不可用：import faster_whisper 失败：%s", exc)
        return ASRTranscriptionResult(
            "failed", [], "asr_provider_unavailable")

    expected_digest = (expected_model_digest or "").removeprefix(
        "sha256:").lower()
    if expected_model_digest is not None:
        if len(expected_digest) != 64 or any(
            char not in "0123456789abcdef" for char in expected_digest
        ):
            return ASRTranscriptionResult(
                "failed", [], "asr_model_digest_invalid")
        try:
            actual_digest = local_model_tree_sha256(model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ASR 本地模型清单不可用：%s", str(exc)[:200])
            return ASRTranscriptionResult(
                "failed", [], "asr_model_artifact_unavailable")
        if actual_digest != expected_digest:
            logger.warning("ASR 本地模型摘要与冻结 profile 不匹配")
            return ASRTranscriptionResult(
                "failed", [], "asr_model_digest_mismatch")

    # P2-a 修正：加载与转写必须在**同一**设备尝试回路内——faster-whisper
    # 的模型是惰性初始化，构造成功不代表该设备真能转写（实测 GPU 构造
    # 成功、首次转写才抛 cublas 缺库）。回退判定必须以"真实转写一次"为准。
    pref = os.environ.get("ASR_DEVICE", "auto").lower()
    attempts = [("cuda", "float16"), ("cpu", "int8")] if pref in ("auto", "cuda")         else [("cpu", "int8")]

    media_asset_id = Path(video_path).name or "unknown"
    media_hash = _media_sha256(video_path)
    now = int(time.time())
    last_exc: Exception | None = None
    offset_us = round(source_time_offset_seconds * _TIMEBASE_US)

    for device, compute in attempts:
        try:
            whisper_cls = _whisper_cls()
            model = whisper_cls(model_name, device=device, compute_type=compute)
            segments, _info = model.transcribe(
                video_path,
                language=language,
                vad_filter=True,
            )

            observations: list[FilmObservation] = []
            for index, seg in enumerate(segments):
                text = (seg.text or "").strip()
                if not text:
                    continue
                observations.append(
                    FilmObservation(
                        observation_id=f"asr_{index:04d}",
                        media_asset_id=media_asset_id,
                        media_hash=media_hash,
                        start_frame=(
                            int(float(seg.start) * _TIMEBASE_US) + offset_us),
                        end_frame=(
                            int(float(seg.end) * _TIMEBASE_US) + offset_us),
                        timebase=_TIMEBASE_US,
                        timebase_unit=TimebaseUnit.MICROSECONDS,
                        source_stream_index=source_stream_index,
                        observation_type="speech_transcript",
                        claim=text,
                        provider="faster_whisper",
                        model_version=model_version or model_name,
                        prompt_version="n/a",
                        confidence=_DEFAULT_CONFIDENCE,
                        review_state="auto_generated",
                        claim_kind=ClaimKind.MODEL_OBSERVATION,
                        schema_version="1.0",
                        project_id="unknown",
                        created_at=now,
                        producer="faster_whisper",
                        source_ref=video_path,
                    )
                )
            logger.info("ASR 转写成功: device=%s, %d 段", device, len(observations))
            return ASRTranscriptionResult(
                "observed" if observations else "completed_empty", observations)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            # 只有被 source manifest 明确标记无音轨时才可判为 completed_empty。
            # 已知有音轨却发生解码越界属于失败，不能冒充静音/无对白。
            if isinstance(exc, IndexError) or "tuple index out of range" in str(exc):
                logger.warning("ASR 音轨解码失败 (video=%s)：%s", video_path, exc)
                return ASRTranscriptionResult(
                    "failed", [], "asr_audio_decode_failed")
            if device == "cuda":
                logger.warning(
                    "ASR GPU 不可用（缺 CUDA 运行库或显存不足？），回退 CPU：%s",
                    str(exc)[:200],
                )
            else:
                logger.warning(
                    "ASR 在 device=%s 上失败 (video=%s)：%s",
                    device, video_path, str(exc)[:200],
                )

    logger.warning("ASR 全部设备尝试失败 (video=%s)：%s",
                   video_path, str(last_exc)[:200])
    return ASRTranscriptionResult("failed", [], "asr_transcription_failed")


def transcribe(
    video_path: str,
    model_name: str | None = None,
    language: str | None = None,
) -> list[FilmObservation]:
    """Backward-compatible fail-soft interface for single-media callers."""
    return transcribe_with_status(
        video_path, model_name=model_name, language=language
    ).observations

