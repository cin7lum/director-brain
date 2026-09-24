"""M1.3 离线 ASR 语音转写：faster-whisper → FilmObservation。

用 CTranslate2 后端的 faster-whisper 对视频做语音转写，每个带时间戳的
segment 映射成一条 :class:`~director_brain.models.film_observation.FilmObservation`。

保持 **fail-soft** 的返回约定（任何失败都返回空列表，绝不向上抛异常），但
不再静默吞错：faster_whisper import 失败、模型权重缺失/加载失败、转写运行期
异常都会通过 ``logging.warning`` 输出原因，便于区分「模型不可用」与「视频无
语音」。视频无语音 / 转写结果为空属于正常空结果，不打 warning。时间以微秒为
整数帧存储，``timebase=1_000_000``。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from director_brain._utils import file_sha256
from director_brain.config import load_settings
from director_brain.models.film_observation import ClaimKind, FilmObservation

logger = logging.getLogger(__name__)

#: ASR 时间基：微秒。faster-whisper 的 segment.start/end 为秒，乘此系数得整数微秒。
_TIMEBASE_US = 1_000_000

#: ASR 默认置信度（faster-whisper 不直接给出归一化置信度，这里给一个保守默认值）。
_DEFAULT_CONFIDENCE = 0.8

#: 视频文件不可读（mock 测试 / 文件缺失）时 media_hash 的回退占位符。
_MEDIA_HASH_PLACEHOLDER = "asr_placeholder"


def _media_sha256(video_path: str) -> str:
    """流式计算视频文件的 sha256。

    复用 :func:`director_brain._utils.file_sha256`；文件不存在或不可读时
    返回 :data:`_MEDIA_HASH_PLACEHOLDER`，不阻断转写主流程（mock 测试传入的
    是不存在的假路径）。
    """
    try:
        return file_sha256(video_path)
    except Exception:
        return _MEDIA_HASH_PLACEHOLDER


def transcribe(
    video_path: str,
    model_name: str | None = None,
    language: str | None = None,
) -> list[FilmObservation]:
    """把视频中的语音转写成时间戳化的 FilmObservation 列表。

    Args:
        video_path: 视频文件路径。
        model_name: faster-whisper 模型名或本地权重目录；为 None 时从
            ``director_brain.config`` 读取 ``ASR_MODEL_PATH``（默认指向 GEN-1
            已下载的 large-v3-turbo）。
        language: 可选语言代码（如 ``"zh"``）；为 None 时由模型自动检测。

    Returns:
        每个转写片段对应一条 ``observation_type="speech_transcript"`` 的观测；
        任何失败都返回空列表，不抛异常，但模型加载 / 转写失败会打 warning。
    """
    if model_name is None:
        model_name = load_settings().asr_model_path

    # 模型不可用（faster_whisper 未安装 / import 失败）：降级空列表并 warning
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        logger.warning("ASR 不可用：import faster_whisper 失败：%s", exc)
        return []

    # 模型加载失败（路径不存在 / 权重损坏 / 下载失败）：降级空列表并 warning
    try:
        # CPU + int8 是最稳的降级路径；GPU 不可用时不崩
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as exc:
        logger.warning("ASR 模型加载失败 (model=%s)：%s", model_name, exc)
        return []

    # 转写阶段异常：降级空列表并 warning；视频无语音 / 空 segments 属正常空结果
    try:
        segments, _info = model.transcribe(
            video_path,
            language=language,
            vad_filter=True,
        )

        media_asset_id = Path(video_path).name or "unknown"
        media_hash = _media_sha256(video_path)
        now = int(time.time())

        observations: list[FilmObservation] = []
        for index, seg in enumerate(segments):
            text = (seg.text or "").strip()
            observations.append(
                FilmObservation(
                    observation_id=f"asr_{index:04d}",
                    media_asset_id=media_asset_id,
                    media_hash=media_hash,
                    start_frame=int(float(seg.start) * _TIMEBASE_US),
                    end_frame=int(float(seg.end) * _TIMEBASE_US),
                    timebase=_TIMEBASE_US,
                    observation_type="speech_transcript",
                    claim=text,
                    provider="faster_whisper",
                    model_version=model_name,
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
        return observations
    except Exception as exc:
        # 无音轨视频在 faster-whisper 内部表现为 PyAV 取音频流越界（IndexError /
        # "tuple index out of range"），属「视频无语音」的正常空结果，不打 warning。
        if isinstance(exc, IndexError) or "tuple index out of range" in str(exc):
            logger.debug("ASR 无音轨/空转写结果 (video=%s)：%s", video_path, exc)
        else:
            logger.warning("ASR 转写失败 (video=%s)：%s", video_path, exc)
        return []
