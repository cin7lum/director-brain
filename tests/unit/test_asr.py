"""M1.3 ASR 语音转写单元测试。

严格覆盖 fail-soft 行为与 FilmObservation 字段映射。所有 faster_whisper
模型加载与转写均通过 ``unittest.mock.patch`` 打桩，绝不实际下载模型权重
或运行真实转写（仅第 1 例用 ffmpeg 生成一个真实的无音轨视频文件作为输入路径）。

覆盖 6 个场景：
1. 无音频视频返回空列表（不抛异常）
2. faster_whisper.WhisperModel 抛 ImportError 时 degrade 为空列表
3. 转写结果时间戳为整数帧（秒→微秒），timebase=1_000_000
4. claim_kind=MODEL_OBSERVATION，provider=faster_whisper
5. 不存在路径返回空列表，不抛异常
6. 转写文本正确映射到 FilmObservation.claim
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# T3：ASR 是可选依赖（pyproject [asr] 组）。缺 faster_whisper 时整模块
# skip（不是 fail）——"可选功能缺依赖优雅降级"的标准形态。
pytest.importorskip("faster_whisper")

from director_brain.models.film_observation import ClaimKind, FilmObservation
from observation_service.asr import transcribe


def _fake_segments():
    """构造两个假转写片段（start/end 单位为秒）。"""
    return [
        SimpleNamespace(start=0.5, end=1.5, text="hello", avg_logprob=-0.3),
        SimpleNamespace(start=2.0, end=3.0, text="world", avg_logprob=-0.4),
    ]


def _patch_whisper(segments, info=None):
    """返回一个已配置好返回值的 WhisperModel patch 上下文工厂。

    用法::

        with _patch_whisper([...]) as m:
            ...
    """
    patcher = patch("faster_whisper.WhisperModel")
    mock_model_cls = patcher.start()
    mock_model_cls.return_value.transcribe.return_value = (
        segments,
        info if info is not None else SimpleNamespace(language="zh"),
    )
    return patcher


# ---------------------------------------------------------------------------
# 1. 无音频视频返回空列表
# ---------------------------------------------------------------------------
def test_no_audio_video_returns_empty(tmp_path):
    video = tmp_path / "no_audio.mp4"
    # testsrc 只有视频流、无音轨；vad 后 segments 为空
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi", "-i",
            "testsrc=duration=3:size=320x240:rate=25",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video), "-y",
        ],
        check=True,
        capture_output=True,
    )
    assert video.exists()

    patcher = patch("faster_whisper.WhisperModel")
    mock_model_cls = patcher.start()
    try:
        # 模拟无音轨 → 空 segments
        mock_model_cls.return_value.transcribe.return_value = (
            [],
            SimpleNamespace(language=None),
        )
        result = transcribe(str(video))
    finally:
        patcher.stop()

    assert result == []


# ---------------------------------------------------------------------------
# 2. ASR 库 import / 模型加载失败时 degrade
# ---------------------------------------------------------------------------
def test_model_load_failure_degrades_to_empty():
    patcher = patch("faster_whisper.WhisperModel")
    mock_model_cls = patcher.start()
    try:
        # 模拟 faster_whisper 不可用 / 权重加载失败
        mock_model_cls.side_effect = ImportError("simulated import failure")
        result = transcribe("anything.mp4")
    finally:
        patcher.stop()

    assert result == []


# ---------------------------------------------------------------------------
# 3. 转写结果时间戳为整数帧
# ---------------------------------------------------------------------------
def test_timestamps_are_integer_microseconds():
    patcher = patch("faster_whisper.WhisperModel")
    mock_model_cls = patcher.start()
    try:
        mock_model_cls.return_value.transcribe.return_value = (
            _fake_segments(),
            SimpleNamespace(language="zh"),
        )
        result = transcribe("fake.mp4")
    finally:
        patcher.stop()

    assert len(result) == 2
    first, second = result
    # 秒 → 微秒：0.5s=500_000us, 1.5s=1_500_000us
    assert first.start_frame == 500_000
    assert first.end_frame == 1_500_000
    assert second.start_frame == 2_000_000
    assert second.end_frame == 3_000_000
    # 必须是 int（bool 不是这里的取值，但显式排除 float）
    assert type(first.start_frame) is int
    assert type(first.end_frame) is int
    assert first.timebase == 1_000_000
    assert second.timebase == 1_000_000
    # observation_id 按序号格式化
    assert first.observation_id == "asr_0000"
    assert second.observation_id == "asr_0001"


# ---------------------------------------------------------------------------
# 4. claim_kind=MODEL_OBSERVATION 且 provider 正确
# ---------------------------------------------------------------------------
def test_claim_kind_and_provider():
    patcher = patch("faster_whisper.WhisperModel")
    mock_model_cls = patcher.start()
    try:
        mock_model_cls.return_value.transcribe.return_value = (
            [SimpleNamespace(start=0.0, end=1.0, text="hi")],
            SimpleNamespace(language="zh"),
        )
        result = transcribe("clip.mp4", model_name="base")
    finally:
        patcher.stop()

    assert len(result) == 1
    obs = result[0]
    assert obs.claim_kind == ClaimKind.MODEL_OBSERVATION
    assert obs.provider == "faster_whisper"
    assert obs.producer == "faster_whisper"
    assert obs.observation_type == "speech_transcript"
    assert obs.review_state == "auto_generated"
    assert obs.model_version == "base"
    assert obs.prompt_version == "n/a"
    assert obs.media_hash == "asr_placeholder"


# ---------------------------------------------------------------------------
# 5. 不存在路径返回空列表，不抛异常
# ---------------------------------------------------------------------------
def test_nonexistent_path_returns_empty():
    patcher = patch("faster_whisper.WhisperModel")
    mock_model_cls = patcher.start()
    try:
        # 真实 whisper 打开不存在的文件会报错；这里模拟该异常被 fail-soft 捕获
        mock_model_cls.return_value.transcribe.side_effect = FileNotFoundError(
            "no such file: nonexistent.mp4"
        )
        result = transcribe("nonexistent.mp4")
    finally:
        patcher.stop()

    assert result == []


# ---------------------------------------------------------------------------
# 6. 转写文本正确
# ---------------------------------------------------------------------------
def test_transcript_text_mapped_to_claim():
    patcher = patch("faster_whisper.WhisperModel")
    mock_model_cls = patcher.start()
    try:
        mock_model_cls.return_value.transcribe.return_value = (
            [SimpleNamespace(start=1.0, end=2.0, text="  测试转写内容  ")],
            SimpleNamespace(language="zh"),
        )
        result = transcribe("clip.mp4")
    finally:
        patcher.stop()

    assert len(result) == 1
    # 文本应被 strip
    assert result[0].claim == "测试转写内容"
    assert isinstance(result[0], FilmObservation)


# ---------------------------------------------------------------------------
# 7. 模型加载失败时打 warning（不再静默吞错）
# ---------------------------------------------------------------------------
def test_model_load_failure_logs_warning(caplog):
    patcher = patch("faster_whisper.WhisperModel")
    mock_model_cls = patcher.start()
    try:
        mock_model_cls.side_effect = RuntimeError("simulated load failure")
        with caplog.at_level(logging.WARNING, logger="observation_service.asr"):
            result = transcribe("anything.mp4", model_name="/nonexistent/model")
    finally:
        patcher.stop()

    assert result == []
    assert any("模型加载失败" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# 8. 真实转写（integration / slow）：默认跳过，需 RUN_REAL_ASR=1 显式开启
# ---------------------------------------------------------------------------
_REAL_TRAILER = r"D:\新建豆包\gen1-roughcut\m5_real\sintel_trailer.mp4"
_RUN_REAL = bool(os.environ.get("RUN_REAL_ASR"))


@pytest.mark.skipif(not _RUN_REAL, reason="需设置 RUN_REAL_ASR=1 显式运行真实转写")
def test_real_transcribe_sintel_trailer():
    if not Path(_REAL_TRAILER).exists():
        pytest.skip(f"真实视频不在位: {_REAL_TRAILER}")

    # 不传 model_name，走 config 默认的 large-v3-turbo
    obs = transcribe(_REAL_TRAILER)
    assert len(obs) > 0
    claims = " ".join(o.claim for o in obs)
    assert "What brings you" in claims
    assert "I'm searching for someone" in claims
