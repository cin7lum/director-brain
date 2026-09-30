"""ffprobe 统一入口 + 归因测试（架构体检候选⑤）。

评审指认：全仓 4 份 ffprobe、失败语义不一；context_gateway 静默吞失败
把探测失败记成 fps=0 的"事实"（归因裂缝）。本文件验证：media_info 是
唯一权威探测，失败带归因，消费方显式降级。
"""
from __future__ import annotations

import json
import subprocess
import time

from observation_service.media_info import probe_audio_stream, probe_media_meta


def _make_video(path: str, duration_sec: int = 3) -> None:
    subprocess.run(
        ["ffmpeg", "-f", "lavfi",
         "-i", f"testsrc=duration={duration_sec}:size=320x240:rate=25",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path, "-y"],
        capture_output=True, text=True, check=True)


def _make_obs(obs_id: str) -> object:
    from director_brain.models import ClaimKind, FilmObservation
    return FilmObservation(
        observation_id=obs_id, media_asset_id="asset_1",
        media_hash="hash_" + obs_id, start_frame=0, end_frame=25,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim="{}", provider="deterministic_opencv", model_version="v1",
        prompt_version="n/a", confidence=1.0, review_state="final",
        claim_kind=ClaimKind.MEASURED, schema_version="1.0",
        project_id="t", created_at=int(time.time()), producer="test",
        source_ref="t.mp4")


def test_probe_media_meta_ok(tmp_path):
    """正常探测：时长/帧率确定性事实。"""
    video = str(tmp_path / "v.mp4")
    _make_video(video, 3)
    meta = probe_media_meta(video)
    assert meta.ok
    assert abs(meta.duration_us - 3_000_000) < 500_000
    assert meta.fps > 0
    assert not meta.has_audio  # testsrc 无音轨


def test_probe_media_meta_failure_attributed(tmp_path):
    """探测失败：ok=False + reason 归因，绝不静默返回零值"事实"。"""
    meta = probe_media_meta(str(tmp_path / "不存在.mp4"))
    assert not meta.ok
    assert meta.reason  # 归因非空


def test_probe_audio_stream_wrapper(tmp_path):
    """音轨归因薄包装：ok=False 透传原因（T4 ASR 归因链路不回退）。"""
    miss = probe_audio_stream(str(tmp_path / "nope.mp4"))
    assert not miss.ok and miss.reason

    video = str(tmp_path / "v.mp4")
    _make_video(video, 2)
    info = probe_audio_stream(video)
    assert info.ok and not info.has_audio


def test_context_gateway_records_probe_attribution(tmp_path):
    """ASSET 层快照：探测失败时 sampling_config 携带归因（不再装事实）。"""
    import observation_service.media_info as mi
    from director_brain.context_gateway import build_asset_context

    video = str(tmp_path / "v.mp4")
    _make_video(video, 2)

    # monkeypatch 深度探测失败 → 快照归因可见
    import director_brain.context_gateway as cg

    def _fail_probe(path):
        return {"duration_us": 0, "fps": 0.0, "has_audio": False,
                "probe_ok": False, "probe_reason": "ffprobe 退出码 1: mock"}

    orig = cg._probe_video_meta
    cg._probe_video_meta = _fail_probe
    try:
        snap = build_asset_context(video, [_make_obs("o1")])
    finally:
        cg._probe_video_meta = orig

    cfg = snap.sampling_config
    assert cfg["probe_ok"] is False
    assert cfg["probe_reason"]


def test_context_gateway_normal_probe_ok(tmp_path):
    """正常探测：probe_ok=True（归因字段恒在——可复核）。"""
    from director_brain.context_gateway import build_asset_context

    video = str(tmp_path / "v.mp4")
    _make_video(video, 2)
    snap = build_asset_context(video, [_make_obs("o1")])
    assert snap.sampling_config["probe_ok"] is True
