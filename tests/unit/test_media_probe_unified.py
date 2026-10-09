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
    assert meta.time_map["mapping_state"] == "complete"
    assert meta.time_map["audio_streams"] == []


def test_local_only_probe_restricts_ffprobe_input_protocol(monkeypatch):
    from observation_service import media_info

    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps({"streams": [], "format": {"duration": "1"}}),
            stderr="",
        )

    monkeypatch.setattr(media_info.subprocess, "run", fake_run)
    assert probe_media_meta("authorized-local-source.mp4", local_only=True).ok

    assert len(commands) == 1
    command = commands[0]
    whitelist_index = command.index("-protocol_whitelist")
    input_index = command.index("authorized-local-source.mp4")
    assert command[whitelist_index + 1] == "file"
    assert whitelist_index < input_index


def test_context_gateway_requests_local_only_probe(monkeypatch):
    import observation_service.media_info as media_info_module
    import director_brain.context_gateway as context_gateway

    calls = []

    def fake_probe(path, *, local_only=False):
        calls.append((path, local_only))
        return media_info_module.MediaMeta(
            ok=True, duration_us=1_000_000, fps=25.0,
            r_frame_rate="25/1",
        )

    monkeypatch.setattr(media_info_module, "probe_media_meta", fake_probe)
    result = context_gateway._probe_video_meta("authorized-local-source.mp4")

    assert calls == [("authorized-local-source.mp4", True)]
    assert result["probe_ok"] is True


def test_probe_media_meta_preserves_exact_ffprobe_clocks(monkeypatch):
    """Fractional rates and stream timebase survive without float conversion."""
    from observation_service import media_info

    response = {
        "streams": [{
            "codec_type": "video",
            "codec_name": "h264",
            "r_frame_rate": "30000/1001",
            "avg_frame_rate": "24000/1001",
            "time_base": "1/30000",
        }],
        "format": {"duration": "2.002000"},
    }

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(response), stderr="")

    monkeypatch.setattr(media_info.subprocess, "run", fake_run)
    meta = probe_media_meta("synthetic://fractional-rate")

    assert meta.ok
    assert meta.fps == 29.97  # Compatibility summary; not authoritative clock data.
    assert meta.r_frame_rate == "30000/1001"
    assert meta.avg_frame_rate == "24000/1001"
    assert meta.stream_time_base == "1/30000"


def test_probe_media_meta_maps_multiple_audio_streams_to_container_origin(monkeypatch):
    from observation_service import media_info

    response = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "time_base": "1/90000",
                "start_pts": 9000,
                "start_time": "0.100000",
                "duration_ts": 180000,
                "duration": "2.000000",
                "disposition": {"default": 1},
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "time_base": "1/48000",
                "start_pts": 16800,
                "start_time": "0.350000",
                "duration_ts": 48000,
                "duration": "1.000000",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
                "tags": {"language": "eng"},
                "disposition": {"default": 1},
            },
            {
                "index": 2,
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "time_base": "1/44100",
                "start_pts": 37485,
                "start_time": "0.850000",
                "duration_ts": 44100,
                "duration": "1.000000",
                "sample_rate": "44100",
                "channels": 1,
                "channel_layout": "mono",
                "tags": {"language": "spa"},
                "disposition": {"default": 0},
            },
        ],
        "format": {"start_time": "0.100000", "duration": "2.100000"},
    }

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(response), stderr="")

    monkeypatch.setattr(media_info.subprocess, "run", fake_run)
    meta = probe_media_meta("synthetic://offset-multiaudio")

    assert meta.ok
    assert meta.has_audio is True
    assert meta.time_map["mapping_state"] == "complete"
    assert meta.time_map["video_stream"]["source_start_offset_state"] == (
        "mapped_from_pts")
    assert meta.time_map["container_start_time_seconds"] == "0.100000"
    assert meta.time_map["video_stream"]["time_base"] == "1/90000"
    assert meta.time_map["video_stream"]["start_pts"] == 9000
    audio = meta.time_map["audio_streams"]
    assert [item["stream_index"] for item in audio] == [1, 2]
    assert [(item["source_start_offset_numerator"],
             item["source_start_offset_denominator"]) for item in audio] == [
        (1, 4), (3, 4)]
    assert [item["sample_rate"] for item in audio] == [48000, 44100]
    assert [item["time_base"] for item in audio] == ["1/48000", "1/44100"]
    assert [item["start_pts"] for item in audio] == [16800, 37485]
    assert [item["language"] for item in audio] == ["eng", "spa"]
    assert [item["is_default"] for item in audio] == [True, False]


def test_probe_media_meta_uses_start_time_when_pts_clock_is_missing(monkeypatch):
    from observation_service import media_info

    response = {
        "streams": [{
            "index": 0,
            "codec_type": "video",
            "start_time": "0.500000",
            "duration": "1.000000",
        }],
        "format": {"start_time": "0.250000", "duration": "1.250000"},
    }

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(response), stderr="")

    monkeypatch.setattr(media_info.subprocess, "run", fake_run)
    meta = probe_media_meta("synthetic://decimal-clock-fallback")

    assert meta.time_map["mapping_state"] == "complete"
    video = meta.time_map["video_stream"]
    assert video["source_start_offset_state"] == "mapped_from_start_time"
    assert (video["source_start_offset_numerator"],
            video["source_start_offset_denominator"]) == (1, 4)


def test_probe_media_meta_keeps_clocks_unavailable_without_container_origin(monkeypatch):
    from observation_service import media_info

    response = {
        "streams": [{
            "index": 0,
            "codec_type": "video",
            "time_base": "1/90000",
            "start_pts": 9000,
            "start_time": "0.100000",
        }, {
            "index": 1,
            "codec_type": "audio",
            "time_base": "1/48000",
            "start_pts": 24000,
            "start_time": "0.500000",
            "sample_rate": "48000",
        }],
        "format": {"duration": "2.000000"},
    }

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(response), stderr="")

    monkeypatch.setattr(media_info.subprocess, "run", fake_run)
    meta = probe_media_meta("synthetic://missing-container-origin")

    assert meta.time_map["mapping_state"] == "unavailable"
    assert meta.time_map["container_start_time_seconds"] is None
    assert meta.time_map["video_stream"]["source_start_offset_numerator"] is None
    assert all(item["source_start_offset_state"] == "unavailable"
               for item in [meta.time_map["video_stream"]]
               + meta.time_map["audio_streams"])


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
