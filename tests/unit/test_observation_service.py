"""M1.1 observation_service 单元测试。

三窗口分离：本文件先于实现落地，覆盖镜头切分与确定性分析的核心契约。

用 tmp_path 经 ffmpeg ``testsrc`` 滤镜现场生成 3-5s 彩色测试视频，不依赖 GEN-1 仓库。
覆盖：
1. discover_shots 返回非空列表
2. 每个镜头 in/out 为非负整数且 out > in
3. analyze_shot 返回 FilmObservation（MEASURED / 确定性 provider / confidence=1.0）
4. analyze_media 端到端：观测数 == 镜头数
5. 不存在路径不抛异常（discover_shots / analyze_media 均返回空列表）
6. 镜头总时长与 ffprobe 实际时长接近（1% 误差）
"""
from __future__ import annotations

import subprocess

import pytest

from director_brain.models import ClaimKind, FilmObservation
from observation_service import analyze_media, analyze_shot, discover_shots


# ---------------------------------------------------------------------------
# 测试视频构造辅助
# ---------------------------------------------------------------------------

def _make_test_video(path: str, duration_sec: int = 4) -> None:
    """用 ffmpeg testsrc 滤镜生成一段彩色测试视频（H.264 / yuv420p）。"""
    cmd = [
        "ffmpeg", "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=320x240:rate=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path, "-y",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def _ffprobe_duration_us(path: str) -> int:
    """用 ffprobe 读取视频时长（微秒）。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(round(float(r.stdout.strip()) * 1_000_000))


@pytest.fixture()
def test_video(tmp_path):
    """现场生成的 4s 测试视频路径（str）。"""
    p = tmp_path / "obs_test.mp4"
    _make_test_video(str(p), duration_sec=4)
    return str(p)


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_discover_shots_returns_nonempty(test_video):
    """1. 镜头切分返回非空列表。"""
    shots = discover_shots(test_video)
    assert isinstance(shots, list)
    assert len(shots) >= 1


def test_shots_have_valid_bounds(test_video):
    """2. 每个镜头 in/out 为非负整数且 out > in。"""
    shots = discover_shots(test_video)
    assert len(shots) >= 1
    for s in shots:
        assert isinstance(s["source_in_us"], int)
        assert isinstance(s["source_out_us"], int)
        assert s["source_in_us"] >= 0
        assert s["source_out_us"] > s["source_in_us"]
        assert s["duration_us"] == s["source_out_us"] - s["source_in_us"]
        assert s["shot_id"].startswith("shot_")
        assert isinstance(s["source_media_hash"], str) and s["source_media_hash"]
        assert isinstance(s["keyframes"], list)


def test_analyze_shot_returns_film_observation(test_video):
    """3. deterministic 分析返回 FilmObservation（MEASURED）。"""
    shots = discover_shots(test_video)
    assert shots, "前置：至少切出一个镜头"
    obs = analyze_shot(test_video, shots[0])
    assert isinstance(obs, FilmObservation)
    assert obs.claim_kind == ClaimKind.MEASURED
    assert obs.provider == "deterministic_opencv"
    assert obs.confidence == 1.0
    assert obs.observation_type == "deterministic_technical"


def test_pipeline_end_to_end(test_video):
    """4. analyze_media 返回 list[FilmObservation]，数量与镜头数一致。"""
    shots = discover_shots(test_video)
    obs_list = analyze_media(test_video)
    assert isinstance(obs_list, list)
    assert len(obs_list) == len(shots)
    for obs in obs_list:
        assert isinstance(obs, FilmObservation)


def test_nonexistent_path_does_not_raise():
    """5. 不存在路径不抛异常，discover_shots / analyze_media 返回空列表。"""
    assert discover_shots("nonexistent_video_xyz.mp4") == []
    assert analyze_media("nonexistent_video_xyz.mp4") == []


def test_total_shot_duration_matches_video(test_video):
    """6. 镜头总时长接近视频实际时长（1% 误差）。"""
    actual_us = _ffprobe_duration_us(test_video)
    shots = discover_shots(test_video)
    covered_us = sum(s["source_out_us"] - s["source_in_us"] for s in shots)
    assert actual_us > 0
    # 总覆盖时长应等于视频时长（首尾相接、无间隙），允许 1% 误差。
    assert abs(covered_us - actual_us) <= actual_us * 0.01
