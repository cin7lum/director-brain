"""M1.5 context_gateway ASSET 层单元测试。

三窗口分离：本文件先于实现落地，覆盖 FilmContextSnapshot ASSET 层的构建与读取契约。

用 tmp_path 经 ffmpeg ``testsrc`` 滤镜现场生成 3s 测试视频；observations 用最小合法
FilmObservation mock。覆盖：
1. build_asset_index 返回 FilmContextSnapshot，layers 含 ContextLayer.ASSET
2. sampling_config 含 observation_count，evidence_refs 长度等于 observations 数量
3. 空 observations 返回空索引（不抛异常）
4. 不存在视频路径 degrade（ffprobe 失败不抛异常）
5. get_asset_context 从 repository 读取 / None 透传
6. 相同输入生成相同 context_id（稳定性）
7. source_content_hashes 含真实文件哈希
"""
from __future__ import annotations

import subprocess

from director_brain.context_gateway import build_asset_context as build_asset_index, get_asset_context
from director_brain.models import ClaimKind, ContextLayer, FilmContextSnapshot, FilmObservation


# ---------------------------------------------------------------------------
# 测试数据辅助
# ---------------------------------------------------------------------------

def _make_test_video(path: str, duration_sec: int = 3) -> None:
    """用 ffmpeg testsrc 滤镜生成一段彩色测试视频（H.264 / yuv420p / 25fps）。"""
    cmd = [
        "ffmpeg", "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=320x240:rate=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path, "-y",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def _make_obs(obs_id: str, media_asset_id: str = "asset_1") -> FilmObservation:
    """构造最小合法 FilmObservation（mock，不连真实模型）。"""
    return FilmObservation(
        observation_id=obs_id,
        media_asset_id=media_asset_id,
        media_hash="hash_" + obs_id,
        start_frame=0,
        end_frame=25,
        timebase=1_000_000,
        observation_type="deterministic_technical",
        claim="test claim",
        provider="deterministic_opencv",
        model_version="v1",
        prompt_version="n/a",
        confidence=1.0,
        review_state="final",
        claim_kind=ClaimKind.MEASURED,
        project_id="unknown",
        created_at=1_700_000_000,
        producer="test",
        source_ref="test.mp4",
    )


def _two_observations() -> list[FilmObservation]:
    return [_make_obs("obs_001"), _make_obs("obs_002")]


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_build_asset_index_returns_snapshot_with_asset_layer(tmp_path):
    """1. 返回 FilmContextSnapshot，layers 含 ContextLayer.ASSET。"""
    video = str(tmp_path / "asset_test.mp4")
    _make_test_video(video, duration_sec=3)

    snap = build_asset_index(video, _two_observations())

    assert isinstance(snap, FilmContextSnapshot)
    assert ContextLayer.ASSET in snap.layers
    assert snap.provider == "context_gateway"
    assert snap.model == "asset_context_v1"
    assert snap.timebase == 1_000_000
    assert snap.coverage == "asset_level"
    assert snap.rights_scope == "internal"
    assert snap.cache_state == "fresh"
    assert snap.invalidated_at is None


def test_asset_index_has_count_and_evidence_refs(tmp_path):
    """2. sampling_config 含 observation_count，evidence_refs 长度等于观测数。"""
    video = str(tmp_path / "asset_test.mp4")
    _make_test_video(video, duration_sec=3)
    obs = _two_observations()

    snap = build_asset_index(video, obs)

    assert snap.sampling_config["observation_count"] == 2
    assert len(snap.evidence_refs) == 2
    assert set(snap.evidence_refs) == {"obs_001", "obs_002"}


def test_empty_observations_returns_empty_index(tmp_path):
    """3. 空 observations 返回 FilmContextSnapshot，evidence_refs 为空，不抛异常。"""
    video = str(tmp_path / "asset_test.mp4")
    _make_test_video(video, duration_sec=3)

    snap = build_asset_index(video, [])

    assert isinstance(snap, FilmContextSnapshot)
    assert snap.evidence_refs == []
    assert snap.sampling_config["observation_count"] == 0
    assert ContextLayer.ASSET in snap.layers


def test_nonexistent_video_degrades():
    """4. 不存在路径不抛异常，仍返回 FilmContextSnapshot（ffprobe degrade）。"""
    snap = build_asset_index("nonexistent_video_xyz.mp4", [])

    assert isinstance(snap, FilmContextSnapshot)
    assert snap.layers == [ContextLayer.ASSET]
    assert snap.evidence_refs == []
    # degrade：时长 / 帧率归零
    assert snap.sampling_config["video_duration_us"] == 0
    assert snap.sampling_config["fps"] == 0


def test_get_asset_context_reads_from_repository(tmp_path):
    """5. repository 命中返回该实例；未命中返回 None。"""
    from unittest.mock import Mock

    video = str(tmp_path / "asset_test.mp4")
    _make_test_video(video, duration_sec=3)
    expected = build_asset_index(video, _two_observations())

    repo_hit = Mock()
    repo_hit.get.return_value = expected
    got = get_asset_context(expected.context_id, repo_hit)
    assert got is expected
    repo_hit.get.assert_called_once_with(FilmContextSnapshot, expected.context_id)

    repo_miss = Mock()
    repo_miss.get.return_value = None
    assert get_asset_context("ctx_nonexistent", repo_miss) is None


def test_context_id_stable_across_calls(tmp_path):
    """6. 相同 video_path + 相同 observations 生成相同 context_id。"""
    video = str(tmp_path / "asset_test.mp4")
    _make_test_video(video, duration_sec=3)

    first = build_asset_index(video, _two_observations())
    second = build_asset_index(video, _two_observations())

    assert first.context_id == second.context_id
    assert first.context_id.startswith("ctx_asset_")


def test_context_id_changes_when_observation_ids_differ(tmp_path):
    """6b. 观测数量相同但 observation_id 不同时，context_id 必须不同。

    回归 P1-1：旧实现 context_id 仅依赖 path + len(observations)，
    两组同数量、不同 observation_id 的输入会撞出同一个 context_id。
    """
    video = str(tmp_path / "asset_test.mp4")
    _make_test_video(video, duration_sec=3)

    group_a = [_make_obs("obs_AAA"), _make_obs("obs_BBB")]
    group_b = [_make_obs("obs_XXX"), _make_obs("obs_YYY")]
    assert len(group_a) == len(group_b)  # 数量相同，仅 ID 不同

    snap_a = build_asset_index(video, group_a)
    snap_b = build_asset_index(video, group_b)

    assert snap_a.context_id != snap_b.context_id


def test_source_content_hashes_nonempty_for_real_file(tmp_path):
    """7. 对存在的视频文件，source_content_hashes 非空。"""
    video = str(tmp_path / "asset_test.mp4")
    _make_test_video(video, duration_sec=3)

    snap = build_asset_index(video, [])

    assert isinstance(snap.source_content_hashes, list)
    assert len(snap.source_content_hashes) == 1
    assert snap.source_content_hashes[0]
