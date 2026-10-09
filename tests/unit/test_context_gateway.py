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
from types import SimpleNamespace

import pytest

import director_brain.context_gateway as context_gateway
from director_brain.context_gateway import (
    build_asset_context as build_asset_index,
    get_asset_context,
    project_analysis_profile,
    project_asset_analysis_fingerprint,
)
from director_brain.models import (
    ClaimKind,
    ContextLayer,
    FilmContextSnapshot,
    FilmObservation,
    ProjectAnalysisProfile,
    ProjectAsset,
    TimebaseUnit,
)
from director_brain.models.film_observation import FILM_OBSERVATION_SCHEMA_VERSION


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
        timebase_unit=TimebaseUnit.MICROSECONDS,
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
    assert snap.timebase_unit == TimebaseUnit.MICROSECONDS
    assert snap.timeline_scope == "single_asset"
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


def test_local_vlm_analysis_profile_binds_model_runtime_prompt_and_sampling(
    monkeypatch,
):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_MODEL", "qwen3-vl:4b")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_DIGEST", "a" * 64)
    monkeypatch.setenv("PROJECT_LOCAL_VLM_RUNTIME_VERSION", "0.32.14")

    profile = project_analysis_profile(ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1)

    assert "model=qwen3-vl:4b@sha256:" + "a" * 64 in profile
    assert "ollama=0.32.14" in profile
    assert "prompt=vlm_prompt_v5_shot_scale:" in profile
    assert (
        "sampling=shot-relative-v2:0.15,0.50,0.85:max-width-2560:lanczos"
        in profile
    )
    assert (
        "generation=format=json,temperature=0.1,num_ctx=8192,num_predict=1024,"
        "think=false,"
        "unsupported_options=fail_closed"
    ) in profile
    assert "observation_map=film_observation_mapping_v2" in profile
    assert f"observation_schema=FilmObservation/{FILM_OBSERVATION_SCHEMA_VERSION}" in profile
    assert "FilmObservation/1.1" not in profile
    assert project_analysis_profile(
        ProjectAnalysisProfile.DETERMINISTIC_V1) != profile


def test_local_vlm_analysis_profile_rejects_unpinned_or_remote_runtime(
    monkeypatch,
):
    monkeypatch.setenv("PROJECT_LOCAL_VLM_MODEL", "qwen3-vl:4b")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_DIGEST", "not-a-digest")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_RUNTIME_VERSION", "0.32.14")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    with pytest.raises(ValueError, match="pinned 64-character model digest"):
        project_analysis_profile(ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1)

    monkeypatch.setenv("PROJECT_LOCAL_VLM_DIGEST", "a" * 64)
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://example.com")
    with pytest.raises(ValueError, match="loopback Ollama URL"):
        project_analysis_profile(ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1)


def test_local_asr_profile_binds_local_model_tree_and_runtime(
    tmp_path, monkeypatch
):
    import importlib.metadata

    model_path = tmp_path / "large-v3-turbo"
    model_path.mkdir()
    (model_path / "model.bin").write_bytes(b"frozen-local-model")
    monkeypatch.setattr(
        context_gateway,
        "load_settings",
        lambda: SimpleNamespace(asr_model_path=str(model_path)),
    )
    versions = {
        "faster-whisper": "1.2.1",
        "ctranslate2": "4.6.0",
        "av": "15.1.0",
    }
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: versions[name])

    profile = project_analysis_profile(ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1)

    assert "profile=local_asr_shadow_v1" in profile
    assert "provider=faster_whisper" in profile
    assert "model=large-v3-turbo@sha256:" in profile
    assert "faster-whisper=1.2.1;ctranslate2=4.6.0;av=15.1.0" in profile
    assert "audio_clock=first_stream_offset_from_video_v1" in profile

    (model_path / "model.bin").write_bytes(b"different-model")
    assert project_analysis_profile(
        ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1) != profile


def test_local_asr_profile_rejects_model_alias_without_local_directory(
    monkeypatch,
):
    monkeypatch.setattr(
        context_gateway,
        "load_settings",
        lambda: SimpleNamespace(asr_model_path="large-v3-turbo"),
    )

    with pytest.raises(ValueError, match="absolute local model directory"):
        project_analysis_profile(ProjectAnalysisProfile.LOCAL_ASR_SHADOW_V1)


@pytest.mark.parametrize(
    "identity_dimension",
    [
        "model_tag",
        "model_digest",
        "runtime_version",
        "prompt_version",
        "prompt_hash",
        "sampling_profile",
        "generation_profile",
        "observation_mapper",
        "observation_schema",
    ],
)
def test_local_vlm_cache_identity_changes_when_any_bound_component_changes(
    monkeypatch, identity_dimension,
):
    import director_brain.context_gateway as context_gateway
    import observation_service.keyframe as keyframe
    import observation_service.ollama_vlm_adapter as vlm_adapter

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_MODEL", "qwen3-vl:4b")
    monkeypatch.setenv("PROJECT_LOCAL_VLM_DIGEST", "a" * 64)
    monkeypatch.setenv("PROJECT_LOCAL_VLM_RUNTIME_VERSION", "0.32.14")
    asset = ProjectAsset(
        asset_id="asset-a",
        source_ref="/authorized/asset-a.mp4",
        source_content_hash="c" * 64,
        size_bytes=1,
        order=0,
        probe_ok=True,
        rights={
            "state": "local_processing_allowed",
            "basis": "owner_permission",
            "evidence_ref": "test-fixture://rights/asset-a",
        },
    )
    baseline_profile = project_analysis_profile(
        ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1)
    baseline_fingerprint = project_asset_analysis_fingerprint(
        asset, ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1)

    if identity_dimension == "model_tag":
        monkeypatch.setenv("PROJECT_LOCAL_VLM_MODEL", "qwen3-vl:8b")
    elif identity_dimension == "model_digest":
        monkeypatch.setenv("PROJECT_LOCAL_VLM_DIGEST", "b" * 64)
    elif identity_dimension == "runtime_version":
        monkeypatch.setenv("PROJECT_LOCAL_VLM_RUNTIME_VERSION", "0.32.15")
    elif identity_dimension == "prompt_version":
        monkeypatch.setattr(vlm_adapter, "SEMANTIC_PROMPT_VERSION", "vlm_prompt_v5")
    elif identity_dimension == "prompt_hash":
        monkeypatch.setattr(vlm_adapter, "SEMANTIC_PROMPT_SHA256", "b" * 64)
    elif identity_dimension == "sampling_profile":
        monkeypatch.setattr(
            keyframe, "MULTI_FRAME_SAMPLING_PROFILE", "shot-relative-v2")
    elif identity_dimension == "generation_profile":
        monkeypatch.setattr(
            vlm_adapter, "SEMANTIC_GENERATION_PROFILE", "temperature=0")
    elif identity_dimension == "observation_mapper":
        monkeypatch.setattr(
            vlm_adapter, "SEMANTIC_OBSERVATION_MAPPER_VERSION", "mapping_v2")
    elif identity_dimension == "observation_schema":
        monkeypatch.setattr(
            context_gateway, "FILM_OBSERVATION_SCHEMA_VERSION", "2.0")

    changed_profile = project_analysis_profile(
        ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1)
    changed_fingerprint = project_asset_analysis_fingerprint(
        asset, ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1)
    assert changed_profile != baseline_profile
    assert changed_fingerprint != baseline_fingerprint


def test_mixed_observation_timebases_have_no_scalar_timeline(tmp_path):
    video = str(tmp_path / "asset_test.mp4")
    _make_test_video(video, duration_sec=3)
    microseconds = _make_obs("obs_us")
    frames = _make_obs("obs_frames").model_copy(update={
        "timebase": 25,
        "timebase_unit": TimebaseUnit.FRAMES,
    })

    snap = build_asset_index(video, [microseconds, frames])

    assert snap.timeline_scope == "mixed_or_unknown"
    assert snap.timebase is None
    assert snap.timebase_unit is None
