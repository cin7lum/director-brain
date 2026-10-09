"""VLM 结果 → FilmObservation 转换层单元测试。

覆盖：
- vlm_result_to_observation 成功 / 失败两种情况
- batch_vlm_observations 用 mock adapter（含缓存命中）
- claim 是合法 JSON 且含必需字段
"""
from __future__ import annotations

import contextlib
import json
import time

import pytest

from director_brain.analysis_cache import AnalysisCache
from director_brain.models.film_observation import (
    ClaimKind,
    FILM_OBSERVATION_SCHEMA_VERSION,
    TimebaseUnit,
)
from observation_service.vlm_observation import (
    batch_vlm_observations,
    vlm_result_to_observation,
)


def _shot(index: int = 0, in_us: int = 0, out_us: int = 2_000_000) -> dict:
    return {
        "shot_id": f"shot_test_{index:04d}",
        "source_in_us": in_us,
        "source_out_us": out_us,
        "duration_us": out_us - in_us,
        "keyframes": [],
        "source_media_hash": f"hash_test_{index}",
    }


def _vlm_success(**overrides) -> dict:
    base = {
        "shot_function": "ACTION",
        "shot_scale": "medium",
        "sensory_wet_heat": 0.3,
        "sensory_mood_intensity": 0.8,
        "motion_amount": "burst",
        "proposed_role_v2": "hero",
        "frame_description": "一个角色在奔跑",
        "status": "OBSERVED",
        "degraded": False,
        "confidence_type": "SELF_REPORTED",
        "_warnings": [],
    }
    base.update(overrides)
    return base


def _vlm_failure(reason: str = "network timeout") -> dict:
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
        "failure_type": "NETWORK",
        "degrade_reason": reason,
        "_warnings": [],
    }


# ---------------------------------------------------------------------------
# vlm_result_to_observation
# ---------------------------------------------------------------------------

class TestVlmResultToObservation:
    def test_success_produces_model_observation(self):
        obs = vlm_result_to_observation(_vlm_success(), _shot(), "/tmp/frame.jpg")
        assert obs.observation_type == "vlm_semantic"
        assert obs.claim_kind == ClaimKind.MODEL_OBSERVATION
        assert obs.provider == "ollama_qwen3_vl"
        assert obs.model_version == "qwen3-vl:latest"
        assert obs.confidence == 0.7
        assert obs.timebase == 1_000_000
        assert obs.timebase_unit == TimebaseUnit.MICROSECONDS
        assert obs.schema_version == FILM_OBSERVATION_SCHEMA_VERSION
        assert obs.media_asset_id == "shot_test_0000"
        assert obs.media_hash == "hash_test_0"
        assert obs.start_frame == 0
        assert obs.end_frame == 2_000_000

    def test_success_claim_is_valid_json_with_fields(self):
        obs = vlm_result_to_observation(_vlm_success(), _shot(), "/tmp/frame.jpg")
        claim = json.loads(obs.claim)
        assert claim["shot_function"] == "ACTION"
        assert claim["shot_scale"] == "medium"
        assert claim["motion_amount"] == "burst"
        assert claim["proposed_role_v2"] == "hero"
        assert claim["frame_description"] == "一个角色在奔跑"
        assert claim["sensory_wet_heat"] == 0.3
        assert claim["sensory_mood_intensity"] == 0.8

    def test_failure_produces_not_determined(self):
        obs = vlm_result_to_observation(_vlm_failure(), _shot(), "/tmp/frame.jpg")
        assert obs.claim_kind == ClaimKind.NOT_DETERMINED
        assert obs.confidence == 0.3
        claim = json.loads(obs.claim)
        assert claim["error"] == "network timeout"
        assert claim["failure_type"] == "NETWORK"

    def test_failure_still_has_semantic_fields(self):
        """失败时 claim 仍含 fallback 语义字段（值为适配器默认）。"""
        obs = vlm_result_to_observation(_vlm_failure(), _shot(), "/tmp/frame.jpg")
        claim = json.loads(obs.claim)
        assert claim["shot_function"] == "SENSORY_INSERT"
        assert claim["proposed_role_v2"] == "broll"
        assert claim["frame_description"] == ""

    def test_degraded_with_warnings_not_failure(self):
        """degraded=True 但 status=OBSERVED（仅 warning）不算失败。"""
        result = _vlm_success(degraded=True, _warnings=["fallback used"])
        obs = vlm_result_to_observation(result, _shot(), "/tmp/frame.jpg")
        assert obs.claim_kind == ClaimKind.MODEL_OBSERVATION
        assert obs.confidence == 0.7

    def test_non_self_reported_confidence(self):
        result = _vlm_success(confidence_type="OTHER")
        obs = vlm_result_to_observation(result, _shot(), "/tmp/frame.jpg")
        assert obs.confidence == 0.5

    def test_observation_id_prefix(self):
        obs = vlm_result_to_observation(_vlm_success(), _shot(), "/tmp/frame.jpg")
        assert obs.observation_id.startswith("vlm_shot_test_")


# ---------------------------------------------------------------------------
# batch_vlm_observations (mock adapter)
# ---------------------------------------------------------------------------

class _MockAdapter:
    def __init__(self, results: list[dict] | None = None):
        self.model = "qwen3-vl"
        self.calls: list[str] = []
        self._results = results or []
        self._idx = 0

    def analyze_frame(self, image_path: str) -> dict:
        self.calls.append(image_path)
        if self._idx < len(self._results):
            r = self._results[self._idx]
            self._idx += 1
            return r
        return _vlm_success()

    def analyze_frames(self, image_paths: list[str]) -> dict:
        return self.analyze_frame(image_paths[0])


class TestBatchVlmObservations:
    def test_returns_one_obs_per_shot(self, tmp_path):
        shots = [_shot(i, i * 2_000_000, (i + 1) * 2_000_000) for i in range(3)]
        adapter = _MockAdapter()
        # 用不存在的视频路径 → extract_keyframe fail-soft 返回 ""，
        # 但我们 mock 了 adapter，不过 keyframe 抽取会失败。
        # 所以这里需要真实视频或 mock keyframe。用 monkeypatch。
        import observation_service.vlm_observation as mod

        original = mod.extract_keyframes

        @contextlib.contextmanager
        def fake_kfs(video_path, in_us, out_us, positions=(0.15, 0.50, 0.85)):
            yield ["/tmp/fake_frame.jpg"]

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, "extract_keyframes", fake_kfs)
        try:
            cache = AnalysisCache()
            obs_list = batch_vlm_observations(
                "dummy.mp4", shots, adapter=adapter, cache=cache
            )
        finally:
            monkeypatch.undo()

        assert len(obs_list) == 3
        assert all(o.observation_type == "vlm_semantic" for o in obs_list)
        assert all(o.claim_kind == ClaimKind.MODEL_OBSERVATION for o in obs_list)

    def test_cache_hit_skips_adapter_call(self, tmp_path):
        import contextlib
        import observation_service.vlm_observation as mod

        shots = [_shot(0, 0, 2_000_000)]
        adapter = _MockAdapter()

        @contextlib.contextmanager
        def fake_kfs(video_path, in_us, out_us, positions=(0.15, 0.50, 0.85)):
            yield ["/tmp/fake_frame.jpg"]

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, "extract_keyframes", fake_kfs)
        try:
            cache = AnalysisCache()
            # 第一次：调用 adapter
            obs1 = batch_vlm_observations(
                "dummy.mp4", shots, adapter=adapter, cache=cache
            )
            assert len(adapter.calls) == 1

            # 第二次：应命中缓存，不调用 adapter
            adapter.calls.clear()
            obs2 = batch_vlm_observations(
                "dummy.mp4", shots, adapter=adapter, cache=cache
            )
            assert len(adapter.calls) == 0
            assert len(obs2) == 1
            assert obs2[0].observation_id == obs1[0].observation_id
            assert cache.hit_rate() > 0
        finally:
            monkeypatch.undo()

    def test_failure_produces_not_determined(self, tmp_path):
        import contextlib
        import observation_service.vlm_observation as mod

        shots = [_shot(0, 0, 2_000_000)]
        adapter = _MockAdapter(results=[_vlm_failure()])

        @contextlib.contextmanager
        def fake_kfs(video_path, in_us, out_us, positions=(0.15, 0.50, 0.85)):
            yield ["/tmp/fake_frame.jpg"]

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, "extract_keyframes", fake_kfs)
        try:
            cache = AnalysisCache()
            obs_list = batch_vlm_observations(
                "dummy.mp4", shots, adapter=adapter, cache=cache
            )
        finally:
            monkeypatch.undo()

        assert len(obs_list) == 1
        assert obs_list[0].claim_kind == ClaimKind.NOT_DETERMINED

    def test_failure_is_retried_and_only_success_is_cached(self, tmp_path):
        import contextlib
        import observation_service.vlm_observation as mod

        shots = [_shot(0, 0, 2_000_000)]
        adapter = _MockAdapter(results=[_vlm_failure(), _vlm_success()])

        @contextlib.contextmanager
        def fake_kfs(video_path, in_us, out_us, positions=(0.15, 0.50, 0.85)):
            yield ["/tmp/fake_frame.jpg"]

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, "extract_keyframes", fake_kfs)
        try:
            cache = AnalysisCache()
            failed_attempt = batch_vlm_observations(
                "dummy.mp4", shots, adapter=adapter, cache=cache)
            recovered_attempt = batch_vlm_observations(
                "dummy.mp4", shots, adapter=adapter, cache=cache)
            replay = batch_vlm_observations(
                "dummy.mp4", shots, adapter=adapter, cache=cache)
        finally:
            monkeypatch.undo()

        assert failed_attempt[0].claim_kind == ClaimKind.NOT_DETERMINED
        assert recovered_attempt[0].claim_kind == ClaimKind.MODEL_OBSERVATION
        assert replay[0].claim_kind == ClaimKind.MODEL_OBSERVATION
        assert len(adapter.calls) == 2

    def test_keyframe_failure_degrades_gracefully(self, tmp_path):
        """extract_keyframe 返回空串时产出 NOT_DETERMINED，不抛异常。"""
        import contextlib
        import observation_service.vlm_observation as mod

        shots = [_shot(0, 0, 2_000_000)]
        adapter = _MockAdapter()

        @contextlib.contextmanager
        def fake_kfs(video_path, in_us, out_us, positions=(0.15, 0.50, 0.85)):
            yield []  # 模拟 ffmpeg 失败

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, "extract_keyframes", fake_kfs)
        try:
            cache = AnalysisCache()
            obs_list = batch_vlm_observations(
                "dummy.mp4", shots, adapter=adapter, cache=cache
            )
        finally:
            monkeypatch.undo()

        assert len(obs_list) == 1
        assert obs_list[0].claim_kind == ClaimKind.NOT_DETERMINED
        claim = json.loads(obs_list[0].claim)
        assert "keyframe" in claim["error"]
