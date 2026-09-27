"""director_brain 包级公共契约测试（T5）。

包入口必须提供稳定的 facade（compile_brief / generate_plan /
propose_revision / validate_plan / repair_plan），外部代码不得深引用
内部私有路径。
"""
from __future__ import annotations

import json
import time

import director_brain
from director_brain import (
    compile_brief,
    file_sha256,
    propose_revision,
    repair_plan,
    short_hash,
    validate_plan,
)
from director_brain.models.film_observation import ClaimKind, FilmObservation


def _obs(asset_id: str, start: int, end: int, blur: float) -> FilmObservation:
    claim = json.dumps({"blur_score": blur, "exposure_ok": True, "shake_score": 0.05})
    return FilmObservation(
        observation_id=f"obs_{asset_id}", media_asset_id=asset_id,
        media_hash=f"hash_{asset_id}", start_frame=start, end_frame=end,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim=claim, provider="test", model_version="test", prompt_version="test",
        confidence=1.0, review_state="auto_verified", claim_kind=ClaimKind.MEASURED,
        schema_version="1.0", project_id="facade_proj", created_at=int(time.time()),
        producer="test", source_ref="test.mp4",
    )


def test_package_version_and_exports():
    assert director_brain.__version__ == "0.1.0"
    for name in director_brain.__all__:
        assert callable(getattr(director_brain, name)) or isinstance(
            getattr(director_brain, name), str
        ), f"__all__ 中的 {name} 不可从包入口访问"


def test_facade_compile_and_validate_roundtrip():
    """facade 走通 compile_brief → generate_plan → validate_plan 全链。

    素材为四幕各一镜头（3s/个，时间轴 93s），目标 8s：每幕镜头源长均
    不小于该幕配额（1.2/2.8/2.4/1.6s），总时长应恰为 8s（±10% 内）。
    """
    observations = [
        _obs("shot_a", 0, 3_000_000, 90.0),
        _obs("shot_b", 20_000_000, 23_000_000, 120.0),
        _obs("shot_c", 60_000_000, 63_000_000, 110.0),
        _obs("shot_d", 90_000_000, 93_000_000, 100.0),
    ]
    brief = compile_brief(
        "facade_proj", "test.mp4", observations, target_duration_us=8_000_000
    )
    from director_brain.story_graph_builder import build_story_graph

    g = build_story_graph(brief, observations)
    edl, plan = director_brain.generate_plan(brief, g, observations)
    assert plan.sequence == [e.source_asset_id for e in edl.ordered_edits]
    ok, errors = validate_plan(edl, plan, observations)
    assert ok, f"errors={errors}"
    assert callable(propose_revision) and callable(repair_plan)


def test_facade_hash_tools_match_module():
    from director_brain.utils import file_sha256 as f2, short_hash as s2

    assert short_hash is s2 and file_sha256 is f2
    assert short_hash("abc") == short_hash("abc") and len(short_hash("abc")) == 16
