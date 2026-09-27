"""集成冒烟测试（F-5b）：observation → brief → heuristic EDL → validator 全链路。

纯 Python 层，不渲染、不调 ffmpeg/ollama/VLM。用构造的确定性观测跑通：
compile_brief 编译简报 → HeuristicBaseline 生成 EDL → validate_plan 校验通过。
"""
from __future__ import annotations

import json
import time

from director_brain.brief_compiler import compile_brief
from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.plan_validator import validate_plan
from gen1_adapter.heuristic_baseline import generate_edl


def _tech_obs(asset_id: str, start_us: int, end_us: int, blur: float) -> FilmObservation:
    claim = json.dumps({"blur_score": blur, "shake_score": 0.05, "exposure_ok": True})
    return FilmObservation(
        observation_id=f"obs_{asset_id}", media_asset_id=asset_id,
        media_hash=f"hash_{asset_id}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim=claim, provider="test", model_version="test", prompt_version="test",
        confidence=1.0, review_state="auto_verified", claim_kind=ClaimKind.MEASURED,
        schema_version="1.0", project_id="pipeline_proj", created_at=int(time.time()),
        producer="test", source_ref="test.mp4",
    )


def _candidate(asset_id: str, start_us: int, end_us: int, blur: float) -> dict:
    return {
        "source_shot_id": asset_id,
        "source_media_hash": f"hash_{asset_id}",
        "source_in_us": start_us,
        "source_out_us": end_us,
        "duration_us": end_us - start_us,
        "technical_usable": True,
        "blur_score": blur,
        "vlm_shot_function": None, "vlm_motion": None, "vlm_role": None,
    }


def test_pipeline_smoke_observations_to_validated_edl():
    # 3 个清晰镜头，各 3s
    observations = [
        _tech_obs("shot_a", 0, 3_000_000, 90.0),
        _tech_obs("shot_b", 4_000_000, 7_000_000, 60.0),
        _tech_obs("shot_c", 8_000_000, 11_000_000, 120.0),
    ]

    # 1. brief_compiler 消费观测
    brief = compile_brief("pipeline_proj", "test.mp4", observations, target_duration_us=6_000_000)
    assert brief.target_duration == 6_000_000
    assert brief.visual_language == "consistent_exposure"

    # 2. heuristic 由观测派生候选并生成 EDL
    candidates = [
        _candidate("shot_a", 0, 3_000_000, 90.0),
        _candidate("shot_b", 4_000_000, 7_000_000, 60.0),
        _candidate("shot_c", 8_000_000, 11_000_000, 120.0),
    ]
    edl = generate_edl("pipeline_proj", candidates, 6_000_000)
    assert len(edl.ordered_edits) >= 1

    # 3. validator 校验 EDL（plan 与 EDL 逐位一致，符合 T1 新校验契约）
    edl_ids = [e.source_asset_id for e in edl.ordered_edits]
    plan = DirectorDecisionPlan(
        schema_version="1.0", project_id="pipeline_proj", created_at=int(time.time()),
        producer="test", source_ref="test", plan_id="plan", version="0.1",
        brief_version="0.1", film_state_version="0.1", sequence=edl_ids,
        decisions=[
            Decision(decision_id=f"dec_{i}", purpose="select_shot", shot_refs=[aid])
            for i, aid in enumerate(edl_ids)
        ],
        constraints=["target_duration_us=6000000"], open_questions=[],
        validation_status="pending", approval_state="draft")
    ok, errors = validate_plan(edl, plan, observations)
    assert ok is True, f"pipeline EDL 未通过校验: {errors}"

    # 选片顺序应按 blur 降序（120 > 90 > 60），首段为 shot_c
    assert edl.ordered_edits[0].source_asset_id == "shot_c"
