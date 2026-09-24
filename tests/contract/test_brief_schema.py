"""合同测试（F-5a）：Plan/EDL schema 校验契约。

面向 validate_plan 的公共契约：合法 Plan 必须通过；违反 schema 的 Plan
（空 EDL / 反向时间区间 / 未知素材）必须被拒绝且错误信息可定位。
这是契约层断言，区别于 tests/unit 的逐条规则穷举。
"""
from __future__ import annotations

import time

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.plan_validator import validate_plan


def _obs(asset_id: str) -> FilmObservation:
    return FilmObservation(
        observation_id=f"obs_{asset_id}", media_asset_id=asset_id,
        media_hash=f"h_{asset_id}", start_frame=0, end_frame=10_000_000,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim="{}", provider="t", model_version="t", prompt_version="t",
        confidence=1.0, review_state="auto_verified", claim_kind=ClaimKind.MEASURED,
        schema_version="1.0", project_id="p", created_at=int(time.time()),
        producer="t", source_ref="t.mp4",
    )


def _edit(asset_id: str, in_f: int, out_f: int) -> EditItem:
    return EditItem(source_asset_id=asset_id, source_media_hash=f"h_{asset_id}",
                    in_frame=in_f, out_frame=out_f, timebase=1_000_000)


def _edl(edits):
    return EditorialDecisionList(
        schema_version="1.0", project_id="p", created_at=int(time.time()), producer="t",
        source_ref="t", edl_id="e", version="0.1", brief_version="0.1", context_id="c",
        source_asset_hashes=[f"h_{e.source_asset_id}" for e in edits],
        timebase=1_000_000, ordered_edits=edits,
        expected_duration=sum(e.out_frame - e.in_frame for e in edits),
        approval_state="draft")


def _plan():
    return DirectorDecisionPlan(
        schema_version="1.0", project_id="p", created_at=int(time.time()), producer="t",
        source_ref="t", plan_id="plan", version="0.1", brief_version="0.1",
        film_state_version="0.1", sequence=[], decisions=[], constraints=[],
        open_questions=[], validation_status="pending", approval_state="draft")


OBS = [_obs("a"), _obs("b")]


def test_valid_accepted():
    edl = _edl([_edit("a", 1_000_000, 3_000_000), _edit("b", 5_000_000, 8_000_000)])
    ok, errors = validate_plan(edl, _plan(), OBS)
    assert ok is True, errors
    assert errors == []


def test_empty_edl_rejected_with_reason():
    ok, errors = validate_plan(_edl([]), _plan(), OBS)
    assert ok is False
    assert any("empty EDL" in e for e in errors)


def test_reverse_interval_rejected():
    edl = _edl([_edit("a", 5_000_000, 2_000_000)])  # in > out
    ok, errors = validate_plan(edl, _plan(), OBS)
    assert ok is False
    assert any("in_frame" in e and "out_frame" in e for e in errors)


def test_unknown_asset_rejected_with_asset_id():
    edl = _edl([_edit("ghost", 1_000_000, 3_000_000)])
    ok, errors = validate_plan(edl, _plan(), OBS)
    assert ok is False
    assert any("unknown source_asset_id" in e and "ghost" in e for e in errors)
