"""M2.3 Plan Validator 单元测试。

验证 validate_plan 的 8 条规则：
1. 总时长 > 0（空 EDL → "empty EDL: no edits"）
2. 每个片段 in_frame < out_frame
3. 无重叠片段（按 in_frame 排序后相邻检查）
4. 间隙可接受（不报错）
5. source_asset_id 存在性
6. 时长在目标范围内（±10%，仅当 plan.constraints 含 target_duration_us 时检查）
7. plan.sequence 与 EDL source_asset_id 序列逐位一致（T1 修复新增）
8. decision.shot_refs 摊平后与 EDL 镜头集合一致（T1 修复新增）
"""
from __future__ import annotations

import time

from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList, TransitionSpec
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.plan_validator import validate_plan
from director_brain.providers.heuristic import generate_edl


# ---------------------------------------------------------------------------
# Fixture 辅助
# ---------------------------------------------------------------------------

def _obs(asset_id: str, start: int = 0, end: int = 10_000_000) -> FilmObservation:
    return FilmObservation(
        observation_id=f"obs_{asset_id}",
        media_asset_id=asset_id,
        media_hash=f"hash_{asset_id}",
        start_frame=start,
        end_frame=end,
        timebase=1_000_000,
        observation_type="deterministic_technical",
        claim="{}",
        provider="test",
        model_version="test",
        prompt_version="test",
        confidence=1.0,
        review_state="auto_verified",
        claim_kind=ClaimKind.MEASURED,
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="test",
        source_ref="test.mp4",
    )


def _edit(
    asset_id: str,
    in_frame: int,
    out_frame: int,
    media_hash: str | None = None,
) -> EditItem:
    return EditItem(
        source_asset_id=asset_id,
        source_media_hash=media_hash or f"hash_{asset_id}",
        in_frame=in_frame,
        out_frame=out_frame,
        timebase=1_000_000,
    )


def _edl(edits: list[EditItem]) -> EditorialDecisionList:
    hashes = []
    seen = set()
    for e in edits:
        if e.source_media_hash not in seen:
            seen.add(e.source_media_hash)
            hashes.append(e.source_media_hash)
    return EditorialDecisionList(
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="test",
        source_ref="test",
        edl_id="test_edl",
        version="0.1",
        brief_version="0.1",
        context_id="ctx_test",
        source_asset_hashes=hashes,
        timebase=1_000_000,
        ordered_edits=edits,
        expected_duration=sum(e.out_frame - e.in_frame for e in edits),
        approval_state="draft",
    )


def _plan(
    constraints: list[str] | None = None,
    sequence: list[str] | None = None,
    decisions: list[Decision] | None = None,
) -> DirectorDecisionPlan:
    return DirectorDecisionPlan(
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="test",
        source_ref="test",
        plan_id="test_plan",
        version="0.1",
        brief_version="0.1",
        film_state_version="0.1",
        sequence=sequence if sequence is not None else [],
        decisions=decisions if decisions is not None else [],
        constraints=constraints or [],
        open_questions=[],
        validation_status="pending",
        approval_state="draft",
    )


def _decisions_for(edl: EditorialDecisionList) -> list[Decision]:
    """构造与 EDL 逐位一致的 decisions（每片段一条 select_shot）。"""
    return [
        Decision(
            decision_id=f"dec_{i}",
            purpose="select_shot",
            shot_refs=[e.source_asset_id],
        )
        for i, e in enumerate(edl.ordered_edits)
    ]


OBS = [_obs("shot_a"), _obs("shot_b"), _obs("shot_c")]


# ---------------------------------------------------------------------------
# Rule 1: empty EDL
# ---------------------------------------------------------------------------

def test_empty_edl_errors():
    edl = _edl([])
    plan = _plan()
    is_valid, errors = validate_plan(edl, plan, OBS)
    assert is_valid is False
    assert any("empty EDL" in e for e in errors)


# ---------------------------------------------------------------------------
# Rule 2: in_frame < out_frame
# ---------------------------------------------------------------------------

def test_in_ge_out_caught():
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=500_000),  # in > out
        _edit("shot_b", in_frame=2_000_000, out_frame=2_000_000),  # in == out
    ]
    edl = _edl(edits)
    plan = _plan()
    is_valid, errors = validate_plan(edl, plan, OBS)
    assert is_valid is False
    assert any("in_frame" in e and "out_frame" in e for e in errors)
    # 两条都应报错
    bad_count = sum(1 for e in errors if "in_frame" in e and "out_frame" in e)
    assert bad_count == 2


# ---------------------------------------------------------------------------
# Rule 3: overlap
# ---------------------------------------------------------------------------

def test_overlap_caught():
    # shot_a: 1M-5M, shot_b: 4M-8M → 重叠 4M-5M
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=5_000_000),
        _edit("shot_b", in_frame=4_000_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    is_valid, errors = validate_plan(edl, plan, OBS)
    assert is_valid is False
    assert any("overlap" in e for e in errors)


def test_gap_is_acceptable():
    # shot_a: 1M-3M, shot_b: 5M-8M → 间隙 3M-5M，不报错
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()
    is_valid, errors = validate_plan(edl, plan, OBS)
    overlap_errors = [e for e in errors if "overlap" in e]
    assert overlap_errors == []


# ---------------------------------------------------------------------------
# Rule 5: unknown source_asset_id
# ---------------------------------------------------------------------------

def test_unknown_asset_id_caught():
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_ghost", in_frame=4_000_000, out_frame=6_000_000),  # 不在 OBS
    ]
    edl = _edl(edits)
    plan = _plan()
    is_valid, errors = validate_plan(edl, plan, OBS)
    assert is_valid is False
    assert any("unknown source_asset_id" in e and "shot_ghost" in e for e in errors)


# ---------------------------------------------------------------------------
# Valid EDL (no target duration constraint)
# ---------------------------------------------------------------------------

def test_valid_edl_no_constraint():
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=8_000_000),
        _edit("shot_c", in_frame=10_000_000, out_frame=12_000_000),
    ]
    edl = _edl(edits)
    plan = _plan(
        constraints=[],  # 无 target_duration_us
        sequence=[e.source_asset_id for e in edl.ordered_edits],
        decisions=_decisions_for(edl),
    )
    is_valid, errors = validate_plan(edl, plan, OBS)
    assert is_valid is True, f"errors={errors}"
    assert errors == []


def test_xfade_must_fit_both_adjacent_source_clips():
    edits = [
        _edit("shot_a", in_frame=0, out_frame=2_000_000),
        _edit("shot_b", in_frame=3_000_000, out_frame=3_800_000),
    ]
    edits[0].transition = TransitionSpec(
        type="xfade", name="dissolve", duration_us=500_000)
    edl = _edl(edits)
    plan = _plan(
        sequence=[edit.source_asset_id for edit in edits],
        decisions=_decisions_for(edl),
    )

    valid, errors = validate_plan(edl, plan, OBS)

    assert valid is True, f"errors={errors}"

    edits[0].transition = TransitionSpec(
        type="xfade", name="dissolve", duration_us=800_000)
    edl = _edl(edits)
    plan = _plan(
        sequence=[edit.source_asset_id for edit in edits],
        decisions=_decisions_for(edl),
    )
    valid, errors = validate_plan(edl, plan, OBS)

    assert valid is False
    assert any("xfade duration must be shorter than both adjacent" in error
               for error in errors)


def test_xfade_requires_a_following_edit():
    edit = _edit("shot_a", in_frame=0, out_frame=2_000_000)
    edit.transition = TransitionSpec(
        type="xfade", name="dissolve", duration_us=500_000)
    edl = _edl([edit])
    plan = _plan(
        sequence=[edit.source_asset_id], decisions=_decisions_for(edl))

    valid, errors = validate_plan(edl, plan, OBS)

    assert valid is False
    assert any("xfade requires a following edit" in error for error in errors)


def test_edl_source_hash_manifest_must_match_used_sources_in_first_use_order():
    edits = [
        _edit("shot_b", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_a", in_frame=4_000_000, out_frame=6_000_000),
    ]
    edl = _edl(edits)
    plan = _plan(
        sequence=[e.source_asset_id for e in edits],
        decisions=_decisions_for(edl),
    )

    valid, errors = validate_plan(edl, plan, OBS)
    assert valid is True, errors

    edl.source_asset_hashes = ["hash_shot_a", "hash_shot_b"]
    valid, errors = validate_plan(edl, plan, OBS)
    assert valid is False
    assert any("EDL source_asset_hashes" in error for error in errors)


def test_heuristic_edl_deduplicates_source_hashes_used_by_multiple_edits():
    edl = generate_edl(
        "test_proj",
        [
            {
                "source_shot_id": "shot_a",
                "source_media_hash": "shared-source",
                "source_in_us": 0,
                "source_out_us": 2_000_000,
                "technical_usable": True,
                "blur_score": 1.0,
            },
            {
                "source_shot_id": "shot_b",
                "source_media_hash": "shared-source",
                "source_in_us": 3_000_000,
                "source_out_us": 5_000_000,
                "technical_usable": True,
                "blur_score": 0.9,
            },
        ],
        target_duration_us=3_000_000,
    )

    assert len(edl.ordered_edits) == 2
    assert edl.source_asset_hashes == ["shared-source"]


# ---------------------------------------------------------------------------
# Rule 6: target duration ±10%
# ---------------------------------------------------------------------------

def test_duration_outside_target_range_caught():
    # 总时长 = 2M + 3M + 2M = 7M us
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=8_000_000),
        _edit("shot_c", in_frame=10_000_000, out_frame=12_000_000),
    ]
    edl = _edl(edits)
    # 目标 20M us，±10% → [18M, 22M]；实际 7M 远不在范围内
    plan = _plan(constraints=["target_duration_us=20000000"])
    is_valid, errors = validate_plan(edl, plan, OBS)
    assert is_valid is False
    assert any("outside target range" in e for e in errors)


def test_duration_within_target_range_ok():
    # 总时长 = 9M + 9M = 18M us
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=10_000_000),
        _edit("shot_b", in_frame=11_000_000, out_frame=20_000_000),
    ]
    edl = _edl(edits)
    # 目标 20M，±10% → [18M, 22M]；实际 18M 恰好在下界
    plan = _plan(constraints=["target_duration_us=20000000"])
    is_valid, errors = validate_plan(edl, plan, OBS)
    duration_errors = [e for e in errors if "outside target range" in e]
    assert duration_errors == [], f"unexpected duration error: {errors}"


def test_no_target_duration_skips_rule6():
    # 总时长 7M，即使没有约束也不报错
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan(
        constraints=["other_constraint=foo"],  # 无 target_duration_us
        sequence=[e.source_asset_id for e in edl.ordered_edits],
        decisions=_decisions_for(edl),
    )
    is_valid, errors = validate_plan(edl, plan, OBS)
    # 不应有 target range 错误
    assert not any("outside target range" in e for e in errors)
    assert is_valid is True


# ---------------------------------------------------------------------------
# Rule 7: plan.sequence 与 EDL 序列一致（T1 修复新增）
# ---------------------------------------------------------------------------

def test_plan_sequence_mismatch_caught():
    """历史 P0-1 分裂场景：EDL 少了一个片段而 plan 未更新 → 必须判红。"""
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    # plan 仍记录 3 个镜头（repair 删镜头后未回写的分裂形态）
    plan = _plan(
        sequence=["shot_a", "shot_b", "shot_c"],
        decisions=_decisions_for(edl) + [
            Decision(decision_id="dec_orphan", purpose="select_shot", shot_refs=["shot_c"])
        ],
    )
    is_valid, errors = validate_plan(edl, plan, OBS)
    assert is_valid is False
    assert any("plan/edl split" in e and "plan.sequence" in e for e in errors)


def test_plan_shot_refs_mismatch_caught():
    """decisions.shot_refs 与 EDL 镜头集合不一致 → 必须判红。"""
    edits = [_edit("shot_a", in_frame=1_000_000, out_frame=3_000_000)]
    edl = _edl(edits)
    plan = _plan(
        sequence=[e.source_asset_id for e in edl.ordered_edits],
        decisions=[Decision(decision_id="dec_0", purpose="select_shot", shot_refs=["shot_b"])],
    )
    is_valid, errors = validate_plan(edl, plan, OBS)
    assert is_valid is False
    assert any("plan/edl split" in e and "shot_refs" in e for e in errors)


def test_consistent_plan_and_edl_passes():
    """plan 与 EDL 完全一致时，Rule 7/8 不报错。"""
    edits = [
        _edit("shot_a", in_frame=1_000_000, out_frame=3_000_000),
        _edit("shot_b", in_frame=5_000_000, out_frame=8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan(
        sequence=[e.source_asset_id for e in edl.ordered_edits],
        decisions=_decisions_for(edl),
    )
    is_valid, errors = validate_plan(edl, plan, OBS)
    split_errors = [e for e in errors if "plan/edl split" in e]
    assert split_errors == [], f"unexpected split errors: {split_errors}"
    assert is_valid is True


def test_empty_edl_with_nonempty_plan_sequence_caught():
    """空 EDL + 非空 plan.sequence 也是分裂 → 必须判红。"""
    edl = _edl([])
    plan = _plan(sequence=["shot_a"], decisions=[
        Decision(decision_id="dec_0", purpose="select_shot", shot_refs=["shot_a"])
    ])
    is_valid, errors = validate_plan(edl, plan, OBS)
    assert is_valid is False
    assert any("plan/edl split" in e for e in errors)
