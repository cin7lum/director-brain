"""M3.2 Revision Engine 单元测试。

验证 propose_revision / apply_revision / list_revisions：
- 四种修订类型（extend_peak / remove_low_quality / adjust_duration / reorder）
  都能生成合法的 RevisionProposal；
- remove_low_quality 正确识别 blur 低于阈值的镜头；
- apply_revision 对 EDL / Plan 产生可验证的、非原地（深拷贝）的修改；
- reorder 后 ordered_edits 按 in_frame 升序；
- list_revisions 从 repository 读取。
"""
from __future__ import annotations

import json
import time

import pytest

from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.models.revision import RevisionProposal
from director_brain.revision_engine import (
    apply_revision,
    list_revisions,
    propose_revision,
)


# ---------------------------------------------------------------------------
# Fixture 辅助（风格对齐 tests/unit/test_plan_validator.py）
# ---------------------------------------------------------------------------

def _obs(asset_id: str, claim: str = "{}") -> FilmObservation:
    return FilmObservation(
        observation_id=f"obs_{asset_id}",
        media_asset_id=asset_id,
        media_hash=f"hash_{asset_id}",
        start_frame=0,
        end_frame=15_000_000,
        timebase=1_000_000,
        observation_type="deterministic_technical",
        claim=claim,
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


def _edit(asset_id: str, in_frame: int, out_frame: int) -> EditItem:
    return EditItem(
        source_asset_id=asset_id,
        source_media_hash=f"hash_{asset_id}",
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


def _decision(asset_id: str) -> Decision:
    return Decision(
        decision_id=f"dec_{asset_id}",
        purpose=f"purpose_{asset_id}",
        shot_refs=[asset_id],
        evidence_refs=[],
        rationale=None,
        alternatives=[],
        confidence=0.8,
        requires_approval=False,
    )


def _plan(
    edits: list[EditItem],
    constraints: list[str] | None = None,
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
        sequence=[e.source_asset_id for e in edits],
        decisions=[_decision(e.source_asset_id) for e in edits],
        constraints=constraints or [],
        open_questions=[],
        validation_status="pending",
        approval_state="draft",
    )


# 三套镜头：
#   shot_a: in=0,      out=4M   (时长 4M)
#   shot_b: in=8M,     out=11M  (时长 3M)  —— 总时长 15M 的 50%~80% = [7.5M, 12M] 内
#   shot_c: in=12M,    out=15M  (时长 3M)
# 总时长 = 4M + 3M + 3M = 10M us；max(out) = 15M us。
EDITS = [
    _edit("shot_a", 0, 4_000_000),
    _edit("shot_b", 8_000_000, 11_000_000),
    _edit("shot_c", 12_000_000, 15_000_000),
]

# 观测：shot_a blur 过低（30），shot_b / shot_c 正常（80 / 90）
OBS = [
    _obs("shot_a", '{"blur_score": 30.0, "exposure_ok": false}'),
    _obs("shot_b", '{"blur_score": 80.0, "exposure_ok": true}'),
    _obs("shot_c", '{"blur_score": 90.0, "exposure_ok": true}'),
]


# ---------------------------------------------------------------------------
# propose_revision：四种类型都生成合法 RevisionProposal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "revision_type,reason",
    [
        ("extend_peak", "高潮幕张力不足"),
        ("remove_low_quality", "blur<50 的镜头质量不足"),
        ("adjust_duration", "总时长偏离目标"),
        ("reorder", "叙事弧顺序错乱"),
    ],
)
def test_produce_valid_proposal_for_all_types(revision_type, reason):
    edl = _edl(EDITS)
    plan = _plan(EDITS, constraints=["target_duration_us=10000000"])

    proposal = propose_revision(edl, plan, OBS, revision_type, reason)

    assert isinstance(proposal, RevisionProposal)
    assert proposal.proposal_id.startswith("rev_")
    # change_summary 以 "revision_type:" 开头承载类型
    assert proposal.change_summary == f"{revision_type}: {reason}"
    assert proposal.approval_state == "pending"
    assert proposal.regression_risks == []
    assert proposal.source_finding_ids == []
    assert proposal.result_refs == []
    assert proposal.expected_effect is not None
    # BaseRecord 字段
    assert proposal.schema_version == "1.0"
    assert proposal.project_id == edl.project_id
    assert proposal.producer == "revision_engine_v0.1"
    assert proposal.source_ref == edl.source_ref


# ---------------------------------------------------------------------------
# remove_low_quality：正确识别低 blur 镜头
# ---------------------------------------------------------------------------

def test_remove_low_quality_detects_blur_below_threshold():
    edl = _edl(EDITS)
    plan = _plan(EDITS)

    proposal = propose_revision(edl, plan, OBS, "remove_low_quality", "低质量镜头")

    assert proposal.change_summary.startswith("remove_low_quality:")
    # shot_a blur=30 < 50 → 受影响；shot_b / shot_c blur >= 50 → 不受影响
    assert proposal.target_decision_ids == ["shot_a"]
    assert "低质量" in proposal.expected_effect


def test_remove_low_quality_empty_when_all_high_quality():
    edl = _edl(EDITS)
    plan = _plan(EDITS)
    obs_all_good = [
        _obs("shot_a", '{"blur_score": 60.0}'),
        _obs("shot_b", '{"blur_score": 80.0}'),
        _obs("shot_c", '{"blur_score": 90.0}'),
    ]

    proposal = propose_revision(edl, plan, obs_all_good, "remove_low_quality", "x")

    # 没有低质量镜头时，target_decision_ids 为空（合法）
    assert proposal.target_decision_ids == []


# ---------------------------------------------------------------------------
# extend_peak：识别 peak 幕镜头
# ---------------------------------------------------------------------------

def test_extend_peak_targets_peak_shot():
    edl = _edl(EDITS)
    plan = _plan(EDITS)

    proposal = propose_revision(edl, plan, OBS, "extend_peak", "延长高潮")

    assert proposal.change_summary.startswith("extend_peak:")
    # max(out)=15M，peak 范围 [7.5M, 12M]；shot_b.in=8M 落在范围内
    assert "shot_b" in proposal.target_decision_ids
    # shot_a.in=0 不在范围内
    assert "shot_a" not in proposal.target_decision_ids


# ---------------------------------------------------------------------------
# adjust_duration：目标从 constraints 解析
# ---------------------------------------------------------------------------

def test_adjust_duration_targets_last_edit():
    edl = _edl(EDITS)
    plan = _plan(EDITS, constraints=["target_duration_us=12000000"])

    proposal = propose_revision(edl, plan, OBS, "adjust_duration", "时长偏差")

    assert proposal.change_summary.startswith("adjust_duration:")
    # 最后一个 edit 是 shot_c
    assert proposal.target_decision_ids == ["shot_c"]
    assert "12000000" in proposal.expected_effect


# ---------------------------------------------------------------------------
# reorder：所有镜头都受影响
# ---------------------------------------------------------------------------

def test_reorder_targets_all_edits():
    edl = _edl(EDITS)
    plan = _plan(EDITS)

    proposal = propose_revision(edl, plan, OBS, "reorder", "恢复叙事弧")

    assert proposal.change_summary.startswith("reorder:")
    assert set(proposal.target_decision_ids) == {"shot_a", "shot_b", "shot_c"}


# ---------------------------------------------------------------------------
# apply_revision：深拷贝 + 可验证变化
# ---------------------------------------------------------------------------

def test_apply_remove_low_quality_removes_edit_and_decision():
    edl = _edl(EDITS)
    plan = _plan(EDITS)
    original_edits = list(edl.ordered_edits)
    original_decisions = list(plan.decisions)

    proposal = propose_revision(edl, plan, OBS, "remove_low_quality", "低质量")
    new_edl, new_plan = apply_revision(edl, plan, proposal)

    # 原对象未被修改（深拷贝）
    assert len(edl.ordered_edits) == 3
    assert len(plan.decisions) == 3
    assert edl.ordered_edits is not new_edl.ordered_edits

    # 新对象：shot_a 被移除
    remaining_ids = [e.source_asset_id for e in new_edl.ordered_edits]
    assert remaining_ids == ["shot_b", "shot_c"]
    decision_ids = [d.shot_refs[0] for d in new_plan.decisions]
    assert decision_ids == ["shot_b", "shot_c"]
    assert new_plan.sequence == ["shot_b", "shot_c"]
    # expected_duration 重新计算 = 3M + 3M = 6M
    assert new_edl.expected_duration == 6_000_000
    assert new_plan.validation_status == "revised_pending_validation"


def test_apply_remove_low_quality_empty_target_leaves_edl_unchanged():
    edl = _edl(EDITS)
    plan = _plan(EDITS)
    obs_all_good = [
        _obs("shot_a", '{"blur_score": 80.0}'),
        _obs("shot_b", '{"blur_score": 80.0}'),
        _obs("shot_c", '{"blur_score": 90.0}'),
    ]
    proposal = propose_revision(edl, plan, obs_all_good, "remove_low_quality", "x")
    assert proposal.target_decision_ids == []

    new_edl, new_plan = apply_revision(edl, plan, proposal)

    assert len(new_edl.ordered_edits) == 3
    assert new_edl.expected_duration == edl.expected_duration


def test_apply_extend_peak_increases_out_frame():
    edl = _edl(EDITS)
    plan = _plan(EDITS)
    proposal = propose_revision(edl, plan, OBS, "extend_peak", "延长高潮")
    peak_id = proposal.target_decision_ids[0]  # shot_b

    new_edl, new_plan = apply_revision(edl, plan, proposal)

    peak_edit = next(e for e in new_edl.ordered_edits if e.source_asset_id == peak_id)
    original_peak = next(e for e in edl.ordered_edits if e.source_asset_id == peak_id)
    # 延长最多 2_000_000 us
    assert peak_edit.out_frame - original_peak.out_frame == 2_000_000
    # 原对象未变
    assert original_peak.out_frame == 11_000_000


def test_apply_adjust_duration_extends_when_below_target():
    # 当前总时长 = 10M；目标 14M → 延长最后一个 edit (shot_c)
    edl = _edl(EDITS)
    plan = _plan(EDITS, constraints=["target_duration_us=14000000"])
    proposal = propose_revision(edl, plan, OBS, "adjust_duration", "延长")

    new_edl, new_plan = apply_revision(edl, plan, proposal)

    # 最后一个 edit = shot_c，out_frame 从 15M → 19M（补 4M）
    last = new_edl.ordered_edits[-1]
    assert last.source_asset_id == "shot_c"
    assert last.out_frame - last.in_frame == 3_000_000 + 4_000_000  # 原 3M + 补 4M
    # 总时长 = 14M
    assert new_edl.expected_duration == 14_000_000


def test_apply_adjust_duration_truncates_when_above_target():
    # 当前总时长 = 10M；目标 7M → 截断最后一个 edit
    edl = _edl(EDITS)
    plan = _plan(EDITS, constraints=["target_duration_us=7000000"])
    proposal = propose_revision(edl, plan, OBS, "adjust_duration", "缩短")

    new_edl, new_plan = apply_revision(edl, plan, proposal)

    last = new_edl.ordered_edits[-1]
    # shot_c 原 in=12M, out=15M；总时长要降到 7M → 前两段 4M+3M=7M，shot_c 应被截到 in
    # 但至少保留 1us：out = in + 1
    assert last.out_frame > last.in_frame
    assert new_edl.expected_duration == 4_000_000 + 3_000_000 + (last.out_frame - last.in_frame)


def test_apply_reorder_sorts_by_in_frame():
    # 故意打乱顺序
    shuffled = [EDITS[2], EDITS[0], EDITS[1]]  # shot_c, shot_a, shot_b
    edl = _edl(shuffled)
    plan = _plan(shuffled)
    proposal = propose_revision(edl, plan, OBS, "reorder", "恢复叙事弧")

    new_edl, new_plan = apply_revision(edl, plan, proposal)

    ins = [e.in_frame for e in new_edl.ordered_edits]
    assert ins == sorted(ins)
    assert [e.source_asset_id for e in new_edl.ordered_edits] == [
        "shot_a", "shot_b", "shot_c",
    ]
    # plan.sequence 同步重排
    assert new_plan.sequence == ["shot_a", "shot_b", "shot_c"]
    # decisions 按 shot_refs[0] 同步重排
    assert [d.shot_refs[0] for d in new_plan.decisions] == [
        "shot_a", "shot_b", "shot_c",
    ]


# ---------------------------------------------------------------------------
# list_revisions：从 repository 读取
# ---------------------------------------------------------------------------

class _FakeRepo:
    """最小 repository mock：只实现 list。"""

    def __init__(self, items: list[RevisionProposal]) -> None:
        self._items = items

    def list(self, entity_cls, project_id=None):
        return [
            it for it in self._items
            if project_id is None or it.project_id == project_id
        ]


def test_list_revisions_reads_from_repository():
    edl = _edl(EDITS)
    plan = _plan(EDITS)
    p1 = propose_revision(edl, plan, OBS, "reorder", "a")
    p2 = propose_revision(edl, plan, OBS, "extend_peak", "b")
    # 另一项目的提案不应被返回
    other = p1.model_copy(deep=True, update={"proposal_id": "rev_other", "project_id": "other_proj"})

    repo = _FakeRepo([p1, p2, other])
    result = list_revisions("test_proj", repo)

    assert len(result) == 2
    ids = {r.proposal_id for r in result}
    assert p1.proposal_id in ids
    assert p2.proposal_id in ids
    assert all(isinstance(r, RevisionProposal) for r in result)


# ---------------------------------------------------------------------------
# 边界：未知 revision_type 抛错
# ---------------------------------------------------------------------------

def test_unknown_revision_type_raises():
    edl = _edl(EDITS)
    plan = _plan(EDITS)
    with pytest.raises(ValueError, match="revision_type"):
        propose_revision(edl, plan, OBS, "teleport", "?")
