"""M3.3 Decision Ledger + 审计追踪 单元测试。

验证：
1. log_decision 成功保存并返回 DecisionLedgerEntry
2. get_decision_history 按时间升序返回正确条目
3. get_decision_history 过滤不同 decision_id 互不干扰
4. generate_audit_report 含三个必需字段
5. evidence_integrity 正确识别 missing assets
"""
from __future__ import annotations

import time

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.audit_trail import (
    generate_audit_report,
    get_decision_history,
    log_decision,
)
from storage import get_repository
from storage.repository import DecisionLedgerEntry


# ---------------------------------------------------------------------------
# Fixture 辅助（参考 test_plan_validator 风格）
# ---------------------------------------------------------------------------

def _repo():
    """每个测试用全新的内存 sqlite 库。"""
    return get_repository("sqlite", db_path=":memory:")


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
        shot_function="heuristic_selected",
        rationale="test rationale",
    )


def _edl(edits: list[EditItem]) -> EditorialDecisionList:
    hashes: list[str] = []
    seen: set[str] = set()
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


def _plan() -> DirectorDecisionPlan:
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
        sequence=[],
        decisions=[],
        constraints=[],
        open_questions=[],
        validation_status="pending",
        approval_state="draft",
    )


# ---------------------------------------------------------------------------
# 1. log_decision 保存并返回
# ---------------------------------------------------------------------------

def test_log_decision_saves_and_returns_entry():
    repo = _repo()
    entry = log_decision(
        repo,
        decision_id="plan_001",
        action="plan_generated",
        detail={"producer": "heuristic", "shot_count": 7},
    )

    # 返回值类型正确
    assert isinstance(entry, DecisionLedgerEntry)
    assert entry.decision_id == "plan_001"
    assert entry.action == "plan_generated"
    assert entry.detail == {"producer": "heuristic", "shot_count": 7}
    assert entry.ledger_id.startswith("ledger_")
    assert isinstance(entry.timestamp, int)

    # 已持久化：可从 repo 读回
    stored = repo.list(DecisionLedgerEntry)
    assert len(stored) == 1
    assert stored[0].ledger_id == entry.ledger_id
    assert stored[0].decision_id == "plan_001"


# ---------------------------------------------------------------------------
# 2. get_decision_history 按时间升序
# ---------------------------------------------------------------------------

def test_get_decision_history_sorted_ascending():
    repo = _repo()
    log_decision(repo, "plan_001", "plan_generated", {"producer": "heuristic"})
    time.sleep(0.02)  # 确保时间戳递增
    log_decision(repo, "plan_001", "revision_applied", {"revision_type": "remove_low_quality"})
    time.sleep(0.02)
    log_decision(repo, "plan_001", "plan_validated", {"is_valid": True})

    history = get_decision_history(repo, "plan_001")
    assert len(history) == 3
    assert [h.action for h in history] == [
        "plan_generated",
        "revision_applied",
        "plan_validated",
    ]
    # 时间戳严格升序
    assert history[0].timestamp <= history[1].timestamp <= history[2].timestamp


# ---------------------------------------------------------------------------
# 3. get_decision_history 按 decision_id 过滤互不干扰
# ---------------------------------------------------------------------------

def test_get_decision_history_filters_by_decision_id():
    repo = _repo()
    # plan_001 的两条
    log_decision(repo, "plan_001", "plan_generated", {})
    log_decision(repo, "plan_001", "plan_validated", {"is_valid": True})
    # plan_002 的一条
    log_decision(repo, "plan_002", "plan_generated", {})

    h1 = get_decision_history(repo, "plan_001")
    h2 = get_decision_history(repo, "plan_002")

    assert len(h1) == 2
    assert len(h2) == 1
    assert all(h.decision_id == "plan_001" for h in h1)
    assert h2[0].decision_id == "plan_002"

    # 不存在的 decision_id 返回空列表
    assert get_decision_history(repo, "plan_999") == []


# ---------------------------------------------------------------------------
# 4. generate_audit_report 含三个必需字段
# ---------------------------------------------------------------------------

def test_generate_audit_report_has_required_sections():
    repo = _repo()
    obs = [_obs("shot_a"), _obs("shot_b")]
    edits = [
        _edit("shot_a", 1_000_000, 3_000_000),
        _edit("shot_b", 5_000_000, 8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()

    # 记录与该 plan 相关的决策
    log_decision(repo, plan.plan_id, "plan_generated", {"producer": "heuristic"})
    log_decision(repo, plan.plan_id, "plan_validated", {"is_valid": True})

    report = generate_audit_report(edl, plan, obs, repo)

    # 三个必需字段
    assert "edl_summary" in report
    assert "decision_timeline" in report
    assert "evidence_integrity" in report

    # edl_summary 内容
    s = report["edl_summary"]
    assert s["edl_id"] == "test_edl"
    assert s["shot_count"] == 2
    assert s["total_duration_us"] == (3_000_000 - 1_000_000) + (8_000_000 - 5_000_000)
    assert s["timebase"] == 1_000_000
    assert s["approval_state"] == "draft"
    assert s["plan_id"] == "test_plan"
    assert s["validation_status"] == "pending"

    # decision_timeline 含两条且按时间升序
    tl = report["decision_timeline"]
    assert len(tl) == 2
    assert [t["action"] for t in tl] == ["plan_generated", "plan_validated"]
    assert tl[0]["timestamp"] <= tl[1]["timestamp"]
    assert "detail" in tl[0] and "timestamp" in tl[0]


# ---------------------------------------------------------------------------
# 5. evidence_integrity 正确识别 missing assets
# ---------------------------------------------------------------------------

def test_evidence_integrity_detects_missing_assets():
    repo = _repo()
    # observations 只有 shot_a / shot_b
    obs = [_obs("shot_a"), _obs("shot_b")]
    # EDL 引用了一个不存在的 shot_ghost
    edits = [
        _edit("shot_a", 1_000_000, 3_000_000),
        _edit("shot_ghost", 4_000_000, 6_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()

    report = generate_audit_report(edl, plan, obs, repo)
    ei = report["evidence_integrity"]

    assert ei["all_assets_found"] is False
    assert "shot_ghost" in ei["missing_assets"]
    assert "shot_a" not in ei["missing_assets"]
    assert ei["edit_count"] == 2
    assert ei["verified_count"] == 1  # 只有 shot_a 能匹配


def test_evidence_integrity_all_assets_found():
    repo = _repo()
    obs = [_obs("shot_a"), _obs("shot_b")]
    edits = [
        _edit("shot_a", 1_000_000, 3_000_000),
        _edit("shot_b", 5_000_000, 8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()

    report = generate_audit_report(edl, plan, obs, repo)
    ei = report["evidence_integrity"]

    assert ei["all_assets_found"] is True
    assert ei["missing_assets"] == []
    assert ei["edit_count"] == 2
    assert ei["verified_count"] == 2


# ---------------------------------------------------------------------------
# 6. edit_traceability 包含 observation 匹配
# ---------------------------------------------------------------------------

def test_edit_traceability_matches_observation():
    repo = _repo()
    obs = [_obs("shot_a"), _obs("shot_b")]
    edits = [
        _edit("shot_a", 1_000_000, 3_000_000),
        _edit("shot_b", 5_000_000, 8_000_000),
    ]
    edl = _edl(edits)
    plan = _plan()

    report = generate_audit_report(edl, plan, obs, repo)
    trace = report["edit_traceability"]

    assert len(trace) == 2
    # 第一条应匹配到 obs_shot_a
    first = trace[0]
    assert first["source_asset_id"] == "shot_a"
    assert first["observation_id"] == "obs_shot_a"
    assert first["in_frame"] == 1_000_000
    assert first["out_frame"] == 3_000_000
    assert "shot_function" in first
    assert "rationale" in first
