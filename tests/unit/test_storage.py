"""W4 存储层单元测试。

覆盖：
1. save -> get 回读一致（DirectorBrief / FilmObservation / EDL / DirectorDecisionPlan）
2. list(project_id=...) 过滤
3. update 局部更新
4. LocalObjectStore CRUD 与 sha256
5. 工厂函数 get_repository / get_object_store
6. invalidate_context 使上下文快照失效
7. delete 后 get 返回 None
另附 Decision Ledger 往返冒烟测试。
"""
from __future__ import annotations

import hashlib

import pytest
import storage.sqlite_repository as sqlite_repository_module

from director_brain.models import (
    ClaimKind,
    Decision,
    DirectorBrief,
    DirectorDecisionPlan,
    EditItem,
    EditorialDecisionList,
    FilmContextSnapshot,
    FilmObservation,
)
from storage import (
    BrainRepository,
    LocalObjectStore,
    SqliteRepository,
    get_object_store,
    get_repository,
)
from storage.repository import DecisionLedgerEntry


# ---------------------------------------------------------------------------
# 测试数据构造辅助
# ---------------------------------------------------------------------------

def _make_brief(brief_id: str = "b-1", project_id: str = "P-A") -> DirectorBrief:
    return DirectorBrief(
        project_id=project_id,
        created_at=1700000000,
        producer="ut",
        source_ref="brief.md#v1",
        brief_id=brief_id,
        version="1",
        source_text="A heist story.",
        creator_direction="Begin with suspicion; reveal the crew's loyalty later.",
        language="zh",
        intent="create tension",
        audience="adults",
        target_duration=120,
        delivery_profile="online",
        themes=["heist", "loyalty"],
        relationships=["crew"],
        emotional_arc="rise-fall-rise",
        visual_language="neon-noir",
        editing_language="fast cuts",
        sound_language="low drone",
        must_include=["mask reveal"],
        must_avoid=["spoiler"],
        privacy_constraints=["blur faces"],
        approval_state="draft",
    )


def _make_observation(obs_id: str = "o-1", project_id: str = "P-A") -> FilmObservation:
    return FilmObservation(
        project_id=project_id,
        created_at=1700000001,
        producer="ut",
        source_ref="clip.mp4#0",
        observation_id=obs_id,
        media_asset_id="asset-1",
        media_hash="h-abc",
        start_frame=0,
        end_frame=50,
        timebase=25,
        observation_type="motion",
        claim="a car passes",
        provider="test-provider",
        model_version="m-1",
        prompt_version="p-1",
        confidence=0.9,
        evidence_refs=["frame://10"],
        review_state="pending",
        claim_kind=ClaimKind.MEASURED,
    )


def _make_edl(edl_id: str = "e-1", project_id: str = "P-A") -> EditorialDecisionList:
    return EditorialDecisionList(
        project_id=project_id,
        created_at=1700000002,
        producer="ut",
        source_ref="plan-1",
        edl_id=edl_id,
        version="1",
        brief_version="1",
        context_id="ctx-1",
        source_asset_hashes=["h-abc"],
        timebase=25,
        ordered_edits=[
            EditItem(
                source_asset_id="asset-1",
                source_media_hash="h-abc",
                in_frame=0,
                out_frame=24,
                timebase=25,
                transition="cut",
                effect_refs=["fade"],
                shot_function="establish",
                rationale="opens the scene",
            )
        ],
        decision_refs=["d-1"],
        approval_state="draft",
    )


def _make_plan(plan_id: str = "p-1", project_id: str = "P-A") -> DirectorDecisionPlan:
    return DirectorDecisionPlan(
        project_id=project_id,
        created_at=1700000003,
        producer="ut",
        source_ref="edl-1",
        plan_id=plan_id,
        version="1",
        brief_version="1",
        film_state_version="1",
        sequence=["e-1"],
        decisions=[
            Decision(
                decision_id="d-1",
                purpose="build tension",
                shot_refs=["asset-1"],
                evidence_refs=["o-1"],
                rationale="matches arc",
                alternatives=["hold longer"],
                confidence=0.8,
                requires_approval=False,
            )
        ],
        constraints=["stay under 2min"],
        open_questions=["music?"],
        validation_status="ok",
        approval_state="draft",
    )


def _make_context(context_id: str = "ctx-1", project_id: str = "P-A") -> FilmContextSnapshot:
    return FilmContextSnapshot(
        project_id=project_id,
        created_at=1700000004,
        producer="ut",
        source_ref="analyze#1",
        context_id=context_id,
        asset_refs=["asset-1"],
        source_content_hashes=["h-abc"],
        analysis_fingerprint="fp-1",
        provider="test-provider",
        model="m-1",
        prompt_version="p-1",
        sampling_config={"temperature": 0.2},
        timebase=25,
        coverage="full",
        rights_scope="internal",
        evidence_refs=["o-1"],
        cache_state="cached",
    )


@pytest.fixture()
def repo(tmp_path) -> SqliteRepository:
    return SqliteRepository(db_path=str(tmp_path / "brain.db"))


# ---------------------------------------------------------------------------
# 1. save -> get 回读一致
# ---------------------------------------------------------------------------

def test_roundtrip_director_brief(repo: SqliteRepository):
    brief = _make_brief()
    repo.save(brief)
    got = repo.get(DirectorBrief, "b-1")
    assert got is not None
    assert got == brief


def test_roundtrip_film_observation(repo: SqliteRepository):
    obs = _make_observation()
    repo.save(obs)
    got = repo.get(FilmObservation, "o-1")
    assert got is not None
    assert got == obs
    assert got.claim_kind == ClaimKind.MEASURED


def test_roundtrip_edl_with_edit_item(repo: SqliteRepository):
    edl = _make_edl()
    repo.save(edl)
    got = repo.get(EditorialDecisionList, "e-1")
    assert got is not None
    assert got == edl
    from director_brain.models.edl import TransitionSpec
    assert got.ordered_edits[0].transition == TransitionSpec(type="cut")


def test_roundtrip_plan_with_decision(repo: SqliteRepository):
    plan = _make_plan()
    repo.save(plan)
    got = repo.get(DirectorDecisionPlan, "p-1")
    assert got is not None
    assert got == plan
    assert got.decisions[0].decision_id == "d-1"


def test_roundtrip_ledger(repo: SqliteRepository):
    entry = DecisionLedgerEntry(
        ledger_id="lg-1",
        decision_id="d-1",
        action="created",
        timestamp=1700000005,
        detail={"who": "ut"},
    )
    repo.save(entry)
    got = repo.get(DecisionLedgerEntry, "lg-1")
    assert got == entry


# ---------------------------------------------------------------------------
# 2. list 过滤
# ---------------------------------------------------------------------------

def test_list_filters_by_project_id(repo: SqliteRepository):
    repo.save(_make_brief("b-a1", project_id="P-A"))
    repo.save(_make_brief("b-a2", project_id="P-A"))
    repo.save(_make_brief("b-b1", project_id="P-B"))

    only_a = repo.list(DirectorBrief, project_id="P-A")
    assert {b.brief_id for b in only_a} == {"b-a1", "b-a2"}

    all_records = repo.list(DirectorBrief)
    assert {b.brief_id for b in all_records} == {"b-a1", "b-a2", "b-b1"}


# ---------------------------------------------------------------------------
# 3. update 局部更新
# ---------------------------------------------------------------------------

def test_update_partial_fields(repo: SqliteRepository):
    brief = _make_brief()
    repo.save(brief)

    updated = repo.update(
        DirectorBrief,
        "b-1",
        approval_state="approved",
        themes=["heist", "loyalty", "betrayal"],
    )
    assert updated.approval_state == "approved"
    assert updated.themes == ["heist", "loyalty", "betrayal"]
    # 未传入的字段保持不变
    assert updated.intent == "create tension"
    assert updated.target_duration == 120

    got = repo.get(DirectorBrief, "b-1")
    assert got == updated


def test_update_rejects_invalid_field(repo: SqliteRepository):
    brief = _make_brief()
    repo.save(brief)

    with pytest.raises(ValueError, match="nonexistent_field"):
        repo.update(
            DirectorBrief,
            "b-1",
            nonexistent_field="bad",
        )

    # 非法字段未写入数据库：原数据保持不变
    got = repo.get(DirectorBrief, "b-1")
    assert got is not None
    assert got.approval_state == "draft"
    assert not hasattr(got, "nonexistent_field")


def test_update_valid_field_succeeds(repo: SqliteRepository):
    brief = _make_brief()
    repo.save(brief)

    updated = repo.update(DirectorBrief, "b-1", approval_state="approved")
    assert updated.approval_state == "approved"
    # 未传入字段保持原值
    assert updated.intent == "create tension"

    got = repo.get(DirectorBrief, "b-1")
    assert got.approval_state == "approved"


# ---------------------------------------------------------------------------
# 4. 对象存储 CRUD
# ---------------------------------------------------------------------------

def test_object_store_crud(tmp_path):
    store = LocalObjectStore(base_path=str(tmp_path / "objects"))
    payload = b"fake-media-bytes" * 10

    digest = store.put("shots/a/frame1.bin", payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert store.exists("shots/a/frame1.bin") is True
    assert store.get("shots/a/frame1.bin") == payload

    keys = store.list_keys("shots/")
    assert "shots/a/frame1.bin" in keys

    assert store.delete("shots/a/frame1.bin") is True
    assert store.exists("shots/a/frame1.bin") is False
    assert store.delete("shots/a/frame1.bin") is False


def test_object_store_get_nonexistent_returns_none(tmp_path):
    """get() 对不存在的 key 返回 None，不抛异常。"""
    store = LocalObjectStore(base_path=str(tmp_path / "objects"))
    assert store.get("nonexistent/key.bin") is None


def test_object_store_get_or_raise_nonexistent_raises(tmp_path):
    """get_or_raise() 对不存在的 key 抛 FileNotFoundError。"""
    store = LocalObjectStore(base_path=str(tmp_path / "objects"))
    with pytest.raises(FileNotFoundError):
        store.get_or_raise("nonexistent/key.bin")


def test_object_store_get_or_raise_existent_returns_bytes(tmp_path):
    """get_or_raise() 对存在的 key 正常返回 bytes。"""
    store = LocalObjectStore(base_path=str(tmp_path / "objects"))
    payload = b"hello-world"
    store.put("data/k.bin", payload)
    assert store.get_or_raise("data/k.bin") == payload


# ---------------------------------------------------------------------------
# 5. 工厂函数
# ---------------------------------------------------------------------------

def test_factory_functions(tmp_path):
    repo = get_repository("sqlite", db_path=str(tmp_path / "f.db"))
    assert isinstance(repo, SqliteRepository)
    assert isinstance(repo, BrainRepository)

    store = get_object_store(str(tmp_path / "obj"))
    assert isinstance(store, LocalObjectStore)

    with pytest.raises(ValueError):
        get_repository("s3", db_path=str(tmp_path / "x.db"))


def test_sqlite_repository_pins_extra_synchronous_durability(tmp_path, monkeypatch):
    original_connect = sqlite_repository_module.sqlite3.connect

    def connect_with_normal_synchronous(db_path):
        connection = original_connect(db_path)
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    monkeypatch.setattr(
        sqlite_repository_module.sqlite3,
        "connect",
        connect_with_normal_synchronous,
    )
    repo = SqliteRepository(str(tmp_path / "durability.db"))
    try:
        assert repo._conn.execute("PRAGMA synchronous").fetchone()[0] == 3
        repo.save(_make_brief("durability-brief"))
    finally:
        repo.close()


def test_project_story_link_ranking_snapshot_is_immutable_and_persistent(tmp_path):
    db_path = tmp_path / "ranking-cache.db"
    identity = (
        "project-1",
        "candidate-set-1",
        "bge_m3_shadow_v1",
        "bge-m3:latest",
        "a" * 64,
        "cosine_similarity",
    )
    entries = [("candidate-a", 0.91), ("candidate-b", -0.2)]
    repo = SqliteRepository(str(db_path))
    repo.save_project_story_link_ranking_snapshot(*identity, entries)
    repo.save_project_story_link_ranking_snapshot(*identity, entries)
    with pytest.raises(ValueError, match="result drift"):
        repo.save_project_story_link_ranking_snapshot(
            *identity, [("candidate-a", 0.1), ("candidate-b", 0.2)])
    repo.close()

    restarted_repo = SqliteRepository(str(db_path))
    try:
        assert restarted_repo.get_project_story_link_ranking_snapshot(
            *identity) == entries
        assert restarted_repo.get_project_story_link_ranking_snapshot(
            *identity[:2], "different-profile", *identity[3:]) is None
    finally:
        restarted_repo.close()


# ---------------------------------------------------------------------------
# 6. context invalidate
# ---------------------------------------------------------------------------

def test_invalidate_context(repo: SqliteRepository):
    ctx = _make_context()
    repo.save(ctx)
    assert repo.get(FilmContextSnapshot, "ctx-1").invalidated_at is None

    assert repo.invalidate_context("ctx-1") is True
    got = repo.get(FilmContextSnapshot, "ctx-1")
    assert got.invalidated_at is not None
    assert isinstance(got.invalidated_at, int)


# ---------------------------------------------------------------------------
# 7. delete
# ---------------------------------------------------------------------------

def test_delete(repo: SqliteRepository):
    repo.save(_make_observation())
    assert repo.get(FilmObservation, "o-1") is not None
    assert repo.delete(FilmObservation, "o-1") is True
    assert repo.get(FilmObservation, "o-1") is None
    assert repo.delete(FilmObservation, "o-1") is False
