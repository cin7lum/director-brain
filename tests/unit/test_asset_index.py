"""arsenal.asset_index 单元测试。"""
from __future__ import annotations

import pytest

from arsenal.asset_index import AssetIndex


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def idx() -> AssetIndex:
    """内存库 AssetIndex。"""
    return AssetIndex(":memory:")


def _make_shot(shot_id: str, media_hash: str = "hash_test", **overrides) -> dict:
    """构造一个测试用镜头 dict。"""
    base = {
        "shot_id": shot_id,
        "media_hash": media_hash,
        "start_us": 0,
        "end_us": 1_000_000,
        "duration_us": 1_000_000,
        "blur_score": 0.15,
        "exposure_ok": True,
        "technical_usable": True,
        "vlm_shot_function": "establishing",
        "vlm_role": "hero",
        "frame_description": "测试画面",
        "analyzed_at": "2026-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ schema

def test_create_tables_in_memory(idx: AssetIndex) -> None:
    """内存库初始化不报错，表可查询。"""
    cur = idx._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row["name"] for row in cur.fetchall()]
    assert "media" in tables
    assert "shots" in tables


# ------------------------------------------------------------------ media

def test_upsert_media_and_list(idx: AssetIndex) -> None:
    """upsert 一条 media，list_media 返回 1 条且字段正确。"""
    idx.upsert_media({
        "media_hash": "abc123",
        "path": "/tmp/test.mp4",
        "duration_us": 52208333,
        "width": 854,
        "height": 480,
        "fps": 24.0,
        "audio_present": True,
        "file_size": 4372373,
        "imported_at": "2026-01-01T00:00:00+00:00",
    })
    result = idx.list_media()
    assert len(result) == 1
    m = result[0]
    assert m["media_hash"] == "abc123"
    assert m["duration_us"] == 52208333
    assert m["width"] == 854
    assert m["height"] == 480
    assert m["fps"] == 24.0
    assert m["audio_present"] is True
    assert m["file_size"] == 4372373


def test_upsert_media_idempotent(idx: AssetIndex) -> None:
    """同一 media_hash upsert 两次，list_media 仍为 1 条。"""
    media = {"media_hash": "dup", "path": "/a.mp4"}
    idx.upsert_media(media)
    idx.upsert_media(media)
    assert len(idx.list_media()) == 1


# ------------------------------------------------------------------ shots

def test_upsert_shot_15_shots(idx: AssetIndex) -> None:
    """upsert 15 个镜头，get_shots 返回 15 条，blur_score 非空。"""
    for i in range(15):
        idx.upsert_shot(_make_shot(f"shot_{i:03d}", start_us=i * 1_000_000))
    shots = idx.get_shots()
    assert len(shots) == 15
    for s in shots:
        assert s["blur_score"] is not None
        assert s["exposure_ok"] is True
        assert s["technical_usable"] is True


def test_upsert_shot_idempotent(idx: AssetIndex) -> None:
    """同一 shot_id upsert 两次，COUNT 不变。"""
    shot = _make_shot("shot_dup")
    idx.upsert_shot(shot)
    idx.upsert_shot(shot)
    assert len(idx.get_shots()) == 1


def test_get_shot_by_id(idx: AssetIndex) -> None:
    """get_shot 按 ID 查询，不存在返回 None。"""
    idx.upsert_shot(_make_shot("shot_001", blur_score=0.42))
    found = idx.get_shot("shot_001")
    assert found is not None
    assert found["shot_id"] == "shot_001"
    assert found["blur_score"] == 0.42
    assert idx.get_shot("nonexistent") is None


def test_get_shots_by_media_hash(idx: AssetIndex) -> None:
    """两个不同 media_hash 各插入镜头，按 hash 过滤返回正确数量。"""
    for i in range(3):
        idx.upsert_shot(_make_shot(f"a_{i}", media_hash="hash_a"))
    for i in range(5):
        idx.upsert_shot(_make_shot(f"b_{i}", media_hash="hash_b"))
    assert len(idx.get_shots("hash_a")) == 3
    assert len(idx.get_shots("hash_b")) == 5
    assert len(idx.get_shots()) == 8  # None 返回全部


def test_is_analyzed_true_false(idx: AssetIndex) -> None:
    """有镜头的 media_hash 返回 True，无镜头的返回 False。"""
    idx.upsert_shot(_make_shot("s1", media_hash="analyzed_hash"))
    assert idx.is_analyzed("analyzed_hash") is True
    assert idx.is_analyzed("never_seen") is False


def test_bool_fields_roundtrip(idx: AssetIndex) -> None:
    """exposure_ok True/False/None 存入后查询返回对应值。"""
    idx.upsert_shot(_make_shot("s_true", exposure_ok=True))
    idx.upsert_shot(_make_shot("s_false", exposure_ok=False))
    idx.upsert_shot(_make_shot("s_none", exposure_ok=None))
    assert idx.get_shot("s_true")["exposure_ok"] is True
    assert idx.get_shot("s_false")["exposure_ok"] is False
    assert idx.get_shot("s_none")["exposure_ok"] is None


def test_technical_usable_roundtrip(idx: AssetIndex) -> None:
    """technical_usable bool 往返。"""
    idx.upsert_shot(_make_shot("s1", technical_usable=False))
    assert idx.get_shot("s1")["technical_usable"] is False


# ------------------------------------------------------------------ lifecycle

def test_close(idx: AssetIndex) -> None:
    """close() 后连接关闭。"""
    idx.close()
    assert idx._conn is None


def test_file_based_db(tmp_path) -> None:
    """文件库：upsert 后关闭重开，数据持久化。"""
    db_path = str(tmp_path / "test_index.db")
    idx1 = AssetIndex(db_path)
    idx1.upsert_media({"media_hash": "persist", "path": "/p.mp4"})
    idx1.upsert_shot(_make_shot("persist_shot", media_hash="persist"))
    idx1.close()

    idx2 = AssetIndex(db_path)
    assert len(idx2.list_media()) == 1
    assert len(idx2.get_shots()) == 1
    assert idx2.is_analyzed("persist") is True
    idx2.close()
