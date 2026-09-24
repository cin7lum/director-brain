"""M2.3 EDL Generator 单元测试。

验证：
- 空 selected_shots → 返回空 EDL，expected_duration=0，不抛异常
- 正常 selected_shots → ordered_edits 数量正确，timebase 正确
- 每个 EditItem 字段正确映射（source_asset_id/hash/in/out/timebase/shot_function/rationale）
- producer == "edl_generator_v0.1"
- edl_id / version / brief_version / context_id / approval_state 字段
- source_asset_hashes 去重且保序
- expected_duration == sum(out_frame - in_frame)
- ordered_edits 顺序与 selected_shots 一致（不重排）
"""
from __future__ import annotations

import time

from director_brain.edl_generator import generate_edl
from director_brain.models.edl import EditorialDecisionList, EditItem


def _shot(
    idx: int,
    *,
    in_frame: int = 1_000_000,
    out_frame: int = 3_000_000,
    asset_id: str | None = None,
    media_hash: str | None = None,
    shot_function: str | None = None,
    rationale: str | None = None,
) -> dict:
    return {
        "source_asset_id": asset_id or f"shot_{idx:08d}",
        "source_media_hash": media_hash or f"hash_{idx}",
        "in_frame": in_frame,
        "out_frame": out_frame,
        "shot_function": shot_function,
        "rationale": rationale,
    }


def test_empty_returns_empty_edl():
    edl = generate_edl("proj_x", selected_shots=[])
    assert isinstance(edl, EditorialDecisionList)
    assert edl.ordered_edits == []
    assert edl.expected_duration == 0
    assert edl.approval_state == "draft"
    assert edl.source_asset_hashes == []


def test_normal_shots_count_and_timebase():
    shots = [_shot(0), _shot(1), _shot(2)]
    edl = generate_edl("proj_x", selected_shots=shots, timebase=1_000_000)
    assert isinstance(edl, EditorialDecisionList)
    assert len(edl.ordered_edits) == 3
    assert edl.timebase == 1_000_000
    for e in edl.ordered_edits:
        assert e.timebase == 1_000_000


def test_edititem_field_mapping():
    shots = [
        _shot(
            0,
            in_frame=1_500_000,
            out_frame=4_500_000,
            asset_id="shot_AAA",
            media_hash="hash_AAA",
            shot_function="hook",
            rationale="best blur",
        )
    ]
    edl = generate_edl("proj_y", selected_shots=shots, timebase=1_000_000)
    e = edl.ordered_edits[0]
    assert isinstance(e, EditItem)
    assert e.source_asset_id == "shot_AAA"
    assert e.source_media_hash == "hash_AAA"
    assert e.in_frame == 1_500_000
    assert e.out_frame == 4_500_000
    assert e.timebase == 1_000_000
    assert e.shot_function == "hook"
    assert e.rationale == "best blur"


def test_producer_field():
    edl = generate_edl("proj_z", selected_shots=[_shot(0)])
    assert edl.producer == "edl_generator_v0.1"


def test_metadata_fields():
    edl = generate_edl("proj_meta", selected_shots=[_shot(0)])
    assert edl.edl_id.startswith("edl_proj_meta_")
    assert edl.version == "0.1"
    assert edl.brief_version == "0.1"
    assert edl.context_id == "ctx_proj_meta"
    assert edl.approval_state == "draft"
    assert edl.project_id == "proj_meta"
    assert isinstance(edl.created_at, int)
    # created_at 应接近当前时间
    assert abs(edl.created_at - int(time.time())) < 5


def test_source_asset_hashes_dedup_preserving_order():
    # 同一 hash 出现两次，应只保留第一次出现
    shots = [
        _shot(0, media_hash="hash_A"),
        _shot(1, media_hash="hash_B"),
        _shot(2, media_hash="hash_A"),  # 重复
        _shot(3, media_hash="hash_C"),
    ]
    edl = generate_edl("proj_dedup", selected_shots=shots)
    assert edl.source_asset_hashes == ["hash_A", "hash_B", "hash_C"]


def test_expected_duration_sum():
    shots = [
        _shot(0, in_frame=0, out_frame=2_000_000),
        _shot(1, in_frame=5_000_000, out_frame=8_000_000),
        _shot(2, in_frame=10_000_000, out_frame=10_500_000),
    ]
    edl = generate_edl("proj_dur", selected_shots=shots)
    # 2M + 3M + 0.5M = 5.5M
    assert edl.expected_duration == 5_500_000


def test_order_preserved_not_resorted():
    # selected_shots 故意乱序（out < in 也没关系，generator 不校验）
    shots = [
        _shot(9, in_frame=9_000_000, out_frame=10_000_000),
        _shot(1, in_frame=1_000_000, out_frame=2_000_000),
        _shot(5, in_frame=5_000_000, out_frame=6_000_000),
    ]
    edl = generate_edl("proj_order", selected_shots=shots)
    assert [e.source_asset_id for e in edl.ordered_edits] == [
        "shot_00000009",
        "shot_00000001",
        "shot_00000005",
    ]


def test_default_timebase():
    shots = [_shot(0)]
    edl = generate_edl("proj_tb", selected_shots=shots)
    assert edl.timebase == 1_000_000
    assert edl.ordered_edits[0].timebase == 1_000_000


def test_source_ref_used_when_provided():
    shots = [_shot(0)]
    edl = generate_edl("proj", shots, source_ref="/path/to/video.mp4")
    assert edl.source_ref == "/path/to/video.mp4"


def test_source_ref_defaults_to_project_id():
    shots = [_shot(0)]
    edl = generate_edl("proj", shots)
    assert edl.source_ref == "proj"
