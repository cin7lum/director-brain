"""EDL Exporter 单元测试。

验证 CMX 3600 EDL 导出：
- 微秒转时间码的基本换算与帧边界钳制；
- 头部 TITLE / FCM；
- 记录行数、字段宽度与时间码累加；
- 输出目录自动创建；空 EDL 只输出头部。
"""
from __future__ import annotations

import pytest

from director_brain.models.edl import EditorialDecisionList, EditItem
from execution.edl_exporter import _us_to_tc, export_to_davinci_edl


def _make_edl(edits, edl_id="edl_test"):
    return EditorialDecisionList(
        schema_version="1.0", project_id="test", created_at=0,
        producer="test", source_ref="test.mp4",
        edl_id=edl_id, version="0.1", brief_version="0.1",
        context_id="ctx_test", timebase=1_000_000,
        ordered_edits=edits, approval_state="draft",
    )


def _edit(in_f, out_f, asset_id="shot_001_long_name"):
    return EditItem(
        source_asset_id=asset_id, source_media_hash="hash1",
        in_frame=in_f, out_frame=out_f, timebase=1_000_000,
    )


def test_us_to_tc_basic():
    assert _us_to_tc(0, 24.0) == "00:00:00:00"
    assert _us_to_tc(1_500_000, 24.0) == "00:00:01:12"
    assert _us_to_tc(3_000_000, 24.0) == "00:00:03:00"


def test_us_to_tc_frame_clamp():
    # 接近整秒的边界值：浮点误差可能使 FF == fps，需钳制到 fps-1
    fps = 24.0
    # 23 帧对应的微秒：23/24 秒 ≈ 958_333us
    tc = _us_to_tc(958_333, fps)
    frame = int(tc.split(":")[-1])
    assert frame < fps
    # 整秒附近：1,000,000us 应恰好为下一秒第 0 帧
    assert _us_to_tc(1_000_000, fps) == "00:00:01:00"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def test_edl_header(tmp_path):
    edl = _make_edl([_edit(0, 1_000_000)], edl_id="my_timeline")
    out = tmp_path / "out.edl"
    export_to_davinci_edl(edl, str(out))
    lines = _read(str(out))
    assert lines[0] == "TITLE: my_timeline"
    assert lines[1] == "FCM: NON-DROP FRAME"


def test_edl_record_count(tmp_path):
    edits = [_edit(i * 1_000_000, (i + 1) * 1_000_000) for i in range(3)]
    edl = _make_edl(edits)
    out = tmp_path / "out.edl"
    export_to_davinci_edl(edl, str(out))
    lines = _read(str(out))
    assert len(lines) == 2 + 2 * 3


def test_edl_record_fields(tmp_path):
    edl = _make_edl([_edit(0, 1_500_000, asset_id="shot_001_long_name")])
    out = tmp_path / "out.edl"
    export_to_davinci_edl(edl, str(out))
    lines = _read(str(out))
    rec = lines[2].split()
    assert rec[0] == "001"
    assert rec[1] == "shot_001"  # source_asset_id 前 8 位
    assert rec[2] == "V"
    assert rec[3] == "C"
    assert rec[4] == "00:00:00:00"  # 源入点
    assert rec[5] == "00:00:01:12"  # 源出点 1.5s@24
    assert lines[3] == "* FROM CLIP NAME: shot_001_long_name"


def test_record_timecode_accumulation(tmp_path):
    # 第一条 1.5s（=36 帧 @24fps），第二条从记录 00:00:01:12 开始
    edits = [
        _edit(10_000_000, 11_500_000),  # 时长 1.5s
        _edit(20_000_000, 21_000_000),  # 时长 1.0s
    ]
    edl = _make_edl(edits)
    out = tmp_path / "out.edl"
    export_to_davinci_edl(edl, str(out))
    lines = _read(str(out))
    rec1 = lines[2].split()
    rec2 = lines[4].split()
    # 第一条记录入点从 0 开始
    assert rec1[6] == "00:00:00:00"
    # 第一条记录出点 = 1.5s = 00:00:01:12
    assert rec1[7] == "00:00:01:12"
    # 第二条记录入点 = 第一条时长 = 00:00:01:12
    assert rec2[6] == "00:00:01:12"
    # 第二条记录出点 = 1.5 + 1.0 = 2.5s = 00:00:02:12
    assert rec2[7] == "00:00:02:12"


def test_output_dir_creation(tmp_path):
    edl = _make_edl([_edit(0, 1_000_000)])
    out = tmp_path / "new_dir" / "sub" / "out.edl"
    assert not out.parent.exists()
    result = export_to_davinci_edl(edl, str(out))
    assert out.exists()
    assert result == str(out)


def test_empty_edl(tmp_path):
    edl = _make_edl([])
    out = tmp_path / "empty.edl"
    export_to_davinci_edl(edl, str(out))
    lines = _read(str(out))
    assert lines == ["TITLE: edl_test", "FCM: NON-DROP FRAME"]
