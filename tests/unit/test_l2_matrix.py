"""scripts/l2_matrix.py 单元测试（聚合逻辑，不跑真实剪辑）。"""
from __future__ import annotations

from scripts.l2_matrix import aggregate, run_one


def _produced(tag: str, hard: bool = False, dev: float = 5.0) -> dict:
    return {"tag": tag, "material": "m1", "status": "produced",
            "video": f"{tag}.mp4", "target_seconds": 15, "intent": None,
            "roughcut_exit": 0, "hard_defect": hard, "duration_s": 14.0,
            "duration_deviation_pct": dev, "shots": 4, "asl_s": 3.5,
            "black_segments": 1 if hard else 0, "freeze_segments": 0,
            "has_audio": False}


def test_aggregate_counts_and_table():
    rows = [
        _produced("a", hard=True),
        _produced("b"),
        {"tag": "c", "material": "m2", "status": "refused",
         "reason": "target_duration_unreachable"},
    ]
    report, summary = aggregate(rows)
    assert summary == {
        "total": 3, "produced": 2, "refused": 1, "errored": 0,
        "produced_with_hard_defect": 1,
        "refusal_reasons": ["target_duration_unreachable"],
    }
    assert "| **拒绝** |" in report
    assert "⚠硬伤" in report
    assert "fail-closed" in report  # 拒绝语义必须向读者解释


def test_run_one_missing_material_is_data_not_crash(tmp_path):
    row = run_one({"tag": "x", "video": str(tmp_path / "nope.mp4")}, tmp_path)
    assert row["status"] == "missing_material"
    assert "不存在" in row["reason"]
