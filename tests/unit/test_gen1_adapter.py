"""W5 V0.1 导入适配器 + heuristic baseline 测试。

覆盖 7 个场景：
1. 导入真实 proposal 文件，EDL.ordered_edits 数量与 slots 一致
2. 字段映射正确性（source_asset_id / in_frame / rationale / timebase）
3. 导入→导出→再导入往返，关键字段一致
4. heuristic 生成总时长在 target 的 90%-110% 范围内
5. heuristic VLM 权重排序正确（ATMOSPHERIC_EVIDENCE vs discard）
6. 空 slots 不抛异常，返回空 ordered_edits
7. 缺少 source_shot_id 的 slot 被跳过并在 import_log 记录警告
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList
from gen1_adapter.heuristic_baseline import HeuristicBaseline, compute_clip_window
from gen1_adapter.v01_exporter import export_to_v01
from gen1_adapter.v01_importer import import_v01_proposal

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SMALL_FIXTURE = FIXTURES_DIR / "prop_f96e39f69870.json"
MULTI_FIXTURE = FIXTURES_DIR / "prop_15e408499380.json"


# ---------------------------------------------------------------------------
# 1. 导入真实文件
# ---------------------------------------------------------------------------

class TestImportRealFile:
    """场景 1：导入真实 proposal JSON，slots 数 == ordered_edits 数。"""

    def test_small_proposal_slot_count(self):
        edl, plan, log = import_v01_proposal(str(SMALL_FIXTURE))
        assert isinstance(edl, EditorialDecisionList)
        assert isinstance(plan, DirectorDecisionPlan)
        assert isinstance(log, list)
        with open(SMALL_FIXTURE, encoding="utf-8") as f:
            raw = json.load(f)
        assert len(edl.ordered_edits) == len(raw["slots"])
        assert len(plan.decisions) == len(raw["slots"])

    def test_multi_slot_proposal_slot_count(self):
        edl, plan, log = import_v01_proposal(str(MULTI_FIXTURE))
        with open(MULTI_FIXTURE, encoding="utf-8") as f:
            raw = json.load(f)
        assert len(edl.ordered_edits) == len(raw["slots"])
        assert len(plan.decisions) == len(raw["slots"])

    def test_public_fields_filled(self):
        edl, plan, log = import_v01_proposal(str(SMALL_FIXTURE))
        with open(SMALL_FIXTURE, encoding="utf-8") as f:
            raw = json.load(f)
        assert edl.project_id == raw["project_id"]
        assert edl.producer == raw["ai_model"]
        assert edl.timebase == 1000000
        assert edl.approval_state == "imported"
        assert edl.source_ref == str(SMALL_FIXTURE)
        assert plan.project_id == raw["project_id"]
        assert plan.producer == raw["ai_model"]
        assert plan.validation_status == "unverified"
        assert plan.approval_state == "imported"


# ---------------------------------------------------------------------------
# 2. 字段映射
# ---------------------------------------------------------------------------

class TestFieldMapping:
    """场景 2：逐个字段映射断言。"""

    def test_edit_item_mapping(self):
        edl, plan, log = import_v01_proposal(str(MULTI_FIXTURE))
        with open(MULTI_FIXTURE, encoding="utf-8") as f:
            raw = json.load(f)
        # 按 position 排序后逐 slot 对比
        sorted_slots = sorted(raw["slots"], key=lambda s: s["position"])
        for item, slot in zip(edl.ordered_edits, sorted_slots):
            assert item.source_asset_id == slot["source_shot_id"]
            assert item.source_media_hash == slot["source_media_hash"]
            assert item.in_frame == slot["proposed_in_us"]
            assert item.out_frame == slot["proposed_out_us"]
            assert item.timebase == 1000000
            # reason 兼容两种字段名
            expected_reason = slot.get("reason") or slot.get("proposed_reason")
            assert item.rationale == expected_reason
            assert item.shot_function == slot.get("proposed_role")

    def test_decision_mapping(self):
        edl, plan, log = import_v01_proposal(str(MULTI_FIXTURE))
        with open(MULTI_FIXTURE, encoding="utf-8") as f:
            raw = json.load(f)
        sorted_slots = sorted(raw["slots"], key=lambda s: s["position"])
        for dec, slot in zip(plan.decisions, sorted_slots):
            assert dec.decision_id == f"dec_{slot['slot_id']}"
            expected_reason = slot.get("reason") or slot.get("proposed_reason")
            assert dec.rationale == expected_reason
            assert dec.confidence == slot.get("confidence_value")
            assert dec.requires_approval is False


# ---------------------------------------------------------------------------
# 3. 导出往返
# ---------------------------------------------------------------------------

class TestExportRoundTrip:
    """场景 3：导入→导出→再导入，关键字段一致。"""

    def test_round_trip_key_fields(self):
        edl1, plan1, _ = import_v01_proposal(str(MULTI_FIXTURE))
        exported = export_to_v01(edl1, plan1)
        # 导出结果可直接 json.dumps
        json_str = json.dumps(exported)
        reloaded = json.loads(json_str)

        # 写临时文件再导入
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(reloaded, tf)
            tmp_path = tf.name
        try:
            edl2, plan2, _ = import_v01_proposal(tmp_path)
        finally:
            os.unlink(tmp_path)

        assert len(edl2.ordered_edits) == len(edl1.ordered_edits)
        for it1, it2, dec1, dec2 in zip(
            edl1.ordered_edits, edl2.ordered_edits,
            plan1.decisions, plan2.decisions,
        ):
            assert it1.source_asset_id == it2.source_asset_id
            assert it1.in_frame == it2.in_frame
            assert it1.out_frame == it2.out_frame
            assert it1.rationale == it2.rationale
            assert dec1.confidence == dec2.confidence

    def test_exported_slot_structure(self):
        edl1, plan1, _ = import_v01_proposal(str(SMALL_FIXTURE))
        exported = export_to_v01(edl1, plan1)
        assert "proposal_id" in exported
        assert "project_id" in exported
        assert "slots" in exported
        assert "ai_model" in exported
        assert "rough_cut_id" in exported
        slot = exported["slots"][0]
        for key in (
            "slot_id", "position", "source_shot_id", "source_media_hash",
            "proposed_in_us", "proposed_out_us", "reason",
            "clip_instance_id", "confidence_type", "confidence_value",
        ):
            assert key in slot, f"missing key {key}"
        assert slot["clip_instance_id"], "clip_instance_id must be non-empty"

    def test_export_has_clip_instance_id_and_rough_cut_id(self):
        edl1, plan1, _ = import_v01_proposal(str(MULTI_FIXTURE))
        exported = export_to_v01(edl1, plan1)
        # rough_cut_id 非空且以 rc_ 开头
        assert exported["rough_cut_id"], "rough_cut_id must be non-empty"
        assert exported["rough_cut_id"].startswith("rc_")
        # 每个 slot 的 clip_instance_id 存在且非空
        for slot in exported["slots"]:
            cid = slot.get("clip_instance_id")
            assert cid, f"slot {slot.get('slot_id')} missing clip_instance_id"
            assert cid.startswith("clip_")


# ---------------------------------------------------------------------------
# 4. heuristic 时长
# ---------------------------------------------------------------------------

def _make_candidate(
    shot_id: str,
    blur: float,
    src_in: int = 0,
    src_dur: int = 3_000_000,
    technical_usable: bool = True,
    **vlm_kwargs,
) -> dict:
    """构造 heuristic 候选 dict。"""
    c = {
        "source_shot_id": shot_id,
        "source_media_hash": f"sha256:{shot_id}",
        "technical_usable": technical_usable,
        "blur_score": blur,
        "source_in_us": src_in,
        "source_out_us": src_in + src_dur,
        "duration_us": src_dur,
        "vlm_shot_function": None,
        "vlm_motion": None,
        "vlm_role": None,
    }
    c.update(vlm_kwargs)
    return c


class TestHeuristicDuration:
    """场景 4：生成总时长在 target 的 90%-110% 范围内。"""

    def test_total_duration_band(self):
        candidates = [
            _make_candidate(f"shot_{i:02d}", blur=100.0 + i * 10, src_dur=3_000_000)
            for i in range(10)
        ]
        target = 10_000_000
        result = HeuristicBaseline.generate(
            project_id="test_proj", candidates=candidates, target_duration_us=target
        )
        total = sum(
            s["proposed_out_us"] - s["proposed_in_us"] for s in result["slots"]
        )
        assert 0.9 * target <= total <= 1.1 * target, (
            f"total={total} outside [0.9*target, 1.1*target]"
        )

    def test_producer_is_v02(self):
        candidates = [_make_candidate("shot_01", blur=100.0)]
        result = HeuristicBaseline.generate(
            project_id="test_proj", candidates=candidates, target_duration_us=5_000_000
        )
        assert result["ai_model"] == "heuristic_v0.2"
        assert result["prompt_version"] == "heuristic_v0.2"

    def test_skip_too_short_shots(self):
        # shot_dur < MIN_CLIP_US (800_000) 的镜头应被跳过
        candidates = [
            _make_candidate("shot_short", blur=999.0, src_dur=500_000),
            _make_candidate("shot_ok", blur=100.0, src_dur=3_000_000),
        ]
        result = HeuristicBaseline.generate(
            project_id="test_proj", candidates=candidates, target_duration_us=5_000_000
        )
        shot_ids = [s["source_shot_id"] for s in result["slots"]]
        assert "shot_short" not in shot_ids
        assert "shot_ok" in shot_ids


# ---------------------------------------------------------------------------
# 5. heuristic 权重
# ---------------------------------------------------------------------------

class TestHeuristicWeights:
    """场景 5：VLM 权重排序正确。"""

    def test_atmospheric_ranked_above_discard(self):
        # target > 15s 时 ATMOSPHERIC_EVIDENCE 获 x1.15 加成
        # discard 获 x0.2 降权
        atmospheric = _make_candidate(
            "shot_atmos", blur=100.0, src_dur=3_000_000,
            vlm_shot_function="ATMOSPHERIC_EVIDENCE",
        )
        discard = _make_candidate(
            "shot_discard", blur=200.0, src_dur=3_000_000,
            vlm_role="discard",
        )
        plain = _make_candidate("shot_plain", blur=150.0, src_dur=3_000_000)
        candidates = [discard, plain, atmospheric]
        result = HeuristicBaseline.generate(
            project_id="test_proj",
            candidates=candidates,
            target_duration_us=20_000_000,  # > 15_000_000 触发 atmospheric
        )
        shot_ids = [s["source_shot_id"] for s in result["slots"]]
        # atmospheric score = 100 * 1.15 = 115
        # plain score = 150 * 1.0 = 150
        # discard score = 200 * 0.2 = 40
        # 排序应为 plain > atmospheric > discard
        assert shot_ids[0] == "shot_plain"
        assert shot_ids[1] == "shot_atmos"
        assert shot_ids[2] == "shot_discard"

    def test_confidence_type_vlm_vs_heuristic(self):
        vlm_cand = _make_candidate(
            "shot_vlm", blur=100.0, vlm_role="hero"
        )
        plain_cand = _make_candidate("shot_plain", blur=50.0)
        result = HeuristicBaseline.generate(
            project_id="test_proj",
            candidates=[vlm_cand, plain_cand],
            target_duration_us=10_000_000,
        )
        ctype_map = {s["source_shot_id"]: s["confidence_type"] for s in result["slots"]}
        assert ctype_map["shot_vlm"] == "SELF_REPORTED"
        assert ctype_map["shot_plain"] == "HEURISTIC"

    def test_compute_clip_window_midpoint(self):
        # 取中点居中裁剪
        lo, hi = compute_clip_window(0, 10_000_000, 4_000_000)
        assert lo == 3_000_000
        assert hi == 7_000_000

    def test_compute_clip_window_clamp(self):
        # clip 比 shot 长时钳制到边界（调用方保证 clip_dur <= shot_dur，
        # 但这里验证中点偏移的钳制逻辑）
        lo, hi = compute_clip_window(0, 1_000_000, 1_000_000)
        assert lo == 0
        assert hi == 1_000_000


# ---------------------------------------------------------------------------
# 6. 空输入不抛异常
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """场景 6：空 slots 不抛异常。"""

    def test_empty_slots_no_exception(self, tmp_path):
        empty_proposal = {
            "project_id": "empty_proj",
            "proposal_id": "prop_empty",
            "rough_cut_id": "rc_empty",
            "slots": [],
            "ai_model": "heuristic_v0.1",
            "prompt_version": "heuristic_v0.1",
        }
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(empty_proposal), encoding="utf-8")
        edl, plan, log = import_v01_proposal(str(p))
        assert edl.ordered_edits == []
        assert plan.decisions == []
        assert isinstance(log, list)


# ---------------------------------------------------------------------------
# 7. import_log 记录
# ---------------------------------------------------------------------------

class TestImportLog:
    """场景 7：缺少 source_shot_id 的 slot 被跳过并记录警告。"""

    def test_missing_source_shot_id_warns(self, tmp_path):
        bad_proposal = {
            "project_id": "bad_proj",
            "proposal_id": "prop_bad",
            "rough_cut_id": "rc_bad",
            "slots": [
                {
                    "slot_id": "slot_01",
                    "position": 0,
                    "source_media_hash": "sha256:abc",
                    "proposed_in_us": 0,
                    "proposed_out_us": 1_000_000,
                    "reason": "heuristic:blur=100",
                    "confidence_type": "HEURISTIC",
                    "confidence_value": None,
                },
                {
                    "slot_id": "slot_02",
                    "position": 1,
                    "source_shot_id": "shot_good",
                    "source_media_hash": "sha256:def",
                    "proposed_in_us": 0,
                    "proposed_out_us": 2_000_000,
                    "reason": "heuristic:blur=200",
                    "confidence_type": "HEURISTIC",
                    "confidence_value": None,
                },
            ],
            "ai_model": "heuristic_v0.1",
            "prompt_version": "heuristic_v0.1",
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad_proposal), encoding="utf-8")
        edl, plan, log = import_v01_proposal(str(p))
        # 只有 good slot 被导入
        assert len(edl.ordered_edits) == 1
        assert edl.ordered_edits[0].source_asset_id == "shot_good"
        # log 中有警告
        warnings = [entry for entry in log if entry.get("level") == "warning"]
        assert len(warnings) >= 1
        assert any("slot_01" in str(w.get("slot_id", "")) for w in warnings)
