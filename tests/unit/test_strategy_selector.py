"""M3.1 多方案对比与选择 单元测试。

覆盖：
- generate_variants：对每个 config 用不同 target_duration 复制 brief 并调用推理器
- compare_plans：duration_us / shot_count / avg_blur / narrative_coverage /
  quality_distribution 字段与数值计算
- select_best：duration（选最接近 target_duration_us）与 quality（选最高 avg_blur）两种优先级

fixture 风格对齐 tests/unit/test_plan_validator.py（_obs / _edit / _edl / _plan）。
"""
from __future__ import annotations

import json
import time

import pytest

from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain import strategy_selector
from director_brain.strategy_selector import (
    compare_plans,
    generate_variants,
    select_best,
)


# ---------------------------------------------------------------------------
# Fixture 辅助
# ---------------------------------------------------------------------------

def _obs(
    asset_id: str,
    claim: dict,
    start: int = 0,
    end: int = 10_000_000,
) -> FilmObservation:
    return FilmObservation(
        observation_id=f"obs_{asset_id}",
        media_asset_id=asset_id,
        media_hash=f"hash_{asset_id}",
        start_frame=start,
        end_frame=end,
        timebase=1_000_000,
        observation_type="deterministic_technical",
        claim=json.dumps(claim),
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


def _brief(target_duration: int = 15_000_000) -> DirectorBrief:
    return DirectorBrief(
        schema_version="1.0",
        project_id="test_proj",
        created_at=int(time.time()),
        producer="test",
        source_ref="test.mp4",
        brief_id="test_brief",
        version="0.1",
        source_text="test",
        language="zh",
        intent="test",
        audience="test",
        target_duration=target_duration,
        delivery_profile="test",
        themes=[],
        relationships=[],
        emotional_arc="test",
        visual_language="test",
        editing_language="test",
        sound_language="test",
        must_include=[],
        must_avoid=[],
        privacy_constraints=[],
        approval_state="draft",
    )


# ---------------------------------------------------------------------------
# generate_variants
# ---------------------------------------------------------------------------

class _FakeReasoner:
    """记录每次 generate_plan 收到的 brief.target_duration。"""

    def __init__(self) -> None:
        self.received_targets: list[int] = []

    def generate_plan(self, brief, graph, observations):
        self.received_targets.append(brief.target_duration)
        edl = _edl([_edit("shot_a", 0, 1_000_000)])
        return edl, _plan()


def test_generate_variants_one_per_config(monkeypatch):
    fake = _FakeReasoner()
    monkeypatch.setattr(strategy_selector, "HeuristicDirectorReasoner", lambda blur_threshold=10.0: fake)

    brief = _brief(target_duration=15_000_000)
    variants = generate_variants(
        brief,
        graph=None,
        observations=[],
        configs=[
            {"target_duration_us": 20_000_000},
            {"target_duration_us": 10_000_000},
        ],
    )

    assert len(variants) == 2
    assert all(isinstance(v[0], EditorialDecisionList) for v in variants)
    assert all(isinstance(v[1], DirectorDecisionPlan) for v in variants)
    # 推理器收到与 config 一致的目标时长
    assert fake.received_targets == [20_000_000, 10_000_000]
    # 原 brief 不被修改
    assert brief.target_duration == 15_000_000


# ---------------------------------------------------------------------------
# compare_plans
# ---------------------------------------------------------------------------

#: total_us = 10_000_000 → 幕边界
#:   hook    [0,      1_500_000)
#:   develop  [1_500_000, 5_000_000)
#:   peak     [5_000_000, 8_000_000)
#:   resolve  [8_000_000, 10_000_000]
def _make_obs() -> list[FilmObservation]:
    return [
        _obs("shot_a", {"blur_score": 80.0, "exposure_ok": True}, 500_000, 1_200_000),
        _obs("shot_b", {"blur_score": 60.0, "exposure_ok": True}, 3_000_000, 4_000_000),
        _obs("shot_c", {"blur_score": 40.0, "exposure_ok": False}, 6_000_000, 7_000_000),
        _obs("shot_d", {"blur_score": 20.0, "exposure_ok": True}, 8_500_000, 10_000_000),
    ]


def _make_edl() -> EditorialDecisionList:
    # 时长: 0.2 + 0.6 + 0.4 + 0.6 + 0.5 = 2.3M
    # in_frame 幕归属: hook / develop / develop / peak / resolve
    return _edl([
        _edit("shot_a", 800_000, 1_000_000),
        _edit("shot_b", 3_200_000, 3_800_000),
        _edit("shot_b", 3_500_000, 3_900_000),
        _edit("shot_c", 6_200_000, 6_800_000),
        _edit("shot_d", 9_000_000, 9_500_000),
    ])


def test_compare_plans_fields_and_math():
    obs = _make_obs()
    edl = _make_edl()
    cards = compare_plans([(edl, _plan())], obs)

    assert len(cards) == 1
    sc = cards[0]
    # 必备字段
    for key in ("duration_us", "shot_count", "avg_blur",
                "narrative_coverage", "quality_distribution"):
        assert key in sc, f"missing field {key}"

    assert sc["duration_us"] == 2_300_000
    assert sc["shot_count"] == 5
    # blur: (80 + 60 + 60 + 40 + 20) / 5 = 52.0
    assert sc["avg_blur"] == pytest.approx(52.0)
    # 四幕全覆盖
    assert sc["narrative_coverage"] == {
        "hook": 1, "develop": 2, "peak": 1, "resolve": 1,
    }
    # exposure_ok: a=T, b=T, b=T, c=F, d=T → 4/5 = 0.8
    assert sc["quality_distribution"]["exposure_ok_ratio"] == pytest.approx(0.8)


def test_narrative_coverage_all_keys_present_when_empty():
    # 空 EDL → 四幕全 0，字段仍齐全
    edl = _edl([])
    cards = compare_plans([(edl, _plan())], _make_obs())
    sc = cards[0]
    assert sc["narrative_coverage"] == {
        "hook": 0, "develop": 0, "peak": 0, "resolve": 0,
    }
    assert sc["duration_us"] == 0
    assert sc["shot_count"] == 0
    assert sc["avg_blur"] == 0.0


def test_unmatched_asset_skipped_in_blur_and_exposure():
    # edit 引用了无对应观测的 shot_ghost：不计入 avg_blur 分母，也不算 exposure_ok
    obs = _make_obs()
    edits = [
        _edit("shot_a", 800_000, 1_000_000),   # blur 80, exposure T
        _edit("shot_ghost", 2_000_000, 2_500_000),  # 无观测
    ]
    edl = _edl(edits)
    cards = compare_plans([(edl, _plan())], obs)
    sc = cards[0]
    # 只有 shot_a 被计入 blur → 80.0
    assert sc["avg_blur"] == pytest.approx(80.0)
    # exposure: 总 edits=2，只有 shot_a=True → 1/2 = 0.5
    assert sc["quality_distribution"]["exposure_ok_ratio"] == pytest.approx(0.5)


def test_malformed_claim_falls_back():
    # claim 非法 JSON → 解析失败兜底，不抛异常
    bad_obs = [
        FilmObservation(
            observation_id="obs_bad",
            media_asset_id="shot_bad",
            media_hash="hash_bad",
            start_frame=0,
            end_frame=10_000_000,
            timebase=1_000_000,
            observation_type="deterministic_technical",
            claim="{not json",
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
    ]
    edl = _edl([_edit("shot_bad", 1_000_000, 2_000_000)])
    cards = compare_plans([(edl, _plan())], bad_obs)
    sc = cards[0]
    assert sc["avg_blur"] == 0.0
    assert sc["quality_distribution"]["exposure_ok_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------

def test_select_best_duration_picks_closest_to_target():
    cards = [
        {"duration_us": 5_000_000, "avg_blur": 50.0},
        {"duration_us": 10_000_000, "avg_blur": 80.0},
        {"duration_us": 15_000_000, "avg_blur": 60.0},
    ]
    # target=12s → 15s(idx2) 距离 3s，10s(idx1) 距离 2s → 选 idx1
    assert select_best(cards, priority="duration",
                       target_duration_us=12_000_000) == 1
    # target=8s → 10s(idx1) 距离 2s，5s(idx0) 距离 3s → 选 idx1
    assert select_best(cards, priority="duration",
                       target_duration_us=8_000_000) == 1
    # target=14s → 15s(idx2) 距离 1s → 选 idx2
    assert select_best(cards, priority="duration",
                       target_duration_us=14_000_000) == 2


def test_select_best_duration_requires_target():
    cards = [{"duration_us": 10_000_000, "avg_blur": 50.0}]
    with pytest.raises(ValueError):
        select_best(cards, priority="duration")


def test_select_best_duration_returns_legal_index():
    cards = [
        {"duration_us": 12_000_000, "avg_blur": 10.0},
        {"duration_us": 18_000_000, "avg_blur": 20.0},
    ]
    idx = select_best(cards, priority="duration",
                      target_duration_us=15_000_000)
    assert 0 <= idx < len(cards)


def test_select_best_quality_picks_highest_blur():
    cards = [
        {"duration_us": 10_000_000, "avg_blur": 30.0},
        {"duration_us": 10_000_000, "avg_blur": 90.0},  # 最高 blur
        {"duration_us": 10_000_000, "avg_blur": 50.0},
    ]
    assert select_best(cards, priority="quality") == 1



# ---------------------------------------------------------------------------
# 多方案真实差异（blur_threshold 维度）
# ---------------------------------------------------------------------------

def _make_eight_obs():
    """8 个镜头，每 2 个一组：高质量(blur=100)+ 中质量(blur=8)，每个 5s。"""
    obs = []
    for i in range(8):
        blur = 100.0 if i % 2 == 0 else 8.0
        obs.append(_obs(
            f"shot_{i}",
            {"blur_score": blur, "exposure_ok": True},
            start=i * 5_000_000,
            end=(i + 1) * 5_000_000,
        ))
    return obs


def test_generate_variants_blur_threshold_changes_edl():
    """严格(20)与宽松(5)阈值应产生不同 EDL（非仅时长不同）。

    4 幕各 2 候选(blur 100/8)，总目标 20s：
    - threshold=20：每幕仅高 blur 可用，develop/peak 各 1 槽 → 共 4 槽
    - threshold=5：全部可用，develop/peak 目标需 2 槽 → 共 6 槽
    """
    from director_brain.models.story_graph import (
        StoryGraph, StoryNode, StoryNodeType,
    )

    obs = _make_eight_obs()
    brief = _brief(target_duration=20_000_000)
    graph = StoryGraph(
        graph_id="g", version="0.1",
        project_id="test_proj", created_at=int(time.time()),
        producer="test", source_ref="test.mp4",
        nodes=[
            StoryNode(node_id="n1", node_type=StoryNodeType.ACT, ref_id="r1",
                      attributes={"act": "hook", "shot_ids": ["shot_0", "shot_1"]}),
            StoryNode(node_id="n2", node_type=StoryNodeType.ACT, ref_id="r2",
                      attributes={"act": "develop", "shot_ids": ["shot_2", "shot_3"]}),
            StoryNode(node_id="n3", node_type=StoryNodeType.ACT, ref_id="r3",
                      attributes={"act": "peak", "shot_ids": ["shot_4", "shot_5"]}),
            StoryNode(node_id="n4", node_type=StoryNodeType.ACT, ref_id="r4",
                      attributes={"act": "resolve", "shot_ids": ["shot_6", "shot_7"]}),
        ],
    )
    configs = [
        {"target_duration_us": 20_000_000, "blur_threshold": 20.0},
        {"target_duration_us": 20_000_000, "blur_threshold": 5.0},
    ]
    variants = generate_variants(brief, graph, obs, configs)
    edl0 = [e.source_asset_id for e in variants[0][0].ordered_edits]
    edl1 = [e.source_asset_id for e in variants[1][0].ordered_edits]
    assert edl0 != edl1, f"阈值不同但 EDL 相同: {edl0} vs {edl1}"
    assert len(edl0) < len(edl1), f"严格阈值应槽更少: {len(edl0)} vs {len(edl1)}"
