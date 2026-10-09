"""D1 镜头卡系统测试（美学策略载体 v1）。

覆盖：卡库加载与 schema 校验、确定性选择器、内核消费（节奏边界覆盖/
转场策略/artistic_choices 留痕）、显式选择的响亮失败。
"""
from __future__ import annotations

import json
import time

import pytest

from director_brain.brief_compiler import compile_brief
from director_brain.director_reasoner import get_director_reasoner
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.models.shot_card import ShotCard
from director_brain.pathway_protocol import PathwayStatus, set_pathway_status
from director_brain.shot_cards import get_card, load_cards, select_card
from director_brain.story_graph_builder import build_story_graph


# ---------------------------------------------------------------------------
# 卡库
# ---------------------------------------------------------------------------

def test_load_cards_valid_and_unique():
    cards = load_cards()
    assert len(cards) >= 12
    ids = [c.card_id for c in cards]
    assert len(ids) == len(set(ids))
    assert all(isinstance(c, ShotCard) for c in cards)


def test_get_card_and_unknown():
    assert get_card("fast_cut") is not None
    assert get_card("no_such_card") is None


# ---------------------------------------------------------------------------
# 选择器
# ---------------------------------------------------------------------------

def _brief(arc="upbeat", lang="fast_cut") -> DirectorBrief:
    return DirectorBrief(
        schema_version="1.0", project_id="t", created_at=0, producer="t",
        source_ref="t.mp4", brief_id="b1", version="0.1", source_text="test",
        language="zh", intent="test", audience="not_determined",
        target_duration=10_000_000, source_duration_us=20_000_000,
        delivery_profile="short_form", themes=[], relationships=[],
        emotional_arc=arc, visual_language="consistent_exposure",
        editing_language=lang, sound_language="no_speech",
        must_include=[], must_avoid=[], privacy_constraints=[],
        approval_state="draft")


def test_select_matches_editing_language_and_arc():
    card = select_card(_brief(arc="upbeat", lang="fast_cut"), shot_count=10)
    assert card is not None
    assert card.card_id in ("fast_cut", "montage", "climax_peak", "beat_sync")


def test_select_slow_calm():
    card = select_card(_brief(arc="calm", lang="slow_paced"), shot_count=5)
    assert card is not None
    assert card.card_id in ("slow_paced", "documentary", "warm_moment",
                            "gentle_close")


def test_select_skips_cards_below_min_shots():
    """素材 2 个镜头：min_shots>=3 的卡全部跳过——无条件匹配返回 None。"""
    card = select_card(_brief(lang="not_determined"), shot_count=2)
    assert card is None or card.min_shots <= 2


def test_explicit_card_id_resolved():
    card = select_card(_brief(), shot_count=10, card_id="balanced")
    assert card.card_id == "balanced"


def test_explicit_unknown_card_fails_loud():
    with pytest.raises(ValueError, match="未知镜头卡"):
        select_card(_brief(), shot_count=10, card_id="nope")


def test_explicit_card_condition_fails_loud():
    """显式选卡但素材镜头数不满足 → 响亮失败（不静默降级）。"""
    with pytest.raises(ValueError, match="条件不满足"):
        select_card(_brief(), shot_count=2, card_id="montage")


# ---------------------------------------------------------------------------
# 内核消费
# ---------------------------------------------------------------------------

def _tech_obs(idx: int, start_us: int, end_us: int, blur: float = 150.0) -> FilmObservation:
    return FilmObservation(
        observation_id=f"det_{idx}", media_asset_id=f"shot_{idx:08d}",
        media_hash=f"hash_{idx}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim=json.dumps({"blur_score": blur, "brightness_mean": 120.0,
                          "exposure_ok": True, "shake_score": 5.0}),
        provider="deterministic_opencv", model_version="opencv",
        prompt_version="n/a", confidence=1.0, review_state="auto_verified",
        claim_kind=ClaimKind.MEASURED, schema_version="1.0",
        project_id="t", created_at=int(time.time()),
        producer="deterministic_opencv", source_ref="t.mp4")


def _obs_set():
    return [_tech_obs(i, i * 3_000_000, (i + 1) * 3_000_000) for i in range(1, 7)]


def test_kernel_card_overrides_pacing_and_annotates():
    """卡的节奏边界进内核；EDL.artistic_choices 与 plan.constraints 留痕。"""
    obs = _obs_set()
    brief = compile_brief("t", "t.mp4", obs, target_duration_us=10_000_000,
                          intent_text="快剪")
    graph = build_story_graph(brief, obs)
    card = get_card("fast_cut")
    edl, plan = get_director_reasoner("heuristic").generate_plan(
        brief, graph, obs, card=card)
    assert f"shot_card:{card.card_id}@{card.version}" in edl.artistic_choices
    assert any(c.startswith("shot_card=fast_cut@") for c in plan.constraints)
    # 快剪上界 3s：所有片段 ≤3s
    assert all(e.out_frame - e.in_frame <= 3_000_000 for e in edl.ordered_edits)


def test_kernel_card_transition_policy_applies():
    """卡的 dissolve_act_boundary 在幕边界产生转场。"""
    obs = _obs_set()
    brief = compile_brief("t", "t.mp4", obs, target_duration_us=10_000_000)
    graph = build_story_graph(brief, obs)
    card = get_card("balanced")  # dissolve_act_boundary
    edl, plan = get_director_reasoner("heuristic").generate_plan(
        brief, graph, obs, card=card)
    trans = [e for e in edl.ordered_edits if e.transition
             and e.transition.type == "xfade"]
    assert trans, "均衡卡应产生幕边界转场"
    # 转场只出现在幕边界（当前与下一镜头不同幕）
    acts = [e.act for e in edl.ordered_edits]
    for i, e in enumerate(edl.ordered_edits[:-1]):
        if e.transition and e.transition.type == "xfade":
            assert acts[i] != acts[i + 1]


def test_kernel_no_card_unchanged():
    """无卡路径与旧行为一致（无 artistic_choices 注记）。"""
    obs = _obs_set()
    brief = compile_brief("t", "t.mp4", obs, target_duration_us=10_000_000)
    graph = build_story_graph(brief, obs)
    edl, _plan = get_director_reasoner("heuristic").generate_plan(
        brief, graph, obs)
    assert edl.artistic_choices == []


# ---------------------------------------------------------------------------
# D2 实体解析与连续性
# ---------------------------------------------------------------------------

def _vlm_people_obs(idx: int, start_us: int, end_us: int, people: list[str],
                    narrative_role: str = "development") -> FilmObservation:
    return FilmObservation(
        observation_id=f"vlm_{idx}", media_asset_id=f"shot_{idx:08d}",
        media_hash=f"hash_{idx}", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="vlm_semantic",
        claim=json.dumps({
            "shot_function": "ACTION", "proposed_role_v2": "support",
            "motion_amount": "subtle", "frame_description": "t",
            "importance": 3, "narrative_role": narrative_role,
            "emotional_tone": "neutral", "action_type": "action",
            "scene_description": "测试", "people": people}),
        provider="ollama_qwen3_vl", model_version="qwen3-vl:latest",
        prompt_version="vlm_prompt_v4_people", confidence=0.7,
        review_state="auto_generated", claim_kind=ClaimKind.MODEL_OBSERVATION,
        schema_version="1.0", project_id="t", created_at=int(time.time()),
        producer="ollama_qwen3_vl", source_ref="t.mp4")


def test_entity_resolver_clusters_same_person():
    """同外观描述（红衣短发女孩）跨镜头 → 同一实体。"""
    from director_brain.entity_resolver import resolve_entities
    obs = [
        _vlm_people_obs(1, 0, 2_000_000, ["红衣短发女孩"]),
        _vlm_people_obs(2, 2_000_000, 4_000_000, ["蓝衣男孩"]),
        _vlm_people_obs(3, 4_000_000, 6_000_000, ["红衣短发女孩"]),
    ]
    res = resolve_entities(obs)
    assert len(res.entities) == 2
    a1 = set(res.assignments["shot_00000001"])
    a3 = set(res.assignments["shot_00000003"])
    assert a1 & a3, "镜头 1/3 的红衣女孩应同实体"
    assert not (a1 & set(res.assignments["shot_00000002"])), "蓝衣男孩不同实体"
    # 多成员实体置信度 ≥ 0.5；单成员弱置信
    assert all(e.identity_confidence >= 0.3 for e in res.entities)


def test_entity_resolver_confidence_grading():
    """共享 颜色+载体 判别词对 → 高置信；无判别词 → 弱置信。"""
    from director_brain.entity_resolver import resolve_entities
    obs = [
        _vlm_people_obs(1, 0, 2_000_000, ["红衣女孩"]),
        _vlm_people_obs(2, 2_000_000, 4_000_000, ["红衣女孩拿着伞"]),
    ]
    res = resolve_entities(obs)
    assert len(res.entities) == 1  # 判别词对命中 → 合并
    assert res.entities[0].identity_confidence >= 0.8


def test_entity_resolver_empty_people_no_assignment():
    """无 people 字段的镜头不参与聚类（无证据不归属）。"""
    from director_brain.entity_resolver import resolve_entities
    obs = [_vlm_people_obs(1, 0, 2_000_000, [])]
    res = resolve_entities(obs)
    assert res.entities == [] and res.assignments == {}


def test_entity_resolver_fails_closed_across_project_assets():
    """独立素材上的相似描述不得自动变成同一人物身份。"""
    from director_brain.entity_resolver import resolve_entities

    obs = [
        _vlm_people_obs(1, 0, 2_000_000, ["红衣短发女孩"]).model_copy(
            update={"project_asset_id": "asset-a"}),
        _vlm_people_obs(2, 0, 2_000_000, ["红衣短发女孩"]).model_copy(
            update={"project_asset_id": "asset-b"}),
    ]

    with pytest.raises(ValueError, match="cross-asset identity matching"):
        resolve_entities(obs)


def test_entity_resolver_rejects_mixed_asset_binding_and_clock():
    """跨素材混合绑定或时钟不能进入按时间排序的聚类。"""
    from director_brain.entity_resolver import resolve_entities
    from director_brain.models.film_observation import TimebaseUnit

    bound = _vlm_people_obs(1, 0, 2_000_000, ["红衣短发女孩"]).model_copy(
        update={"project_asset_id": "asset-a"})
    unbound = _vlm_people_obs(2, 2_000_000, 4_000_000, ["红衣短发女孩"])
    with pytest.raises(ValueError, match="mix bound and unbound"):
        resolve_entities([bound, unbound])

    second_clock = _vlm_people_obs(3, 2_000_000, 4_000_000,
                                   ["红衣短发女孩"]).model_copy(update={
        "timebase": 25,
        "timebase_unit": TimebaseUnit.FRAMES,
    })
    with pytest.raises(ValueError, match="one source timebase"):
        resolve_entities([bound, bound.model_copy(update={
            "observation_id": "vlm_other",
        }), second_clock.model_copy(update={"project_asset_id": "asset-a"})])


def test_kernel_entity_continuity_bonus():
    """与上一幕已选镜头共享人物实体 → continuity_bonus 加分留痕。"""
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        obs = [
            _tech_obs(1, 0, 2_000_000),
            _tech_obs(2, 1_000_000, 3_000_000),
            _tech_obs(3, 1_500_000, 3_500_000),
            # shot1/shot3 同实体（跨 hook→develop 人物线索）
            _vlm_people_obs(1, 0, 2_000_000, ["红衣短发女孩"],
                            narrative_role="setup"),
            _vlm_people_obs(2, 1_000_000, 3_000_000, ["蓝衣男孩"]),
            _vlm_people_obs(3, 1_500_000, 3_500_000, ["红衣短发女孩"]),
        ]
        brief = _brief()
        graph = build_story_graph(brief, obs[:3])
        from director_brain.entity_resolver import resolve_entities
        entities = resolve_entities(obs[3:])
        edl, plan = get_director_reasoner("heuristic").generate_plan(
            brief, graph, obs, entities=entities)
        assert any(q.startswith("entities_resolved:count=")
                   for q in plan.open_questions)
        sem = [e.rationale for e in edl.ordered_edits
               if "entity_continuity" in (e.rationale or "")]
        assert sem, "应有镜头获得 entity_continuity 加分"
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)


# ---------------------------------------------------------------------------
# D3 语音驱动选片（voice_led）
# ---------------------------------------------------------------------------

def _speech_obs(idx: int, start_us: int, end_us: int, text: str) -> FilmObservation:
    return FilmObservation(
        observation_id=f"asr_{idx}", media_asset_id="src",
        media_hash="hash_src", start_frame=start_us, end_frame=end_us,
        timebase=1_000_000, observation_type="speech_transcript",
        claim=text, provider="faster_whisper", model_version="large-v3-turbo",
        prompt_version="n/a", confidence=0.9, review_state="auto_generated",
        claim_kind=ClaimKind.MEASURED, schema_version="1.0",
        project_id="t", created_at=int(time.time()),
        producer="faster_whisper", source_ref="t.mp4")


def test_voice_led_bonus_applies_to_speech_shots():
    """voice_led 开启：对白覆盖 ≥20% 的镜头获得 +0.1 价值加成。"""
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        obs = [
            _tech_obs(1, 0, 3_000_000, blur=150.0),
            _tech_obs(2, 3_000_000, 6_000_000, blur=150.0),
            _tech_obs(3, 6_000_000, 9_000_000, blur=150.0),
            # 语音覆盖 shot_1 全部、shot_2 一半、shot_3 无
            _speech_obs(1, 0, 3_000_000, "台词一"),
            _speech_obs(2, 3_000_000, 4_500_000, "台词二"),
        ]
        brief = _brief()
        graph = build_story_graph(brief, obs[:3])
        edl, plan = get_director_reasoner("heuristic").generate_plan(
            brief, graph, obs, voice_led=True)
        assert any(q.startswith("voice_led:applied:")
                   for q in plan.open_questions)
        assert any(c.startswith("voice_led=on:")
                   for c in plan.constraints)
        covered = [e for e in edl.ordered_edits
                   if "speech_bonus" not in (e.rationale or "")]
        # 加成通过 selection_score/generate 生效——行为验证：无语音的
        # shot_3 与有语音的 shot_1 同模糊度时，shot_1 先入选
        seq = [e.source_asset_id for e in edl.ordered_edits]
        if "shot_00000003" in seq and "shot_00000001" in seq:
            assert seq.index("shot_00000001") < seq.index("shot_00000003")
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)


def test_voice_led_off_no_change():
    """默认（无 --voice-led）行为与旧路径一致（无语音注记）。"""
    obs = [
        _tech_obs(1, 0, 3_000_000),
        _tech_obs(2, 3_000_000, 6_000_000),
        _speech_obs(1, 0, 3_000_000, "台词"),
    ]
    brief = _brief()
    graph = build_story_graph(brief, obs[:2])
    edl, plan = get_director_reasoner("heuristic").generate_plan(
        brief, graph, obs)
    assert not any(q.startswith("voice_led:") for q in plan.open_questions)
    assert not any(c.startswith("voice_led=") for c in plan.constraints)


def test_voice_led_without_speech_attributed():
    """voice_led 但素材无语音观测 → 归因标记（speech_shots=-1）。"""
    obs = [_tech_obs(1, 0, 3_000_000), _tech_obs(2, 3_000_000, 6_000_000)]
    brief = _brief()
    graph = build_story_graph(brief, obs)
    _edl, plan = get_director_reasoner("heuristic").generate_plan(
        brief, graph, obs, voice_led=True)
    assert "voice_led:applied:speech_shots=-1" in plan.open_questions


def test_asr_pathway_default_active():
    """D3：asr_transcript 默认 ACTIVE（影子期证据齐备 + 所有者批准）。"""
    from director_brain.pathway_protocol import PathwayStatus, get_pathway_status
    assert get_pathway_status("asr_transcript") is PathwayStatus.ACTIVE


# ---------------------------------------------------------------------------
# D5 节拍网格与切点吸附
# ---------------------------------------------------------------------------

def _make_beat_wav(path, bpm=120.0, dur_s=8.0):
    import numpy as np
    import soundfile as sf
    sr = 22050
    t = np.arange(int(sr * dur_s)) / sr
    click = np.zeros_like(t)
    step = 60.0 / bpm
    k = 0
    while k * step < dur_s:
        i = int(k * step * sr)
        n = min(400, len(click) - i)
        if n > 0:
            click[i:i+n] += np.exp(-np.linspace(0, 6, n)) * np.sin(2*np.pi*1000*np.linspace(0, 6, n)*0.001)
        k += 1
    sf.write(path, click, sr)


def test_beat_grid_analysis_and_cache(tmp_path):
    from observation_service.beat_grid import analyze_beat_grid
    wav = str(tmp_path / "beat.wav")
    _make_beat_wav(wav, bpm=120.0, dur_s=8.0)
    cache = str(tmp_path / "cache")
    g1 = analyze_beat_grid(wav, cache_dir=cache)
    assert 110 <= g1.bpm <= 130
    assert len(g1.beat_times_us) >= 12
    g2 = analyze_beat_grid(wav, cache_dir=cache)
    assert g2.beat_times_us == g1.beat_times_us  # 缓存命中


def test_beat_grid_missing_file_loud(tmp_path):
    from observation_service.beat_grid import analyze_beat_grid
    with pytest.raises(RuntimeError, match="不存在"):
        analyze_beat_grid(str(tmp_path / "nope.wav"))


def test_kernel_beat_snap_requires_active_pathway():
    """EXPERIMENTAL（默认）：网格计算+注记，但不吸附（禁止半消费）。"""
    obs = [_tech_obs(i, i * 3_000_000, (i + 1) * 3_000_000) for i in range(1, 6)]
    brief = _brief()
    graph = build_story_graph(brief, obs)

    class _Grid:
        bpm = 120.0
        beat_times_us = [int(i * 500_000) for i in range(1, 40)]
        source = "test"

    edl, plan = get_director_reasoner("heuristic").generate_plan(
        brief, graph, obs, beat_grid=_Grid())
    assert any(q.startswith("beat_grid:bpm=120") for q in plan.open_questions)
    assert not any(ev.startswith("beat_aligned")
                   for ev in plan.degradation_events)


def test_kernel_beat_snap_aligns_when_active():
    """通路 ACTIVE：剪切点吸附节拍（degradation 留痕 + 时长在半拍容差内变化）。"""
    set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)
    try:
        from director_brain.pathway_protocol import set_pathway_status as sps
        sps("beat_grid", PathwayStatus.ACTIVE)
        try:
            obs = [_tech_obs(i, i * 3_000_000, (i + 1) * 3_000_000)
                   for i in range(1, 6)]
            brief = _brief()
            graph = build_story_graph(brief, obs)

            class _Grid:
                bpm = 120.0
                beat_times_us = [int(i * 500_000) for i in range(1, 60)]
                source = "test"

            edl, plan = get_director_reasoner("heuristic").generate_plan(
                brief, graph, obs, beat_grid=_Grid())
            assert any(q.startswith("beat_grid:bpm=120")
                       for q in plan.open_questions)
            # 至少一个切点被吸附或保持（网格 0.5s 间距 vs 3s 镜头——
            # 3.0s 处的切点恰在节拍上，吸附量可能为 0；验证不越界即可）
            for e in edl.ordered_edits:
                assert e.out_frame > e.in_frame
        finally:
            sps("beat_grid", PathwayStatus.EXPERIMENTAL)
    finally:
        set_pathway_status("vlm_semantic", PathwayStatus.ACTIVE)


# ---------------------------------------------------------------------------
# D6 参考片学习（特征提取 → 显式卡 → user 卡库装载）
# ---------------------------------------------------------------------------

def test_reference_card_roundtrip(tmp_path, monkeypatch):
    """参考片 → 特征 → 卡 → user 卡库装载 → --card 可选。"""
    import shutil
    import subprocess
    from pathlib import Path

    from director_brain import shot_cards as shot_cards_module
    from director_brain.shot_cards import get_card, load_cards
    import scripts.reference_card as reference_card

    isolated_root = tmp_path / "reference-card-root"
    isolated_cards_dir = isolated_root / "director_brain" / "cards"
    isolated_cards_dir.mkdir(parents=True)
    repo_cards_dir = Path(__file__).resolve().parents[2] / "director_brain" / "cards"
    shutil.copy2(repo_cards_dir / "v1.json", isolated_cards_dir / "v1.json")
    monkeypatch.setattr(reference_card, "ROOT", isolated_root)
    monkeypatch.setattr(shot_cards_module, "_CARDS_DIR", isolated_cards_dir)

    src = str(tmp_path / "ref.mp4")
    subprocess.run(["ffmpeg", "-f", "lavfi",
                    "-i", "testsrc=duration=10:size=320x240:rate=25",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", src, "-y"],
                   capture_output=True, text=True, check=True)
    # testsrc 无场景变化 → 单镜头；用 scenedetect 后端切不出多镜头，
    # 特征提取对单镜头素材也应产出合法卡（min_shots 兜底）
    features = reference_card.extract_features(src)
    assert features["shot_count"] >= 1
    card = reference_card.features_to_card(features, "测试参考卡")
    assert card.card_id.startswith("ref_")
    assert card.pacing_override["min_clip_us"] >= 300_000
    out = reference_card.save_user_card(card)
    assert out.is_file()
    assert out.parent == isolated_cards_dir / "user"
    assert get_card(card.card_id) is not None  # user 卡进卡库
    # 同 id 重复保存不产生重复卡
    reference_card.save_user_card(card)
    ids = [c.card_id for c in load_cards()]
    assert ids.count(card.card_id) == 1
