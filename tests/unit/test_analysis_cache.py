"""AnalysisCache 与 compute_fingerprint 验收测试。

覆盖 M1.4 的 8 项行为：
1. 相同输入返回相同 hash
2. 任一输入变化返回不同 hash（参数化）
3. put 后 get 命中
4. 未命中返回 None
5. invalidate 后 get 返回 None；invalidate 不存在的指纹返回 False
6. hit_rate 统计正确
7. 持久化：重新实例化后从 JSON 文件加载
8. 空 sampling_config 参与指纹计算
"""
from __future__ import annotations

import pytest

from director_brain.analysis_cache import AnalysisCache, compute_fingerprint
from director_brain.models.film_observation import ClaimKind, FilmObservation


# --- FilmObservation 最小合法字段（与 test_models.py 的 COMMON/MINIMAL 对齐） ---
_COMMON = dict(
    project_id="proj-001",
    created_at=1700000000,
    producer="pytest",
    source_ref="brief://v1",
)

_OBS_MINIMAL = dict(
    observation_id="obs-001",
    media_asset_id="asset-001",
    media_hash="hash-123",
    start_frame=0,
    end_frame=25,
    timebase=25,
    observation_type="action",
    claim="人物走入画面",
    provider="zhipu",
    model_version="glm-4.6v",
    prompt_version="1",
    confidence=0.9,
    review_state="pending",
    claim_kind=ClaimKind.MODEL_OBSERVATION,
)


def _make_observation(**overrides) -> FilmObservation:
    kwargs = dict(_COMMON)
    kwargs.update(_OBS_MINIMAL)
    kwargs.update(overrides)
    return FilmObservation(**kwargs)


# 指纹计算的一组基准参数
BASE_KWARGS = dict(
    source_content_hash="sha256:abc123",
    provider="zhipu",
    model_version="glm-4.6v",
    prompt_version="v1",
    sampling_config={"temperature": 0.7, "top_p": 0.9},
    timebase=25,
    schema_version="1.0",
)


# ---------------------------------------------------------------------------
# 1. 相同输入返回相同 hash
# ---------------------------------------------------------------------------
def test_fingerprint_stable_for_same_input():
    fp1 = compute_fingerprint(**BASE_KWARGS)
    fp2 = compute_fingerprint(**BASE_KWARGS)
    assert fp1 == fp2
    assert isinstance(fp1, str)
    # 统一为 SHA-256 前 16 位（经 director_brain._utils.short_hash）
    assert len(fp1) == 16


# ---------------------------------------------------------------------------
# 2. 任一输入变化返回不同 hash（参数化）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "override",
    [
        {"source_content_hash": "sha256:different"},
        {"provider": "openai"},
        {"model_version": "gpt-4o"},
        {"prompt_version": "v2"},
        {"sampling_config": {"temperature": 0.5}},
        {"timebase": 30},
        {"schema_version": "2.0"},
    ],
)
def test_fingerprint_changes_on_any_input(override):
    base_fp = compute_fingerprint(**BASE_KWARGS)
    changed = dict(BASE_KWARGS)
    changed.update(override)
    changed_fp = compute_fingerprint(**changed)
    assert changed_fp != base_fp


# ---------------------------------------------------------------------------
# 3. put 后 get 命中
# ---------------------------------------------------------------------------
def test_put_then_get_returns_observations():
    cache = AnalysisCache()
    obs1 = _make_observation(observation_id="obs-001", claim="人物走入画面")
    obs2 = _make_observation(observation_id="obs-002", claim="镜头切换")
    fingerprint = "fp-hit"

    cache.put(fingerprint, [obs1, obs2])
    got = cache.get(fingerprint)

    assert got is not None
    assert len(got) == 2
    assert got[0].observation_id == "obs-001"
    assert got[0].claim == "人物走入画面"
    assert got[1].observation_id == "obs-002"
    assert got[1].claim_kind is ClaimKind.MODEL_OBSERVATION


# ---------------------------------------------------------------------------
# 4. 未命中返回 None
# ---------------------------------------------------------------------------
def test_get_miss_returns_none():
    cache = AnalysisCache()
    assert cache.get("nonexistent-fp") is None


# ---------------------------------------------------------------------------
# 5. invalidate 后 get 返回 None；invalidate 不存在返回 False
# ---------------------------------------------------------------------------
def test_invalidate_removes_entry():
    cache = AnalysisCache()
    obs = _make_observation()
    cache.put("fp-to-invalidate", [obs])

    assert cache.get("fp-to-invalidate") is not None
    assert cache.invalidate("fp-to-invalidate") is True
    assert cache.get("fp-to-invalidate") is None


def test_invalidate_nonexistent_returns_false():
    cache = AnalysisCache()
    assert cache.invalidate("never-existed") is False


# ---------------------------------------------------------------------------
# 6. hit_rate 统计正确
#    序列：get(不存在) → put → get(存在) → get(存在) => 2 hit / 3 total
# ---------------------------------------------------------------------------
def test_hit_rate_stats():
    cache = AnalysisCache()
    obs = _make_observation()

    miss = cache.get("missing-fp")       # miss, total=1
    assert miss is None

    cache.put("real-fp", [obs])
    cache.get("real-fp")                # hit, total=2
    cache.get("real-fp")                # hit, total=3

    assert cache.hit_rate() == pytest.approx(2 / 3)


def test_hit_rate_zero_when_no_calls():
    cache = AnalysisCache()
    assert cache.hit_rate() == 0.0


# ---------------------------------------------------------------------------
# 7. 持久化：重新实例化后从 JSON 文件加载
# ---------------------------------------------------------------------------
def test_persist_roundtrip(tmp_path):
    persist_file = tmp_path / "cache.json"
    fp = "fp-persist"
    obs1 = _make_observation(observation_id="obs-p1")
    obs2 = _make_observation(observation_id="obs-p2", claim="第二个观测")

    cache1 = AnalysisCache(persist_path=str(persist_file))
    cache1.put(fp, [obs1, obs2])
    cache1.flush()

    # 重新实例化，应从磁盘加载
    cache2 = AnalysisCache(persist_path=str(persist_file))
    got = cache2.get(fp)

    assert got is not None
    assert len(got) == 2
    assert got[0].observation_id == "obs-p1"
    assert got[1].claim == "第二个观测"
    # claim_kind 枚举应被正确还原
    assert got[0].claim_kind is ClaimKind.MODEL_OBSERVATION


# ---------------------------------------------------------------------------
# 8. 空 sampling_config 参与计算
# ---------------------------------------------------------------------------
def test_empty_sampling_config_changes_fingerprint():
    with_empty = compute_fingerprint(
        source_content_hash="sha256:abc123",
        provider="zhipu",
        model_version="glm-4.6v",
        prompt_version="v1",
        sampling_config={},
        timebase=25,
    )
    with_nonempty = compute_fingerprint(
        source_content_hash="sha256:abc123",
        provider="zhipu",
        model_version="glm-4.6v",
        prompt_version="v1",
        sampling_config={"temperature": 0.7},
        timebase=25,
    )
    assert with_empty != with_nonempty


def test_sampling_config_order_independent():
    """sampling_config 内部 key 顺序不影响指纹（sort_keys 保证）。"""
    a = compute_fingerprint(
        source_content_hash="h",
        provider="p",
        model_version="m",
        prompt_version="v",
        sampling_config={"temperature": 0.7, "top_p": 0.9},
        timebase=25,
    )
    b = compute_fingerprint(
        source_content_hash="h",
        provider="p",
        model_version="m",
        prompt_version="v",
        sampling_config={"top_p": 0.9, "temperature": 0.7},
        timebase=25,
    )
    assert a == b


# ---------------------------------------------------------------------------
# 9. 批量 put 仅在 flush() 时落盘（dirty flag 批写）
# ---------------------------------------------------------------------------
def test_batch_put_persists_only_after_flush(tmp_path):
    """连续 100 次 put 后未 flush，重新实例化读取为空；flush 后数据持久化。"""
    persist_file = tmp_path / "cache.json"
    cache1 = AnalysisCache(persist_path=str(persist_file))

    for i in range(100):
        obs = _make_observation(observation_id=f"obs-batch-{i:03d}")
        cache1.put(f"fp-batch-{i:03d}", [obs])

    # 未调用 flush()：重新实例化应读不到任何数据
    cache_unflushed = AnalysisCache(persist_path=str(persist_file))
    for i in range(100):
        assert cache_unflushed.get(f"fp-batch-{i:03d}") is None

    # 调用 flush() 后数据才写入磁盘
    cache1.flush()
    cache2 = AnalysisCache(persist_path=str(persist_file))
    for i in range(100):
        got = cache2.get(f"fp-batch-{i:03d}")
        assert got is not None
        assert len(got) == 1
        assert got[0].observation_id == f"obs-batch-{i:03d}"


# ---------------------------------------------------------------------------
# 10. _dirty 标志行为：put 置 dirty=True，flush 置 False，未 dirty 不写盘
# ---------------------------------------------------------------------------
def test_dirty_flag_behavior(tmp_path, monkeypatch):
    """put 后 _dirty=True；flush 后 _dirty=False；未 dirty 时 flush 不触发写盘。"""
    persist_file = tmp_path / "cache.json"
    cache = AnalysisCache(persist_path=str(persist_file))

    # 初始状态：未 dirty
    assert cache._dirty is False

    # 计 _flush 调用次数
    flush_calls = 0
    original_flush = cache._flush

    def counting_flush():
        nonlocal flush_calls
        flush_calls += 1
        original_flush()

    monkeypatch.setattr(cache, "_flush", counting_flush)

    # put 后应标记 dirty，但尚未写盘
    obs = _make_observation()
    cache.put("fp-dirty", [obs])
    assert cache._dirty is True
    assert flush_calls == 0, "put 不应立即触发 _flush"

    # 未 dirty 时 flush() 不触发写入（这里先手动重置 dirty=False）
    cache._dirty = False
    cache.flush()
    assert flush_calls == 0, "未 dirty 时 flush() 不应触发 _flush"

    # invalidate 实际删除后也应标记 dirty
    cache._dirty = False
    cache.put("fp-to-inv", [obs])   # 这会把 dirty 设回 True
    cache.flush()                    # 写盘并清 dirty
    assert flush_calls == 1
    assert cache._dirty is False

    cache.invalidate("fp-to-inv")
    assert cache._dirty is True
    cache.flush()
    assert flush_calls == 2
    assert cache._dirty is False
