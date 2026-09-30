"""四幕结构单一事实源测试（架构体检候选⑥收编）。

四处定义（story_graph_builder / director_reasoner / strategy_selector /
semantic_scorer）必须全部引用 director_brain.acts，区间表与配额表
代码级一致（区间 = 配额累积和）。
"""
from director_brain import acts
from director_brain import strategy_selector
from director_brain.director_reasoner import (
    ACT_FUNCTION as REASONER_FUNCTION,
    ACT_ORDER as REASONER_ORDER,
    ACT_RATIO as REASONER_RATIO,
)
from director_brain.semantic_scorer import _ROLE_TO_ACT as SCORER_ROLE_MAP
from director_brain.story_graph_builder import _ACTS as GRAPH_INTERVALS


def test_intervals_derive_from_ratios():
    """区间表由配额表累积派生——改配额即改区间，不可能漂移。"""
    lo = 0.0
    for name, start, end, _label in acts.ACT_INTERVALS:
        assert start == lo
        assert abs((end - start) - acts.ACT_RATIO[name]) < 1e-9
        lo = end
    assert abs(lo - 1.0) < 1e-9


def test_all_importers_share_single_source():
    """四个消费方与 acts 同源（对象同一性，不是值巧合）。"""
    assert REASONER_RATIO is acts.ACT_RATIO
    assert REASONER_ORDER is acts.ACT_ORDER
    assert REASONER_FUNCTION is acts.ACT_FUNCTION
    assert GRAPH_INTERVALS is acts.ACT_INTERVALS
    assert strategy_selector._ACTS == tuple(
        (n, lo, hi) for n, lo, hi, _ in acts.ACT_INTERVALS
    )
    assert SCORER_ROLE_MAP is acts.ROLE_TO_ACT


def test_role_to_act_covers_vlm_vocabulary():
    """VLM 叙事角色词表全部有幕映射（未知角色回退由内核处理）。"""
    assert set(acts.ROLE_TO_ACT) == {
        "setup", "development", "climax", "resolution", "transition",
    }
    assert set(acts.ROLE_TO_ACT.values()) <= set(acts.ACT_NAMES)
