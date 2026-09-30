"""M3.1 多方案对比与选择。

围绕同一份 Brief / 故事图 / 观测，按不同目标时长生成多个 EDL 变体，
逐方案计算评分卡（时长、镜头数、平均清晰度、四幕覆盖、曝光通过率），
再按 ``duration`` 或 ``quality`` 优先级选出最优方案索引。

时间统一微秒（timebase=1_000_000）。本模块不引入新依赖，仅用标准库 + pydantic。
"""
from __future__ import annotations

import json

from director_brain.director_reasoner import HeuristicDirectorReasoner
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_observation import FilmObservation
from director_brain.models.story_graph import StoryGraph

#: 四幕时间比例：单一事实源 acts.ACT_INTERVALS 派生（候选⑥收编）。
from director_brain.acts import ACT_INTERVALS  # noqa: E402

_ACTS: tuple[tuple[str, float, float], ...] = tuple(
    (name, lo, hi) for name, lo, hi, _label in ACT_INTERVALS
)
_AC_ORDER: tuple[str, ...] = tuple(name for name, _lo, _hi, _l in ACT_INTERVALS)
_RESOLVE_RATIO = _ACTS[3][1]


def _parse_claim(claim: str) -> dict:
    """解析观测 claim JSON；失败或非 dict 时返回空 dict。"""
    try:
        data = json.loads(claim)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def generate_variants(
    brief: DirectorBrief,
    graph: StoryGraph,
    observations: list[FilmObservation],
    configs: list[dict],
    narrative: dict | None = None,
) -> list[tuple[EditorialDecisionList, DirectorDecisionPlan]]:
    """对每个 config 生成一个 EDL 变体。

    每个 config 至少含 ``{"target_duration_us": int}``，可选
    ``{"blur_threshold": float}``（默认 10.0）：用
    ``brief.model_copy(update={"target_duration": ...})`` 复制简报后交给
    ``HeuristicDirectorReasoner(blur_threshold=...)`` 生成 (EDL, Plan)。
    不同 blur_threshold 会真实改变选片结果（非仅时长不同）。原 brief 不被修改。

    Returns:
        与 ``configs`` 等长的 (EDL, Plan) 列表。
    """
    variants: list[tuple[EditorialDecisionList, DirectorDecisionPlan]] = []
    for cfg in configs:
        target = int(cfg["target_duration_us"])
        threshold = float(cfg.get("blur_threshold", 10.0))
        reasoner = HeuristicDirectorReasoner(blur_threshold=threshold)
        brief_copy = brief.model_copy(update={"target_duration": target})
        edl, plan = reasoner.generate_plan(brief_copy, graph, observations,
                                           narrative=narrative)
        variants.append((edl, plan))
    return variants


def _act_for_time(t_us: int, total_us: int) -> str | None:
    """按源时间线比例判断 ``t_us`` 属于哪一幕（对齐 story_graph_builder）。"""
    if total_us <= 0:
        return None
    ratio = t_us / total_us
    if ratio >= _RESOLVE_RATIO - 1e-9:
        return "resolve"
    for name, lo, hi in _ACTS[:3]:
        if lo <= ratio < hi:
            return name
    return None


def compare_plans(
    plans: list[tuple[EditorialDecisionList, DirectorDecisionPlan]],
    observations: list[FilmObservation],
) -> list[dict]:
    """对每个 (EDL, Plan) 计算一张评分卡。

    评分卡字段：
    - ``duration_us``：sum(out_frame - in_frame)
    - ``shot_count``：len(ordered_edits)
    - ``avg_blur``：按 edit 的 source_asset_id 匹配 deterministic_technical
      观测，取 claim 中的 blur_score 求平均（无匹配观测的 edit 跳过；全部
      无匹配时为 0.0）
    - ``narrative_coverage``：四幕各有多少镜头（按 edit.in_frame 落在哪个
      幕时间区间统计）
    - ``quality_distribution``：含 ``exposure_ok_ratio``（曝光通过的 edit 数
      / 总 edit 数）
    """
    tech_obs = [
        o for o in observations if o.observation_type == "deterministic_technical"
    ]
    claim_by_asset: dict[str, dict] = {}
    for o in tech_obs:
        # 同一 asset 若有多条观测，保留第一条（与单镜头单观测的约定一致）
        claim_by_asset.setdefault(o.media_asset_id, _parse_claim(o.claim))

    total_us = max((o.end_frame for o in tech_obs), default=0)

    scorecards: list[dict] = []
    for edl, _plan in plans:
        edits = edl.ordered_edits
        duration_us = sum(e.out_frame - e.in_frame for e in edits)
        shot_count = len(edits)

        blur_sum = 0.0
        blur_n = 0
        exposure_ok_count = 0
        coverage: dict[str, int] = {act: 0 for act in _AC_ORDER}
        for e in edits:
            claim = claim_by_asset.get(e.source_asset_id, {})
            blur = claim.get("blur_score")
            if isinstance(blur, (int, float)):
                blur_sum += float(blur)
                blur_n += 1
            if claim.get("exposure_ok") is True:
                exposure_ok_count += 1
            act = _act_for_time(e.in_frame, total_us)
            if act is not None:
                coverage[act] += 1

        avg_blur = blur_sum / blur_n if blur_n else 0.0
        exposure_ratio = (exposure_ok_count / shot_count) if shot_count else 0.0

        scorecards.append({
            "duration_us": duration_us,
            "shot_count": shot_count,
            "avg_blur": avg_blur,
            "narrative_coverage": coverage,
            "quality_distribution": {
                "exposure_ok_ratio": exposure_ratio,
                "exposure_ok_count": exposure_ok_count,
                "total_edits": shot_count,
            },
        })
    return scorecards


def select_best(
    scorecards: list[dict],
    priority: str = "duration",
    target_duration_us: int | None = None,
) -> int:
    """按优先级选出最优方案索引。

    - ``priority="duration"``：选 |duration_us - target_duration_us| 最小者
      （距离并列取首个）。``target_duration_us`` 必填，缺省抛 ValueError。
    - ``priority="quality"``：选 avg_blur 最高者（并列取首个）。

    Returns:
        合法索引 ``0 <= idx < len(scorecards)``。
    """
    if not scorecards:
        raise ValueError("select_best: empty scorecards")

    if priority == "duration":
        if target_duration_us is None:
            raise ValueError(
                "select_best priority='duration' requires target_duration_us")
        return min(
            range(len(scorecards)),
            key=lambda i: abs(scorecards[i]["duration_us"] - target_duration_us),
        )
    if priority == "quality":
        return max(
            range(len(scorecards)),
            key=lambda i: scorecards[i]["avg_blur"],
        )
    raise ValueError(f"unknown priority: {priority!r}")
