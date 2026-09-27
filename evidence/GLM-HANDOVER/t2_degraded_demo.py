# -*- coding: utf-8 -*-
"""T2 验收演示：静默降级显式化 + fail-closed。

用确定性合成观测（不依赖素材/模型）演示：
  正样本：blur 全部低于阈值但曝光合格 → plan.degraded=True + relax 事件留痕；
  负样本：候选 >=2 却无 2 个曝光合格 → EvidenceTooPoorError（历史行为是静默注水）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

from director_brain.brief_compiler import compile_brief                 # noqa: E402
from director_brain.director_reasoner import (                          # noqa: E402
    EvidenceTooPoorError,
    HeuristicDirectorReasoner,
)
from director_brain.models.film_observation import ClaimKind, FilmObservation  # noqa: E402
from director_brain.story_graph_builder import build_story_graph        # noqa: E402


def _obs(index: int, start: int, end: int, blur: float, exposure_ok: bool) -> FilmObservation:
    claim = json.dumps({
        "blur_score": blur, "brightness_mean": 120.0,
        "exposure_ok": exposure_ok, "shake_score": 5.0,
    })
    return FilmObservation(
        observation_id=f"det_{index:08d}", media_asset_id=f"shot_{index:08d}",
        media_hash=f"hash_{index}", start_frame=start, end_frame=end,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim=claim, provider="deterministic_opencv", model_version="opencv_5.0",
        prompt_version="n/a", confidence=1.0, review_state="auto_verified",
        claim_kind=ClaimKind.MEASURED, schema_version="1.0", project_id="t2",
        created_at=int(time.time()), producer="deterministic_opencv",
        source_ref="t2.mp4",
    )


def main() -> int:
    rz = HeuristicDirectorReasoner()

    # ---- 正样本：低画质但可用 → 显式降级 ----
    obs_ok = [
        _obs(0, 0, 2_000_000, blur=5.0, exposure_ok=True),
        _obs(1, 20_000_000, 22_000_000, blur=5.0, exposure_ok=True),
    ]
    brief = compile_brief("t2", "t2.mp4", obs_ok)
    graph = build_story_graph(brief, obs_ok)
    edl, plan = rz.generate_plan(brief, graph, obs_ok)
    print("=== 正样本：blur 低于阈值（放宽 level=1）===")
    print(f"  plan.degraded          = {plan.degraded}")
    print(f"  plan.degradation_events= {plan.degradation_events}")
    print(f"  plan.open_questions    = {plan.open_questions}")
    print(f"  选片                   = {[e.source_asset_id for e in edl.ordered_edits]}")

    # ---- 负样本：证据不足 → fail-closed ----
    obs_bad = [
        _obs(0, 0, 2_000_000, blur=200.0, exposure_ok=False),
        _obs(1, 2_000_000, 4_000_000, blur=200.0, exposure_ok=False),
    ]
    brief2 = compile_brief("t2", "t2.mp4", obs_bad)
    graph2 = build_story_graph(brief2, obs_bad)
    print("\n=== 负样本：0/2 曝光合格（历史行为：静默强制可用）===")
    try:
        rz.generate_plan(brief2, graph2, obs_bad)
        print("  [FAIL] 未抛错——静默注水仍然存在！")
        return 1
    except EvidenceTooPoorError as exc:
        print(f"  [PASS] EvidenceTooPoorError: {exc}")

    if plan.degraded and plan.degradation_events:
        print("\n>>> T2 演示通过：降级显式留痕 + 证据不足 fail-closed。")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
