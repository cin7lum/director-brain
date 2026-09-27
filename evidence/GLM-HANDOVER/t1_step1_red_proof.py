# -*- coding: utf-8 -*-
"""T1 commit-1 证据：validator 新增 plan↔EDL 交叉校验「能判红」的证明。

复刻历史 P0-1 分裂形态（旧 repair 的 rule4/6 行为：EDL 删了一个片段、
plan 一个字段不更新），证明：
  1. 加校验前：该分裂样本 validate_plan = PASS（历史 bug）；
  2. 加校验后：同一分裂样本 validate_plan = FAIL，并给出 plan/edl split 错误；
  3. 一致样本仍 PASS（不误伤）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from director_brain.models.director_plan import Decision, DirectorDecisionPlan
from director_brain.models.edl import EditorialDecisionList, EditItem
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.plan_validator import validate_plan


def _obs(asset_id: str) -> FilmObservation:
    return FilmObservation(
        observation_id=f"obs_{asset_id}", media_asset_id=asset_id,
        media_hash=f"hash_{asset_id}", start_frame=0, end_frame=10_000_000,
        timebase=1_000_000, observation_type="deterministic_technical",
        claim="{}", provider="test", model_version="test", prompt_version="test",
        confidence=1.0, review_state="auto_verified", claim_kind=ClaimKind.MEASURED,
        schema_version="1.0", project_id="t1", created_at=int(time.time()),
        producer="test", source_ref="t1.mp4",
    )


def _edit(asset_id: str, in_us: int, out_us: int) -> EditItem:
    return EditItem(
        source_asset_id=asset_id, source_media_hash=f"hash_{asset_id}",
        in_frame=in_us, out_frame=out_us, timebase=1_000_000,
    )


def _edl(edits: list[EditItem]) -> EditorialDecisionList:
    return EditorialDecisionList(
        schema_version="1.0", project_id="t1", created_at=int(time.time()),
        producer="test", source_ref="t1.mp4", edl_id="edl_t1", version="0.1",
        brief_version="0.1", context_id="ctx", timebase=1_000_000,
        ordered_edits=edits,
        expected_duration=sum(e.out_frame - e.in_frame for e in edits),
        approval_state="draft",
    )


def _plan_for(edits: list[EditItem]) -> DirectorDecisionPlan:
    ids = [e.source_asset_id for e in edits]
    return DirectorDecisionPlan(
        schema_version="1.0", project_id="t1", created_at=int(time.time()),
        producer="test", source_ref="t1.mp4", plan_id="plan_t1", version="0.1",
        brief_version="0.1", film_state_version="0.1", sequence=list(ids),
        decisions=[
            Decision(decision_id=f"dec_{i}", purpose="select_shot", shot_refs=[aid])
            for i, aid in enumerate(ids)
        ],
        constraints=[], open_questions=[],
        validation_status="pending", approval_state="draft",
    )


def main() -> int:
    obs = [_obs("shot_a"), _obs("shot_b"), _obs("shot_c")]
    edits = [
        _edit("shot_a", 1_000_000, 3_000_000),
        _edit("shot_b", 5_000_000, 8_000_000),
        _edit("shot_c", 10_000_000, 12_000_000),
    ]
    edl = _edl(edits)
    plan = _plan_for(edits)

    ok, errors = validate_plan(edl, plan, obs)
    print(f"[1] 一致样本（修复前）: valid={ok} errors={errors}")
    if not ok:
        print("FAIL: 一致样本被误判")
        return 2

    # ---- 复刻旧 repair 的分裂行为：EDL 删掉最后一个片段，plan 不动 ----
    split_edl = _edl(edits[:-1])
    ok2, errors2 = validate_plan(split_edl, plan, obs)
    print(f"[2] 分裂样本（EDL 2 段 / plan 3 段）: valid={ok2}")
    for e in errors2:
        print(f"    - {e}")
    split_hit = any("plan/edl split" in e for e in errors2)
    if ok2 or not split_hit:
        print("FAIL: 新校验未能判红历史分裂形态")
        return 1

    print("\n[PASS] 校验对历史分裂形态判红（能变红），对一致样本放行（不误伤）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
