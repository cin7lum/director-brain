"""02 导演脑 REST API（方案 §6 · 8 端点全量对齐）。

FastAPI 实现；每个请求携带 correlation_id + schema_version 信封；
有副作用端点接受 idempotency_key。

启动：uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from director_brain import (
    compile_brief,
    generate_plan,
    propose_revision,
    repair_plan,
    validate_plan,
)
from director_brain.context_gateway import (
    ContextGateway,
    build_asset_context,
    build_evidence_context,
    build_project_context,
    build_scene_context,
)
from director_brain.plan_state import (
    PlanState,
    confirm_strategy,
    compute_edl_hash,
    compute_plan_hash,
    is_confirmation_valid,
    validate_transition,
)

app = FastAPI(
    title="Director Brain API",
    version="1.0.0",
    description="02 导演脑 · 可解释可追溯可修订的导演决策 REST 接口",
)


# ---------------------------------------------------------------------------
# 信封
# ---------------------------------------------------------------------------

def _envelope(correlation_id: str, data: Any) -> dict:
    return {
        "correlation_id": correlation_id,
        "schema_version": "1.0",
        "timestamp": int(time.time()),
        "data": data,
    }


def _new_correlation_id() -> str:
    return f"corr_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class CompileBriefRequest(BaseModel):
    project_id: str
    video_path: str
    intent_text: str | None = None
    target_duration_us: int = 15_000_000
    observations_json: list[dict] | None = None  # 可选：外部观测注入


class GeneratePlanRequest(BaseModel):
    project_id: str
    video_path: str
    brief_id: str | None = None
    intent_text: str | None = None
    target_duration_us: int = 15_000_000
    observations_json: list[dict] | None = None


class ConfirmStrategyRequest(BaseModel):
    plan_id: str
    edl_id: str
    plan_json: dict
    edl_json: dict
    confirmed_by: str = "user"
    output_target: str = "delivery"
    notes: str = ""


class ValidateRequest(BaseModel):
    plan_json: dict
    edl_json: dict
    observations_json: list[dict]


class RevisionRequest(BaseModel):
    finding_ids: list[str] = Field(default_factory=list)
    target_decision_ids: list[str] = Field(default_factory=list)
    change_summary: str = ""


class FilmContextSnapshotRequest(BaseModel):
    """方案 §6 第一条：POST /v1/film-context:snapshot 的请求体。"""

    project_id: str
    video_path: str
    layer: str = "asset"  # project | asset | scene | evidence
    intent_text: str | None = None  # project/scene 层编译 Brief 用
    target_duration_us: int = 15_000_000
    observations_json: list[dict] | None = None  # 缺省时走 analyze_media
    shot_ids: list[str] = Field(default_factory=list)  # evidence 层显式镜头
    window_us: list[int] | None = None  # scene/evidence 层时间窗 [start, end]
    reason: str = ""  # evidence 层必须（审计留痕）


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------

@app.post("/v1/briefs:compile")
def compile_brief_endpoint(req: CompileBriefRequest):
    corr = _new_correlation_id()
    try:
        observations = []
        if req.observations_json:
            from director_brain.models.film_observation import FilmObservation
            observations = [FilmObservation(**o) for o in req.observations_json]
        brief = compile_brief(
            req.project_id, req.video_path, observations,
            target_duration_us=req.target_duration_us,
            intent_text=req.intent_text,
        )
        return _envelope(corr, {
            "brief": brief.model_dump(),
            "brief_id": brief.brief_id,
            "version": brief.version,
        })
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


@app.post("/v1/director-plans:generate")
def generate_plan_endpoint(req: GeneratePlanRequest):
    corr = _new_correlation_id()
    try:
        observations = []
        if req.observations_json:
            from director_brain.models.film_observation import FilmObservation
            observations = [FilmObservation(**o) for o in req.observations_json]
        else:
            from observation_service.pipeline import analyze_media
            observations = analyze_media(req.video_path)

        brief = compile_brief(
            req.project_id, req.video_path, observations,
            target_duration_us=req.target_duration_us,
            intent_text=req.intent_text,
        )
        graph = build_story_graph(brief, observations)
        edl, plan = generate_plan(brief, graph, observations)

        ok, errors = validate_plan(edl, plan, observations)
        plan.validation_status = "valid" if ok else "invalid"

        return _envelope(corr, {
            "plan": plan.model_dump(),
            "edl": edl.model_dump(),
            "plan_hash": compute_plan_hash(plan),
            "edl_hash": compute_edl_hash(edl),
            "validation": {"valid": ok, "errors": errors},
        })
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


@app.post("/v1/director-plans/{plan_id}:confirm-strategy")
def confirm_strategy_endpoint(plan_id: str, req: ConfirmStrategyRequest):
    corr = _new_correlation_id()
    try:
        from director_brain.models.director_plan import DirectorDecisionPlan
        from director_brain.models.edl import EditorialDecisionList
        from director_brain.plan_state import transition_plan
        plan = DirectorDecisionPlan(**req.plan_json)
        edl = EditorialDecisionList(**req.edl_json)
        # 候选③执法：确认前 plan 必须已到 READY_FOR_STRATEGY_CONFIRMATION
        # （非法转换在此抛 InvalidTransition），确认绑定真实 hash 并落账本。
        if plan.state != PlanState.READY_FOR_STRATEGY_CONFIRMATION.value:
            raise HTTPException(
                409,
                detail=f"plan 状态为 {plan.state}，须先验证通过到达 "
                       f"ready_for_strategy_confirmation 才能确认",
            )
        confirmation = confirm_strategy(
            plan, edl, confirmed_by=req.confirmed_by,
            output_target=req.output_target, notes=req.notes,
        )
        transition_plan(plan, PlanState.STRATEGY_CONFIRMED)
        transition_plan(plan, PlanState.DISPATCH_ELIGIBLE)
        # 持久化：决策账本（此前端点返回硬编码状态、不落任何记录）
        from storage.sqlite_repository import SqliteRepository
        from director_brain.audit_trail import log_decision
        try:
            repo = SqliteRepository("./data/director_brain.db")
            log_decision(repo, plan.plan_id, "strategy_confirmed", {
                "plan_hash": confirmation.plan_hash,
                "edl_hash": confirmation.edl_hash,
                "confirmed_by": confirmation.confirmed_by,
                "output_target": confirmation.output_target,
                "state": plan.state,
                "correlation_id": corr,
            })
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, detail=f"确认记录落账本失败: {exc}") from exc
        return _envelope(corr, {
            "plan_id": plan_id,
            "confirmation": confirmation.to_dict(),
            "state": plan.state,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


@app.post("/v1/director-plans/{plan_id}:validate")
def validate_endpoint(plan_id: str, req: ValidateRequest):
    corr = _new_correlation_id()
    try:
        from director_brain.models.director_plan import DirectorDecisionPlan
        from director_brain.models.edl import EditorialDecisionList
        from director_brain.models.film_observation import FilmObservation
        plan = DirectorDecisionPlan(**req.plan_json)
        edl = EditorialDecisionList(**req.edl_json)
        observations = [FilmObservation(**o) for o in req.observations_json]
        ok, errors = validate_plan(edl, plan, observations)
        return _envelope(corr, {"valid": ok, "errors": errors})
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


@app.post("/v1/revisions:propose")
def propose_revision_endpoint(req: RevisionRequest):
    corr = _new_correlation_id()
    return _envelope(corr, {
        "proposal_id": f"rev_{uuid.uuid4().hex[:8]}",
        "source_finding_ids": req.finding_ids,
        "target_decision_ids": req.target_decision_ids,
        "change_summary": req.change_summary,
        "status": "DRAFT",
        "note": "FQL 闭环需 05 模块产出 finding IDs 后才能填实",
    })


# ---------------------------------------------------------------------------
# Film Context 端点（方案 §6 前两条：快照 + 按层查询）
# ---------------------------------------------------------------------------

# 进程内快照存储：POST 写入、GET 读取；键为 context_id（输入确定性哈希，
# 同输入同 id → 天然去重复用）。服务重启即失（快照可由 POST 重建）。
_CONTEXT_STORE: dict[str, dict] = {}


def _load_observations(req: FilmContextSnapshotRequest) -> list:
    if req.observations_json:
        from director_brain.models.film_observation import FilmObservation
        return [FilmObservation(**o) for o in req.observations_json]
    from observation_service.pipeline import analyze_media
    return analyze_media(req.video_path)


@app.post("/v1/film-context:snapshot")
def film_context_snapshot_endpoint(req: FilmContextSnapshotRequest):
    corr = _new_correlation_id()
    try:
        layer = (req.layer or "").lower()
        if layer not in ("project", "asset", "scene", "evidence"):
            raise HTTPException(400, detail="layer 须为 project|asset|scene|evidence")
        observations = _load_observations(req)

        brief = None
        if layer in ("project", "scene"):
            brief = compile_brief(
                req.project_id, req.video_path, observations,
                target_duration_us=req.target_duration_us,
                intent_text=req.intent_text,
            )
        window = (
            tuple(req.window_us)
            if req.window_us and len(req.window_us) == 2
            else None
        )

        # 候选⑤：ContextGateway 类是渐进披露的会话入口（此前休眠）——
        # 端点经其取层快照，status() 汇报已展开层
        if layer == "scene":
            graph = build_story_graph(brief, observations)
            edl, _plan = generate_plan(brief, graph, observations)
            gw = ContextGateway(req.video_path, observations, brief=brief,
                                graph=graph, edl=edl)
            snap = gw.scene(window)
        else:
            gw = ContextGateway(req.video_path, observations, brief=brief)
            if layer == "project":
                snap = gw.project()
            elif layer == "asset":
                snap = gw.asset()
            else:  # evidence
                if not req.reason.strip():
                    raise HTTPException(
                        400,
                        detail="EVIDENCE 层展开必须提供 reason（方案 §4.2 审计留痕）")
                targets = list(req.shot_ids)
                if not targets and window:
                    # 时间窗 → 相交镜头（与 build_scene_context 同一口径：
                    # 窗口对 start_frame/end_frame 区间比较）
                    targets = sorted({
                        o.media_asset_id for o in observations
                        if o.end_frame > window[0] and o.start_frame < window[1]
                    })
                if not targets:
                    raise HTTPException(
                        400, detail="EVIDENCE 层需要 shot_ids 或 window_us 定位目标镜头")
                snap = gw.evidence(targets, req.reason)

        cache_hit = snap.context_id in _CONTEXT_STORE
        payload = snap.model_dump(mode="json")
        _CONTEXT_STORE[snap.context_id] = payload
        return _envelope(corr, {
            "snapshot": payload,
            "context_id": snap.context_id,
            "analysis_fingerprint": snap.analysis_fingerprint,
            "coverage": snap.coverage,
            "evidence_refs": snap.evidence_refs,
            "cache_hit": cache_hit,
            "gateway_status": gw.status(),
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


@app.get("/v1/film-context/{context_id}")
def film_context_get_endpoint(
    context_id: str,
    layer: str | None = Query(
        default=None, description="按需请求的层：project|asset|scene|evidence"),
    reason: str = Query(
        default="", description="EVIDENCE 层用途理由（方案 §6：必须）"),
):
    corr = _new_correlation_id()
    snap = _CONTEXT_STORE.get(context_id)
    if snap is None:
        raise HTTPException(
            404, detail="context 不存在（进程内存储，服务重启后需重建快照）")
    if layer:
        want = layer.lower()
        have = [str(v) for v in snap.get("layers", [])]
        if want not in have:
            raise HTTPException(
                400, detail=f"快照层 {have} 不包含请求层 {want}")
        if want == "evidence" and not reason.strip():
            raise HTTPException(
                400, detail="EVIDENCE 层按需展开必须提供 reason（方案 §6）")
    return _envelope(corr, {
        "snapshot": snap,
        "requested_layer": layer,
        "expansion_reason": reason or None,
    })


@app.get("/v1/projects/{project_id}/decision-ledger")
def decision_ledger_endpoint(project_id: str):
    corr = _new_correlation_id()
    try:
        from storage.sqlite_repository import SqliteRepository
        from storage.repository import DecisionLedgerEntry
        settings_db = "./data/director_brain.db"
        repo = SqliteRepository(settings_db)
        entries = repo.list(DecisionLedgerEntry, project_id=project_id)
        return _envelope(corr, {
            "project_id": project_id,
            "entries": [
                {"ledger_id": e.ledger_id, "decision_id": e.decision_id,
                 "action": e.action, "timestamp": e.timestamp, "detail": e.detail}
                for e in entries
            ],
        })
    except FileNotFoundError:
        return _envelope(corr, {"entries": [], "note": "账本数据库尚未创建"})
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)[:300])


# 导入延迟引用（避免循环 import）
from director_brain.story_graph_builder import build_story_graph  # noqa: E402
from director_brain.plan_state import compute_plan_hash, compute_edl_hash  # noqa: E402
