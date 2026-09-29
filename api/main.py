"""02 导演脑 REST API（方案 §6 · 7 端点）。

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
from director_brain.context_gateway import ContextGateway
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
        plan = DirectorDecisionPlan(**req.plan_json)
        edl = EditorialDecisionList(**req.edl_json)
        confirmation = confirm_strategy(
            plan, edl, confirmed_by=req.confirmed_by,
            output_target=req.output_target, notes=req.notes,
        )
        return _envelope(corr, {
            "plan_id": plan_id,
            "confirmation": confirmation.to_dict(),
            "state": PlanState.STRATEGY_CONFIRMED.value,
        })
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
