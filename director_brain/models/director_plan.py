"""导演决策计划 DirectorDecisionPlan 与 Decision 模型。"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from director_brain.models.base import BaseRecord


class Decision(BaseModel):
    """单条编辑决策理由（嵌入式组件，非顶层记录）。"""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    purpose: str
    shot_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    confidence: float | None = None
    requires_approval: bool = False


class DirectorDecisionPlan(BaseRecord):
    """面向执行的导演决策计划。"""

    plan_id: str
    version: str
    brief_version: str
    film_state_version: str
    sequence: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    validation_status: str
    approval_state: str
    supersedes_plan_id: str | None = None
