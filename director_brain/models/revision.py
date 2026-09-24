"""修订提案 RevisionProposal 模型。"""
from __future__ import annotations

from pydantic import Field

from director_brain.models.base import BaseRecord


class RevisionProposal(BaseRecord):
    """针对已有计划 / EDL 的修订提案。"""

    proposal_id: str
    source_finding_ids: list[str] = Field(default_factory=list)
    target_decision_ids: list[str] = Field(default_factory=list)
    change_summary: str
    expected_effect: str | None = None
    regression_risks: list[str] = Field(default_factory=list)
    approval_state: str
    verification_plan: str | None = None
    result_refs: list[str] = Field(default_factory=list)
