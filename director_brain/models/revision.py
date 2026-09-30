"""修订提案 RevisionProposal 模型。"""
from __future__ import annotations

from pydantic import Field

from director_brain.models.base import BaseRecord


class RevisionProposal(BaseRecord):
    """针对已有计划 / EDL 的修订提案（架构体检⑦拆雷后的类型化契约）。

    权限模型（与 T1 裁定分层）：修订是**导演级重规划**——应用产物是
    取代性新草案（supersedes_plan_id/edl_id、state=draft），必须重走
    验证 + 策略确认；绝不原地修改已确认的 plan/EDL。修复器（plan_repair）
    的物理-only 白名单不受影响。
    """

    proposal_id: str
    source_finding_ids: list[str] = Field(default_factory=list)
    target_decision_ids: list[str] = Field(default_factory=list)
    change_summary: str
    #: 修订类型（白名单见 revision_engine.RevisionType；旧数据可空——
    #: 旧协议把类型拼在 change_summary 前缀里，属字符串协议已废弃）
    revision_type: str | None = None
    reason: str = ""
    expected_effect: str | None = None
    regression_risks: list[str] = Field(default_factory=list)
    approval_state: str
    verification_plan: str | None = None
    result_refs: list[str] = Field(default_factory=list)
    #: 被取代的旧 plan/EDL（应用产物是新草案时回填）
    supersedes_plan_id: str | None = None
    supersedes_edl_id: str | None = None
