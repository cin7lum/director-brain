"""故事图 StoryGraph 模型（含嵌入式 StoryNode / StoryEdge）。"""
from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from director_brain.models.base import BaseRecord


class StoryNodeType(str, enum.Enum):
    """故事图节点类型。"""

    PERSON = "person"
    EVENT = "event"
    LOCATION = "location"
    SHOT = "shot"
    OBSERVATION = "observation"
    ACT = "act"


#: StoryEdge.inference_status 模块级常量（不改为枚举，避免破坏兼容性）。
INFERENCE_STATUS_STRUCTURAL = "structural"
INFERENCE_STATUS_INFERRED = "inferred"


class StoryEdgeType(str, enum.Enum):
    """故事图边类型。"""

    TEMPORAL = "temporal"
    CAUSAL_CANDIDATE = "causal_candidate"
    EMOTIONAL_TURN = "emotional_turn"
    NARRATIVE_LINK = "narrative_link"


class StoryNode(BaseModel):
    """故事图节点（嵌入式组件，非顶层记录）。"""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: StoryNodeType
    ref_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class StoryEdge(BaseModel):
    """故事图边（嵌入式组件；每条推断边必须带推断状态与证据）。"""

    model_config = ConfigDict(extra="forbid")

    edge_id: str
    from_node: str
    to_node: str
    edge_type: StoryEdgeType
    inference_status: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float


class StoryGraph(BaseRecord):
    """从观测与实体构建的故事结构。"""

    graph_id: str
    version: str
    nodes: list[StoryNode] = Field(default_factory=list)
    edges: list[StoryEdge] = Field(default_factory=list)
