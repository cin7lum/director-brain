"""影片实体 FilmEntity 与故事关系 StoryRelation 模型。"""
from __future__ import annotations

import enum

from pydantic import Field

from director_brain.models.base import BaseRecord


class EntityType(str, enum.Enum):
    """实体类型。"""

    PERSON = "person"
    LOCATION = "location"
    EVENT = "event"
    OBJECT = "object"


class FilmEntity(BaseRecord):
    """从素材中抽取的故事实体。"""

    entity_id: str
    entity_type: EntityType
    display_name: str
    identity_confidence: float


class StoryRelation(BaseRecord):
    """两个实体之间的关系。"""

    relation_id: str
    from_entity: str
    to_entity: str
    relation_type: str
    evidence_refs: list[str] = Field(default_factory=list)
    confirmation_state: str
    privacy_class: str
