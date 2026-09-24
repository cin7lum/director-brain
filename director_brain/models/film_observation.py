"""影片观测 FilmObservation 模型与 6 级 claim_kind 枚举。

claim_kind 把断言按可信来源分为 6 级，供下游决策区分硬证据与模型猜测。
"""
from __future__ import annotations

import enum

from pydantic import Field

from director_brain.models.base import BaseRecord


class ClaimKind(str, enum.Enum):
    """观测断言的置信级别（共 6 级）。"""

    MEASURED = "measured"
    MODEL_OBSERVATION = "model_observation"
    HUMAN_CONFIRMED = "human_confirmed"
    INFERRED = "inferred"
    USER_ASSERTED = "user_asserted"
    NOT_DETERMINED = "not_determined"


class FilmObservation(BaseRecord):
    """对一段媒体素材的单次可溯源观测。"""

    observation_id: str
    media_asset_id: str
    media_hash: str
    start_frame: int
    end_frame: int
    timebase: int
    observation_type: str
    claim: str
    provider: str
    model_version: str
    prompt_version: str
    confidence: float
    evidence_refs: list[str] = Field(default_factory=list)
    review_state: str
    claim_kind: ClaimKind
