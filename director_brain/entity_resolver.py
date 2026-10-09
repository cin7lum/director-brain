"""实体解析器（D2 · 人物与关系理解 v1）。

方案 §4.4/§5 指引：StoryGraph 管理"人物、地点、事件、关系"；
FilmEntity/StoryRelation 模型已定义、实现此前空白。本模块实现 v1：

**跨镜头"这是同一个人"**——从 VLM 语义观测（vlm_prompt_v4_people 的
people 外观描述字段）做**确定性外观聚类**：颜色+衣物/发型判别性词对
匹配 → 实体归属；identity_confidence 按 §5 要求诚实分级（0.8 共享
判别词对 / 0.5 部分词重叠 / 0.3 弱归属）。

诚实边界（v1 已知局限，账本留痕）：
- 外观词表法对换装/遮挡/相似穿着会误合——confidence 分级可见，不冒充事实；
- StoryRelation 的人际类型（family/friend…）需要叙事证据，v1 不推断
  （NOT_DETERMINED 纪律）；本模块只产出"同实体跨镜头出现"证据；
- 单次出现的实体保留（identity_confidence 低），不强行合并。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from director_brain.models.film_entity import EntityType, FilmEntity
from director_brain.models.film_observation import ClaimKind, FilmObservation

#: 判别性外观词表（v1：颜色 + 衣物/发型载体；可随实测扩充）
_COLOR_TOKENS = (
    "红", "蓝", "白", "黑", "灰", "绿", "黄", "粉", "棕", "紫", "金", "橙",
)
_GARMENT_TOKENS = (
    "衣", "裙", "帽", "发", "袍", "西装", "制服", "甲", "披风", "裤",
    "围巾", "鞋", "靴", "面具", "眼镜",
)
_CONF_PAIR = 0.8    # 共享 颜色+载体 判别词对
_CONF_PARTIAL = 0.5  # 共享单一判别词
_CONF_WEAK = 0.3     # 仅整串包含/兜底同槽


def _tokens(desc: str) -> set[str]:
    """从外观描述提取判别 token（颜色×载体组合 + 单 token）。"""
    tokens: set[str] = set()
    for c in _COLOR_TOKENS:
        if c in desc:
            tokens.add(c)
            for g in _GARMENT_TOKENS:
                if g in desc:
                    tokens.add(f"{c}{g}")
    for g in _GARMENT_TOKENS:
        if g in desc:
            tokens.add(g)
    return tokens


def _norm(desc: str) -> str:
    return re.sub(r"\s+", "", desc or "").lower()


@dataclass
class EntityResolution:
    """一次实体解析的完整产物（账本留痕用）。"""

    entities: list[FilmEntity] = field(default_factory=list)
    #: shot_id → entity_id 列表（一镜多人）
    assignments: dict[str, list[str]] = field(default_factory=dict)
    #: entity_id → 支撑观测 id（证据可追溯）
    evidence: dict[str, list[str]] = field(default_factory=dict)
    #: 聚类过程摘要（诚实边界声明）
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entities": [
                {"entity_id": e.entity_id, "display_name": e.display_name,
                 "identity_confidence": e.identity_confidence}
                for e in self.entities
            ],
            "assignments": self.assignments,
            "evidence": self.evidence,
            "limitations": self.limitations,
        }

    def entity_ids_of(self, shot_id: str) -> frozenset[str]:
        return frozenset(self.assignments.get(shot_id, []))


def resolve_entities(
    vlm_obs: list[FilmObservation],
) -> EntityResolution:
    """从 vlm_semantic 观测做确定性外观聚类（按源时间序贪心）。

    只消费 claim_kind == MODEL_OBSERVATION 的观测；people 字段缺失/为空
    的镜头不参与聚类（无证据不归属）。贪心规则：新外观描述与既有实体
    的**任一**成员描述共享判别词对 → 归入（confidence=_CONF_PAIR）；
    仅共享单 token → 归入（_CONF_PARTIAL）；否则新建实体（_CONF_WEAK
    起步，单成员实体保留 _CONF_WEAK）。
    """
    res = EntityResolution()
    eligible = [
        observation for observation in vlm_obs
        if observation.claim_kind is ClaimKind.MODEL_OBSERVATION
    ]
    project_asset_ids = {observation.project_asset_id for observation in eligible}
    if len(project_asset_ids) > 1:
        if None in project_asset_ids:
            raise ValueError(
                "entity resolution cannot mix bound and unbound observations"
            )
        raise ValueError(
            "entity resolution requires one project_asset_id; cross-asset "
            "identity matching is not admitted by this resolver"
        )
    project_ids = {observation.project_id for observation in eligible}
    if len(project_ids) > 1:
        raise ValueError("entity resolution cannot mix project_id values")
    timebases = {
        (observation.timebase, observation.timebase_unit)
        for observation in eligible
    }
    if len(timebases) > 1:
        raise ValueError(
            "entity resolution requires one source timebase for ordering"
        )

    # 实体成员外观列表：entity_id → [(descriptor, tokens, obs_id)]
    members: dict[str, list[tuple[str, set[str], str]]] = {}
    member_conf: dict[str, float] = {}  # entity_id → 分配时的归属置信
    counter = 0

    for o in sorted(eligible, key=lambda x: x.start_frame):
        try:
            claim = json.loads(o.claim) if o.claim else {}
        except (json.JSONDecodeError, TypeError):
            continue
        people = claim.get("people") or []
        if not isinstance(people, list):
            continue

        for desc in people:
            if not isinstance(desc, str) or not desc.strip():
                continue
            desc = desc.strip()
            toks = _tokens(desc)
            norm = _norm(desc)

            best_eid: str | None = None
            best_conf = 0.0
            for eid, mems in members.items():
                conf = 0.0
                for _d, mtoks, _o in mems:
                    if toks and mtoks:
                        # 判别词对：颜色+载体 复合 token 双方共享
                        pair_hits = any(
                            t in mtoks and len(t) >= 2 and t[0] in _COLOR_TOKENS
                            for t in toks)
                        if pair_hits:
                            conf = max(conf, _CONF_PAIR)
                        else:
                            shared = toks & mtoks
                            # 部分匹配必须共享颜色词或 ≥2 token——
                            # 泛型载体词（衣/发）单命中不构成身份证据
                            color_shared = any(t in shared for t in _COLOR_TOKENS)
                            if color_shared or len(shared) >= 2:
                                conf = max(conf, _CONF_PARTIAL)
                    if norm and (norm in _norm(_d) or (_norm(_d) and _norm(_d) in norm)):
                        conf = max(conf, _CONF_PARTIAL)
                if conf > best_conf:
                    best_conf, best_eid = conf, eid

            if best_eid is None:
                counter += 1
                eid = f"person_{counter:03d}"
                members[eid] = []
                best_eid, best_conf = eid, _CONF_WEAK

            members[best_eid].append((desc, toks, o.observation_id))
            member_conf[best_eid] = max(member_conf.get(best_eid, 0.0), best_conf)
            res.assignments.setdefault(o.media_asset_id, []).append(best_eid)
            res.evidence.setdefault(best_eid, []).append(o.observation_id)

    for eid, mems in members.items():
        descs = [m[0] for m in mems]
        display = max(set(descs), key=descs.count)
        conf = (max(member_conf.get(eid, _CONF_WEAK), _CONF_PARTIAL)
                if len(mems) > 1 else _CONF_WEAK)
        res.entities.append(FilmEntity(
            schema_version="1.0",
            project_id="unknown",
            created_at=0,
            producer="entity_resolver_v1",
            source_ref="vlm_semantic",
            entity_id=eid,
            entity_type=EntityType.PERSON,
            display_name=display,
            identity_confidence=round(conf, 2),
        ))

    res.limitations = [
        "v1 外观词表聚类：换装/遮挡/相似穿着可能误合，confidence 分级可见",
        "人际关系类型（family/friend…）需叙事证据，v1 不推断（NOT_DETERMINED）",
    ]
    return res
