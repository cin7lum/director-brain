"""Project-scoped collection of source-local StoryGraphs.

This contract preserves independent source timelines. It intentionally carries
no inferred cross-asset person, event, or causal links.
"""
from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from director_brain.models.base import BaseRecord
from director_brain.models.film_context import AssetAnalysisState
from director_brain.models.story_graph import StoryGraph
from director_brain.models.project_story_link_review import (
    ProjectStoryLinkAnchor,
    ProjectStoryLinkReview,
    ProjectStoryRelationClaim,
    validate_project_story_link_review_mention_exclusivity,
)


class ProjectAssetStoryGraph(BaseModel):
    """One asset's structural timeline and exact evidence references."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    source_content_hash: str | None = Field(
        default=None, pattern=r"^[a-fA-F0-9]{64}$")
    analysis_state: AssetAnalysisState
    story_graph_state: Literal[
        "constructed",
        "no_deterministic_technical_observations",
        "not_authorized",
        "analysis_incomplete",
    ]
    evidence_refs: list[str] = Field(default_factory=list)
    story_graph: StoryGraph | None = None

    @model_validator(mode="after")
    def validate_story_graph_scope(self) -> Self:
        if self.story_graph_state == "constructed":
            if self.story_graph is None:
                raise ValueError("constructed asset graph requires a StoryGraph")
            if self.story_graph.project_asset_id != self.asset_id:
                raise ValueError("asset StoryGraph scope does not match asset_id")
            if not self.evidence_refs:
                raise ValueError("constructed asset graph requires evidence refs")
            if any(
                ref not in self.evidence_refs
                for edge in self.story_graph.edges
                for ref in edge.evidence_refs
            ):
                raise ValueError("asset StoryGraph edge cites out-of-scope evidence")
        elif self.story_graph is not None:
            raise ValueError("non-constructed asset graph cannot carry a StoryGraph")

        if self.story_graph_state == "not_authorized":
            if self.analysis_state != AssetAnalysisState.NOT_AUTHORIZED:
                raise ValueError("not_authorized graph state requires blocked analysis")
            if self.evidence_refs:
                raise ValueError("unauthorized asset cannot carry evidence refs")
        if self.story_graph_state == "analysis_incomplete" and self.analysis_state not in (
            AssetAnalysisState.FAILED,
            AssetAnalysisState.SOURCE_UNAVAILABLE,
            AssetAnalysisState.SOURCE_CHANGED,
        ):
            raise ValueError("analysis_incomplete graph state requires incomplete analysis")
        return self


class ProjectStoryGraph(BaseRecord):
    """Revision-bound project view containing only independent asset graphs."""

    graph_id: str
    project_manifest_id: str
    project_revision: int = Field(ge=1)
    context_id: str
    analysis_fingerprint: str
    boundary_state: Literal["declared_unverified"] = "declared_unverified"
    timeline_scope: Literal["project_per_asset"] = "project_per_asset"
    relation_scope: Literal["asset_local_structural_only"] = (
        "asset_local_structural_only"
    )
    cross_asset_relations_state: Literal["not_attempted"] = "not_attempted"
    evidence_refs: list[str] = Field(default_factory=list)
    assets: list[ProjectAssetStoryGraph] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_project_aggregation(self) -> Self:
        asset_ids = [item.asset_id for item in self.assets]
        orders = [item.order for item in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("project graph asset ids must be unique")
        if len(orders) != len(set(orders)):
            raise ValueError("project graph asset order values must be unique")
        if orders != sorted(orders):
            raise ValueError("project graph assets must be ordered by manifest order")
        flattened_refs = [ref for asset in self.assets for ref in asset.evidence_refs]
        if len(flattened_refs) != len(set(flattened_refs)):
            raise ValueError("project graph evidence refs must be unique across assets")
        if set(flattened_refs) != set(self.evidence_refs):
            raise ValueError("project graph evidence refs must match asset evidence refs")
        if any(
            asset.story_graph is not None
            and asset.story_graph.project_id != self.project_id
            for asset in self.assets
        ):
            raise ValueError("asset StoryGraph project_id does not match project graph")
        return self


class ProjectStoryGraphRelationMember(BaseModel):
    """One revision-bound source member of a project-level relation group."""

    model_config = ConfigDict(extra="forbid")

    anchor: ProjectStoryLinkAnchor
    node_binding_state: Literal[
        "exact_mention_node", "observation_interval_only"
    ]

    @model_validator(mode="after")
    def validate_binding_state(self) -> Self:
        expected = (
            "exact_mention_node"
            if self.anchor.story_graph_node_id is not None
            else "observation_interval_only"
        )
        if self.node_binding_state != expected:
            raise ValueError("relation member binding state differs from its anchor")
        return self


class ProjectStoryGraphRelationGroup(BaseModel):
    """A caller-asserted identity group projected over independent asset graphs."""

    model_config = ConfigDict(extra="forbid")

    relation_id: str = Field(min_length=1)
    relation_kind: Literal[
        "person_identity", "event_identity", "place_identity"]
    display_label: str = Field(min_length=1, max_length=200)
    members: list[ProjectStoryGraphRelationMember] = Field(min_length=2, max_length=500)
    source_review_id: str = Field(min_length=1)
    source_review_revision: int = Field(ge=1)
    source_link_id: str = Field(min_length=1)
    source_comparison_ids: list[str] = Field(default_factory=list, max_length=50)
    assertion_state: Literal["caller_asserted"] = "caller_asserted"
    identity_verification_state: Literal["not_independently_verified"] = (
        "not_independently_verified"
    )

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        if len({item.anchor.project_asset_id for item in self.members}) < 2:
            raise ValueError("project StoryGraph relation must span multiple assets")
        return self


class ProjectStoryGraphView(BaseModel):
    """Read projection of source-local graphs and current caller graph claims.

    The persisted source graph and immutable caller link review remain separate
    evidence records. This view gives consumers one revision-bound object while
    preserving that provenance boundary; identity groups and semantic relation
    edges remain caller-asserted and are neither inferred nor verified here.
    """

    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1)
    source_graph: ProjectStoryGraph
    cross_asset_link_review: ProjectStoryLinkReview | None = None
    source_mention_review_id: str | None = Field(default=None, min_length=1)
    source_mention_review_revision: int = Field(default=0, ge=0)
    cross_asset_relations: list[ProjectStoryGraphRelationGroup] = Field(
        default_factory=list)
    cross_asset_relation_edges: list[ProjectStoryRelationClaim] = Field(
        default_factory=list)
    link_snapshot_state: Literal["absent", "caller_asserted", "stale"]
    cross_asset_relation_state: Literal[
        "none", "caller_asserted", "needs_reconfirmation"
    ]
    cross_asset_link_count: int = Field(ge=0)
    cross_asset_relation_edge_state: Literal[
        "none", "caller_asserted", "needs_reconfirmation"
    ] = "none"
    cross_asset_relation_edge_count: int = Field(default=0, ge=0)
    automatic_inference_state: Literal["not_attempted"] = "not_attempted"
    schema_version: Literal["1.4"] = "1.4"

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        if (self.source_mention_review_id is None) != (
            self.source_mention_review_revision == 0
        ):
            raise ValueError("active source mention review identity and revision must bind")
        review = self.cross_asset_link_review
        if review is None:
            if self.link_snapshot_state != "absent":
                raise ValueError("absent link review requires absent link state")
            if self.cross_asset_link_count != 0:
                raise ValueError("absent link review cannot declare cross-asset links")
            if self.cross_asset_relations:
                raise ValueError("absent link review cannot project cross-asset relations")
            if self.cross_asset_relation_edges:
                raise ValueError("absent link review cannot project relation edges")
            if self.cross_asset_relation_state != "none":
                raise ValueError("absent link review cannot declare relation state")
            if self.cross_asset_relation_edge_state != "none":
                raise ValueError("absent link review cannot declare relation-edge state")
            if self.cross_asset_relation_edge_count != 0:
                raise ValueError("absent link review cannot declare relation edges")
        else:
            graph = self.source_graph
            validate_project_story_link_review_mention_exclusivity(review)
            if (
                review.project_id != graph.project_id
                or review.project_manifest_id != graph.project_manifest_id
                or review.project_revision != graph.project_revision
                or review.context_id != graph.context_id
                or review.story_graph_id != graph.graph_id
                or review.analysis_fingerprint != graph.analysis_fingerprint
            ):
                raise ValueError("link review must match the exact source graph revision")
            has_mention_dependent_evidence = bool(
                review.links or review.relations or review.comparison_dispositions)
            source_mention_is_current = (
                not has_mention_dependent_evidence
                or (
                    review.source_mention_review_id == self.source_mention_review_id
                    and review.source_mention_review_revision
                    == self.source_mention_review_revision
                )
            )
            expected_link_state = (
                "caller_asserted" if source_mention_is_current else "stale")
            if self.link_snapshot_state != expected_link_state:
                raise ValueError("link snapshot currentness differs from source mention version")
            expected_relation_state = (
                "needs_reconfirmation" if not source_mention_is_current
                else "caller_asserted" if review.links
                else "none"
            )
            expected_active_count = len(review.links) if source_mention_is_current else 0
            expected_relation_edge_state = (
                "needs_reconfirmation"
                if review.relations and not source_mention_is_current
                else "caller_asserted" if review.relations
                else "none"
            )
            expected_relation_edges = (
                review.relations if source_mention_is_current else [])
            if self.cross_asset_relation_state != expected_relation_state:
                raise ValueError("cross-asset relation state differs from its evidence currentness")
            if self.cross_asset_link_count != expected_active_count:
                raise ValueError("cross-asset link count must include only current relations")
            if len(self.cross_asset_relations) != expected_active_count:
                raise ValueError("project relation projection must include only current links")
            if self.cross_asset_relation_edge_state != expected_relation_edge_state:
                raise ValueError("cross-asset relation-edge state differs from its evidence currentness")
            if self.cross_asset_relation_edge_count != len(expected_relation_edges):
                raise ValueError("project relation-edge count must include only current claims")
            if self.cross_asset_relation_edges != expected_relation_edges:
                raise ValueError("project relation-edge projection differs from its caller review")
            graph_nodes = {
                asset.asset_id: {
                    node.node_id: node
                    for node in asset.story_graph.nodes
                }
                for asset in graph.assets
                if asset.story_graph is not None
            }
            active_links = review.links if source_mention_is_current else []
            for projected, link in zip(self.cross_asset_relations, active_links):
                if (
                    projected.relation_id != link.link_id
                    or projected.source_link_id != link.link_id
                    or projected.relation_kind != link.entity_kind
                    or projected.display_label != link.display_label
                    or projected.source_review_id != review.review_id
                    or projected.source_review_revision != review.review_revision
                    or projected.source_comparison_ids != link.source_comparison_ids
                    or [item.anchor for item in projected.members] != link.anchors
                ):
                    raise ValueError("project relation projection differs from its caller link")
                expected_node_type = {
                    "person_identity": "person_mention",
                    "event_identity": "event_mention",
                    "place_identity": "location",
                }[link.entity_kind]
                for member in projected.members:
                    anchor = member.anchor
                    node_id = anchor.story_graph_node_id
                    if node_id is None:
                        continue
                    node = graph_nodes.get(anchor.project_asset_id, {}).get(node_id)
                    source_anchor = node.attributes.get("source_anchor") if node else None
                    node_start = (
                        source_anchor.get("source_start")
                        if isinstance(source_anchor, dict) else None
                    )
                    node_end = (
                        source_anchor.get("source_end")
                        if isinstance(source_anchor, dict) else None
                    )
                    if (
                        node is None
                        or node.node_type.value != expected_node_type
                        or node.ref_id != anchor.observation_id
                        or not isinstance(source_anchor, dict)
                        or source_anchor.get("project_asset_id") != anchor.project_asset_id
                        or str(source_anchor.get("source_content_hash", "")).lower()
                        != anchor.source_content_hash.lower()
                        or source_anchor.get("timebase") != anchor.timebase
                        or source_anchor.get("timebase_unit") != anchor.timebase_unit.value
                        or type(node_start) is not int
                        or type(node_end) is not int
                        or anchor.source_start < node_start
                        or anchor.source_end > node_end
                    ):
                        raise ValueError("project relation member differs from its graph node")

            for relation in self.cross_asset_relation_edges:
                for anchor in (relation.from_anchor, relation.to_anchor):
                    asset_graph = next(
                        (item for item in graph.assets
                         if item.asset_id == anchor.project_asset_id),
                        None,
                    )
                    node = graph_nodes.get(anchor.project_asset_id, {}).get(
                        anchor.story_graph_node_id or "")
                    source_anchor = node.attributes.get("source_anchor") if node else None
                    node_start = (
                        source_anchor.get("source_start")
                        if isinstance(source_anchor, dict) else None
                    )
                    node_end = (
                        source_anchor.get("source_end")
                        if isinstance(source_anchor, dict) else None
                    )
                    if (
                        node is None
                        or asset_graph is None
                        or asset_graph.source_content_hash is None
                        or asset_graph.source_content_hash.lower()
                        != anchor.source_content_hash.lower()
                        or anchor.observation_id not in asset_graph.evidence_refs
                        or node.node_type.value not in {
                            "person_mention", "event_mention", "location"
                        }
                        or node.ref_id != anchor.observation_id
                        or not isinstance(source_anchor, dict)
                        or source_anchor.get("project_asset_id") != anchor.project_asset_id
                        or str(source_anchor.get("source_content_hash", "")).lower()
                        != anchor.source_content_hash.lower()
                        or source_anchor.get("timebase") != anchor.timebase
                        or source_anchor.get("timebase_unit") != anchor.timebase_unit.value
                        or type(node_start) is not int
                        or type(node_end) is not int
                        or anchor.source_start < node_start
                        or anchor.source_end > node_end
                    ):
                        raise ValueError("project relation endpoint differs from its graph node")

        return self
