"""Human-entered cross-asset identity links, kept apart from inferred graphs."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from director_brain.models.base import BaseRecord
from director_brain.models.film_observation import TimebaseUnit


class ProjectStoryLinkComparisonDisposition(BaseModel):
    """Caller-asserted disposition of one immutable model comparison."""

    model_config = ConfigDict(extra="forbid")

    comparison_id: str = Field(min_length=1, max_length=160)
    decision: Literal[
        "accepted_as_caller_asserted",
        "rejected",
        "needs_more_evidence",
    ]
    link_id: str | None = Field(default=None, min_length=1, max_length=128)
    rationale: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_link_binding(self) -> Self:
        if self.decision == "accepted_as_caller_asserted" and self.link_id is None:
            raise ValueError("accepted comparison disposition requires a link_id")
        if self.decision != "accepted_as_caller_asserted" and self.link_id is not None:
            raise ValueError("only an accepted comparison disposition can bind a link_id")
        return self


class ProjectStoryLinkAnchor(BaseModel):
    """A caller-selected interval bounded by one stored source observation."""

    model_config = ConfigDict(extra="forbid")

    project_asset_id: str = Field(min_length=1)
    source_asset_id: str | None = Field(default=None, min_length=1)
    # Optional for compatibility with persisted interval-only reviews. New
    # person/event reviews can bind exact mention nodes; place links remain
    # caller-selected intervals unless a location node is explicitly present.
    story_graph_node_id: str | None = Field(default=None, min_length=1)
    observation_id: str = Field(min_length=1)
    source_content_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    timebase: int
    timebase_unit: TimebaseUnit

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.source_end <= self.source_start:
            raise ValueError("source_end must be greater than source_start")
        return self


class ProjectStoryEntityLink(BaseModel):
    """A caller-asserted identity grouping across independent source assets."""

    model_config = ConfigDict(extra="forbid")

    link_id: str = Field(min_length=1, max_length=128)
    entity_kind: Literal[
        "person_identity", "event_identity", "place_identity"]
    display_label: str = Field(min_length=1, max_length=200)
    anchors: list[ProjectStoryLinkAnchor] = Field(min_length=2, max_length=500)
    source_comparison_ids: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def validate_cross_asset_scope(self) -> Self:
        exact_anchors = [
            (item.project_asset_id, item.story_graph_node_id, item.observation_id,
             item.source_start, item.source_end)
            for item in self.anchors
        ]
        if len(exact_anchors) != len(set(exact_anchors)):
            raise ValueError("a link cannot repeat an exact source anchor")
        bound_mentions = [
            (item.project_asset_id, item.story_graph_node_id)
            for item in self.anchors
            if item.story_graph_node_id is not None
        ]
        if len(bound_mentions) != len(set(bound_mentions)):
            raise ValueError("a link cannot repeat an exact mention node")
        if len({item.project_asset_id for item in self.anchors}) < 2:
            raise ValueError("a cross-asset link requires anchors from at least two assets")
        if len(self.source_comparison_ids) != len(set(self.source_comparison_ids)):
            raise ValueError("a link cannot repeat a source comparison id")
        if self.entity_kind == "place_identity" and self.source_comparison_ids:
            raise ValueError(
                "place links require caller-selected source intervals, not person/event model comparisons")
        return self


class ProjectStoryRelationClaim(BaseModel):
    """One caller-asserted semantic edge between exact cross-asset mentions."""

    model_config = ConfigDict(extra="forbid")

    relation_id: str = Field(min_length=1, max_length=128)
    relation_type: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    from_anchor: ProjectStoryLinkAnchor
    to_anchor: ProjectStoryLinkAnchor
    evidence_refs: list[str] = Field(min_length=2, max_length=2)
    relation_type_state: Literal["caller_supplied_unstandardized"] = (
        "caller_supplied_unstandardized"
    )
    confirmation_state: Literal["caller_asserted"] = "caller_asserted"
    privacy_class: Literal["public", "internal", "private", "sensitive"] = "private"

    @model_validator(mode="after")
    def validate_relation_evidence(self) -> Self:
        if self.from_anchor.project_asset_id == self.to_anchor.project_asset_id:
            raise ValueError("project story relation must connect different assets")
        if (
            self.from_anchor.story_graph_node_id is None
            or self.to_anchor.story_graph_node_id is None
        ):
            raise ValueError("project story relation endpoints require exact graph nodes")
        expected_refs = [
            self.from_anchor.observation_id,
            self.to_anchor.observation_id,
        ]
        if expected_refs[0] == expected_refs[1]:
            raise ValueError("project story relation endpoints require distinct observations")
        if self.evidence_refs != expected_refs:
            raise ValueError("project story relation evidence must match endpoint observations")
        return self


class ProjectStoryLinkReview(BaseRecord):
    """Immutable, revision-bound caller overlay for cross-asset graph claims.

    A comparison disposition can point to an immutable local-model pair result,
    but never changes that result's unreviewed state or proves the relationship.
    The source-local StoryGraph remains unchanged. A later revision replaces the
    complete identity-link and semantic-relation sets while prior snapshots stay
    in the decision ledger.
    """

    # v1.1 makes exact mention membership mutually exclusive across identity
    # groups; v1.2 binds source mention versions; v1.3 adds caller-asserted places;
    # v1.4 also prevents interval-only anchors being assigned to multiple groups;
    # v1.5 adds evidence-bound semantic relation claims.
    # Explicitly loaded v1.0/v1.1 snapshots remain readable below.
    schema_version: str = "1.5"
    review_id: str
    project_manifest_id: str
    project_revision: int = Field(ge=1)
    context_id: str
    story_graph_id: str
    analysis_fingerprint: str
    source_mention_review_id: str | None = Field(default=None, min_length=1)
    source_mention_review_revision: int = Field(default=0, ge=0)
    review_revision: int = Field(ge=1)
    review_scope: Literal[
        "project_cross_asset_identity_overlay", "project_cross_asset_review"
    ] = (
        "project_cross_asset_identity_overlay"
    )
    evidence_state: Literal["caller_asserted"] = "caller_asserted"
    actor_identity_state: Literal["caller_asserted"] = "caller_asserted"
    automatic_inference_state: Literal["not_attempted"] = "not_attempted"
    links: list[ProjectStoryEntityLink] = Field(default_factory=list, max_length=500)
    relations: list[ProjectStoryRelationClaim] = Field(default_factory=list, max_length=500)
    comparison_dispositions: list[ProjectStoryLinkComparisonDisposition] = Field(
        default_factory=list, max_length=500)

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if (self.source_mention_review_id is None) != (
            self.source_mention_review_revision == 0
        ):
            raise ValueError("source mention review identity and revision must be bound")
        link_ids = [item.link_id for item in self.links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("link ids must be unique within a review revision")
        relation_ids = [item.relation_id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation ids must be unique within a review revision")
        if set(link_ids) & set(relation_ids):
            raise ValueError("identity link and semantic relation ids must be distinct")
        relation_keys = [
            (
                item.relation_type,
                item.from_anchor.project_asset_id,
                item.from_anchor.story_graph_node_id,
                item.to_anchor.project_asset_id,
                item.to_anchor.story_graph_node_id,
            )
            for item in self.relations
        ]
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("a review cannot repeat an exact semantic relation")
        if self.relations and self.schema_version != "1.5":
            raise ValueError("semantic relations require review schema 1.5")
        if self.schema_version != "1.0":
            validate_project_story_link_review_mention_exclusivity(
                self,
                include_interval_only=self.schema_version in {"1.4", "1.5"},
            )
        links_by_id = {item.link_id: item for item in self.links}
        dispositions = {
            item.comparison_id: item for item in self.comparison_dispositions
        }
        if len(dispositions) != len(self.comparison_dispositions):
            raise ValueError("comparison dispositions must have unique comparison ids")
        for disposition in self.comparison_dispositions:
            link = links_by_id.get(disposition.link_id or "")
            if disposition.decision == "accepted_as_caller_asserted":
                if link is None or disposition.comparison_id not in link.source_comparison_ids:
                    raise ValueError(
                        "accepted comparison disposition must bind its exact link")
            elif any(
                disposition.comparison_id in item.source_comparison_ids
                for item in self.links
            ):
                raise ValueError(
                    "non-accepted comparison disposition cannot create a link")
        for link in self.links:
            for comparison_id in link.source_comparison_ids:
                disposition = dispositions.get(comparison_id)
                if (
                    disposition is None
                    or disposition.decision != "accepted_as_caller_asserted"
                    or disposition.link_id != link.link_id
                ):
                    raise ValueError(
                        "a source comparison requires a matching caller-accepted disposition")
        return self


def validate_project_story_link_review_mention_exclusivity(
    review: ProjectStoryLinkReview,
    *,
    include_interval_only: bool = True,
) -> None:
    """Reject conflicting source-anchor ownership in any consumed snapshot.

    Schema 1.0 remains constructible for historical readback. Schema 1.4
    extends exact mention exclusivity to exact interval-only anchors. Consumers
    apply the interval check to every active snapshot so older contradictory
    data remains auditable but cannot be projected or used for a Plan.
    """
    mention_owners: dict[tuple[str, str, str], str] = {}
    interval_owners: dict[tuple, str] = {}
    for link in review.links:
        for anchor in link.anchors:
            if anchor.story_graph_node_id is not None:
                mention_key = (
                    link.entity_kind,
                    anchor.project_asset_id,
                    anchor.story_graph_node_id,
                )
                previous_link_id = mention_owners.setdefault(
                    mention_key, link.link_id)
                if previous_link_id != link.link_id:
                    raise ValueError(
                        "an exact mention node cannot belong to multiple identity groups"
                    )
                continue

            if not include_interval_only:
                continue
            interval_key = (
                link.entity_kind,
                anchor.project_asset_id,
                anchor.source_asset_id,
                anchor.observation_id,
                anchor.source_content_hash.lower(),
                anchor.source_start,
                anchor.source_end,
                anchor.timebase,
                anchor.timebase_unit,
            )
            previous_link_id = interval_owners.setdefault(
                interval_key, link.link_id)
            if previous_link_id != link.link_id:
                raise ValueError(
                    "an exact interval anchor cannot belong to multiple identity groups"
                )
