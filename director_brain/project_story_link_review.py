"""Build caller-asserted project link reviews from current stored evidence."""
from __future__ import annotations

import time
from typing import Any

from director_brain.models.film_context import FilmContextSnapshot
from director_brain.models.film_observation import FilmObservation
from director_brain.models.project import FilmProjectManifest, RightsState
from director_brain.models.project_story_graph import ProjectStoryGraph
from director_brain.models.project_story_mention_review import ProjectStoryMentionReview
from director_brain.models.project_story_link_review import (
    ProjectStoryLinkComparisonDisposition,
    ProjectStoryEntityLink,
    ProjectStoryLinkAnchor,
    ProjectStoryLinkReview,
    ProjectStoryRelationClaim,
    validate_project_story_link_review_mention_exclusivity,
)
from director_brain.models.director_plan import DirectorDecisionPlan
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.utils import short_hash


def _anchor_identity(anchor: ProjectStoryLinkAnchor) -> tuple:
    return (
        anchor.project_asset_id,
        anchor.source_asset_id,
        anchor.story_graph_node_id,
        anchor.observation_id,
        anchor.source_content_hash.lower(),
        anchor.source_start,
        anchor.source_end,
        anchor.timebase,
        anchor.timebase_unit,
    )


def _validate_anchor_story_graph_node(
    graph: ProjectStoryGraph,
    entity_kind: str,
    anchor: ProjectStoryLinkAnchor,
) -> None:
    """Validate an optional exact mention-node binding against the graph."""
    if anchor.story_graph_node_id is None:
        return
    expected_node_type = {
        "person_identity": "person_mention",
        "event_identity": "event_mention",
        "place_identity": "location",
    }.get(entity_kind)
    allowed_node_types = (
        {expected_node_type} if expected_node_type is not None else
        {"person_mention", "event_mention", "location"}
        if entity_kind == "semantic_relation" else set()
    )
    if not allowed_node_types:
        raise ValueError("unsupported project story anchor kind")
    asset_graph = next(
        (item for item in graph.assets
         if item.asset_id == anchor.project_asset_id),
        None,
    )
    if asset_graph is None or asset_graph.story_graph is None:
        raise ValueError("link anchor mention node is absent from the project graph")
    node = next(
        (item for item in asset_graph.story_graph.nodes
         if item.node_id == anchor.story_graph_node_id),
        None,
    )
    source_anchor = node.attributes.get("source_anchor") if node else None
    node_start = source_anchor.get("source_start") if isinstance(source_anchor, dict) else None
    node_end = source_anchor.get("source_end") if isinstance(source_anchor, dict) else None
    if (
        node is None
        or node.node_type.value not in allowed_node_types
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
        raise ValueError("link anchor mention node differs from its source anchor")


def validate_project_story_comparison_dispositions(
    review: ProjectStoryLinkReview,
    comparisons: dict[str, Any],
) -> None:
    """Bind each caller disposition to one exact immutable pair comparison.

    The VLM record remains unreviewed and uncalibrated. This only proves that
    the caller's later disposition references its exact pair and project
    evidence; it does not prove that the identity claim is correct.
    """
    dispositions = {item.comparison_id: item for item in review.comparison_dispositions}
    if set(dispositions) != set(comparisons):
        raise ValueError("comparison disposition records are incomplete")
    links_by_id = {item.link_id: item for item in review.links}
    for comparison_id, disposition in dispositions.items():
        comparison = comparisons[comparison_id]
        if (
            comparison.comparison_id != comparison_id
            or comparison.project_id != review.project_id
            or comparison.project_manifest_id != review.project_manifest_id
            or comparison.project_revision != review.project_revision
            or comparison.context_id != review.context_id
            or comparison.story_graph_id != review.story_graph_id
            or comparison.analysis_fingerprint != review.analysis_fingerprint
            or comparison.source_mention_review_id != review.source_mention_review_id
            or comparison.source_mention_review_revision
            != review.source_mention_review_revision
            or comparison.run_state != "completed_unreviewed"
        ):
            raise ValueError("comparison disposition cites stale or incomplete evidence")

        left_key = _anchor_identity(comparison.left_anchor)
        right_key = _anchor_identity(comparison.right_anchor)
        matching_links = [
            link for link in review.links
            if link.entity_kind == comparison.relation_kind
            and left_key in {_anchor_identity(anchor) for anchor in link.anchors}
            and right_key in {_anchor_identity(anchor) for anchor in link.anchors}
        ]
        if disposition.decision == "accepted_as_caller_asserted":
            link = links_by_id.get(disposition.link_id or "")
            if (
                link is None
                or link.entity_kind != comparison.relation_kind
                or comparison_id not in link.source_comparison_ids
                or link not in matching_links
            ):
                raise ValueError("accepted comparison does not match its exact caller link")
        elif matching_links:
            raise ValueError("rejected or deferred comparison cannot remain linked")


def validate_project_story_link_review_source_mentions(
    review: ProjectStoryLinkReview,
    mention_review: ProjectStoryMentionReview | None,
) -> None:
    """Require active link evidence to match the current source-mention review.

    Empty link snapshots do not depend on mention dispositions. Non-empty
    snapshots and comparison dispositions must bind the exact current review.
    Historical link snapshots remain readable but cannot be consumed as current.
    """
    if project_story_link_review_is_stale(review, mention_review):
        raise ValueError("project story link review cites a stale source mention review")
    if not review.links and not review.relations and not review.comparison_dispositions:
        return
    if mention_review is None:
        return
    if (
        mention_review.project_id != review.project_id
        or mention_review.project_manifest_id != review.project_manifest_id
        or mention_review.project_revision != review.project_revision
        or mention_review.context_id != review.context_id
        or mention_review.story_graph_id != review.story_graph_id
        or mention_review.analysis_fingerprint != review.analysis_fingerprint
    ):
        raise ValueError("project story link review mention evidence is out of scope")

    rejected_mentions = [
        decision for decision in mention_review.decisions
        if decision.decision == "rejected"
    ]
    for link in review.links:
        for anchor in link.anchors:
            for decision in rejected_mentions:
                rejected_anchor = decision.anchor
                if decision.relation_kind != link.entity_kind:
                    continue
                exact_rejected_node = (
                    anchor.project_asset_id == rejected_anchor.project_asset_id
                    and anchor.story_graph_node_id
                    == rejected_anchor.story_graph_node_id
                )
                unbound_interval_overlap = (
                    anchor.story_graph_node_id is None
                    and anchor.project_asset_id == rejected_anchor.project_asset_id
                    and anchor.observation_id == rejected_anchor.observation_id
                    and anchor.source_content_hash.lower()
                    == rejected_anchor.source_content_hash.lower()
                    and anchor.timebase == rejected_anchor.timebase
                    and anchor.timebase_unit == rejected_anchor.timebase_unit
                    and anchor.source_start < rejected_anchor.source_end
                    and rejected_anchor.source_start < anchor.source_end
                )
                if exact_rejected_node or unbound_interval_overlap:
                    raise ValueError("project story link cites a rejected source mention")
    for relation in review.relations:
        for anchor in (relation.from_anchor, relation.to_anchor):
            for decision in rejected_mentions:
                rejected_anchor = decision.anchor
                if decision.relation_kind not in {
                    "person_identity", "event_identity", "place_identity"
                }:
                    continue
                exact_rejected_node = (
                    anchor.project_asset_id == rejected_anchor.project_asset_id
                    and anchor.story_graph_node_id
                    == rejected_anchor.story_graph_node_id
                )
                unbound_interval_overlap = (
                    anchor.story_graph_node_id is None
                    and anchor.project_asset_id == rejected_anchor.project_asset_id
                    and anchor.observation_id == rejected_anchor.observation_id
                    and anchor.source_content_hash.lower()
                    == rejected_anchor.source_content_hash.lower()
                    and anchor.timebase == rejected_anchor.timebase
                    and anchor.timebase_unit == rejected_anchor.timebase_unit
                    and anchor.source_start < rejected_anchor.source_end
                    and rejected_anchor.source_start < anchor.source_end
                )
                if exact_rejected_node or unbound_interval_overlap:
                    raise ValueError("project story relation cites a rejected source mention")


def project_story_link_review_is_stale(
    review: ProjectStoryLinkReview,
    mention_review: ProjectStoryMentionReview | None,
) -> bool:
    """Whether this non-empty relation snapshot cites an older mention review."""
    if not review.links and not review.relations and not review.comparison_dispositions:
        return False
    expected_identity = (
        (mention_review.review_id, mention_review.review_revision)
        if mention_review is not None else (None, 0)
    )
    return (
        review.source_mention_review_id,
        review.source_mention_review_revision,
    ) != expected_identity


def validate_project_story_link_review_for_plan(
    review: ProjectStoryLinkReview,
    manifest: FilmProjectManifest,
    context: FilmContextSnapshot,
    graph: ProjectStoryGraph,
    observations: list[FilmObservation],
    mention_review: ProjectStoryMentionReview | None = None,
) -> None:
    """Require a caller link snapshot to match the Plan's exact evidence set."""
    validate_project_story_link_review_mention_exclusivity(review)
    if (
        manifest.project_id != context.project_id
        or context.invalidated_at is not None
        or context.project_manifest_id != manifest.manifest_id
        or context.project_revision != manifest.revision
        or graph.project_id != manifest.project_id
        or graph.project_manifest_id != manifest.manifest_id
        or graph.project_revision != manifest.revision
        or graph.context_id != context.context_id
        or graph.analysis_fingerprint != context.analysis_fingerprint
        or graph.evidence_refs != context.evidence_refs
        or review.project_id != manifest.project_id
        or review.project_manifest_id != manifest.manifest_id
        or review.project_revision != manifest.revision
        or review.context_id != context.context_id
        or review.story_graph_id != graph.graph_id
        or review.analysis_fingerprint != context.analysis_fingerprint
        or review.review_revision < 1
    ):
        raise ValueError("project story link review is not bound to current plan inputs")
    expected_review_id = "project_story_links_" + short_hash(
        f"{graph.graph_id}|{review.review_revision}"
    )
    if review.review_id != expected_review_id:
        raise ValueError("project story link review id does not match its revision")
    validate_project_story_link_review_source_mentions(review, mention_review)

    by_id = {item.observation_id: item for item in observations}
    if len(by_id) != len(observations) or set(by_id) != set(context.evidence_refs):
        raise ValueError("project story link review requires exact current observations")
    assets = {item.asset_id: item for item in manifest.assets}
    graph_refs = set(graph.evidence_refs)
    for link in review.links:
        for anchor in link.anchors:
            observation = by_id.get(anchor.observation_id)
            asset = assets.get(anchor.project_asset_id)
            if observation is None or asset is None:
                raise ValueError("project story link review contains a missing source anchor")
            if (
                asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED
                or asset.source_identity_state != "locally_verified"
                or anchor.observation_id not in graph_refs
                or observation.project_id != manifest.project_id
                or observation.project_asset_id != asset.asset_id
                or (anchor.source_asset_id is not None
                    and observation.media_asset_id != anchor.source_asset_id)
                or observation.media_hash.lower() != anchor.source_content_hash.lower()
                or asset.source_content_hash is None
                or asset.source_content_hash.lower() != anchor.source_content_hash.lower()
                or observation.timebase != anchor.timebase
                or observation.timebase_unit != anchor.timebase_unit
                or anchor.source_start < observation.start_frame
                or anchor.source_end > observation.end_frame
                or anchor.source_start >= anchor.source_end
            ):
                raise ValueError("project story link review anchor differs from current evidence")
            _validate_anchor_story_graph_node(graph, link.entity_kind, anchor)
    for relation in review.relations:
        for anchor in (relation.from_anchor, relation.to_anchor):
            observation = by_id.get(anchor.observation_id)
            asset = assets.get(anchor.project_asset_id)
            if observation is None or asset is None:
                raise ValueError("project story relation contains a missing source anchor")
            if (
                asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED
                or asset.source_identity_state != "locally_verified"
                or anchor.observation_id not in graph_refs
                or observation.project_id != manifest.project_id
                or observation.project_asset_id != asset.asset_id
                or (anchor.source_asset_id is not None
                    and observation.media_asset_id != anchor.source_asset_id)
                or observation.media_hash.lower() != anchor.source_content_hash.lower()
                or asset.source_content_hash is None
                or asset.source_content_hash.lower() != anchor.source_content_hash.lower()
                or observation.timebase != anchor.timebase
                or observation.timebase_unit != anchor.timebase_unit
                or anchor.source_start < observation.start_frame
                or anchor.source_end > observation.end_frame
                or anchor.source_start >= anchor.source_end
            ):
                raise ValueError("project story relation anchor differs from current evidence")
            _validate_anchor_story_graph_node(graph, "semantic_relation", anchor)


def _project_anchor_overlaps_edit(
    anchor: ProjectStoryLinkAnchor,
    edit: EditItem,
) -> bool:
    """Match an anchor to the same manifest media interval in the EDL."""
    if (
        edit.project_asset_id is None
        or edit.source_observation_start is None
        or edit.source_observation_end is None
        or edit.source_timebase is None
        or edit.source_timebase_unit is None
    ):
        return False
    # Observation and source-asset IDs are producer-specific. A semantic
    # mention and a technical edit can describe the same manifest media with
    # different IDs, so bind only on the owning project asset, exact content
    # hash, timebase, and overlapping source interval.
    return (
        anchor.project_asset_id == edit.project_asset_id
        and anchor.source_content_hash.lower() == edit.source_media_hash.lower()
        and anchor.timebase == edit.source_timebase
        and anchor.timebase_unit == edit.source_timebase_unit
        and anchor.source_start < edit.out_frame
        and edit.in_frame < anchor.source_end
    )


def project_story_link_refs_for_edit(
    edit: EditItem,
    review: ProjectStoryLinkReview | None,
) -> list[str]:
    """Return only caller link IDs whose exact anchors overlap this source edit."""
    if review is None:
        return []
    refs = []
    for link in review.links:
        if any(_project_anchor_overlaps_edit(anchor, edit)
               for anchor in link.anchors):
            refs.append(link.link_id)
    return refs


def project_story_relation_refs_for_edit(
    edit: EditItem,
    review: ProjectStoryLinkReview | None,
) -> list[str]:
    """Return caller-asserted relation IDs whose exact endpoint overlaps an edit."""
    if review is None:
        return []
    refs = []
    for relation in review.relations:
        if any(_project_anchor_overlaps_edit(anchor, edit)
               for anchor in (relation.from_anchor, relation.to_anchor)):
            refs.append(relation.relation_id)
    return refs


def bind_project_story_link_review_to_plan(
    edl: EditorialDecisionList,
    plan: DirectorDecisionPlan,
    review: ProjectStoryLinkReview | None,
    manifest: FilmProjectManifest,
    context: FilmContextSnapshot,
    graph: ProjectStoryGraph,
    observations: list[FilmObservation],
    mention_review: ProjectStoryMentionReview | None = None,
) -> tuple[EditorialDecisionList, DirectorDecisionPlan]:
    """Bind reviewed identity metadata to source edits without inferring or ranking."""
    if review is not None:
        validate_project_story_link_review_for_plan(
            review, manifest, context, graph, observations, mention_review)

    updated_edl = EditorialDecisionList.model_validate({
        **edl.model_dump(mode="python"),
        "ordered_edits": [
            {
                **edit.model_dump(mode="python"),
                "project_story_link_refs": project_story_link_refs_for_edit(
                    edit, review),
                "project_story_relation_refs": project_story_relation_refs_for_edit(
                    edit, review),
            }
            for edit in edl.ordered_edits
        ],
    })
    plan_data = plan.model_dump(mode="python")
    plan_data["project_story_link_review_id"] = (
        review.review_id if review is not None else None)
    plan_data["project_story_link_review_revision"] = (
        review.review_revision if review is not None else None)
    if review is not None:
        plan_data["constraints"] = [
            *plan_data["constraints"],
            f"project_story_link_review={review.review_id}@{review.review_revision}:caller_asserted",
        ]
        if review.links:
            plan_data["open_questions"] = [
                *plan_data["open_questions"],
                "caller_asserted_project_links_not_independently_verified",
            ]
        if review.relations:
            plan_data["open_questions"] = [
                *plan_data["open_questions"],
                "caller_asserted_project_relations_not_independently_verified",
            ]
    return updated_edl, DirectorDecisionPlan.model_validate(plan_data)


def build_project_story_link_review(
    manifest: FilmProjectManifest,
    context: FilmContextSnapshot,
    graph: ProjectStoryGraph,
    observations: dict[str, FilmObservation],
    link_selections: list[dict[str, Any]],
    *,
    review_revision: int,
    relation_selections: list[dict[str, Any]] | None = None,
    comparison_dispositions: list[dict[str, Any]] | None = None,
    mention_review: ProjectStoryMentionReview | None = None,
) -> ProjectStoryLinkReview:
    """Resolve user-selected intervals to exact local source identities.

    `link_selections` is request data containing only link labels and
    observation-relative source intervals. Asset ids, hashes, and clocks are
    always derived from the current manifest and stored observations.
    """
    if manifest.project_id != context.project_id or context.project_id != graph.project_id:
        raise ValueError("link review inputs must belong to the same project")
    if (
        context.project_manifest_id != manifest.manifest_id
        or context.project_revision != manifest.revision
        or graph.project_manifest_id != manifest.manifest_id
        or graph.project_revision != manifest.revision
        or graph.context_id != context.context_id
        or graph.analysis_fingerprint != context.analysis_fingerprint
        or graph.evidence_refs != context.evidence_refs
        or graph.timeline_scope != "project_per_asset"
    ):
        raise ValueError("link review requires the current manifest, context, and graph")
    if context.invalidated_at is not None:
        raise ValueError("link review cannot use an invalidated context")

    assets = {item.asset_id: item for item in manifest.assets}
    evidence_refs = set(context.evidence_refs)
    if set(observations) - evidence_refs:
        raise ValueError("link review observations exceed the current project evidence")

    links: list[ProjectStoryEntityLink] = []
    for selection in link_selections:
        anchors: list[ProjectStoryLinkAnchor] = []
        for selected in selection["anchors"]:
            observation_id = selected["observation_id"]
            observation = observations.get(observation_id)
            if observation is None or observation_id not in evidence_refs:
                raise ValueError("link anchor must cite current project evidence")
            if observation.project_id != manifest.project_id:
                raise ValueError("link anchor observation belongs to another project")
            asset = assets.get(observation.project_asset_id or "")
            if asset is None:
                raise ValueError("link anchor observation does not belong to a current asset")
            if (
                asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED
                or asset.source_identity_state != "locally_verified"
                or asset.source_content_hash is None
            ):
                raise ValueError("link anchors require locally authorized, verified assets")
            if observation.media_hash.lower() != asset.source_content_hash.lower():
                raise ValueError("link anchor observation hash differs from its source asset")
            source_start = selected["source_start"]
            source_end = selected["source_end"]
            if (
                source_start < observation.start_frame
                or source_end > observation.end_frame
                or source_end <= source_start
            ):
                raise ValueError("link anchor interval must fit inside its observation")
            anchors.append(ProjectStoryLinkAnchor(
                project_asset_id=asset.asset_id,
                source_asset_id=observation.media_asset_id,
                story_graph_node_id=selected.get("story_graph_node_id"),
                observation_id=observation.observation_id,
                source_content_hash=asset.source_content_hash.lower(),
                source_start=source_start,
                source_end=source_end,
                timebase=observation.timebase,
                timebase_unit=observation.timebase_unit,
            ))
            _validate_anchor_story_graph_node(
                graph, selection["entity_kind"], anchors[-1])
        links.append(ProjectStoryEntityLink(
            link_id=selection["link_id"],
            entity_kind=selection["entity_kind"],
            display_label=selection["display_label"],
            anchors=anchors,
            source_comparison_ids=selection.get("source_comparison_ids", []),
        ))

    relations: list[ProjectStoryRelationClaim] = []
    for selection in relation_selections or []:
        anchors: list[ProjectStoryLinkAnchor] = []
        for endpoint_name in ("from_anchor", "to_anchor"):
            selected = selection[endpoint_name]
            observation_id = selected["observation_id"]
            observation = observations.get(observation_id)
            if observation is None or observation_id not in evidence_refs:
                raise ValueError("relation endpoint must cite current project evidence")
            if observation.project_id != manifest.project_id:
                raise ValueError("relation endpoint observation belongs to another project")
            asset = assets.get(observation.project_asset_id or "")
            if asset is None:
                raise ValueError("relation endpoint does not belong to a current asset")
            if (
                asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED
                or asset.source_identity_state != "locally_verified"
                or asset.source_content_hash is None
            ):
                raise ValueError("relation endpoints require locally authorized, verified assets")
            if observation.media_hash.lower() != asset.source_content_hash.lower():
                raise ValueError("relation endpoint hash differs from its source asset")
            source_start = selected["source_start"]
            source_end = selected["source_end"]
            if (
                source_start < observation.start_frame
                or source_end > observation.end_frame
                or source_end <= source_start
            ):
                raise ValueError("relation endpoint interval must fit inside its observation")
            anchor = ProjectStoryLinkAnchor(
                project_asset_id=asset.asset_id,
                source_asset_id=observation.media_asset_id,
                story_graph_node_id=selected["story_graph_node_id"],
                observation_id=observation.observation_id,
                source_content_hash=asset.source_content_hash.lower(),
                source_start=source_start,
                source_end=source_end,
                timebase=observation.timebase,
                timebase_unit=observation.timebase_unit,
            )
            _validate_anchor_story_graph_node(graph, "semantic_relation", anchor)
            anchors.append(anchor)
        relations.append(ProjectStoryRelationClaim(
            relation_id=selection["relation_id"],
            relation_type=selection["relation_type"],
            from_anchor=anchors[0],
            to_anchor=anchors[1],
            evidence_refs=[anchors[0].observation_id, anchors[1].observation_id],
            confirmation_state="caller_asserted",
            privacy_class=selection.get("privacy_class", "private"),
        ))

    dispositions = [
        ProjectStoryLinkComparisonDisposition.model_validate(item)
        for item in (comparison_dispositions or [])
    ]

    review_id = "project_story_links_" + short_hash(
        f"{graph.graph_id}|{review_revision}"
    )
    if mention_review is not None and (
        mention_review.project_id != manifest.project_id
        or mention_review.project_manifest_id != manifest.manifest_id
        or mention_review.project_revision != manifest.revision
        or mention_review.context_id != context.context_id
        or mention_review.story_graph_id != graph.graph_id
        or mention_review.analysis_fingerprint != context.analysis_fingerprint
    ):
        raise ValueError("source mention review is not bound to the current project graph")
    review = ProjectStoryLinkReview(
        review_id=review_id,
        project_manifest_id=manifest.manifest_id,
        project_revision=manifest.revision,
        context_id=context.context_id,
        story_graph_id=graph.graph_id,
        analysis_fingerprint=context.analysis_fingerprint,
        source_mention_review_id=(
            mention_review.review_id if mention_review is not None else None),
        source_mention_review_revision=(
            mention_review.review_revision if mention_review is not None else 0),
        review_revision=review_revision,
        review_scope=(
            "project_cross_asset_review" if relations
            else "project_cross_asset_identity_overlay"
        ),
        links=links,
        relations=relations,
        comparison_dispositions=dispositions,
        schema_version="1.5",
        project_id=manifest.project_id,
        created_at=int(time.time()),
        producer="director_brain_project_story_link_review_api",
        source_ref=graph.graph_id,
    )
    validate_project_story_link_review_source_mentions(review, mention_review)
    return review
