"""Build a revision-bound collection of source-local project StoryGraphs."""
from __future__ import annotations

from director_brain.models.film_context import (
    AssetAnalysisState,
    FilmContextSnapshot,
)
from director_brain.models.film_observation import FilmObservation
from director_brain.models.project import FilmProjectManifest
from director_brain.models.project_story_graph import (
    ProjectAssetStoryGraph,
    ProjectStoryGraph,
    ProjectStoryGraphRelationGroup,
    ProjectStoryGraphRelationMember,
    ProjectStoryGraphView,
)
from director_brain.models.project_story_mention_review import ProjectStoryMentionReview
from director_brain.models.project_story_link_review import ProjectStoryLinkReview
from director_brain.project_story_link_review import (
    validate_project_story_link_review_source_mentions,
)
from director_brain.story_graph_builder import build_asset_story_graph
from director_brain.utils import short_hash


def project_story_graph_id(context: FilmContextSnapshot) -> str:
    """Return an identity bound to one exact context evidence fingerprint."""
    return "project_graph_" + short_hash(
        f"{context.context_id}|{context.analysis_fingerprint}"
    )


def project_story_graph_view_id(
    graph: ProjectStoryGraph,
    link_review: ProjectStoryLinkReview | None,
    mention_review: ProjectStoryMentionReview | None,
) -> str:
    """Return a stable identity for one graph and both active review snapshots."""
    link_identity = (
        f"{link_review.review_id}@{link_review.review_revision}"
        if link_review is not None else "no-link-review"
    )
    mention_identity = (
        f"{mention_review.review_id}@{mention_review.review_revision}"
        if mention_review is not None else "no-mention-review"
    )
    return "project_graph_view_" + short_hash(
        f"{graph.graph_id}|{link_identity}|{mention_identity}"
    )


def build_project_story_graph_view(
    graph: ProjectStoryGraph,
    link_review: ProjectStoryLinkReview | None,
    mention_review: ProjectStoryMentionReview | None = None,
) -> ProjectStoryGraphView:
    """Compose the source graph and only current caller-asserted link evidence."""
    if mention_review is not None and (
        mention_review.project_id != graph.project_id
        or mention_review.project_manifest_id != graph.project_manifest_id
        or mention_review.project_revision != graph.project_revision
        or mention_review.context_id != graph.context_id
        or mention_review.story_graph_id != graph.graph_id
        or mention_review.analysis_fingerprint != graph.analysis_fingerprint
    ):
        raise ValueError("source mention review does not match the project StoryGraph")
    link_review_is_current = False
    if link_review is not None:
        link_review_is_current = (
            not link_review.links
            and not link_review.relations
            and not link_review.comparison_dispositions
        ) or (
            link_review.source_mention_review_id
            == (mention_review.review_id if mention_review is not None else None)
            and link_review.source_mention_review_revision
            == (mention_review.review_revision if mention_review is not None else 0)
        )
        if link_review_is_current:
            validate_project_story_link_review_source_mentions(
                link_review, mention_review)
    active_links = (
        link_review.links if link_review_is_current and link_review is not None else [])
    active_relation_edges = (
        link_review.relations
        if link_review_is_current and link_review is not None else [])
    link_count = len(active_links)
    relations = [
        ProjectStoryGraphRelationGroup(
            relation_id=link.link_id,
            relation_kind=link.entity_kind,
            display_label=link.display_label,
            members=[
                ProjectStoryGraphRelationMember(
                    anchor=anchor,
                    node_binding_state=(
                        "exact_mention_node"
                        if anchor.story_graph_node_id is not None
                        else "observation_interval_only"
                    ),
                )
                for anchor in link.anchors
            ],
            source_review_id=link_review.review_id,
            source_review_revision=link_review.review_revision,
            source_link_id=link.link_id,
            source_comparison_ids=list(link.source_comparison_ids),
        )
        for link in active_links
    ]
    snapshot_state = (
        "absent" if link_review is None
        else "caller_asserted" if link_review_is_current
        else "stale"
    )
    relation_state = (
        "needs_reconfirmation" if link_review is not None and not link_review_is_current
        else "caller_asserted" if link_count
        else "none"
    )
    return ProjectStoryGraphView(
        view_id=project_story_graph_view_id(graph, link_review, mention_review),
        source_graph=graph,
        cross_asset_link_review=link_review,
        source_mention_review_id=(
            mention_review.review_id if mention_review is not None else None),
        source_mention_review_revision=(
            mention_review.review_revision if mention_review is not None else 0),
        cross_asset_relations=relations,
        cross_asset_relation_edges=active_relation_edges,
        link_snapshot_state=snapshot_state,
        cross_asset_relation_state=relation_state,
        cross_asset_link_count=link_count,
        cross_asset_relation_edge_state=(
            "needs_reconfirmation"
            if link_review is not None
            and link_review.relations
            and not link_review_is_current
            else "caller_asserted" if active_relation_edges else "none"
        ),
        cross_asset_relation_edge_count=len(active_relation_edges),
    )


def build_project_story_graph(
    manifest: FilmProjectManifest,
    context: FilmContextSnapshot,
    observations: list[FilmObservation],
) -> ProjectStoryGraph:
    """Aggregate independent asset-local structures without cross-asset links.

    The manifest establishes only a caller-declared project boundary. The
    context snapshot selects the exact evidence set. This function rejects
    stale, missing, duplicate, or cross-asset evidence rather than filling gaps.
    """
    if manifest.project_id != context.project_id:
        raise ValueError("manifest and context project_id values must match")
    if (
        context.project_manifest_id != manifest.manifest_id
        or context.project_revision != manifest.revision
    ):
        raise ValueError("context does not belong to the supplied manifest revision")
    if context.invalidated_at is not None:
        raise ValueError("invalidated project context cannot build a StoryGraph")
    if context.timeline_scope != "project_per_asset":
        raise ValueError("project StoryGraph requires per-asset timeline scope")

    ordered_assets = sorted(manifest.assets, key=lambda item: item.order)
    asset_ids = [asset.asset_id for asset in ordered_assets]
    if context.asset_refs != asset_ids:
        raise ValueError("context asset refs do not match manifest order and membership")
    expected_hashes = [
        asset.source_content_hash.lower()
        if asset.source_content_hash is not None else None
        for asset in ordered_assets
    ]
    if context.source_content_hashes != expected_hashes:
        raise ValueError("context source hashes do not match the manifest")
    if len(context.asset_coverage) != len(ordered_assets):
        raise ValueError("context coverage must include every manifest asset")
    if [item.asset_id for item in context.asset_coverage] != asset_ids:
        raise ValueError("context coverage order does not match manifest assets")
    if [item.order for item in context.asset_coverage] != [
        asset.order for asset in ordered_assets
    ]:
        raise ValueError("context coverage order values do not match the manifest")

    refs = [ref for coverage in context.asset_coverage for ref in coverage.evidence_refs]
    if len(refs) != len(set(refs)) or len(context.evidence_refs) != len(set(context.evidence_refs)):
        raise ValueError("project context contains duplicate evidence refs")
    if refs != context.evidence_refs:
        raise ValueError("project context evidence refs do not match its asset coverage")

    observations_by_id: dict[str, FilmObservation] = {}
    for observation in observations:
        if observation.observation_id in observations_by_id:
            raise ValueError("duplicate observation record supplied")
        observations_by_id[observation.observation_id] = observation
    if set(observations_by_id) != set(context.evidence_refs):
        raise ValueError("supplied observations must exactly match context evidence refs")

    graph_assets: list[ProjectAssetStoryGraph] = []
    for asset, coverage in zip(ordered_assets, context.asset_coverage, strict=True):
        expected_hash = (
            asset.source_content_hash.lower()
            if asset.source_content_hash is not None else None
        )
        if coverage.source_content_hash != expected_hash:
            raise ValueError(f"asset {asset.asset_id} source hash differs from context")
        if (
            coverage.rights_state != asset.rights.state.value
            or coverage.rights_evidence_state != asset.rights.evidence_state
        ):
            raise ValueError(f"asset {asset.asset_id} rights state differs from manifest")
        asset_observations = [
            observations_by_id[ref] for ref in coverage.evidence_refs
        ]
        for observation in asset_observations:
            if observation.project_id != manifest.project_id:
                raise ValueError("observation belongs to a different project")
            if observation.project_asset_id != asset.asset_id:
                raise ValueError("observation belongs to a different project asset")
            if expected_hash is None or observation.media_hash.lower() != expected_hash:
                raise ValueError("observation source hash does not match manifest asset")
        observation_timebases = sorted({
            (item.timebase, item.timebase_unit)
            for item in asset_observations
        }, key=lambda value: (value[0], value[1].value))
        coverage_timebases = sorted({
            (item.value, item.unit)
            for item in coverage.observation_timebases
        }, key=lambda value: (value[0], value[1].value))
        if observation_timebases != coverage_timebases:
            raise ValueError(f"asset {asset.asset_id} timebase evidence differs from context")

        graph = None
        if coverage.analysis_state == AssetAnalysisState.NOT_AUTHORIZED:
            graph_state = "not_authorized"
        elif coverage.analysis_state in (
            AssetAnalysisState.FAILED,
            AssetAnalysisState.SOURCE_UNAVAILABLE,
            AssetAnalysisState.SOURCE_CHANGED,
        ):
            graph_state = "analysis_incomplete"
        else:
            temporal_observations = [
                item for item in asset_observations
                if item.observation_type == "deterministic_technical"
            ]
            if not temporal_observations:
                graph_state = "no_deterministic_technical_observations"
            else:
                graph = build_asset_story_graph(
                    manifest.project_id,
                    asset.asset_id,
                    asset_observations,
                )
                graph_state = "constructed"

        graph_assets.append(ProjectAssetStoryGraph(
            asset_id=asset.asset_id,
            order=asset.order,
            source_content_hash=expected_hash,
            analysis_state=coverage.analysis_state,
            story_graph_state=graph_state,
            evidence_refs=list(coverage.evidence_refs),
            story_graph=graph,
        ))

    return ProjectStoryGraph(
        graph_id=project_story_graph_id(context),
        project_manifest_id=manifest.manifest_id,
        project_revision=manifest.revision,
        context_id=context.context_id,
        analysis_fingerprint=context.analysis_fingerprint,
        boundary_state="declared_unverified",
        timeline_scope="project_per_asset",
        relation_scope="asset_local_structural_only",
        cross_asset_relations_state="not_attempted",
        evidence_refs=list(context.evidence_refs),
        assets=graph_assets,
        schema_version="1.0",
        project_id=manifest.project_id,
        created_at=context.created_at,
        producer="project_story_graph",
        source_ref=manifest.manifest_id,
    )
