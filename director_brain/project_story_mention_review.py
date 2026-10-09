"""Build and validate caller review snapshots for source-local mentions."""
from __future__ import annotations

from typing import Any

from director_brain.models.film_context import FilmContextSnapshot
from director_brain.models.film_observation import ClaimKind, FilmObservation
from director_brain.models.project import FilmProjectManifest, RightsState
from director_brain.models.project_story_graph import ProjectStoryGraph
from director_brain.models.project_story_link_comparison import (
    ProjectStoryLinkEventEvidence,
)
from director_brain.models.project_story_mention_review import (
    ProjectStoryMentionDecision,
    ProjectStoryMentionReview,
)
from director_brain.models.story_graph import StoryNodeType


_PERSON_NODE = StoryNodeType.PERSON_MENTION
_EVENT_NODE = StoryNodeType.EVENT_MENTION


def validate_project_story_mention_review(
    review: ProjectStoryMentionReview,
    manifest: FilmProjectManifest,
    context: FilmContextSnapshot,
    graph: ProjectStoryGraph,
    observations: dict[str, FilmObservation],
) -> None:
    """Bind every decision to an eligible, exact source-local observation node."""
    if (
        review.project_id != manifest.project_id
        or context.project_id != manifest.project_id
        or graph.project_id != manifest.project_id
        or review.project_manifest_id != manifest.manifest_id
        or review.project_revision != manifest.revision
        or context.project_manifest_id != manifest.manifest_id
        or context.project_revision != manifest.revision
        or graph.project_manifest_id != manifest.manifest_id
        or graph.project_revision != manifest.revision
        or review.context_id != context.context_id
        or graph.context_id != context.context_id
        or review.story_graph_id != graph.graph_id
        or review.analysis_fingerprint != context.analysis_fingerprint
        or graph.analysis_fingerprint != context.analysis_fingerprint
        or context.invalidated_at is not None
        or graph.timeline_scope != "project_per_asset"
    ):
        raise ValueError("mention review is not bound to the current project evidence")

    assets = {item.asset_id: item for item in manifest.assets}
    graph_assets = {item.asset_id: item for item in graph.assets}
    for decision in review.decisions:
        anchor = decision.anchor
        asset = assets.get(anchor.project_asset_id)
        asset_graph = graph_assets.get(anchor.project_asset_id)
        observation = observations.get(anchor.observation_id)
        if (
            asset is None
            or asset_graph is None
            or asset_graph.story_graph is None
            or asset_graph.story_graph_state != "constructed"
            or asset_graph.source_content_hash is None
            or asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED
            or asset.source_identity_state != "locally_verified"
            or asset.source_content_hash is None
            or asset.source_content_hash.lower() != anchor.source_content_hash.lower()
            or anchor.observation_id not in graph.evidence_refs
            or observation is None
        ):
            raise ValueError("mention review target is not an eligible local source")

        node = next((
            item for item in asset_graph.story_graph.nodes
            if item.node_id == anchor.story_graph_node_id
        ), None)
        expected_node_type = (
            _PERSON_NODE if decision.relation_kind == "person_identity"
            else _EVENT_NODE
        )
        attributes = node.attributes if node is not None else {}
        source_anchor: Any = attributes.get("source_anchor")
        if not isinstance(source_anchor, dict):
            raise ValueError("mention review source node has no exact source anchor")
        if (
            node is None
            or node.node_type != expected_node_type
            or node.ref_id != anchor.observation_id
            or attributes.get("provider") != "ollama_qwen3_vl"
            or attributes.get("claim_kind") != ClaimKind.MODEL_OBSERVATION.value
            or attributes.get("review_state") not in {"auto_generated", "unreviewed"}
            or source_anchor.get("project_asset_id") != asset.asset_id
            or str(source_anchor.get("source_content_hash", "")).lower()
            != anchor.source_content_hash.lower()
            or asset_graph.source_content_hash.lower()
            != anchor.source_content_hash.lower()
            or source_anchor.get("source_start") != anchor.source_start
            or source_anchor.get("source_end") != anchor.source_end
            or source_anchor.get("timebase") != anchor.timebase
            or source_anchor.get("timebase_unit") != anchor.timebase_unit.value
            or observation.project_id != manifest.project_id
            or observation.project_asset_id != asset.asset_id
            or (anchor.source_asset_id is not None
                and observation.media_asset_id != anchor.source_asset_id)
            or observation.claim_kind != ClaimKind.MODEL_OBSERVATION
            or observation.observation_type != "vlm_semantic"
            or observation.provider != "ollama_qwen3_vl"
            or observation.review_state not in {"auto_generated", "unreviewed"}
            or observation.media_hash.lower() != anchor.source_content_hash.lower()
            or observation.timebase != anchor.timebase
            or observation.timebase_unit != anchor.timebase_unit
            or observation.start_frame != anchor.source_start
            or observation.end_frame != anchor.source_end
        ):
            raise ValueError("mention review anchor differs from stored graph evidence")

        if decision.relation_kind == "person_identity":
            source_text = attributes.get("description")
            if not isinstance(source_text, str) or not source_text.strip() or len(source_text) > 240:
                raise ValueError("source person mention is malformed")
        else:
            try:
                ProjectStoryLinkEventEvidence.model_validate(
                    attributes.get("semantic_fields"))
            except (TypeError, ValueError) as exc:
                raise ValueError("source event mention is malformed") from exc


def build_project_story_mention_review(
    manifest: FilmProjectManifest,
    context: FilmContextSnapshot,
    graph: ProjectStoryGraph,
    observations: dict[str, FilmObservation],
    decisions: list[dict[str, Any]],
    *,
    review_revision: int,
    created_at: int,
) -> ProjectStoryMentionReview:
    review = ProjectStoryMentionReview(
        project_id=manifest.project_id,
        review_id=(f"project_mention_review_{manifest.project_id}_"
                   f"{graph.graph_id}"),
        project_manifest_id=manifest.manifest_id,
        project_revision=manifest.revision,
        context_id=context.context_id,
        story_graph_id=graph.graph_id,
        analysis_fingerprint=context.analysis_fingerprint,
        review_revision=review_revision,
        decisions=[ProjectStoryMentionDecision.model_validate(item)
                   for item in decisions],
        created_at=created_at,
        producer="director_brain_project_story_mention_review_api",
        source_ref=graph.graph_id,
    )
    validate_project_story_mention_review(
        review, manifest, context, graph, observations)
    return review
