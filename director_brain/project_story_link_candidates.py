"""Deterministic discovery of exact, eligible cross-asset mention pairs."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from director_brain.models.film_observation import TimebaseUnit
from director_brain.models.project import (
    FilmProjectManifest,
    ProjectAsset,
    ProjectAnalysisProfile,
    RightsState,
)
from director_brain.models.project_story_graph import ProjectStoryGraph
from director_brain.models.project_story_graph import ProjectAssetStoryGraph
from director_brain.models.story_graph import StoryNode
from director_brain.models.project_story_link_candidates import (
    ProjectStoryLinkCandidate,
    ProjectStoryLinkCandidateAssetCoverage,
    ProjectStoryLinkCandidateCallerDisposition,
    ProjectStoryLinkCandidateComparisonAttempt,
    ProjectStoryLinkCandidateComparisonProgress,
    ProjectStoryLinkCandidatePage,
    ProjectStoryLinkComparisonRequestData,
)
from director_brain.models.project_story_link_comparison import (
    ProjectStoryLinkComparison,
    ProjectStoryLinkEventEvidence,
)
from director_brain.models.project_story_link_review import (
    ProjectStoryLinkAnchor,
    ProjectStoryLinkReview,
)
from director_brain.models.project_story_mention_review import ProjectStoryMentionReview
from director_brain.utils import project_story_link_comparison_id


_ALGORITHM_VERSION = "exhaustive_cross_asset_pairs_v1"
_LOCAL_PROVIDER = "ollama_qwen3_vl"
_OBSERVATION_CLAIM = "model_observation"
_SUPPORTED_TIMEBASE = 1_000_000


class CandidateAnchorNotEligibleError(ValueError):
    """The requested anchor is not an eligible current source mention."""


class CandidateScopeNotEligibleError(ValueError):
    """The requested counterpart asset scope is invalid for this anchor."""


@dataclass(frozen=True)
class _Mention:
    node_id: str
    observation_id: str
    anchor: ProjectStoryLinkAnchor
    mention_index: int = 0
    person_description: str | None = None
    event_evidence: ProjectStoryLinkEventEvidence | None = None


def _eligible_pair_count(
    left_mentions: list[_Mention], right_mentions: list[_Mention]
) -> int:
    """Count cross-asset pairs, excluding repeated observation identities."""
    if not left_mentions or not right_mentions:
        return 0
    right_counts = Counter(item.observation_id for item in right_mentions)
    duplicate_pairs = sum(
        right_counts[item.observation_id] for item in left_mentions)
    return len(left_mentions) * len(right_mentions) - duplicate_pairs


def _eligible_pair_page(
    left_mentions: list[_Mention],
    right_mentions: list[_Mention],
    *,
    offset: int,
    limit: int,
) -> list[tuple[_Mention, _Mention]]:
    """Page the valid Cartesian pairs without materializing the full product."""
    right_counts = Counter(item.observation_id for item in right_mentions)
    page: list[tuple[_Mention, _Mention]] = []
    remaining_offset = offset
    for left_mention in left_mentions:
        row_count = len(right_mentions) - right_counts[left_mention.observation_id]
        if remaining_offset >= row_count:
            remaining_offset -= row_count
            continue
        for right_mention in right_mentions:
            if left_mention.observation_id == right_mention.observation_id:
                continue
            if remaining_offset:
                remaining_offset -= 1
                continue
            page.append((left_mention, right_mention))
            if len(page) >= limit:
                return page
    return page


def _stable_id(prefix: str, *parts: str) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _mention_fingerprint(mention: _Mention) -> str:
    payload = {
        "node_id": mention.node_id,
        "observation_id": mention.observation_id,
        "anchor": mention.anchor.model_dump(mode="json"),
        "person_description": mention.person_description,
        "event_evidence": (
            mention.event_evidence.model_dump(mode="json", exclude_none=True)
            if mention.event_evidence else None
        ),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _candidate_id_from_mentions(
    graph_id: str,
    relation_kind: str,
    left: _Mention,
    right: _Mention,
    source_mention_review_id: str | None = None,
    source_mention_review_revision: int = 0,
) -> str:
    return _stable_id(
        "link_candidate_",
        _ALGORITHM_VERSION,
        graph_id,
        relation_kind,
        source_mention_review_id or "no-mention-review",
        str(source_mention_review_revision),
        _mention_fingerprint(left),
        _mention_fingerprint(right),
    )


def project_story_link_candidate_id_from_pair(
    graph_id: str,
    relation_kind: Literal["person_identity", "event_identity"],
    left_anchor: ProjectStoryLinkAnchor,
    right_anchor: ProjectStoryLinkAnchor,
    *,
    left_person_description: str | None,
    right_person_description: str | None,
    left_event_evidence: ProjectStoryLinkEventEvidence | None,
    right_event_evidence: ProjectStoryLinkEventEvidence | None,
    source_mention_review_id: str | None = None,
    source_mention_review_revision: int = 0,
) -> str:
    """Recompute one candidate ID from exact validated graph mention data."""
    if not left_anchor.story_graph_node_id or not right_anchor.story_graph_node_id:
        raise ValueError("candidate identity requires exact mention-node bindings")

    def candidate_mention(
        anchor: ProjectStoryLinkAnchor,
        person_description: str | None,
        event_evidence: ProjectStoryLinkEventEvidence | None,
    ) -> _Mention:
        # Comparison anchors add source_asset_id from the saved observation;
        # candidate fingerprints intentionally bind the project asset and
        # source interval without that later-resolved field.
        candidate_anchor = anchor.model_copy(update={"source_asset_id": None})
        return _Mention(
            node_id=anchor.story_graph_node_id or "",
            observation_id=anchor.observation_id,
            anchor=candidate_anchor,
            person_description=person_description,
            event_evidence=event_evidence,
        )

    return _candidate_id_from_mentions(
        graph_id,
        relation_kind,
        candidate_mention(left_anchor, left_person_description, left_event_evidence),
        candidate_mention(right_anchor, right_person_description, right_event_evidence),
        source_mention_review_id,
        source_mention_review_revision,
    )


def _candidate_from_mentions(
    graph: ProjectStoryGraph,
    relation_kind: Literal["person_identity", "event_identity"],
    left_mention: _Mention,
    right_mention: _Mention,
    mention_review: ProjectStoryMentionReview | None,
) -> ProjectStoryLinkCandidate:
    candidate_id = _candidate_id_from_mentions(
        graph.graph_id,
        relation_kind,
        left_mention,
        right_mention,
        mention_review.review_id if mention_review is not None else None,
        mention_review.review_revision if mention_review is not None else 0,
    )
    request = ProjectStoryLinkComparisonRequestData(
        manifest_revision=graph.project_revision,
        story_graph_id=graph.graph_id,
        idempotency_key=candidate_id,
        candidate_id=candidate_id,
        attempt_number=1,
        relation_kind=relation_kind,
        left_observation_id=left_mention.observation_id,
        right_observation_id=right_mention.observation_id,
        left_story_graph_node_id=left_mention.node_id,
        right_story_graph_node_id=right_mention.node_id,
        left_person_description=left_mention.person_description,
        right_person_description=right_mention.person_description,
        source_mention_review_id=(
            mention_review.review_id if mention_review is not None else None),
        source_mention_review_revision=(
            mention_review.review_revision if mention_review is not None else 0),
    )
    return ProjectStoryLinkCandidate(
        candidate_id=candidate_id,
        relation_kind=relation_kind,
        left_anchor=left_mention.anchor,
        right_anchor=right_mention.anchor,
        left_person_description=left_mention.person_description,
        right_person_description=right_mention.person_description,
        left_event_evidence=left_mention.event_evidence,
        right_event_evidence=right_mention.event_evidence,
        comparison_request=request,
    )


def _coverage_state(asset: ProjectAsset, asset_graph: ProjectAssetStoryGraph) -> str:
    if asset.rights.state != RightsState.LOCAL_PROCESSING_ALLOWED:
        return "not_locally_authorized"
    if (
        asset.source_identity_state != "locally_verified"
        or asset.source_content_hash is None
    ):
        return "source_identity_unverified"
    if asset_graph.story_graph_state != "constructed" or asset_graph.story_graph is None:
        return "analysis_incomplete"
    return "eligible_under_manifest_declaration"


def _mention_from_node(
    *,
    asset: ProjectAsset,
    node: StoryNode,
    relation_kind: str,
) -> _Mention | None:
    attributes = node.attributes
    if (
        attributes.get("provider") != _LOCAL_PROVIDER
        or attributes.get("claim_kind") != _OBSERVATION_CLAIM
        or attributes.get("review_state") not in {"auto_generated", "unreviewed"}
    ):
        return None
    anchor_data = attributes.get("source_anchor")
    if not isinstance(anchor_data, dict):
        return None
    if (
        anchor_data.get("project_asset_id") != asset.asset_id
        or str(anchor_data.get("source_content_hash", "")).lower()
        != str(asset.source_content_hash or "").lower()
        or anchor_data.get("timebase") != _SUPPORTED_TIMEBASE
        or anchor_data.get("timebase_unit") != TimebaseUnit.MICROSECONDS.value
    ):
        return None
    try:
        anchor = ProjectStoryLinkAnchor(
            project_asset_id=asset.asset_id,
            story_graph_node_id=node.node_id,
            observation_id=node.ref_id,
            source_content_hash=anchor_data["source_content_hash"].lower(),
            source_start=anchor_data["source_start"],
            source_end=anchor_data["source_end"],
            timebase=anchor_data["timebase"],
            timebase_unit=anchor_data["timebase_unit"],
        )
    except (KeyError, TypeError, ValueError):
        return None

    if relation_kind == "person_identity":
        description = attributes.get("description")
        if not isinstance(description, str) or not description.strip():
            return None
        if len(description) > 240:
            return None
        mention_index = attributes.get("mention_index", 0)
        if not isinstance(mention_index, int) or isinstance(mention_index, bool):
            return None
        return _Mention(
            node_id=node.node_id,
            observation_id=node.ref_id,
            anchor=anchor,
            mention_index=mention_index,
            person_description=description,
        )

    fields = attributes.get("semantic_fields")
    if not isinstance(fields, dict):
        return None
    try:
        event_evidence = ProjectStoryLinkEventEvidence.model_validate(fields)
    except (TypeError, ValueError):
        return None
    return _Mention(
        node_id=node.node_id,
        observation_id=node.ref_id,
        anchor=anchor,
        mention_index=0,
        event_evidence=event_evidence,
    )


def _eligible_mentions(
    manifest: FilmProjectManifest,
    graph: ProjectStoryGraph,
    relation_kind: Literal["person_identity", "event_identity"],
    mention_review: ProjectStoryMentionReview | None,
) -> tuple[list[list[_Mention]], list[ProjectStoryLinkCandidateAssetCoverage]]:
    graph_assets = {item.asset_id: item for item in graph.assets}
    assets_by_id = {item.asset_id: item for item in manifest.assets}
    node_type = "person_mention" if relation_kind == "person_identity" else "event_mention"
    decisions = {
        (item.relation_kind, item.anchor.project_asset_id,
         item.anchor.story_graph_node_id): item
        for item in (mention_review.decisions if mention_review is not None else [])
    }
    asset_groups: list[list[_Mention]] = []
    coverage: list[ProjectStoryLinkCandidateAssetCoverage] = []

    for asset in sorted(manifest.assets, key=lambda item: item.order):
        asset_graph = graph_assets.get(asset.asset_id)
        if asset_graph is None:
            state = "analysis_incomplete"
            mentions: list[_Mention] = []
        else:
            state = _coverage_state(asset, asset_graph)
            mentions = []
            if (
                state == "eligible_under_manifest_declaration"
                and asset_graph.story_graph is not None
            ):
                mentions = []
                for node in asset_graph.story_graph.nodes:
                    if node.node_type.value != node_type:
                        continue
                    mention = _mention_from_node(
                        asset=asset, node=node, relation_kind=relation_kind)
                    if mention is None:
                        continue
                    decision = decisions.get((
                        relation_kind, asset.asset_id, mention.node_id))
                    if decision is not None:
                        source_anchor = decision.anchor.model_copy(
                            update={"source_asset_id": None})
                        if source_anchor != mention.anchor:
                            raise ValueError(
                                "source mention review anchor differs from the current graph")
                        if decision.decision == "rejected":
                            continue
                        if decision.decision == "corrected_by_caller":
                            mention = _Mention(
                                node_id=mention.node_id,
                                observation_id=mention.observation_id,
                                anchor=mention.anchor,
                                mention_index=mention.mention_index,
                                person_description=(
                                    decision.corrected_person_description
                                    if relation_kind == "person_identity"
                                    else None
                                ),
                                event_evidence=(
                                    decision.corrected_event_evidence
                                    if relation_kind == "event_identity"
                                    else None
                                ),
                            )
                    mentions.append(mention)
                mentions.sort(key=lambda item: (
                    item.anchor.source_start,
                    item.anchor.source_end,
                    item.observation_id,
                    item.mention_index,
                    item.node_id,
                ))
                unique_mentions: list[_Mention] = []
                seen_node_ids: set[str] = set()
                for mention in mentions:
                    if mention.node_id not in seen_node_ids:
                        seen_node_ids.add(mention.node_id)
                        unique_mentions.append(mention)
                mentions = unique_mentions
                if not mentions:
                    state = "no_eligible_mentions"
        coverage.append(ProjectStoryLinkCandidateAssetCoverage(
            asset_id=asset.asset_id,
            order=asset.order,
            state=state,
            rights_state=asset.rights.state.value,
            rights_evidence_state=asset.rights.evidence_state,
            source_identity_state=asset.source_identity_state,
            eligible_mention_count=len(mentions),
        ))
        # Keep one group per manifest asset so anchor/counterpart and explicit
        # asset-pair indexes continue to address the same positions when an
        # earlier asset is unavailable or has no eligible mentions.
        asset_groups.append(mentions)

    # A malformed graph must not silently contribute assets absent from the manifest.
    if (
        set(graph_assets) != set(assets_by_id)
        or [(item.asset_id, item.order) for item in graph.assets]
        != [(item.asset_id, item.order)
            for item in sorted(manifest.assets, key=lambda item: item.order)]
    ):
        raise ValueError("StoryGraph asset membership differs from the current manifest")
    return asset_groups, coverage


def resolve_project_story_link_mention_anchor(
    manifest: FilmProjectManifest,
    graph: ProjectStoryGraph,
    *,
    relation_kind: Literal["person_identity", "event_identity"],
    story_graph_node_id: str,
    mention_review: ProjectStoryMentionReview | None = None,
) -> ProjectStoryLinkAnchor:
    """Resolve one current, eligible mention to its exact source interval.

    This is shared by candidate enumeration and the read-only source-preview
    route so both surfaces enforce the same graph, rights, provider, and review
    rules. It does not create or infer a cross-asset relationship.
    """
    asset_groups, _coverage = _eligible_mentions(
        manifest, graph, relation_kind, mention_review)
    matches = [
        mention.anchor
        for group in asset_groups
        for mention in group
        if mention.node_id == story_graph_node_id
    ]
    if len(matches) != 1:
        raise CandidateAnchorNotEligibleError(
            "selected anchor is not one eligible mention in the current graph")
    return matches[0]


def resolve_project_story_link_candidate_from_mentions(
    manifest: FilmProjectManifest,
    graph: ProjectStoryGraph,
    *,
    relation_kind: Literal["person_identity", "event_identity"],
    left_story_graph_node_id: str,
    right_story_graph_node_id: str,
    mention_review: ProjectStoryMentionReview | None = None,
) -> ProjectStoryLinkCandidate:
    """Rebuild one exact current cross-asset candidate from its two mentions.

    This is used by read-only reviewer surfaces. It resolves both mentions
    through the same eligibility and caller-review rules as candidate paging,
    then preserves manifest asset order for the candidate identity.
    """
    if left_story_graph_node_id == right_story_graph_node_id:
        raise CandidateScopeNotEligibleError(
            "candidate preview requires two different mention nodes")
    asset_groups, _coverage = _eligible_mentions(
        manifest, graph, relation_kind, mention_review)
    matches_by_node = {
        node_id: [
            (asset_index, mention)
            for asset_index, group in enumerate(asset_groups)
            for mention in group
            if mention.node_id == node_id
        ]
        for node_id in (left_story_graph_node_id, right_story_graph_node_id)
    }
    left_matches = matches_by_node[left_story_graph_node_id]
    right_matches = matches_by_node[right_story_graph_node_id]
    if len(left_matches) != 1 or len(right_matches) != 1:
        raise CandidateAnchorNotEligibleError(
            "candidate preview mentions are not currently eligible")
    left_index, left_mention = left_matches[0]
    right_index, right_mention = right_matches[0]
    if left_index == right_index:
        raise CandidateScopeNotEligibleError(
            "candidate preview mentions must belong to different assets")
    if left_index > right_index:
        left_mention, right_mention = right_mention, left_mention
    return _candidate_from_mentions(
        graph, relation_kind, left_mention, right_mention, mention_review)


def build_project_story_link_candidate_page(
    manifest: FilmProjectManifest,
    graph: ProjectStoryGraph,
    *,
    relation_kind: Literal["person_identity", "event_identity"],
    limit: int,
    offset: int,
    mention_review: ProjectStoryMentionReview | None = None,
    selected_anchor_story_graph_node_id: str | None = None,
    selected_counterpart_project_asset_id: str | None = None,
    selected_project_asset_pair: tuple[str, str] | None = None,
) -> ProjectStoryLinkCandidatePage:
    """Enumerate eligible pairs in a caller-selected scope without scoring.

    Pairs follow manifest asset order and per-asset source-time order. The
    algorithm skips directly to the requested page and does not materialize the
    complete Cartesian product. A caller may select a mention anchor, optionally
    narrow it to one counterpart, or select a complete asset pair; these scopes
    are explicit and mutually exclusive.
    """
    if manifest.analysis_profile != ProjectAnalysisProfile.LOCAL_VLM_SHADOW_V1:
        raise ValueError("candidate enumeration requires local_vlm_shadow_v1")
    if (
        graph.project_id != manifest.project_id
        or graph.project_manifest_id != manifest.manifest_id
        or graph.project_revision != manifest.revision
    ):
        raise ValueError("candidate enumeration requires the current manifest graph")
    if limit < 1 or limit > 200 or offset < 0:
        raise ValueError("candidate page limit/offset is out of range")

    if mention_review is not None and (
        mention_review.project_id != manifest.project_id
        or mention_review.project_manifest_id != manifest.manifest_id
        or mention_review.project_revision != manifest.revision
        or mention_review.story_graph_id != graph.graph_id
        or mention_review.context_id != graph.context_id
        or mention_review.analysis_fingerprint != graph.analysis_fingerprint
    ):
        raise ValueError("candidate mention review is stale or inconsistent")
    asset_groups, coverage = _eligible_mentions(
        manifest, graph, relation_kind, mention_review)
    eligible_mention_count = sum(len(group) for group in asset_groups)
    selected_anchor: tuple[int, _Mention] | None = None
    if selected_anchor_story_graph_node_id is not None:
        matches = [
            (group_index, mention)
            for group_index, group in enumerate(asset_groups)
            for mention in group
            if mention.node_id == selected_anchor_story_graph_node_id
        ]
        if len(matches) != 1:
            raise CandidateAnchorNotEligibleError(
                "selected anchor is not one eligible mention in the current graph")
        selected_anchor = matches[0]

    counterpart_group_index: int | None = None
    if selected_counterpart_project_asset_id is not None:
        if selected_anchor is None:
            raise CandidateScopeNotEligibleError(
                "counterpart asset selection requires a selected mention anchor")
        ordered_assets = sorted(manifest.assets, key=lambda item: item.order)
        counterpart_indexes = [
            index for index, asset in enumerate(ordered_assets)
            if asset.asset_id == selected_counterpart_project_asset_id
        ]
        if len(counterpart_indexes) != 1:
            raise CandidateScopeNotEligibleError(
                "counterpart asset is not declared in the current manifest")
        counterpart_group_index = counterpart_indexes[0]
        if counterpart_group_index == selected_anchor[0]:
            raise CandidateScopeNotEligibleError(
                "counterpart asset must differ from the selected mention asset")

    selected_asset_pair_indexes: tuple[int, int] | None = None
    canonical_asset_pair: tuple[str, str] | None = None
    if selected_project_asset_pair is not None:
        if (
            len(selected_project_asset_pair) != 2
            or selected_project_asset_pair[0] == selected_project_asset_pair[1]
        ):
            raise CandidateScopeNotEligibleError(
                "selected project asset pair must contain two different assets")
        if selected_anchor is not None or selected_counterpart_project_asset_id is not None:
            raise CandidateScopeNotEligibleError(
                "asset-pair scope cannot be combined with mention-anchor scope")
        ordered_assets = sorted(manifest.assets, key=lambda item: item.order)
        asset_indexes = {
            asset.asset_id: index for index, asset in enumerate(ordered_assets)
        }
        requested_ids = set(selected_project_asset_pair)
        if len(requested_ids) != 2 or not requested_ids.issubset(asset_indexes):
            raise CandidateScopeNotEligibleError(
                "selected project asset pair must exist in the current manifest")
        indexes = sorted(asset_indexes[asset_id] for asset_id in requested_ids)
        selected_asset_pair_indexes = (indexes[0], indexes[1])
        canonical_asset_pair = (
            ordered_assets[indexes[0]].asset_id,
            ordered_assets[indexes[1]].asset_id,
        )

    if selected_asset_pair_indexes is not None:
        left_group_index, right_group_index = selected_asset_pair_indexes
        total = _eligible_pair_count(
            asset_groups[left_group_index], asset_groups[right_group_index])
        selection_key = (
            f"selected-asset-pair:{canonical_asset_pair[0]}:{canonical_asset_pair[1]}")
        selection_semantics = (
            "all_eligible_cross_asset_pairs_for_selected_asset_pair_in_manifest_order")
    elif selected_anchor is None:
        total = sum(
            _eligible_pair_count(left, right)
            for left, right in combinations(asset_groups, 2)
        )
        selection_key = "all-eligible-cross-asset-pairs"
        selection_semantics = "all_eligible_cross_asset_pairs_in_manifest_order"
    else:
        anchor_group_index, anchor_mention = selected_anchor
        total = sum(
            mention.observation_id != anchor_mention.observation_id
            for group_index, group in enumerate(asset_groups)
            if group_index != anchor_group_index
            and (counterpart_group_index is None
                 or group_index == counterpart_group_index)
            for mention in group
        )
        selection_key = f"selected-anchor:{anchor_mention.node_id}"
        if counterpart_group_index is None:
            selection_semantics = (
                "all_eligible_cross_asset_pairs_for_selected_mention_in_manifest_order")
        else:
            selection_key += f":counterpart:{selected_counterpart_project_asset_id}"
            selection_semantics = (
                "all_eligible_pairs_for_anchor_and_counterpart_asset_in_manifest_order")
    candidate_set_parts = [
        _ALGORITHM_VERSION,
        graph.graph_id,
        relation_kind,
    ]
    if selected_anchor is not None or selected_asset_pair_indexes is not None:
        candidate_set_parts.append(selection_key)
    candidate_set_parts.extend([
        (f"{mention_review.review_id}@{mention_review.review_revision}"
         if mention_review is not None else "no-mention-review"),
        json.dumps(
            [[_mention_fingerprint(item) for item in group]
             for group in asset_groups],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    ])
    candidate_set_id = _stable_id("link_candidate_set_", *candidate_set_parts)
    candidates: list[ProjectStoryLinkCandidate] = []
    skip = offset

    def append_candidate(left_mention: _Mention, right_mention: _Mention) -> None:
        candidates.append(_candidate_from_mentions(
            graph, relation_kind, left_mention, right_mention, mention_review))

    if selected_asset_pair_indexes is not None:
        left_group_index, right_group_index = selected_asset_pair_indexes
        for left_mention, right_mention in _eligible_pair_page(
            asset_groups[left_group_index],
            asset_groups[right_group_index],
            offset=skip,
            limit=limit,
        ):
            append_candidate(left_mention, right_mention)
    elif selected_anchor is None:
        for left_mentions, right_mentions in combinations(asset_groups, 2):
            block_size = _eligible_pair_count(left_mentions, right_mentions)
            if skip >= block_size:
                skip -= block_size
                continue
            pair_page = _eligible_pair_page(
                left_mentions,
                right_mentions,
                offset=skip,
                limit=limit - len(candidates),
            )
            skip = 0
            for left_mention, right_mention in pair_page:
                append_candidate(left_mention, right_mention)
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break
    else:
        anchor_group_index, anchor_mention = selected_anchor
        for group_index, other_group in enumerate(asset_groups):
            if group_index == anchor_group_index:
                continue
            if (
                counterpart_group_index is not None
                and group_index != counterpart_group_index
            ):
                continue
            eligible_other_count = sum(
                mention.observation_id != anchor_mention.observation_id
                for mention in other_group
            )
            if skip >= eligible_other_count:
                skip -= eligible_other_count
                continue
            for other_mention in other_group:
                if other_mention.observation_id == anchor_mention.observation_id:
                    continue
                if skip > 0:
                    skip -= 1
                    continue
                if group_index < anchor_group_index:
                    append_candidate(other_mention, anchor_mention)
                else:
                    append_candidate(anchor_mention, other_mention)
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break

    eligible_asset_count = sum(
        item.state == "eligible_under_manifest_declaration"
        for item in coverage)
    return ProjectStoryLinkCandidatePage(
        candidate_set_id=candidate_set_id,
        project_id=manifest.project_id,
        project_manifest_id=manifest.manifest_id,
        project_revision=manifest.revision,
        context_id=graph.context_id,
        analysis_fingerprint=graph.analysis_fingerprint,
        story_graph_id=graph.graph_id,
        source_mention_review_id=(
            mention_review.review_id if mention_review is not None else None),
        source_mention_review_revision=(
            mention_review.review_revision if mention_review is not None else 0),
        source_mention_review_state=(
            "caller_asserted" if mention_review is not None else "absent"),
        relation_kind=relation_kind,
        selected_anchor_story_graph_node_id=(
            selected_anchor[1].node_id if selected_anchor is not None else None),
        selected_counterpart_project_asset_id=(
            selected_counterpart_project_asset_id),
        selected_project_asset_pair=canonical_asset_pair,
        asset_coverage=coverage,
        eligible_asset_count=eligible_asset_count,
        eligible_mention_count=eligible_mention_count,
        candidates=candidates,
        limit=limit,
        offset=offset,
        total=total,
        has_more=offset + len(candidates) < total,
        selection_semantics=selection_semantics,
    )


def bind_project_story_link_candidate_progress(
    page: ProjectStoryLinkCandidatePage,
    comparisons_by_candidate_id: dict[str, list[ProjectStoryLinkComparison]],
    legacy_comparisons_by_id: dict[str, ProjectStoryLinkComparison],
    review_history: list[ProjectStoryLinkReview],
) -> ProjectStoryLinkCandidatePage:
    """Attach exact persisted attempts and caller disposition history.

    Retry records use an indexed candidate ID, while pre-retry records are
    discovered via the original idempotency-derived comparison ID. Every record
    is still checked against the current project, graph, relation, source anchors,
    and evidence fields before it can be attached to a candidate.
    """
    expected_scope = (
        page.project_id,
        page.project_manifest_id,
        page.project_revision,
        page.context_id,
        page.story_graph_id,
        page.analysis_fingerprint,
    )
    for review in review_history:
        if (
            review.project_id,
            review.project_manifest_id,
            review.project_revision,
            review.context_id,
            review.story_graph_id,
            review.analysis_fingerprint,
        ) != expected_scope:
            raise ValueError("candidate progress review history is stale or inconsistent")

    latest_disposition: dict[str, ProjectStoryLinkCandidateCallerDisposition] = {}
    current_review_id = review_history[-1].review_id if review_history else None
    for review in review_history:
        for disposition in review.comparison_dispositions:
            latest_disposition[disposition.comparison_id] = (
                ProjectStoryLinkCandidateCallerDisposition(
                    disposition=disposition,
                    review_id=review.review_id,
                    review_revision=review.review_revision,
                    is_current=review.review_id == current_review_id,
                )
            )

    def same_anchor(candidate: ProjectStoryLinkAnchor,
                    stored: ProjectStoryLinkAnchor) -> bool:
        # source_asset_id is resolved from the persisted FilmObservation by the
        # comparison endpoint; candidate anchors bind the project-asset/source-
        # hash interval before that lookup.
        return (
            candidate.project_asset_id == stored.project_asset_id
            and candidate.story_graph_node_id == stored.story_graph_node_id
            and candidate.observation_id == stored.observation_id
            and candidate.source_content_hash.lower()
            == stored.source_content_hash.lower()
            and candidate.source_start == stored.source_start
            and candidate.source_end == stored.source_end
            and candidate.timebase == stored.timebase
            and candidate.timebase_unit == stored.timebase_unit
        )

    def same_pair(candidate: ProjectStoryLinkCandidate,
                  comparison: ProjectStoryLinkComparison) -> bool:
        return (
            comparison.project_id == page.project_id
            and comparison.project_manifest_id == page.project_manifest_id
            and comparison.project_revision == page.project_revision
            and comparison.context_id == page.context_id
            and comparison.story_graph_id == page.story_graph_id
            and comparison.analysis_fingerprint == page.analysis_fingerprint
            and comparison.source_mention_review_id == page.source_mention_review_id
            and comparison.source_mention_review_revision
            == page.source_mention_review_revision
            and comparison.relation_kind == candidate.relation_kind
            and comparison.candidate_id in {None, candidate.candidate_id}
            and same_anchor(candidate.left_anchor, comparison.left_anchor)
            and same_anchor(candidate.right_anchor, comparison.right_anchor)
            and comparison.left_person_description
            == candidate.left_person_description
            and comparison.right_person_description
            == candidate.right_person_description
            and comparison.left_event_evidence == candidate.left_event_evidence
            and comparison.right_event_evidence == candidate.right_event_evidence
        )

    def retry_request(
        candidate: ProjectStoryLinkCandidate, attempt_number: int
    ) -> ProjectStoryLinkComparisonRequestData:
        return candidate.comparison_request.model_copy(update={
            "idempotency_key": (
                f"{candidate.candidate_id}:retry:{attempt_number}"),
            "attempt_number": attempt_number,
        })

    bound_candidates = []
    for candidate in page.candidates:
        canonical_comparison_id = project_story_link_comparison_id(
            page.project_id, candidate.candidate_id)
        comparisons = list(comparisons_by_candidate_id.get(
            candidate.candidate_id, []))
        comparisons.sort(key=lambda item: (
            item.attempt_number or 0,
            item.created_at or 0,
            item.comparison_id,
        ))
        legacy = legacy_comparisons_by_id.get(canonical_comparison_id)
        conflict_record = None
        legacy_idempotency_conflict = False
        if legacy is not None and all(
            item.comparison_id != legacy.comparison_id for item in comparisons
        ):
            if (
                legacy.candidate_id is None
                and same_pair(candidate, legacy)
            ):
                comparisons.append(legacy)
                comparisons.sort(key=lambda item: (
                    item.attempt_number or 0,
                    item.created_at or 0,
                    item.comparison_id,
                ))
            elif not comparisons:
                conflict_record = legacy
            else:
                legacy_idempotency_conflict = True

        if conflict_record is not None:
            progress = ProjectStoryLinkCandidateComparisonProgress(
                comparison_run_state="idempotency_conflict",
                comparison_id=conflict_record.comparison_id,
                retry_comparison_request=retry_request(candidate, 2),
            )
        elif not comparisons:
            progress = ProjectStoryLinkCandidateComparisonProgress()
        else:
            comparison_attempts = []
            invalid_record = None
            if legacy_idempotency_conflict:
                comparison_attempts.append(
                    ProjectStoryLinkCandidateComparisonAttempt(
                        attempt_number=1,
                        comparison_id=legacy.comparison_id,
                        run_state="idempotency_conflict",
                    )
                )
            expected_attempt_number = len(comparison_attempts) + 1
            for comparison in comparisons:
                attempt_number = comparison.attempt_number or 1
                expected_key = (
                    candidate.candidate_id
                    if attempt_number == 1
                    else f"{candidate.candidate_id}:retry:{attempt_number}"
                )
                if (
                    not same_pair(candidate, comparison)
                    or attempt_number != expected_attempt_number
                    or comparison.idempotency_key_sha256 != hashlib.sha256(
                        expected_key.encode("utf-8")).hexdigest()
                    or (comparison.candidate_id is None and attempt_number != 1)
                ):
                    invalid_record = comparison
                    break
                comparison_attempts.append(
                    ProjectStoryLinkCandidateComparisonAttempt(
                        attempt_number=attempt_number,
                        comparison_id=comparison.comparison_id,
                        run_state=comparison.run_state,
                        failure_code=comparison.failure_code,
                    )
                )
                expected_attempt_number += 1
            if invalid_record is not None:
                raise ValueError(
                    "persisted candidate comparison history is inconsistent")
            latest = comparisons[-1]
            latest_caller_disposition = latest_disposition.get(
                next((item.comparison_id for item in reversed(comparisons)
                      if item.comparison_id in latest_disposition), ""))
            if latest.run_state == "failed":
                progress = ProjectStoryLinkCandidateComparisonProgress(
                    comparison_run_state="failed",
                    comparison_id=latest.comparison_id,
                    attempts=comparison_attempts,
                    retry_comparison_request=retry_request(
                        candidate, latest.attempt_number + 1
                        if latest.attempt_number is not None
                        else len(comparison_attempts) + 1),
                    latest_caller_disposition=latest_caller_disposition,
                )
            else:
                progress = ProjectStoryLinkCandidateComparisonProgress(
                    comparison_run_state="completed_unreviewed",
                    comparison_id=latest.comparison_id,
                    attempts=comparison_attempts,
                    latest_caller_disposition=latest_caller_disposition,
                )
        bound_candidates.append(candidate.model_copy(update={
            "comparison_progress": progress,
        }))

    return page.model_copy(update={"candidates": bound_candidates})
