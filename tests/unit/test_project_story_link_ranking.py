from __future__ import annotations

import pytest

import director_brain.project_story_link_ranking as ranking
from director_brain.models.film_observation import TimebaseUnit
from director_brain.models.project_story_link_candidates import (
    ProjectStoryLinkCandidate,
    ProjectStoryLinkComparisonRequestData,
)
from director_brain.models.project_story_link_review import ProjectStoryLinkAnchor


def _candidate(candidate_id: str, right_description: str):
    left_anchor = ProjectStoryLinkAnchor(
        project_asset_id="asset-a",
        story_graph_node_id="anchor-node",
        observation_id="anchor-observation",
        source_content_hash="a" * 64,
        source_start=0,
        source_end=1_000_000,
        timebase=1_000_000,
        timebase_unit=TimebaseUnit.MICROSECONDS,
    )
    right_anchor = ProjectStoryLinkAnchor(
        project_asset_id="asset-b",
        story_graph_node_id=f"{candidate_id}-node",
        observation_id=f"{candidate_id}-observation",
        source_content_hash="b" * 64,
        source_start=0,
        source_end=1_000_000,
        timebase=1_000_000,
        timebase_unit=TimebaseUnit.MICROSECONDS,
    )
    request = ProjectStoryLinkComparisonRequestData(
        manifest_revision=1,
        story_graph_id="graph-1",
        idempotency_key=candidate_id,
        candidate_id=candidate_id,
        attempt_number=1,
        relation_kind="person_identity",
        left_observation_id=left_anchor.observation_id,
        right_observation_id=right_anchor.observation_id,
        left_story_graph_node_id=left_anchor.story_graph_node_id,
        right_story_graph_node_id=right_anchor.story_graph_node_id,
        left_person_description="anchor",
        right_person_description=right_description,
    )
    return ProjectStoryLinkCandidate(
        candidate_id=candidate_id,
        relation_kind="person_identity",
        left_anchor=left_anchor,
        right_anchor=right_anchor,
        left_person_description="anchor",
        right_person_description=right_description,
        comparison_request=request,
    )


def test_shadow_ranker_uses_loopback_embeddings_and_keeps_all_candidates(monkeypatch):
    digest = "f" * 64
    monkeypatch.setattr(ranking, "_read_json", lambda url, timeout: {
        "models": [{
            "name": ranking.RANKING_MODEL,
            "digest": digest,
            "details": {"embedding_length": 2},
            "capabilities": ["embedding"],
        }],
    })
    calls = []

    def fake_post(url, body, *, timeout):
        calls.append((url, body, timeout))
        assert body["truncate"] is False
        vectors = {
            "anchor": [1.0, 0.0],
            "same description": [1.0, 0.0],
            "unrelated description": [0.0, 1.0],
        }
        return {"embeddings": [vectors[text] for text in body["input"]]}

    monkeypatch.setattr(ranking, "_post_json", fake_post)
    result = ranking.rank_project_story_link_candidates(
        [
            _candidate("candidate-b", "unrelated description"),
            _candidate("candidate-a", "same description"),
        ],
        base_url="http://127.0.0.1:11434",
    )

    assert [item.candidate_id for item in result.candidates] == [
        "candidate-a", "candidate-b"]
    assert [item.ranking_position for item in result.candidates] == [1, 2]
    assert [item.ranking_score for item in result.candidates] == [1.0, 0.0]
    assert all(item.ranking_state == "shadow_ranked_unadmitted"
               for item in result.candidates)
    assert result.model_digest == digest
    assert result.provider_invocation_count == 2
    assert result.embedding_invocation_count == 1
    assert calls[0][1]["model"] == ranking.RANKING_MODEL
    assert len(calls[0][1]["input"]) == 3  # shared anchor embedded once


def test_shadow_ranker_rejects_non_loopback_url_before_provider_call(monkeypatch):
    monkeypatch.setattr(
        ranking, "_read_json",
        lambda *args, **kwargs: pytest.fail("provider must not be called"),
    )
    with pytest.raises(ranking.ProjectStoryLinkRankingError) as error:
        ranking.rank_project_story_link_candidates(
            [], base_url="https://provider.example"
        )
    assert error.value.code == "non_loopback_provider_rejected"


def test_shadow_ranker_fails_closed_on_embedding_shape_mismatch(monkeypatch):
    monkeypatch.setattr(ranking, "_read_json", lambda url, timeout: {
        "models": [{
            "name": ranking.RANKING_MODEL,
            "digest": "e" * 64,
            "details": {"embedding_length": 2},
            "capabilities": ["embedding"],
        }],
    })
    monkeypatch.setattr(
        ranking, "_post_json", lambda url, body, timeout: {"embeddings": [[1.0]]})
    with pytest.raises(ranking.ProjectStoryLinkRankingError) as error:
        ranking.rank_project_story_link_candidates(
            [_candidate("candidate-a", "other")],
            base_url="http://localhost:11434",
        )
    assert error.value.code == "provider_invalid_embeddings"


def test_ranking_snapshot_round_trip_requires_exact_candidate_scope():
    candidates = [
        _candidate("candidate-b", "unrelated"),
        _candidate("candidate-a", "same description"),
    ]
    entries = [("candidate-a", 0.91), ("candidate-b", -0.2)]
    ranked = ranking.apply_project_story_link_ranking_snapshot(candidates, entries)
    assert [item.candidate_id for item in ranked] == [
        "candidate-a", "candidate-b"]
    assert [item.ranking_position for item in ranked] == [1, 2]
    assert ranking.project_story_link_ranking_snapshot_entries(ranked) == entries
    with pytest.raises(ranking.ProjectStoryLinkRankingError) as error:
        ranking.apply_project_story_link_ranking_snapshot(
            candidates, [("candidate-a", 0.91)])
    assert error.value.code == "ranking_cache_scope_mismatch"
