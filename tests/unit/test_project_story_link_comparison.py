"""Contract, adapter, and persistence coverage for shadow link comparisons."""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest
from pydantic import ValidationError

from director_brain.models import (
    ProjectStoryLinkAnchor,
    ProjectStoryLinkComparison,
    ProjectStoryLinkEventEvidence,
    TimebaseUnit,
)
from director_brain.models.project_story_link_review import (
    ProjectStoryEntityLink,
    ProjectStoryLinkReview,
)
from observation_service.ollama_vlm_adapter import (
    OllamaVLMAdapter,
    PROJECT_LINK_COMPARISON_PROMPT_VERSION,
    build_project_link_comparison_prompt,
)
from storage.repository import DecisionLedgerEntry
from storage.sqlite_repository import SqliteRepository


def _anchor(asset_id: str, obs_id: str, start: int = 0) -> ProjectStoryLinkAnchor:
    return ProjectStoryLinkAnchor(
        project_asset_id=asset_id,
        source_asset_id=f"shot-{asset_id}",
        observation_id=obs_id,
        source_content_hash=("a" if asset_id == "asset-a" else "b") * 64,
        source_start=start,
        source_end=start + 1_000_000,
        timebase=1_000_000,
        timebase_unit=TimebaseUnit.MICROSECONDS,
    )


def _comparison(
    *,
    comparison_id: str = "link_cmp_test",
    request_fingerprint: str = "c" * 64,
    key_hash: str = "d" * 64,
    assessment: str = "possible_match",
    candidate_id: str | None = None,
    attempt_number: int | None = None,
    created_at: int = 1_800_000_000,
) -> ProjectStoryLinkComparison:
    return ProjectStoryLinkComparison(
        comparison_id=comparison_id,
        project_manifest_id="manifest-project-r1",
        project_revision=1,
        context_id="ctx-project-r1",
        story_graph_id="graph-project-r1",
        analysis_fingerprint="analysis-fingerprint",
        request_fingerprint=request_fingerprint,
        idempotency_key_sha256=key_hash,
        candidate_id=candidate_id,
        attempt_number=attempt_number,
        relation_kind="person_identity",
        left_anchor=_anchor("asset-a", "obs-a"),
        right_anchor=_anchor("asset-b", "obs-b"),
        left_person_description="red coat, short hair",
        right_person_description="red coat, short hair",
        run_state="completed_unreviewed",
        assessment=assessment,
        evidence_for=["similar jacket color"],
        evidence_against=[],
        limitation="visual similarity is not confirmed identity",
        left_frame_sha256=["1" * 64] * 3,
        right_frame_sha256=["2" * 64] * 3,
        frame_extractor_version="ffmpeg version synthetic-test",
        project_id="project-1",
        created_at=created_at,
        producer="test",
        source_ref="graph-project-r1",
        model_version="qwen3-vl:4b@sha256:" + "e" * 64 + "|ollama:0.32.14",
        prompt_version="project_cross_asset_pair_v1",
        prompt_sha256="f" * 64,
        generation_profile="format=json,temperature=0.1",
        sampling_profile="pair-v1",
        observation_schema_version="1.0",
    )


def test_comparison_contract_rejects_same_asset_and_unbound_clock():
    base = _comparison().model_dump(mode="python")
    base["right_anchor"] = _anchor("asset-a", "obs-b")
    with pytest.raises(ValidationError, match="two distinct assets"):
        ProjectStoryLinkComparison.model_validate(base)

    base = _comparison().model_dump(mode="python")
    base["right_anchor"]["timebase_unit"] = TimebaseUnit.FRAMES
    with pytest.raises(ValidationError, match="microseconds"):
        ProjectStoryLinkComparison.model_validate(base)


def test_event_comparison_contract_requires_both_exact_stored_event_mentions():
    base = _comparison().model_dump(mode="python")
    base.update({
        "relation_kind": "event_identity",
        "left_person_description": None,
        "right_person_description": None,
        "prompt_version": "project_cross_asset_pair_v2",
        "left_event_evidence": {
            "action_type": "action",
            "scene_description": "A person enters the hall.",
        },
        "right_event_evidence": {
            "action_type": "action",
            "temporal_notes": "The same person enters after guests turn.",
        },
    })
    comparison = ProjectStoryLinkComparison.model_validate(base)
    assert comparison.left_event_evidence.scene_description == (
        "A person enters the hall.")

    base["right_event_evidence"] = None
    with pytest.raises(ValidationError, match="both"):
        ProjectStoryLinkComparison.model_validate(base)

    with pytest.raises(ValidationError, match="stored scene or temporal"):
        ProjectStoryLinkEventEvidence(action_type="action")

    legacy = _comparison().model_dump(mode="python")
    legacy.update({
        "relation_kind": "event_identity",
        "left_person_description": None,
        "right_person_description": None,
        "prompt_version": "project_cross_asset_pair_v1",
    })
    assert ProjectStoryLinkComparison.model_validate(
        legacy).left_event_evidence is None


def test_person_comparison_prompt_omits_untrusted_person_descriptions():
    left = "女性，年轻，戴眼镜，穿浅色上衣"
    right = "男性，年长，戴眼镜，穿深色外套"
    prompt = build_project_link_comparison_prompt(
        "person_identity", left, right,
    )
    normalized_prompt = " ".join(prompt.split())
    assert PROJECT_LINK_COMPARISON_PROMPT_VERSION == "project_cross_asset_person_v2"
    assert left not in prompt
    assert right not in prompt
    assert "Do not infer or state age, sex, gender, race, or ethnicity" in normalized_prompt
    assert "If multiple people are visible and the target cannot be identified" in normalized_prompt
    assert "insufficient_evidence" in normalized_prompt


def test_local_vlm_pair_comparison_binds_six_frames_and_unreviewed_prompt(
    tmp_path, monkeypatch
):
    paths = []
    for index in range(6):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(f"jpeg-placeholder-{index}".encode())
        paths.append(str(path))

    adapter = OllamaVLMAdapter(
        model="qwen3-vl:4b",
        base_url="http://localhost:11434",
        model_digest="e" * 64,
        runtime_version="0.32.14",
        enforce_loopback=True,
    )
    runtime_calls = []
    monkeypatch.setattr(
        adapter, "verify_runtime_binding", lambda: runtime_calls.append("verified"))
    captured = {}

    def fake_chat(body):
        captured.update(body)
        return json.dumps({
            "assessment": "possible_match",
            "evidence_for": ["same distinctive sleeve pattern"],
            "evidence_against": [],
            "limitation": "not an identity confirmation",
        }), None

    monkeypatch.setattr(adapter, "_chat", fake_chat)
    result = adapter.compare_cross_asset_frames(
        paths[:3], paths[3:],
        relation_kind="person_identity",
        left_person_description="red coat, short hair",
        right_person_description="red coat, short hair",
    )

    assert runtime_calls == ["verified"]
    assert len(captured["messages"][0]["images"]) == 6
    prompt = captured["messages"][0]["content"]
    assert "first 3 images are source A" in prompt
    assert "Clothing alone is not enough" in prompt
    assert "red coat, short hair" not in prompt
    assert captured["options"] == {
        "temperature": 0.1,
        "num_ctx": 16384,
        "num_predict": 384,
    }
    assert captured["think"] is False
    assert result["assessment"] == "possible_match"
    assert result["review_state"] == "unreviewed"
    assert result["confidence_type"] == "UNCALIBRATED_MODEL_ASSESSMENT"


def test_local_vlm_event_comparison_binds_exact_stored_event_mentions(
    tmp_path, monkeypatch
):
    paths = []
    for index in range(6):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(b"placeholder")
        paths.append(str(path))
    adapter = OllamaVLMAdapter()
    captured = {}

    def invalid_reply(body):
        captured.update(body)
        return '{"assessment":"same"}', None

    monkeypatch.setattr(adapter, "_chat", invalid_reply)

    result = adapter.compare_cross_asset_frames(
        paths[:3], paths[3:],
        relation_kind="event_identity",
        left_event_evidence={
            "action_type": "action",
            "scene_description": "A person enters the hall.",
        },
        right_event_evidence={
            "action_type": "action",
            "temporal_notes": "The person enters after guests turn.",
        },
    )
    prompt = captured["messages"][0]["content"]
    assert '"scene_description": "A person enters the hall."' in prompt
    assert '"temporal_notes": "The person enters after guests turn."' in prompt
    assert result["status"] == "FAILED"
    assert result["failure_type"] == "PARSE"
    assert "assessment" not in result


def test_comparison_storage_is_immutable_idempotent_and_survives_reopen(tmp_path):
    db_path = str(tmp_path / "comparison.sqlite")
    repo = SqliteRepository(db_path)
    saved, reused = repo.save_project_story_link_comparison(_comparison())
    assert reused is False
    replay, reused = repo.save_project_story_link_comparison(_comparison())
    assert reused is True
    assert replay == saved
    repo.close()

    reopened = SqliteRepository(db_path)
    persisted = reopened.get(ProjectStoryLinkComparison, "link_cmp_test")
    assert persisted == saved
    assert reopened.list(ProjectStoryLinkComparison, "project-1") == [saved]

    conflicting = _comparison(request_fingerprint="9" * 64)
    with pytest.raises(ValueError, match="idempotency key"):
        reopened.save_project_story_link_comparison(conflicting)
    assert reopened.get(ProjectStoryLinkComparison, "link_cmp_test") == saved
    reopened.close()


def test_event_evidence_survives_repository_reopen(tmp_path):
    db_path = str(tmp_path / "event-comparison.sqlite")
    data = _comparison().model_dump(mode="python")
    data.update({
        "comparison_id": "link_cmp_event_test",
        "relation_kind": "event_identity",
        "left_person_description": None,
        "right_person_description": None,
        "left_event_evidence": {
            "action_type": "action",
            "scene_description": "A person opens the door.",
        },
        "right_event_evidence": {
            "action_type": "action",
            "temporal_notes": "The door opens before the group enters.",
        },
    })
    comparison = ProjectStoryLinkComparison.model_validate(data)
    repo = SqliteRepository(db_path)
    saved, reused = repo.save_project_story_link_comparison(comparison)
    assert reused is False
    repo.close()

    reopened = SqliteRepository(db_path)
    persisted = reopened.get(ProjectStoryLinkComparison, comparison.comparison_id)
    assert persisted == saved
    assert persisted.left_event_evidence.scene_description == (
        "A person opens the door.")
    reopened.close()


def test_candidate_attempt_index_migrates_legacy_comparison_table(tmp_path):
    db_path = str(tmp_path / "legacy-comparison.sqlite")
    legacy = _comparison().model_dump(mode="json")
    legacy.pop("candidate_id")
    legacy.pop("attempt_number")
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE project_story_link_comparisons ("
        "comparison_id TEXT PRIMARY KEY, project_id TEXT, created_at INTEGER, "
        "data TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO project_story_link_comparisons "
        "(comparison_id, project_id, created_at, data) VALUES (?, ?, ?, ?)",
        ("link_cmp_test", "project-1", 1_800_000_000,
         json.dumps(legacy, ensure_ascii=False)),
    )
    connection.commit()
    connection.close()

    repo = SqliteRepository(db_path)
    restored_legacy = repo.get(ProjectStoryLinkComparison, "link_cmp_test")
    assert restored_legacy.candidate_id is None
    assert restored_legacy.attempt_number is None

    candidate_id = "link_candidate_retry_migration"
    first = _comparison(
        comparison_id="candidate-attempt-1",
        key_hash=hashlib.sha256(candidate_id.encode()).hexdigest(),
        candidate_id=candidate_id,
        attempt_number=1,
        created_at=1_800_000_001,
    )
    retry = _comparison(
        comparison_id="candidate-attempt-2",
        key_hash=hashlib.sha256(
            f"{candidate_id}:retry:2".encode()).hexdigest(),
        candidate_id=candidate_id,
        attempt_number=2,
        created_at=1_800_000_002,
    )
    repo.save_project_story_link_comparison(first)
    repo.save_project_story_link_comparison(retry)
    indexed = repo.get_project_story_link_comparisons_by_candidate_ids(
        "project-1", [candidate_id])
    assert indexed[candidate_id] == [first, retry]
    repo.close()


def test_project_link_review_rejects_one_exact_mention_in_multiple_groups(
    tmp_path,
):
    first_group = ProjectStoryEntityLink(
        link_id="identity-group-1",
        entity_kind="person_identity",
        display_label="person A",
        anchors=[
            _anchor("asset-a", "obs-a").model_copy(
                update={"story_graph_node_id": "mention-a"}),
            _anchor("asset-b", "obs-b").model_copy(
                update={"story_graph_node_id": "mention-b"}),
        ],
    )
    conflicting_group = ProjectStoryEntityLink(
        link_id="identity-group-2",
        entity_kind="person_identity",
        display_label="different person",
        anchors=[
            _anchor("asset-a", "obs-a").model_copy(
                update={"story_graph_node_id": "mention-a"}),
            _anchor("asset-b", "obs-b").model_copy(
                update={"story_graph_node_id": "mention-c"}),
        ],
    )

    review_values = {
        "review_id": "review-1",
        "project_manifest_id": "manifest-1",
        "project_revision": 1,
        "context_id": "context-1",
        "story_graph_id": "graph-1",
        "analysis_fingerprint": "analysis-1",
        "review_revision": 1,
        "links": [first_group, conflicting_group],
        "comparison_dispositions": [],
        "project_id": "project-1",
        "created_at": 1_800_000_000,
        "producer": "test",
        "source_ref": "graph-1",
    }
    with pytest.raises(ValueError, match="mention node cannot belong to multiple"):
        ProjectStoryLinkReview(**review_values, schema_version="1.1")

    # Existing v1.0 snapshots stay readable and are not retroactively certified.
    legacy_review = ProjectStoryLinkReview(**review_values, schema_version="1.0")
    assert len(legacy_review.links) == 2

    repo = SqliteRepository(str(tmp_path / "legacy-review.sqlite"))
    repo.save(DecisionLedgerEntry(
        ledger_id="legacy-story-links-1",
        decision_id=legacy_review.story_graph_id,
        action="project_story_links_revised",
        timestamp=legacy_review.created_at,
        project_id=legacy_review.project_id,
        detail={"review": legacy_review.model_dump(mode="json")},
    ))
    restored = repo.list_project_story_link_reviews(
        legacy_review.project_id, legacy_review.story_graph_id)
    assert restored == [legacy_review]
    repo.close()


def test_project_link_review_allows_distinct_mentions_with_identical_intervals():
    review = ProjectStoryLinkReview(
        schema_version="1.1",
        review_id="review-1",
        project_manifest_id="manifest-1",
        project_revision=1,
        context_id="context-1",
        story_graph_id="graph-1",
        analysis_fingerprint="analysis-1",
        review_revision=1,
        links=[
            ProjectStoryEntityLink(
                link_id="identity-group-1",
                entity_kind="person_identity",
                display_label="person A",
                anchors=[
                    _anchor("asset-a", "obs-a").model_copy(
                        update={"story_graph_node_id": "mention-a1"}),
                    _anchor("asset-b", "obs-b").model_copy(
                        update={"story_graph_node_id": "mention-b1"}),
                ],
            ),
            ProjectStoryEntityLink(
                link_id="identity-group-2",
                entity_kind="person_identity",
                display_label="person B",
                anchors=[
                    _anchor("asset-a", "obs-a").model_copy(
                        update={"story_graph_node_id": "mention-a2"}),
                    _anchor("asset-b", "obs-b").model_copy(
                        update={"story_graph_node_id": "mention-b2"}),
                ],
            ),
        ],
        comparison_dispositions=[],
        project_id="project-1",
        created_at=1_800_000_000,
        producer="test",
        source_ref="graph-1",
    )

    assert [link.display_label for link in review.links] == ["person A", "person B"]
