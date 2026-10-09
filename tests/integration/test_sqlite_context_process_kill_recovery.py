"""SQLite project Context atomicity under abrupt process termination.

This is a local storage durability regression using synthetic records. It
simulates process death during a repository transaction; it is not a hardware
power-loss test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from director_brain.models import FilmContextSnapshot, FilmObservation
from director_brain.models.project import (
    FilmProjectManifest,
    ProcessingRights,
    ProjectAsset,
)
from director_brain.utils import short_hash
from storage import SqliteRepository


def _manifest(project_id: str) -> FilmProjectManifest:
    return FilmProjectManifest(
        manifest_id=f"manifest_{short_hash(project_id)}_r00000001",
        revision=1,
        boundary_basis="dataset_project_id",
        boundary_source_ref="test-fixture://project/process-kill",
        boundary_evidence_refs=["test-fixture://boundary/process-kill"],
        assets=[
            ProjectAsset(
                asset_id="asset-1",
                source_ref="dataset://synthetic/process-kill-asset",
                source_identity_state="declared_unverified",
                order=0,
                rights=ProcessingRights(state="unverified"),
            )
        ],
        project_id=project_id,
        created_at=1_700_000_000,
        producer="integration-test",
        source_ref="test-fixture://manifest/process-kill",
    )


def _observation(observation_id: str, project_id: str) -> FilmObservation:
    return FilmObservation(
        project_id=project_id,
        created_at=1_700_000_001,
        producer="integration-test",
        source_ref=f"test-fixture://observation/{observation_id}",
        observation_id=observation_id,
        media_asset_id="asset-1",
        project_asset_id="asset-1",
        media_hash="synthetic-content-hash",
        start_frame=0,
        end_frame=1,
        timebase=25,
        observation_type="test_observation",
        claim=observation_id,
        provider="synthetic-test",
        model_version="none",
        prompt_version="none",
        confidence=0.5,
        evidence_refs=[f"test-fixture://evidence/{observation_id}"],
        review_state="pending",
        claim_kind="measured",
    )


def _context(
    context_id: str,
    project_id: str,
    manifest: FilmProjectManifest,
    observation_ids: list[str],
) -> FilmContextSnapshot:
    return FilmContextSnapshot(
        project_id=project_id,
        created_at=1_700_000_002,
        producer="integration-test",
        source_ref=f"test-fixture://context/{context_id}",
        context_id=context_id,
        project_manifest_id=manifest.manifest_id,
        project_revision=manifest.revision,
        asset_refs=["asset-1"],
        source_content_hashes=[None],
        analysis_fingerprint=f"fingerprint-{context_id}",
        provider="synthetic-test",
        model="none",
        prompt_version="none",
        sampling_config={},
        timebase=25,
        coverage="full",
        rights_scope="unverified",
        evidence_refs=observation_ids,
        cache_state="computed",
    )


def test_killed_context_transaction_recovers_without_partial_evidence(tmp_path):
    """A killed writer leaves the prior Context intact and permits retry."""
    project_id = "project-context-process-kill"
    database = tmp_path / "brain.sqlite3"
    ready_marker = tmp_path / "writer-entered-transaction"
    release_marker = tmp_path / "release-writer"
    manifest = _manifest(project_id)
    existing_observation = _observation("observation-existing", project_id)
    existing_context = _context(
        "context-existing", project_id, manifest,
        [existing_observation.observation_id],
    )

    repository = SqliteRepository(str(database))
    repository.save_project_manifest(manifest, expected_revision=0)
    repository.save_project_context(
        existing_context, [existing_observation], expected_manifest_revision=1,
    )
    repository.close()

    incoming_observations = [
        _observation("observation-before-kill", project_id),
        _observation("observation-blocked-by-trigger", project_id),
    ]
    incoming_context = _context(
        "context-after-retry", project_id, manifest,
        [item.observation_id for item in incoming_observations],
    )
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps({
            "context": incoming_context.model_dump(mode="json"),
            "observations": [
                item.model_dump(mode="json") for item in incoming_observations
            ],
            "ready_marker": str(ready_marker),
            "release_marker": str(release_marker),
        }),
        encoding="utf-8",
    )
    child_code = r'''
import json
import sys
import time
from pathlib import Path
from director_brain.models import FilmContextSnapshot, FilmObservation
from storage import SqliteRepository

database, records_path = sys.argv[1:3]
records = json.loads(Path(records_path).read_text(encoding="utf-8"))
ready = Path(records["ready_marker"])
release = Path(records["release_marker"])
repository = SqliteRepository(database)

def block_writer():
    ready.write_text("inside uncommitted transaction", encoding="utf-8")
    while not release.exists():
        time.sleep(0.01)
    return 0

repository._conn.create_function("block_writer", 0, block_writer)
repository._conn.execute(
    "CREATE TEMP TRIGGER interrupt_context_write "
    "BEFORE INSERT ON film_observations "
    "WHEN NEW.observation_id = 'observation-blocked-by-trigger' "
    "BEGIN SELECT block_writer(); END"
)
snapshot = FilmContextSnapshot.model_validate(records["context"])
observations = [FilmObservation.model_validate(item)
                for item in records["observations"]]
repository.save_project_context(snapshot, observations, 1)
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(database), str(records_path)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready_marker.exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    "writer exited before entering its transaction; "
                    f"stdout={stdout!r}, stderr={stderr!r}"
                )
            if time.monotonic() >= deadline:
                raise AssertionError("writer did not reach the transaction trigger")
            time.sleep(0.01)
        process.kill()
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)

    recovered = SqliteRepository(str(database))
    try:
        assert recovered.get_project_manifest(project_id, 1) == manifest
        assert recovered.get(FilmContextSnapshot, existing_context.context_id) == (
            existing_context
        )
        assert recovered.get(
            FilmContextSnapshot, incoming_context.context_id
        ) is None
        assert recovered.get(FilmObservation, existing_observation.observation_id) == (
            existing_observation
        )
        assert recovered.get(
            FilmObservation, incoming_observations[0].observation_id
        ) is None
        assert recovered.get(
            FilmObservation, incoming_observations[1].observation_id
        ) is None

        # The recovered database must accept an ordinary retry of the same bundle.
        recovered.save_project_context(
            incoming_context, incoming_observations,
            expected_manifest_revision=1,
        )
        assert recovered.get(
            FilmContextSnapshot, incoming_context.context_id
        ) == incoming_context
        assert all(
            recovered.get(FilmObservation, item.observation_id) == item
            for item in incoming_observations
        )
    finally:
        recovered.close()
