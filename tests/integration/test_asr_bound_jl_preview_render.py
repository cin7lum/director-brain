"""Real local FFmpeg preview checks for source-bound J/L Plan offsets."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from director_brain.brief_compiler import compile_brief
from director_brain.director_reasoner import HeuristicDirectorReasoner
from director_brain.models.film_observation import (
    ClaimKind,
    FilmObservation,
    TimebaseUnit,
)
from director_brain.story_graph_builder import build_story_graph


def _technical_observation(
    index: int,
    start_us: int,
    end_us: int,
    media_hash: str,
    source_path: Path,
) -> FilmObservation:
    return FilmObservation(
        observation_id=f"preview-tech-{index}",
        media_asset_id=f"shot_{index}",
        media_hash=media_hash,
        start_frame=start_us,
        end_frame=end_us,
        timebase=1_000_000,
        timebase_unit=TimebaseUnit.MICROSECONDS,
        observation_type="deterministic_technical",
        claim=json.dumps({
            "blur_score": 100.0,
            "brightness_mean": 120.0,
            "exposure_ok": True,
            "shake_score": 1.0,
        }),
        provider="deterministic_fixture",
        model_version="fixture-1",
        prompt_version="n/a",
        confidence=1.0,
        review_state="auto_verified",
        claim_kind=ClaimKind.MEASURED,
        schema_version="1.0",
        project_id="preview-jl",
        created_at=int(time.time()),
        producer="deterministic_fixture",
        source_ref=str(source_path),
    )


@pytest.mark.parametrize(
    ("audio_style", "offset_field", "target_index", "start_delta", "end_delta", "expected_offset"),
    [
        ("j_cut", "audio_lead_us", 1, -200_000, 350_000, 200_000),
        ("l_cut", "audio_tail_us", 0, -350_000, 250_000, 250_000),
    ],
)
def test_source_bound_jl_plan_renders_to_decodable_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audio_style: str,
    offset_field: str,
    target_index: int,
    start_delta: int,
    end_delta: int,
    expected_offset: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("local FFmpeg and ffprobe are required for preview verification")

    source_path = tmp_path / "synthetic-source.mp4"
    output_path = tmp_path / f"{audio_style}-preview.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=14",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=14",
            "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(source_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    media_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    observations = [
        _technical_observation(index, start_us, end_us, media_hash, source_path)
            for index, (start_us, end_us) in enumerate((
                (0, 3_000_000),
                (3_000_000, 6_000_000),
                (6_000_000, 9_000_000),
                (9_000_000, 12_000_000),
            ))
    ]
    brief = compile_brief(
        "preview-jl",
        str(source_path),
        observations,
        target_duration_us=10_900_000,
    )
    graph = build_story_graph(brief, observations)
    narrative = {
        "suggested_order_resolved": ["shot_3", "shot_2", "shot_1", "shot_0"],
        "act_boundaries_resolved": [],
    }
    reasoner = HeuristicDirectorReasoner()
    base_edl, _ = reasoner.generate_plan(
        brief,
        graph,
        observations,
        narrative=narrative,
        narrative_order_priority=True,
    )
    target = base_edl.ordered_edits[target_index]
    boundary = target.in_frame if audio_style == "j_cut" else target.out_frame
    source_observation = next(
        item for item in observations
        if item.media_asset_id == target.source_asset_id
    )
    speech = source_observation.model_copy(update={
        "observation_id": f"preview-asr-{audio_style}",
        "media_asset_id": "synthetic-source-audio",
        "source_observation_id": source_observation.observation_id,
        "start_frame": boundary + start_delta,
        "end_frame": boundary + end_delta,
        "observation_type": "speech_transcript",
        "claim": "Synthetic fixture metadata; no transcript content is evaluated.",
        "provider": "local_asr_fixture",
        "model_version": "fixture-1",
        "prompt_version": "fixture-1",
        "review_state": "auto_generated",
        "claim_kind": ClaimKind.MODEL_OBSERVATION,
    })
    edl, plan = reasoner.generate_plan(
        brief,
        graph,
        [*observations, speech],
        narrative=narrative,
        narrative_order_priority=True,
        audio_style=audio_style,
    )
    selected = next(
        edit for edit in edl.ordered_edits
        if edit.source_asset_id == target.source_asset_id
    )
    assert getattr(selected, offset_field) == expected_offset
    assert selected.audio_evidence_refs == [speech.observation_id]
    assert edl.audio_evidence_refs == [speech.observation_id]
    assert speech.observation_id in next(
        decision for decision in plan.decisions
        if decision.shot_refs == [selected.source_asset_id]
    ).evidence_refs
    assert selected.project_asset_id is None
    from director_brain.plan_validator import validate_plan

    plan_valid, plan_errors = validate_plan(edl, plan, [*observations, speech])
    assert plan_valid, plan_errors

    from execution import renderer

    monkeypatch.setattr(renderer, "_nvenc_available", lambda: False)
    renderer.render_edl(edl, str(source_path), str(output_path))
    probe = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries",
            "stream=codec_type:format=duration", "-of", "json", str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    assert {stream["codec_type"] for stream in metadata["streams"]} >= {
        "video", "audio",
    }
    expected_duration_us = sum(
        item.out_frame - item.in_frame for item in edl.ordered_edits
    )
    assert edl.expected_duration == expected_duration_us
    assert float(metadata["format"]["duration"]) == pytest.approx(
        expected_duration_us / 1_000_000,
        abs=0.5,
    )
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output_path), "-f", "null", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
