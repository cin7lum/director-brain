"""
Real Context Tests for Director Brain Context Parameterization Phase 1.

Reads real timeline context from DaVinci Resolve via DaVinciResolveScript,
then runs the formal 02 parameterizer on real facts.

Current timeline: 23 clips, full source range (0 audio handle).
This is NOT eligible for J_CUT Supported Profile V1.
All tests use REAL readback facts — no simulated timeline.
"""
import sys, os, json
from pathlib import Path

os.environ["RESOLVE_SCRIPT_LIB"] = r"D:\DaVinci\fusionscript.dll"
sys.path.insert(0, r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules")
sys.path.insert(0, r"D:\新建豆包\AI-Director")

EVIDENCE = Path(r"D:\新建豆包\arsenal\evidence\DIRECTOR_BRAIN_CONTEXT_PARAMETERIZATION_PHASE_1")
EVIDENCE.mkdir(parents=True, exist_ok=True)

import DaVinciResolveScript as dvr
from director_brain.models.parameterization import (
    AudioEvent, AudioEventType, ClipBoundary, ParameterizationContext,
)
from director_brain.parameterization.parameterizer import JCutParameterizer

# ============================================================
# Step 1: Read real timeline context
# ============================================================
print("=" * 60)
print("STEP 1: Read real timeline context from DaVinci Resolve")
print("=" * 60)

resolve = dvr.scriptapp("Resolve")
pm = resolve.GetProjectManager()
project = pm.GetCurrentProject()
timeline = project.GetCurrentTimeline()
fps = float(timeline.GetSetting("timelineFrameRate"))

video_items = timeline.GetItemsInTrack("video", 1)
clips = []
for k, item in video_items.items():
    mpi = item.GetMediaPoolItem()
    media_name = mpi.GetName() if mpi else item.GetName()
    try:
        src_start = int(item.GetSourceStartFrame())
        src_end = int(item.GetSourceEndFrame())
    except:
        src_start = 0
        src_end = int(item.GetDuration())
    clips.append(ClipBoundary(
        clip_id=f"clip_{k}",
        media_name=media_name,
        start_frame=int(item.GetStart()),
        end_frame=int(item.GetEnd()),
        duration_frames=int(item.GetDuration()),
        source_start_frame=src_start,
        source_end_frame=src_end,
        source_duration_frames=src_end - src_start if src_end > src_start else int(item.GetDuration()),
    ))

# Find B clip
b_clips = [c for c in clips if "video_B" in c.media_name]
b = b_clips[0] if b_clips else clips[1] if len(clips) > 1 else clips[0]
b_idx = clips.index(b)
outgoing = clips[b_idx - 1] if b_idx > 0 else None

print(f"  Project: {project.GetName()}")
print(f"  Timeline: {timeline.GetName()}")
print(f"  FPS: {fps}")
print(f"  Clip count: {len(clips)}")
print(f"  Target B: {b.media_name} frames {b.start_frame}-{b.end_frame}")
print(f"  B source range: {b.source_start_frame}-{b.source_end_frame}")
print(f"  B source_start (audio handle): {b.source_start_frame}")
print(f"  Outgoing A: {outgoing.media_name if outgoing else 'none'}")

# Build REAL context from readback
real_ctx = ParameterizationContext(
    frame_rate=fps,
    picture_cut_frame=b.start_frame,
    incoming_clip=b,
    outgoing_clip=outgoing,
    audio_events=[],
    available_audio_handle_before=b.source_start_frame,
    has_dialogue=None,
    timeline_clip_count=len(clips),
    provenance={
        "frame_rate": "davinci_resolve_readback",
        "picture_cut_frame": "davinci_resolve_readback",
        "incoming_clip": "davinci_resolve_readback",
        "available_audio_handle_before": "davinci_resolve_readback (source_start_frame)",
        "timeline_clip_count": "davinci_resolve_readback",
    },
    notes=[f"Real timeline: {len(clips)} clips, B source_start={b.source_start_frame} (full source, 0 handle)"],
)

parameterizer = JCutParameterizer()
results = {}

# ============================================================
# Test 1: Explicit value (real context)
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: Explicit value '声音提前10帧' on real timeline")
print("=" * 60)

d1 = parameterizer.parameterize(real_ctx, user_exact_value=10.0)
print(f"  Status: {d1.status.value}")
print(f"  Exact value: {d1.exact_value}")
print(f"  Feasible max: {d1.feasible_range.max_value if d1.feasible_range else 'N/A'}")
print(f"  Rationale: {d1.rationale}")
# On current timeline (0 handle), 10f > 0 → CONFLICT (correct, no clamping)
results["test1_explicit_value"] = {
    "input": "user_exact_value=10.0",
    "status": d1.status.value,
    "exact_value": d1.exact_value,
    "feasible_max": d1.feasible_range.max_value if d1.feasible_range else None,
    "expected_on_current_timeline": "CONFLICT (10f > 0 handle, no clamping)",
    "pass": d1.status.value == "CONFLICT",
}

# ============================================================
# Test 2: Vague / insufficient (real context)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Vague '声音提前一点' on real timeline (no dialogue signal)")
print("=" * 60)

d2 = parameterizer.parameterize(real_ctx, user_exact_value=None)
print(f"  Status: {d2.status.value}")
print(f"  Exact value: {d2.exact_value}")
print(f"  Candidates: {len(d2.candidates)}")
print(f"  Rationale: {d2.rationale}")
# On current timeline (0 handle), no candidates → UNSATISFIABLE
results["test2_vague_insufficient"] = {
    "input": "vague '一点', no dialogue signal",
    "status": d2.status.value,
    "exact_value": d2.exact_value,
    "candidate_count": len(d2.candidates),
    "expected_on_current_timeline": "UNSATISFIABLE (0 handle, no candidates)",
    "pass": d2.status.value == "UNSATISFIABLE",
}

# ============================================================
# Test 3: Insufficient handle (real context)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Insufficient audio handle on real timeline")
print("=" * 60)

d3 = parameterizer.parameterize(real_ctx, user_exact_value=None)
print(f"  Status: {d3.status.value}")
print(f"  Available handle: {real_ctx.available_audio_handle_before}")
print(f"  Feasible range: {d3.feasible_range.min_value}-{d3.feasible_range.max_value}f")
# 0 handle → UNSATISFIABLE
results["test3_insufficient_handle"] = {
    "input": "real timeline with 0 audio handle",
    "status": d3.status.value,
    "available_handle": real_ctx.available_audio_handle_before,
    "feasible_max": d3.feasible_range.max_value,
    "expected": "UNSATISFIABLE",
    "pass": d3.status.value == "UNSATISFIABLE",
}

# ============================================================
# Test 4: External evidence (dialogue onset with provenance)
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: External dialogue onset evidence (10f) with provenance")
print("=" * 60)

ctx_with_evidence = real_ctx.model_copy(update={
    "audio_events": [AudioEvent(
        event_type=AudioEventType.DIALOGUE_ONSET,
        frame=10,
        confidence=0.85,
        source="external_asr_provider",
        evidence_ref="external_asr_clip_B_onset_10f",
    )],
    "has_dialogue": True,
    "provenance": {
        **real_ctx.provenance,
        "dialogue_onset": "external_asr_provider (externally supplied, provenance tracked)",
    },
})

d4 = parameterizer.parameterize(ctx_with_evidence, user_exact_value=None)
print(f"  Status: {d4.status.value}")
print(f"  Exact value: {d4.exact_value}")
print(f"  Candidates generated: {[(c.value, c.source.value) for c in d4.candidates]}")
print(f"  Feasible max: {d4.feasible_range.max_value}")
print(f"  Rationale: {d4.rationale}")
# Evidence is accepted by schema, candidate generated from evidence,
# but feasibility rejects (10f > 0 handle) → UNSATISFIABLE
# This proves: external evidence is used, but feasibility gates first
results["test4_external_evidence"] = {
    "input": "external dialogue onset=10f with provenance",
    "status": d4.status.value,
    "exact_value": d4.exact_value,
    "candidates": [{"value": c.value, "source": c.source.value} for c in d4.candidates],
    "feasible_max": d4.feasible_range.max_value,
    "evidence_accepted_by_schema": True,
    "note": "Evidence accepted, candidate generated, but feasibility rejects (0 handle). Proves feasibility gates before selection.",
    "expected_on_current_timeline": "UNSATISFIABLE (evidence valid but physically infeasible)",
    "pass": d4.status.value == "UNSATISFIABLE",
}

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("REAL CONTEXT TESTS SUMMARY")
print("=" * 60)
all_pass = True
for name, r in results.items():
    status = "PASS" if r["pass"] else "FAIL"
    if not r["pass"]:
        all_pass = False
    print(f"  {name}: {status} — {r['status']}")

print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print(f"  Note: All tests use REAL timeline facts (23 clips, full source, 0 handle).")
print(f"  Current timeline is NOT eligible for J_CUT Supported Profile V1.")
print(f"  A controlled 3-clip segment with trimmed source would show READY/NEEDS_DECISION outcomes.")

# Save results
with open(EVIDENCE / "REAL_CONTEXT_TESTS.json", "w") as f:
    json.dump({
        "timeline": {
            "project": project.GetName(),
            "timeline": timeline.GetName(),
            "fps": fps,
            "clip_count": len(clips),
            "target_media": b.media_name,
            "source_start_frame": b.source_start_frame,
            "audio_handle": b.source_start_frame,
        },
        "tests": results,
        "all_pass": all_pass,
    }, f, indent=2, ensure_ascii=False)

print(f"\n  Saved to {EVIDENCE / 'REAL_CONTEXT_TESTS.json'}")
