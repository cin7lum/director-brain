# EDL Boundary Analysis

## Frozen principle
DirectorDecision ≠ EDL.

- DirectorDecision: creative intent, semantic relations, constraints, uncertainty.
- EDL / OTIO: execution representation — concrete clips, tracks, time ranges.

## Current EDL limitations
- Video-centric: ordered_edits have no audio track or audio offset.
- No field for director_decision_id lineage.
- No field for semantic relation (audio_precedes_picture etc.).

## Future boundary (not implemented this POC)
DirectorDecision → Arsenal Film Capability Resolution → OTIO Projection → EDL.
The EDL should carry a reference to the originating DirectorDecision, but should
NOT embed creative intent directly. Execution parameters (frame numbers, track
indices) are derived downstream, not authored by the Director Brain.
