# Current Brain Limitation (pre-POC)

The Vertical Slice V1 failure proved the existing Heuristic Director Brain cannot
express audio/picture temporal relationship decisions:

1. `_parse_intent()` uses fixed keyword lists that don't include any audio-overlap
   concept. "声音提前进入" matches zero keywords.
2. `DirectorBrief` has no field for audio_offset, audio_overlap, j_cut, l_cut,
   or any timing-only edit instruction.
3. `HeuristicDirectorReasoner` only produces `select_shot` decisions
   (blur_score sorting + four-act shot selection). No audio editing code path.
4. EDL is video-centric: ordered_edits have source_asset_id, start_time_us,
   duration_us — no audio track, no audio offset.

Result: the system captures negative constraints (must_avoid) but loses the
positive creative intent. This POC proves that a mature LLM + thin schema
solves this without modifying production code.
