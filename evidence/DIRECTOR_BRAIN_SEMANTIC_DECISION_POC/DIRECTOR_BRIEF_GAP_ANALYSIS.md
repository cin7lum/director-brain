# DirectorBrief Gap Analysis

Field-by-field assessment of current DirectorBrief vs. DirectorDecision POC schema.

| DirectorBrief Field | Assessment | Notes |
|---------------------|-----------|-------|
| brief_id | KEEP | Identity |
| version | KEEP | Versioning |
| source_text | KEEP | Original user input |
| language | KEEP | Useful for downstream |
| intent | EXTEND | Currently only "user_provided"/"auto_compiled"; should carry creative_intent |
| audience | KEEP | Useful metadata |
| target_duration | KEEP | Constraint |
| delivery_profile | KEEP | Useful metadata |
| themes | MOVE_TO_CONTEXT | Derived from observations, not director intent |
| emotional_arc | EXTEND | Too coarse; DirectorDecision.creative_intent is richer |
| visual_language | MOVE_TO_CONTEXT | Derived from observations |
| editing_language | EXTEND | Currently keyword-only; needs semantic relations |
| sound_language | EXTEND | No audio-overlap concept; this is the core gap |
| must_include | KEEP | Works |
| must_avoid | KEEP | Works (captured negative constraints in V1) |
| approval_state | KEEP | Workflow |

**Core gap**: No field expresses desired_relation_or_change (audio_precedes_picture,
outgoing_audio_continues_after_cut, etc.), parameterization certainty, or
required_context. DirectorDecision fills this gap.
