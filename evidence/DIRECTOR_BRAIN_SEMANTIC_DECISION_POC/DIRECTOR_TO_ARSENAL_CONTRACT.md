# Director → Arsenal Contract (minimum)

Arsenal needs from the Director Brain:

1. **semantic_relation**: What should happen (audio_precedes_picture,
   extend_visible_duration, etc.) — controlled film vocabulary.
2. **target**: Which film-semantic objects (incoming_dialogue, reaction_shot, etc.).
3. **creative_intent**: Why (natural language, for audit and downstream context).
4. **preservation_constraints**: What must NOT change (picture_cut_position,
   shot_identity).
5. **avoidance_constraints**: What must be avoided (transition, audio_lead).
6. **parameter_certainty**: exact_value + unit if explicit; magnitude + null if
   vague; required_context if unresolved.
7. **decision_lineage**: decision_id, evidence (user input excerpts).
8. **status**: READY / NEEDS_CONTEXT / UNDERSPECIFIED / UNSUPPORTED /
   CONFLICTING_CONSTRAINTS.

Arsenal does NOT need: MCP tool names, DaVinci API parameters, frame numbers,
track indices, OTIO objects. Those are derived downstream.
