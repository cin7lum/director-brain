# Frozen Schema Identity

- Source: director_brain/models/director_decision.py
- Method: DirectorDecision.model_json_schema()
- SHA256 (sorted JSON): B66CD89758522ACB25AF57A8CEE0765C0433E66764C1D6F294EBDE7B16492B5F
- extra=forbid
- 12 fields: decision_id, creative_intent, target, desired_relation_or_change,
  must_preserve, must_avoid, parameterization, required_context, confidence,
  evidence, status, user_terminology
- DecisionStatus enum: READY, NEEDS_CONTEXT, UNDERSPECIFIED, UNSUPPORTED, CONFLICTING_CONSTRAINTS
- Same schema used for ALL candidates (single source of truth)
