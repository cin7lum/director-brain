# HeuristicDirectorReasoner Role Change

## New role: SHOT_SELECTION_SPECIALIST
HeuristicDirectorReasoner continues to handle:
- Deterministic shot ranking (blur_score, four-act structure)
- Shot selection
- Baseline / fallback reference

## What it does NOT do (formalized)
- Natural language director intent parsing
- Audio/picture temporal relationship decisions
- Semantic relation extraction

## Physical rename
NOT done this phase. Class name remains HeuristicDirectorReasoner.
Role is frozen in documentation/API boundary. Physical rename deferred.

## No competition
SemanticDirectorReasoner and HeuristicDirectorReasoner do NOT produce
competing DirectorPlans. They handle different concerns:
- Semantic: "what does the director want?" (film semantics)
- Heuristic: "which shot to use?" (shot selection)
