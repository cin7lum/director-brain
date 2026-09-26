# Heuristic Fallback Check

## Verified: NO silent heuristic fallback

1. `SemanticDirectorReasoner` source code does not import or instantiate
   `HeuristicDirectorReasoner` (verified by R1-7 test)
2. `SemanticDirectorService` does not import or use heuristic
3. On LLM failure, `process_direction()` returns `SEMANTIC_REASONER_UNAVAILABLE`
   with decision=None — it does NOT construct a heuristic decision
4. `HeuristicDirectorReasoner` still exists and is importable
   (SHOT_SELECTION_SPECIALIST role, unchanged)

## Heuristic role
- Continues to handle: deterministic shot ranking, shot selection, baseline
- Does NOT handle: natural language intent parsing, semantic relations
- Physical rename deferred; role frozen in documentation
