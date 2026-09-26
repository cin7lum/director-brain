# DirectorBrief Boundary

## Two distinct concepts
- **DirectorBrief**: project-level user goals, themes, constraints, audience
- **DirectorDecision**: one specific creative/editing decision with semantic relations

## Current state
DirectorBrief exists and is used by brief_compiler. DirectorDecision is new.
They are NOT merged. DirectorBrief is NOT modified in this phase.

## Future integration (not this phase)
DirectorBrief may reference DirectorDecision IDs, or DirectorDecision may carry
brief_version. This requires Chief decision.

## What was NOT done
- DirectorBrief fields unchanged
- brief_compiler unchanged
- No DirectorDecision forced into DirectorBrief
