# Architecture Boundary

## Component responsibilities

| Component | Responsibility | NOT responsible |
|-----------|---------------|-----------------|
| SemanticDirectorReasoner | NL → DirectorDecision | Shot ranking, execution, scoring |
| HeuristicDirectorReasoner | Shot selection (SHOT_SELECTION_SPECIALIST) | Natural language understanding |
| LLMAdapter | Model invocation, structured output, validation | Semantic reasoning, prompt design |
| DirectorDecision | Film-level decision representation | Implementation details |
| DirectorBrief | Project-level user goals | Per-decision semantics |
| Arsenal (03) | Film Capability resolution, safety, execution | Natural language understanding |
| EDL/OTIO | Execution representation | Creative intent |

## Data flow
User NL → SemanticDirectorReasoner → DirectorDecision → (future) Arsenal → Film Capability → Execution

## Key boundary rules
- DirectorDecision never contains tool/API names
- DirectorDecision ≠ EDL (no audio_offset, no frame numbers)
- SemanticDirectorReasoner never falls back to keyword parser
- Heuristic remains separate specialist, not competing brain
