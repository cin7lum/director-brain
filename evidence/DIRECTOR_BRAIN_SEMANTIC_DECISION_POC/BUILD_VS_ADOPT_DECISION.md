# Build-vs-Adopt Decision

## Natural Language Understanding
**ADOPT**: Mature LLM + Structured Output / JSON Schema.
- Do NOT build: NLU classifier, tokenizer, intent engine, semantic parser framework.
- Rationale: The LLM handles paraphrase, negation, ambiguity, and relation direction
  that would require enormous keyword tables to replicate.

## Film-specific Layer
**KEEP (thin)**: DirectorDecision schema + film reasoning constraints.
- Schema: 12 fields, controlled vocabulary for semantic relations.
- Constraints: no tool names, no fabricated parameters, ambiguity → NEEDS_CONTEXT.

## HeuristicDirectorReasoner
**REPOSITION**: SHOT_SELECTION_SPECIALIST.
- Continues to handle deterministic shot ranking, shot selection, baseline, fallback.
- Does NOT attempt to be a full natural-language director brain.
- Production code not modified in this POC.

## What NOT to build
- Generic Tool Catalog (use MCP discovery)
- Custom Timeline Model (use OTIO)
- Custom DaVinci Tool Layer (use Blackmagic official MCP)
