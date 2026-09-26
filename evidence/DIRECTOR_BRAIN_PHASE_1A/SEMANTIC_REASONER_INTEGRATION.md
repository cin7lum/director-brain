# SemanticDirectorReasoner Integration

## Location
`director_brain/semantic_reasoner.py`

## Interface
```python
reasoner = SemanticDirectorReasoner(llm_adapter=LLMAdapter(model="llama3.2:3b"))
result = reasoner.reason(user_direction="...", context="...", decision_id="...")
# result.decision: DirectorDecision | None
# result.error: str | None (SEMANTIC_REASONER_UNAVAILABLE on failure)
```

## Design decisions
- Thin wrapper over LLMAdapter, no business logic
- Fail-closed: LLM error → None decision + error, no fallback
- trace_id for audit
- context parameter reserved for future Film Context Parameterizer

## Not integrated with
- brief_compiler (separate concern)
- director_reasoner (Heuristic, separate specialist)
- edl_generator (execution layer)
- Arsenal (03, separate project)
