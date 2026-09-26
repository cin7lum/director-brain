1. **Local model underperforms POC model on negative constraints**: qwen2.5:14b
   achieves 62.5% negative constraint recall vs POC's 93.8% (Doubao). This is a
   model capability gap, not an architecture issue. All safety properties hold.

2. **Status calibration**: Model tends to classify vague-but-actionable requests
   as READY instead of NEEDS_CONTEXT. Human gold standard requires NEEDS_CONTEXT
   when exact timing is unspecified. Status match = 65.6%.

3. **Conflict detection weakness**: SD-28 ("don't change cut point but start B
   0.5s earlier") is not detected as CONFLICTING_CONSTRAINTS by the local model.
   The POC's Doubao detected it correctly.

4. **cv2 not installed**: 5 pre-existing test failures + 6 collection errors
   in observation_service tests. Unrelated to Phase-1A/R1.

5. **ollama schema format fallback**: If ollama rejects JSON Schema format,
   adapter falls back to plain "json" mode with runtime Pydantic validation.
   schema_constrained=False in that case. Currently qwen2.5:14b accepts
   schema format (schema_constrained=True for all 32 cases).

6. **required_context often empty**: Model frequently returns empty
   required_context even when status=NEEDS_CONTEXT. This is a model behavior
   issue; the schema field exists and is validated.
