1. **No externally-callable strong model available**: Doubao (POC-proven) has no
   API key/endpoint. This is the primary blocker for model admission.

2. **qwen2.5:14b negative constraint recall (62.5%)**: Partly evaluator term-matching
   (model uses different vocabulary for same semantic), partly genuine model weakness
   on multi-constraint sentences and conflict detection.

3. **qwen2.5:14b status calibration (65.6%)**: Model says READY when human gold
   says NEEDS_CONTEXT for vague-but-actionable requests. This could lead to
   premature parameterization if not handled by downstream context gathering.

4. **qwen3-vl JSON schema incompatibility**: ~22% empty content rate under
   format=JSON Schema. Vision model's thinking mode conflicts with constrained
   generation. Not suitable for text-only structured output.

5. **qwen3-vl tool leakage**: "J-cut" appeared in creative_intent (SD-06),
   violating the no-tool-leakage safety property.

6. **Evaluator exact string matching**: Negative constraint recall may
   undercount semantically-correct outputs that use different vocabulary.
   This affects all models equally but is more visible with weaker models.

7. **cv2 not installed**: 5 pre-existing test failures unrelated to this gate.
