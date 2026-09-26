# Known Issues

## 1. No callable model meets Admission Gate
- All locally-installable and externally-callable models fail semantic thresholds
- Doubao POC reference passes but is not formally callable (no API key)
- This is the primary blocker for Director Brain Phase-1B (Context Parameterizer)

## 2. glm-4-flash field placement errors
- Systematically puts avoid_* relations in desired_relation_or_change instead of must_avoid
- Outputs READY status when NEEDS_CONTEXT is correct for vague parameters
- required_context frequently empty even when status should indicate context needed
- These are model capability issues, not adapter issues

## 3. qwen2.5:14b status semantics weakness
- Tends to output READY for requests with vague parameters ("一点", "稍微")
- Should output NEEDS_CONTEXT per schema definition
- Negative constraint recall 62.5% below 90% threshold

## 4. Zhipu API only supports json_object mode
- No full JSON Schema constrained generation (unlike ollama's format=schema)
- Mitigation: canonical schema included in system prompt (transport adaptation)
- Schema validity still 100% with runtime model_validate_json()
- This is an API limitation, not a product issue

## 5. Hardware limitation for 30B+ class models
- RTX 3060 12GB VRAM insufficient for 30B Q4 models with reasonable speed
- User confirmed "32B跑不了"
- Would require GPU upgrade (>=24GB VRAM) for feasible local inference

## 6. Zhipu paid models inaccessible
- glm-4-air, glm-4-plus return 429 error 1113 (insufficient balance)
- Only glm-4-flash free tier is callable with existing key

## 7. Tool leakage evaluator false positive
- Evaluator checks for "reorder" as substring, which matches valid "reorder_story_beat" semantic relation
- SD-14 and SD-15 flagged as leakage but are actually valid relation vocabulary
- Real leakage: only SD-24 contains "otio" in output
- This is an evaluator precision issue, not a model issue; does not affect admission decision

## 8. Doubao POC results require re-verification
- Per task instruction, Doubao POC results labeled REFERENCE_OUTPUT / FEASIBILITY_REFERENCE
- If a real Doubao/Ark endpoint becomes available, all 32 cases must be re-run
- Old results must not be directly inherited as verified model results
