# P2 Response Contract Implementation Plan

**Goal:** Continue the authorized Production Candidate P2 work by making incomplete provider output fail closed with a safe, same-call failure receipt, and aligning generation instructions with the existing exact-shot contract.

**Architecture:** Retain the existing Ollama/OpenAI-compatible transport, dynamic schemas, hierarchical Reasoner and application validators. Add only bounded response-envelope handling and non-semantic failure context; never repair missing emotion labels, retain runtime content, retry schema failures, or promote pathways. P0 through P5 remain the full goal; P3 requires real independent 04 binding, P4 independent FQL integration, and P5 independent project acceptance.

**Tech Stack:** Existing Python, urllib, pytest, JSON Schema and local Ollama. No new dependency, checkout or subsystem.

## Admission and alternatives (2026-10-09)

Original outcome: return evidence-bound project strategy candidates or an actionable safe failure without inventing missing per-shot output. Existing 2026-10-04 P2 admission and mature-editorial-assistant comparison remain relevant; this is a bounded correction inside that authorized SHADOW path.

| Candidate | Coverage and reliability | Maturity, maintenance, integration and extensibility | Security, performance and observability | Ecosystem, license/cost and architecture fit | Decision |
| --- | --- | --- | --- | --- | --- |
| Existing provider structured outputs plus validators | Already covers generation and exact shot cardinality; does not prove actual runtime success | Existing integration; no new dependency | Local route; bounded requests; failure context needs repair | Installed provider and current product contracts | Adopt/configure; thin handling correction |
| Native Ollama API | Official structured generation; equivalent schemas require validation | Mature API, but switching endpoints does not establish a solution to the observed short array | Local-only possible; same model and generation costs | Compatible candidate, no evidenced need to replace working transport | Keep available; no migration justified |
| General structured-output/retry framework | Could validate/retry; no evidence it solves this invocation | Adds dependency and integration/support surface | Retries add calls and can hide failure; does not own project evidence binding | Requires separate admission; no exact unmet need beyond current seam | Do not adopt for this fix |
| Eddie/Adobe editorial products | Existing source screen does not establish exact fail-closed project contract | Product APIs/routes would need separate admitted PoC | No current permitted private-media route or equivalent receipt established | Reviewed in repo-local 2026-10-04 comparison | No new product integration |

Official sources checked: [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility). These recommend schema grounding and post-generation validation. They do not explain historical failure receipts or prove the installed runtime obeys this contract.

Exact local gaps to reproduce before fixing: provider truncation is not distinguished from JSON/schema failure; an exception from ordinary project reasoning does not carry the call count and terminal stage; dynamic shot cardinality is in the schema but the segment user request lacks a concise explicit count. No new runtime, retry loop, inference subsystem, thresholds or model is admitted.

## Task 1: Response-envelope regression

- Modify `tests/unit/test_llm_local_transport.py` with a loopback HTTP provider returning `finish_reason=length` and syntactically valid partial content.
- Assert `LLMStructuredOutputError.failure_code == provider_output_truncated`; assert one call, no retry and no content in exception text. Run it before the fix and retain only the test verdict.
- Modify `director_brain/llm_adapter.py` to reject truncated structured output before content parsing. Verify ordinary successful and schema-rejection transport cases.

## Task 2: Same-call failure context and generation contract

- Modify `tests/unit/test_narrative_analyzer.py` to drive the public project analyzer through a short segment array and assert safe stage/call-count context, including a failure after a prior successful call.
- Modify `director_brain/narrative_analyzer.py` to preserve safe normalized failure context through ordinary project execution. Count attempted structured provider calls, including transport and parse failures, without exposing prompts, content or emotion values.
- Add dynamic exact-shot instructions to the existing segment request, with no inferred repair or retry. Keep validators authoritative and source identities unchanged.
- Verify all hierarchy/source-binding/diversity regressions and failure privacy assertions.

## Task 3: Runtime, review and checkpoint

- Run one ordinary local hierarchical product call on the existing frozen development snapshot if the input and local provider identities remain available. Keep SQLite read-only; capture only bounded stage/code/count, artifact IDs/counts and hashes. No Attempt 05 collector, prompt/raw response/emotion retention or media upload.
- Report runtime result separately from deterministic software tests; Director Quality and P2 exit remain NOT_PROVEN absent independent evidence.
- Run appropriate existing regression suite, inspect the exact diff and preserve all pre-existing untracked/ignored assets.
- Commit independently reviewable capability blocks locally on `codex/02-production-candidate`; push only a reviewed major-stage checkpoint to that branch. No master edits, PR merge, deployment, Resolve/NAS writes.

## 2026-10-09 synthetic hierarchical strategy failure

The first bounded local hierarchical run on eight synthetic semantic observations failed after one `segment` call with fixed code `segment_strategies_identical`. The run retained no generated response, prose, emotion values, or prompt. It confirms that the output validator rejects a duplicate strategy pair; it does not establish the cause of the earlier CoMind run or any Director Quality result.

Alternatives considered:

1. **Tighten the existing segment prompt's structural self-check** — smallest change within the existing Reasoner and validator; preserves fail-closed behavior.
2. Replace the validator's duplicate rejection with a warning — rejected because duplicate candidates are not a meaningful strategy comparison.
3. Synthesize a second candidate from the heuristic baseline — deferred because that would change candidate provenance and is not required to correct this prompt contract.

Decision: use option 1. Prompt version 2.14 asks the segment model to compare only `suggested_order` and included-source sets, which are fields available in that response schema, while forbidding unsupported changes made only to create a contrast. A first runtime on 2.13 returned two candidates, but its prompt referred to `act_boundaries`, which the segment schema does not expose; that run does not verify the corrected self-check. The validator remains authoritative. If the model still cannot produce two evidence-supported structures, the request fails closed; the result must not be promoted to a valid comparison. Verify the prompt contract in unit tests and run one bounded hierarchical call using synthetic observations only. This is software/runtime evidence, not quality acceptance.

## Implementation and verification (2026-10-09)

- Prompt version **2.14** now limits the local structural self-check to fields exposed by the segment response schema: source order and the included-source set. Unit coverage also locks out the previous schema-mismatched `act_boundaries` reference.
- The corrected bounded runtime (`evidence/P2_SYNTHETIC_HIERARCHICAL_STRATEGY_CONTRAST_2026-10-09/attempt-03/`) used eight fully synthetic semantic observations with explicit fictional descriptions, two assets, the pinned local Ollama `qwen2.5:7b` model and runtime, and three calls (`segment`, `segment`, `project_synthesis`). It returned two candidates; both had unique EDL execution signatures and both differed from the heuristic baseline. No source media, raw response, generated text, or prompt was retained. The result remained `confirmable=false`, `quality_acceptance=NOT_PROVEN`.
- The earlier 2.13 runtime used the same enriched fixture but had the invalid `act_boundaries` instruction; it is retained only as history. The original `segment_strategies_identical` failure used sparse fixture descriptions. This is not evidence that prompt text alone caused the outcome; the evidence and prompt changed between those runs.
- Focused `narrative_analyzer` plus `project_story_graph` regressions: **136 passed**. The full repository suite was rerun on exact pushed HEAD `bdae0e01ba0626ffd544968a9c55501f2f1dcab4` (documentation-only commit on source implementation `82f6af9`): **985 passed, 6 skipped, 8 warnings in 243.88s**, using Python 3.12.8 / pytest 8.4.2 and the recorded temporary dependency environment. The local run summary is preserved at `../../evidence/P2_CURRENT_COMMIT_FULL_SUITE_2026-10-09/attempt-01/`; full stdout was not saved. GitHub check-runs API returned 0 for this HEAD. CI triggers only on master pushes and pull requests, so this is local verification, not GitHub Actions.
- The CoMind TRAIN input remains `declared_unverified` and is not authorized for new model calls. Its earlier response-contract failure remains historical; the later CoMind attempt without a terminal receipt remains `INCOMPLETE`. P2 quality, admitted real-project generalization, pathway admission, and Production Candidate acceptance remain **NOT_PROVEN**.

## Historical 32-shot failure reconciliation (2026-10-09)

The sealed Attempt 06 `failure.json` is accessible and verifies six adjacent SHA-256 sidecars. It binds one call to `phase=project_shadow_reasoner_call`, `failure_code=segment_emotions`, and a 32-shot segment whose parsed dictionary contained two nonempty strings in `emotional_trajectory`; no other invalid positions were observed. The frozen runner used `segment_max_shots=32`, `max_output_tokens_per_call=4096`, and temperature 0. The recorded request was 6,680 UTF-8 bytes, below its 24 KiB request-byte limit. The response envelope's `finish_reason` and actual completion length were not retained.

The separate no-provider capacity audits are for the entire 772-observation project request: 43,169 input tokens against the 32,768-token context, and a compact valid full-project output estimate of 10,591 tokens (or 6,052 for an exploratory, non-approved two-order serialization) against the 4,096 completion cap. They prove a whole-project flat request/output capacity mismatch, but do not measure the failing 32-shot segment.

A new no-provider, schema-valid 32-shot segment probe with the current schema and application validator counted 1,709 body tokens for a deliberately minimal response, below 4,096. It rules out only an unavoidable minimum-shape overflow; it cannot estimate the missing historical response's semantic prose or explain why that provider response stopped after two labels. The frozen Attempt 06 product-source hashes match neither current branch commits nor the preserved pre-sync checkout, so its exact prompt implementation is unavailable. The historical direct cause therefore remains **NOT_PROVEN**. Do not label this a confirmed token truncation or a model defect.

Current code retains the bounded fix already present on this branch: exact dynamic per-segment shot counts are stated in the user prompt, the leaf ceiling is 16, and structured responses explicitly marked `finish_reason=length` fail with `provider_output_truncated`. A current-code 32-shot synthetic run at the unmodified 16-shot leaf ceiling completed two segment calls plus one project-synthesis call. Both strategy Plans validated, both EDLs bound to supplied synthetic source identities, both differed from the heuristic baseline and each other, and the comparison remained non-confirmable. This verifies the software path on synthetic evidence only. It does not retroactively prove the cause of Attempt 06 or accept real-project Director Quality.

Attempt 06's TRAIN input remains `declared_unverified`; no further provider call on it was made. Real-project revalidation needs an admitted dataset with an explicitly permitted local-processing route. Historical Attempt 06 evidence, capacity audits, the new static probe, and the current synthetic runtime are separate records; none substitutes for the missing raw response envelope or real authorized project acceptance.

## Real multi-asset follow-up and hierarchy contract correction (2026-10-09)

The authorized local NASA SVS 14948 diagnostic reached the Director Reasoner with two observed asset-local StoryGraphs and persisted Film Context. On the saved-context reasoner-only run `20261009T052529Z`, the pinned local Qwen2.5:7b made one `segment` call and the application rejected the result with `segment_strategies_identical`; synthesis did not run, and no Plan/EDL was produced. The raw response was not retained, so this proves only that the returned pair had the same structural signature under the current validator. It does not establish whether the model lacked a supported alternative or failed to follow the prompt.

Code review found that every multi-shot leaf was required to produce exactly two distinct local structures before project synthesis. That is stricter than the product requirement: 02 must produce two materially different project-level candidates when the evidence supports them, but each bounded segment need not independently contain two valid edits. Requiring local contrast can prevent project-level composition from seeing a legitimate single local choice and can pressure the model to invent a local distinction.

Revised contract: a one-shot segment returns one local hypothesis; a multi-shot segment may return one or two. A single local hypothesis must include an explicit limitation. Duplicate structural hypotheses remain rejected. The project-level synthesizer still must produce two distinct, evidence-bound candidate structures; all source rationales and source identities remain validated, and final materialized EDLs must be checked for distinct execution signatures. If project-level alternatives cannot be grounded, fail closed. Do not use heuristic fill, inferred source facts, or automatic retries to create a comparison. This revises only the internal leaf cardinality decision; it does not relax the user-facing P2 or Production Candidate acceptance requirements.

The NASA diagnostic remains technical SHADOW evidence only: rights state is `declared_unverified`, cross-asset relationships remain unknown, and quality acceptance is `NOT_PROVEN`. The correction requires focused regression coverage and one bounded local run on the already persisted observations; neither a successful model call nor two generated candidates would prove artistic quality or generalization.

### Follow-up run and rationale contract (2026-10-09)

One bounded local reasoner run after the leaf-cardinality change (`20261009T055432Z`) reused the same saved context and made one provider call, then failed closed at `segment_source_rationale_source`. No project synthesis or Plan/EDL followed. That legacy failure code combined an out-of-range focus index with a duplicate focus index; the model response was not retained, so the exact malformed shape and direct cause remain `NOT_PROVEN`.

Review found an inconsistency in the segment prompt example: for a two-shot segment, each example strategy showed only one `source_rationales` entry at `shot_idx=0`, while prose and the schema required complete per-source coverage. The prompt now shows both focus indices and requires the distinct index set to cover the local segment exactly. The validator still requires exact one-to-one coverage; its safe failure codes now distinguish duplicate focus indices from invalid indices without retaining response text. This is a targeted contract correction supported by the failure code and source review, not proof that the example caused the observed output.

The materializer now rejects a candidate comparison when two generated candidates resolve to the same full EDL execution signature, and records `project_edl_candidates_identical` at `candidate_materialization`. Project-level source, rationale, hypothesis, and EDL uniqueness checks remain strict. Focused narrative, StoryGraph, and API regressions pass **183 tests / 1 warning**.

The bounded local follow-up `20261009T055953Z` used the corrected two-rationale example and made six segment calls before failing at `segment_source_rationale_evidence`; the first five segment calls passed application validation. It did not reach project synthesis and produced no Plan/EDL. The raw response is intentionally absent, so that original category could not identify the exact citation defect. The follow-up below isolated a missing-focus citation; the current failure taxonomy keeps a separate code only for that observed defect and retains the existing grouped codes for other source/evidence errors.

The planned diagnostic repeat was completed as `20261009T060519Z`; it identified `segment_source_rationale_focus_missing` after six segment calls. This closes the failure-code ambiguity for that attempt, but not the direct cause of the model's omission. Subsequent controlled runs and their disposition are recorded below.

### 2026-10-09 goal alignment and bounded NASA follow-up

The user's final goal remains the full Production Candidate across P2→P3→P4→P5. The approved architecture assigns creative intent, evidence-bound StoryGraph, explainable Plan/EDL, versions and revisions to 02; capability routing to 03; formal Resolve execution/readback to 04; and independent quality/revision evaluation to 05. The immediate need remains a real multi-asset P2 Plan/EDL with correct source binding and substantive alternatives. The current work is still on that critical path, with a local risk of spending too much effort on failure taxonomy and supporting evidence; do not expand those mechanisms beyond what is needed to repair and verify the observed Reasoner failure. P2 remains SHADOW until quality and cross-project evidence support promotion.

The canonical schema indeed accepted a rationale whose `shot_idx` was absent from `source_indices`, while the application validator rejected it. Experimental `anyOf`/`contains` schema constraints were tried locally, but a minimal provider probe returned two rationale items that failed required-field validation; the provider-side cross-field constraint is therefore not proven. The probe used 51 prompt tokens and 44 completion tokens, stopped normally, and retained no generated content. The experimental schema, leaf-size and output-cap changes were removed from the product path.

Three controlled NASA Reasoner attempts reused the same persisted local observations and did not rerun visual analysis: `20261009T061258Z` stopped in the first segment at 4,096 completion tokens / 3,067 prompt tokens; `20261009T061626Z` stopped at 4,096 / 2,326 after the experimental 8-shot split; `20261009T062006Z` stopped at 8,192 / 2,326 after temporarily increasing the completion ceiling. All reported `finish_reason=length`, produced no Plan/EDL, and kept quality `NOT_PROVEN`. These attempts establish output-cap exhaustion for those exact runs, not the cause of the historical 32-shot `segment_emotions` receipt. The 8-shot and 8,192-token changes and the unproven conditional schema were reverted; the branch retains the 16-shot/4,096-token product path, strict application validation, corrected prompt example, and a focused failure code for the observed missing-focus defect. Other citation errors keep their existing grouped attribution. No further same-input Reasoner retry is justified without a materially different, evidence-backed correction.

Goal alignment is unchanged: complete the full Production Candidate, with 02's real director reasoning first, then P3 formal execution, P4 independent FQL revision, and P5 product acceptance. The immediate P2 product requirement has not been met on NASA: no real Plan/EDL was produced. The focused regression set `test_narrative_analyzer.py`, `test_project_story_graph.py`, and `test_api_endpoints.py` passes **179 tests with 1 existing warning** on the current working tree. This is software evidence only; P2 stays SHADOW and quality/generalization remain NOT_PROVEN.
