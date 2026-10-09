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
- Focused `narrative_analyzer` plus `project_story_graph` regressions: **136 passed**. Full repository suite: **985 passed, 6 skipped, 8 warnings**. The run used Python 3.12.8 in a temporary environment with project-declared `fastapi` and `httpx2` dependencies; the checkout has no lockfile. GitHub CI was not run.
- The CoMind TRAIN input remains `declared_unverified` and is not authorized for new model calls. Its earlier response-contract failure remains historical; the later CoMind attempt without a terminal receipt remains `INCOMPLETE`. P2 quality, admitted real-project generalization, pathway admission, and Production Candidate acceptance remain **NOT_PROVEN**.
