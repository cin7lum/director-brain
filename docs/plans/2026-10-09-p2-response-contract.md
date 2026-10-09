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
