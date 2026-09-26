"""Thin LLM adapter for structured DirectorDecision output.

Uses ollama HTTP API with canonical Pydantic JSON Schema for constrained
structured output. Fail-closed: any error → SEMANTIC_REASONER_UNAVAILABLE,
no silent fallback.

Schema source of truth: DirectorDecision.model_json_schema() — single source.
No hand-maintained JSON Schema copy.

This adapter is intentionally thin. No model router, no multi-model voting,
no retry labyrinth, no agent framework.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from director_brain.models.director_decision import DirectorDecision

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"  # primary model path; installed locally
DEFAULT_TIMEOUT = 120  # seconds (local 7B model can be slow on first load)
PROMPT_VERSION = "1.1"  # bumped for schema-constrained output

SYSTEM_PROMPT = """You are a film editing director's semantic interpreter.

## Your job
Convert natural-language director requests into a structured DirectorDecision.
You express WHAT the director wants to happen at the film-semantic level.

## Hard boundaries
- NEVER output tool names, API names, software names, or implementation details.
  Forbidden in core fields: J_CUT, L_CUT, HOLD, REORDER, MCP, DaVinci, OTIO,
  recordFrame, mediaType, trackIndex, resolve, ffmpeg.
- NEVER invent exact numeric values from vague language.
  "一点", "稍微", "多一点" → magnitude=slight, exact_value=null.
  Only set exact_value when the user explicitly gives a number.
- NEVER choose a specific editing action for ambiguous requests.
  "自然一点", "舒服一点", "高级一点" → status=UNDERSPECIFIED or NEEDS_CONTEXT.
- ALWAYS separate positive intent (desired_relation_or_change) from
  negative constraints (must_avoid, must_preserve).
- If user explicitly uses a technical term like "J-cut", preserve it in
  user_terminology only — do NOT use it as a desired_relation.
- If the request is outside editing decisions (e.g. "make this person younger"),
  status=UNSUPPORTED.
- If constraints conflict (e.g. "don't change cut point but start B 0.5s earlier"),
  status=CONFLICTING_CONSTRAINTS and describe the conflict.

## Semantic relation vocabulary (film-level, not tool-level)
Use as appropriate; combine multiple:
- audio_precedes_picture — incoming sound starts before incoming picture
- outgoing_audio_continues_after_cut — previous sound carries over the cut
- extend_visible_duration — show a shot longer
- shorten_visible_duration — show a shot less long
- reorder_story_beat — change sequence of shots
- preserve_picture_cut — keep the video cut point unchanged
- avoid_transition — use hard cut, no dissolve/transition
- preserve_shot_identity — don't substitute media
- adjust_pacing — change rhythm (only when no more specific relation fits)
- allow_transition — permit a transition at this point

## Target objects (film semantics)
incoming_dialogue, incoming_picture, outgoing_audio, outgoing_picture,
current_picture_cut, reaction_shot, shot_order, transition_point,
source_media, pacing, speech_cadence

## Field usage rules (critical)
- desired_relation_or_change: POSITIVE actions only. What SHOULD happen.
  Examples: audio_precedes_picture, extend_visible_duration, reorder_story_beat.
  NEVER put negative statements here (no "avoid_...", "don't_...").
- must_preserve: what MUST remain unchanged.
  Examples: picture_cut_position, shot_identity, shot_order, source_media.
  Use for "不要改变X", "保留X", "X别动".
- must_avoid: what MUST NOT happen.
  Examples: transition, audio_lead, shot_substitution, duration_extension.
  Use for "不要X", "禁止X", "别X".
- Negative requests ("不要延长") → must_avoid=[duration_extension],
  desired_relation_or_change=[]. Do NOT put extend_visible_duration.
- Preservation requests ("不要改切点") → must_preserve=[picture_cut_position],
  desired_relation_or_change=[].

## Status definitions (use exactly one)
- READY: semantic intent is clear AND parameters are explicit enough to proceed.
  Use when the user gives concrete direction with no ambiguity.
- NEEDS_CONTEXT: semantic intent is clear BUT parameters are vague
  ("一点", "稍微", "多一点") or need timing/context facts to finalize.
  This is the most common status for natural-language directions.
- UNDERSPECIFIED: request is too vague to determine any specific editing action
  ("自然一点", "高级一点", "舒服一点"). No desired_relation should be set.
- UNSUPPORTED: request is outside editing decisions (e.g. "make person younger").
- CONFLICTING_CONSTRAINTS: ONLY when the user explicitly states two or more
  requirements that cannot both be satisfied (e.g. "don't change cut point but
  start B 0.5s earlier"). Do NOT use this for negative statements or simple
  positive requests.

## Output
Strict JSON matching the provided DirectorDecision schema. No prose, no markdown.
"""


@dataclass
class LLMResult:
    """Result of an LLM structured output call."""
    decision: DirectorDecision | None
    error: str | None = None
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    schema_version: str = ""
    raw_response: str = ""
    latency_ms: int = 0
    schema_constrained: bool = False


class LLMAdapter:
    """Thin adapter for LLM structured output.

    Primary path: ollama HTTP API with canonical Pydantic JSON Schema.
    Fail-closed on any error — no silent heuristic fallback.

    Schema chain:
      DirectorDecision (canonical Pydantic model)
        → model_json_schema()  (single source of truth)
        → ollama format param   (constrained generation)
        → model_validate_json() (runtime validation)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Canonical schema — generated from the single source of truth
        self._schema = DirectorDecision.model_json_schema()
        self.schema_version = f"pydantic_{DirectorDecision.model_config.get('title', 'DirectorDecision')}"

    @property
    def json_schema(self) -> dict:
        """Return the canonical JSON Schema (single source of truth)."""
        return self._schema

    def generate_decision(
        self,
        user_input: str,
        context: str | None = None,
        decision_id: str | None = None,
    ) -> LLMResult:
        """Generate a DirectorDecision from natural language input.

        Uses canonical DirectorDecision.model_json_schema() as ollama format
        for schema-constrained generation, then model_validate_json() for
        runtime validation.

        Args:
            user_input: The director's natural language request.
            context: Optional available context (timeline state, etc.).
            decision_id: Optional explicit ID; auto-generated if None.

        Returns:
            LLMResult with validated decision or error.
            Never raises — fail-closed with error field set.
        """
        if decision_id is None:
            decision_id = f"sd_{int(time.time() * 1000)}"

        user_message = user_input
        if context:
            user_message = f"Available context:\n{context}\n\nDirector request:\n{user_input}"

        # Canonical schema as ollama format — single source of truth
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "format": self._schema,  # canonical Pydantic JSON Schema
            "stream": False,
            "options": {"temperature": 0.1},
        }

        start = time.time()
        schema_constrained = True
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            # If schema format is rejected by this ollama version, fall back
            # to plain JSON mode but keep runtime Pydantic validation.
            if e.code == 400 and "format" in (e.read().decode("utf-8", errors="replace").lower()):
                schema_constrained = False
                payload["format"] = "json"
                try:
                    req2 = urllib.request.Request(
                        f"{self.base_url}/api/chat",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req2, timeout=self.timeout) as resp2:
                        raw = resp2.read().decode("utf-8")
                except Exception as e2:
                    latency = int((time.time() - start) * 1000)
                    return LLMResult(
                        decision=None,
                        error=f"SEMANTIC_REASONER_UNAVAILABLE: ollama request failed: {e2}",
                        model=self.model,
                        latency_ms=latency,
                    )
            else:
                latency = int((time.time() - start) * 1000)
                return LLMResult(
                    decision=None,
                    error=f"SEMANTIC_REASONER_UNAVAILABLE: ollama HTTP {e.code}: {e}",
                    model=self.model,
                    latency_ms=latency,
                )
        except urllib.error.URLError as e:
            latency = int((time.time() - start) * 1000)
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: ollama connection failed: {e}",
                model=self.model,
                latency_ms=latency,
            )
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: {type(e).__name__}: {e}",
                model=self.model,
                latency_ms=latency,
            )

        latency = int((time.time() - start) * 1000)

        # Parse ollama response
        try:
            resp_data = json.loads(raw)
            content = resp_data.get("message", {}).get("content", "")
        except (json.JSONDecodeError, KeyError):
            return LLMResult(
                decision=None,
                error="SEMANTIC_REASONER_UNAVAILABLE: invalid ollama response format",
                model=self.model,
                raw_response=raw[:500],
                latency_ms=latency,
            )

        if not content.strip():
            return LLMResult(
                decision=None,
                error="SEMANTIC_REASONER_UNAVAILABLE: model returned empty content",
                model=self.model,
                raw_response=raw[:500],
                latency_ms=latency,
            )

        # Runtime validation using canonical model — model_validate_json
        try:
            decision = DirectorDecision.model_validate_json(content)
            # Always override with the caller-provided decision_id (model may
            # generate its own placeholder IDs; caller-provided is authoritative)
            if decision.decision_id != decision_id:
                data = decision.model_dump()
                data["decision_id"] = decision_id
                decision = DirectorDecision(**data)
        except Exception as e:
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: schema validation failed: {e}",
                model=self.model,
                raw_response=content[:500],
                latency_ms=latency,
                schema_constrained=schema_constrained,
            )

        return LLMResult(
            decision=decision,
            model=self.model,
            raw_response=content,
            latency_ms=latency,
            schema_constrained=schema_constrained,
            schema_version=self.schema_version,
        )
