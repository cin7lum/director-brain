"""Thin LLM adapter for structured DirectorDecision output.

Uses ollama HTTP API with format=json for structured output.
Fail-closed: any error → SEMANTIC_REASONER_UNAVAILABLE, no silent fallback.

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
DEFAULT_MODEL = "llama3.2:3b"  # primary model path; override via config
DEFAULT_TIMEOUT = 60  # seconds

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

## Output
Strict JSON matching the DirectorDecision schema. No prose, no markdown.
"""


@dataclass
class LLMResult:
    """Result of an LLM structured output call."""
    decision: DirectorDecision | None
    error: str | None = None
    model: str = ""
    prompt_version: str = "1.0"
    raw_response: str = ""
    latency_ms: int = 0


class LLMAdapter:
    """Thin adapter for LLM structured output.

    Primary path: ollama HTTP API.
    Fail-closed on any error — no silent heuristic fallback.
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

    def generate_decision(
        self,
        user_input: str,
        context: str | None = None,
        decision_id: str | None = None,
    ) -> LLMResult:
        """Generate a DirectorDecision from natural language input.

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

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }

        start = time.time()
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
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

        # Parse and validate JSON against schema
        try:
            data = json.loads(content)
            # Ensure decision_id is set
            data.setdefault("decision_id", decision_id)
            decision = DirectorDecision(**data)
        except json.JSONDecodeError as e:
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: model returned invalid JSON: {e}",
                model=self.model,
                raw_response=content[:500],
                latency_ms=latency,
            )
        except Exception as e:
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: schema validation failed: {e}",
                model=self.model,
                raw_response=content[:500],
                latency_ms=latency,
            )

        return LLMResult(
            decision=decision,
            model=self.model,
            raw_response=content,
            latency_ms=latency,
        )
