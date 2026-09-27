"""Thin Zhipu (GLM) model provider adapter for DirectorDecision.

OpenAI-compatible HTTP API with response_format=json_object.
Runtime validation via DirectorDecision.model_validate_json().

This is intentionally a transport-only adapter. No semantic repair,
no keyword patch, no model-specific answer correction.
Semantic logic remains in SemanticDirectorReasoner / the model.

Schema source of truth: DirectorDecision.model_json_schema() — same
canonical schema used by ollama adapter. No second schema copy.
"""
from __future__ import annotations

import http.client as http_client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from director_brain.llm_adapter import LLMResult, SYSTEM_PROMPT, PROMPT_VERSION
from director_brain.models.director_decision import DirectorDecision

ZHIPU_BASE = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4-flash"
DEFAULT_TIMEOUT = 60


def _build_system_prompt(schema: dict) -> str:
    """Append canonical schema to system prompt for APIs that don't support
    schema-constrained generation (Zhipu only supports json_object mode).
    This is a transport adaptation, not a semantic prompt change."""
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    return SYSTEM_PROMPT + f"\n\n## Required JSON Schema\nOutput must match this exact schema:\n```json\n{schema_json}\n```"


class ZhipuLLMAdapter:
    """Thin transport adapter for Zhipu OpenAI-compatible API.

    Uses response_format=json_object + runtime Pydantic validation.
    Does NOT support full JSON Schema constrained generation (Zhipu API
    only supports json_object mode), but model_validate_json() catches
    invalid output at runtime.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = ZHIPU_BASE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._schema = DirectorDecision.model_json_schema()
        self.schema_version = f"pydantic_{DirectorDecision.model_config.get('title', 'DirectorDecision')}"

    @property
    def json_schema(self) -> dict:
        return self._schema

    def generate_decision(
        self,
        user_input: str,
        context: str | None = None,
        decision_id: str | None = None,
    ) -> LLMResult:
        if decision_id is None:
            decision_id = f"sd_{int(time.time() * 1000)}"

        user_message = user_input
        if context:
            user_message = f"Available context:\n{context}\n\nDirector request:\n{user_input}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _build_system_prompt(self._schema)},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        start = time.time()
        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = ""
            # 窄例外：读错误响应体失败的已知来源是 socket 层读错误与
            # HTTP 层不完整读取；外层仍返回 fail-closed，不吞其他异常。
            try:
                body = e.read().decode("utf-8", errors="replace")
            except (OSError, http_client.HTTPException):
                pass
            latency = int((time.time() - start) * 1000)
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: zhipu HTTP {e.code}: {body[:200]}",
                model=self.model,
                latency_ms=latency,
            )
        except urllib.error.URLError as e:
            latency = int((time.time() - start) * 1000)
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: zhipu connection failed: {e}",
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

        # Parse API response → extract content
        try:
            api_resp = json.loads(raw)
            content = api_resp["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: failed to parse API response: {e}",
                model=self.model,
                raw_response=raw[:500],
                latency_ms=latency,
            )

        # Runtime Pydantic validation — same canonical schema
        try:
            decision = DirectorDecision.model_validate_json(content)
        except Exception as e:
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: schema validation failed: {e}",
                model=self.model,
                raw_response=content[:500],
                latency_ms=latency,
                schema_constrained=False,
            )

        return LLMResult(
            decision=decision,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            schema_version=self.schema_version,
            raw_response=content,
            latency_ms=latency,
            schema_constrained=True,
        )
