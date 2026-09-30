"""Thin Volcano Ark (Doubao) model provider adapter for DirectorDecision.

OpenAI-compatible HTTP API (ark.cn-beijing.volces.com/api/v3) with
response_format=json_object. Runtime validation via
DirectorDecision.model_validate_json().

This is intentionally a transport-only adapter — same discipline as
:mod:`director_brain.zhipu_adapter`. No semantic repair, no keyword patch,
no model-specific answer correction. Semantic logic remains in
SemanticDirectorReasoner / the model.

Schema source of truth: DirectorDecision.model_json_schema() — same
canonical schema used by ollama/zhipu adapters. No second schema copy.

阶段 7（模型准入）：按市调结论先测 doubao-seed-2.0-lite（单轮基准约
0.1 元），不过线换 doubao-seed-2.1-pro。API key 从环境变量 ARK_API_KEY
读取（.env 本地文件，不入库）。
"""
from __future__ import annotations

import http.client as http_client
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from director_brain.input_sanitizer import sanitize_untrusted
from director_brain.llm_adapter import LLMResult, SYSTEM_PROMPT, PROMPT_VERSION
from director_brain.models.director_decision import DirectorDecision

ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2.0-lite"
DEFAULT_TIMEOUT = 60


def _build_system_prompt(schema: dict) -> str:
    """Append canonical schema for APIs without schema-constrained generation.

    Same transport adaptation as the zhipu adapter (documented in
    KNOWN_ISSUES): json_object mode + schema in system prompt. Not a
    semantic prompt change — the frozen SYSTEM_PROMPT text is untouched.
    """
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    return SYSTEM_PROMPT + (
        f"\n\n## Required JSON Schema\nOutput must match this exact schema:\n"
        f"```json\n{schema_json}\n```"
    )


@dataclass
class ArkLLMAdapter:
    """Thin transport adapter for Volcano Ark OpenAI-compatible API."""

    api_key: str
    model: str = DEFAULT_ARK_MODEL
    base_url: str = ARK_BASE
    timeout: int = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._schema = DirectorDecision.model_json_schema()
        self.schema_version = (
            f"pydantic_{DirectorDecision.model_config.get('title', 'DirectorDecision')}"
        )

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

        # S5 安全边界：用户请求与上下文（可能携带视频内容派生文本）均不可信
        safe_request = sanitize_untrusted(user_input)
        safe_context = sanitize_untrusted(context) if context else None
        user_message = safe_request
        if safe_context:
            user_message = f"Available context:\n{safe_context}\n\nDirector request:\n{safe_request}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _build_system_prompt(self._schema)},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 2048,
            # 传输层配置（方舟官方参数）：seed 系列默认深度思考与
            # json_object 约束生成冲突，实测约 50% 空内容——关闭 thinking
            # 后消除。不是 prompt/用例/金标修改。
            "thinking": {"type": "disabled"},
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
                error=f"SEMANTIC_REASONER_UNAVAILABLE: ark HTTP {e.code}: {body[:200]}",
                model=self.model,
                latency_ms=latency,
            )
        except urllib.error.URLError as e:
            latency = int((time.time() - start) * 1000)
            return LLMResult(
                decision=None,
                error=f"SEMANTIC_REASONER_UNAVAILABLE: ark connection failed: {e}",
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
        # 传输层清理：json_object 模式下模型偶发用 markdown 围栏包裹 JSON
        # （实测 ~20% 用例），剥围栏属传输适配，不改语义内容。
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        try:
            decision = DirectorDecision.model_validate_json(cleaned)
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
