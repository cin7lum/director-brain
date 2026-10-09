from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

from director_brain.llm_adapter import (
    LLMAdapter, LLMStructuredOutputError, LLMTransportError, post_chat_json,
)


def test_structured_chat_rejects_token_limit_even_when_content_is_valid_json():
    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.server.call_count += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "choices": [{
                    "finish_reason": "length",
                    "message": {"content": '{"private_marker":true}'},
                }],
            }).encode())

        def log_message(self, *_args):
            pass

    server, thread = _serve(ModelHandler)
    server.call_count = 0
    try:
        with pytest.raises(LLMStructuredOutputError) as exc_info:
            post_chat_json(
                f"http://127.0.0.1:{server.server_port}/v1",
                "local-only", "test-model", "system", "user",
                response_schema={"type": "object"},
            )
        assert exc_info.value.failure_code == "provider_output_truncated"
        assert "private_marker" not in str(exc_info.value)
        assert server.call_count == 1
    finally:
        _stop(server, thread)


def _serve(handler_type):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_loopback_chat_bypasses_configured_proxy(monkeypatch):
    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.server.model_requests += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "choices": [{"message": {"content": '{"ok":true}'}}],
            }).encode())

        def log_message(self, *_args):
            pass

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.server.proxy_requests += 1
            self.send_error(502)

        def log_message(self, *_args):
            pass

    model_server, model_thread = _serve(ModelHandler)
    proxy_server, proxy_thread = _serve(ProxyHandler)
    model_server.model_requests = 0
    proxy_server.proxy_requests = 0
    try:
        proxy_url = f"http://127.0.0.1:{proxy_server.server_port}"
        monkeypatch.setattr(urllib.request, "getproxies", lambda: {"http": proxy_url})
        monkeypatch.setattr(urllib.request, "proxy_bypass", lambda _host: False)
        monkeypatch.setattr(urllib.request, "_opener", None)

        content = post_chat_json(
            f"http://127.0.0.1:{model_server.server_port}/v1",
            "local-only", "test-model", "system", "user", timeout=3,
        )

        assert content == '{"ok":true}'
        assert model_server.model_requests == 1
        assert proxy_server.proxy_requests == 0
    finally:
        _stop(model_server, model_thread)
        _stop(proxy_server, proxy_thread)


def test_chat_captures_only_provider_reported_identity_metadata():
    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "model": "provider/model-v7",
                "system_fingerprint": "fp_123",
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "debug_text": "must not enter provenance",
            }).encode())

        def log_message(self, *_args):
            pass

    server, thread = _serve(ModelHandler)
    metadata = {}
    try:
        content = post_chat_json(
            f"http://127.0.0.1:{server.server_port}/v1",
            "local-only", "requested-alias", "system", "user",
            response_metadata=metadata,
        )
        assert content == '{"ok":true}'
        assert metadata == {
            "model": "provider/model-v7",
            "system_fingerprint": "fp_123",
        }
        assert "debug_text" not in metadata
        assert "ok" not in json.dumps(metadata)
    finally:
        _stop(server, thread)


def test_loopback_chat_sends_openai_json_schema_response_format():
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "pattern": r"\S"},
            "code": {"type": "string", "pattern": "^[A-Z]{2}$"},
            "items": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {"type": "string"},
            },
        },
        "required": ["label", "code", "items"],
    }

    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.server.requests.append(json.loads(body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "choices": [{"message": {"content": (
                    '{"label":"ok","code":"AB","items":["a","b"]}'
                )}}],
            }).encode())

        def log_message(self, *_args):
            pass

    server, thread = _serve(ModelHandler)
    server.requests = []
    try:
        content = post_chat_json(
            f"http://127.0.0.1:{server.server_port}/v1",
            "local-only", "test-model", "system", "user",
            response_schema=schema,
        )
        assert content == '{"label":"ok","code":"AB","items":["a","b"]}'
        assert len(server.requests) == 1
        assert server.requests[0]["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "director_brain_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "code": {"type": "string", "pattern": "^[A-Z]{2}$"},
                        "items": {
                            "type": "array", "minItems": 2, "maxItems": 2,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["label", "code", "items"],
                },
            },
        }
        assert schema["properties"]["label"]["pattern"] == r"\S"
    finally:
        _stop(server, thread)


def test_loopback_chat_schema_rejection_fails_closed_without_json_retry():
    class RejectSchemaHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.server.requests.append(json.loads(body))
            self.send_error(400)

        def log_message(self, *_args):
            pass

    server, thread = _serve(RejectSchemaHandler)
    server.requests = []
    try:
        with pytest.raises(
            LLMTransportError, match="chat structured schema http 400") as exc_info:
            post_chat_json(
                f"http://127.0.0.1:{server.server_port}/v1",
                "local-only", "test-model", "system", "user",
                response_schema={"type": "object", "properties": {}},
            )
        assert exc_info.value.failure_code == "provider_http_error"
        assert len(server.requests) == 1
        assert server.requests[0]["response_format"]["type"] == "json_schema"
    finally:
        _stop(server, thread)


def test_loopback_chat_does_not_follow_redirects():
    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(302)
            self.send_header("Location", self.server.redirect_target)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    class DestinationHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.server.received += 1
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    destination, destination_thread = _serve(DestinationHandler)
    redirect, redirect_thread = _serve(RedirectHandler)
    destination.received = 0
    redirect.redirect_target = f"http://127.0.0.1:{destination.server_port}/collect"
    try:
        with pytest.raises(LLMTransportError, match="chat http 302") as exc_info:
            post_chat_json(
                f"http://127.0.0.1:{redirect.server_port}/v1",
                "local-only", "test-model", "system", "user", timeout=3,
            )
        assert exc_info.value.failure_code == "provider_http_error"
        assert destination.received == 0
    finally:
        _stop(redirect, redirect_thread)
        _stop(destination, destination_thread)


def test_loopback_chat_connection_failure_has_stable_code(monkeypatch):
    def fail_request(_request, *, timeout):
        raise urllib.error.URLError("local test connection refusal")

    monkeypatch.setattr("director_brain.llm_adapter._open_local_request", fail_request)
    with pytest.raises(LLMTransportError) as exc_info:
        post_chat_json(
            "http://127.0.0.1:11434/v1",
            "local-only", "test-model", "system", "user", timeout=1,
        )
    assert exc_info.value.failure_code == "provider_connection_error"


def test_loopback_chat_invalid_provider_envelope_has_stable_code(monkeypatch):
    response = MagicMock()
    response.read.return_value = b"not-json"
    response.__enter__.return_value = response
    monkeypatch.setattr(
        "director_brain.llm_adapter._open_local_request",
        lambda _request, timeout: response,
    )

    with pytest.raises(LLMTransportError) as exc_info:
        post_chat_json(
            "http://127.0.0.1:11434/v1",
            "local-only", "test-model", "system", "user", timeout=1,
        )
    assert exc_info.value.failure_code == "provider_exchange_error"


def test_loopback_chat_empty_content_has_stable_code(monkeypatch):
    response = MagicMock()
    response.read.return_value = json.dumps({
        "choices": [{"message": {"content": "  "}}],
    }).encode()
    response.__enter__.return_value = response
    monkeypatch.setattr(
        "director_brain.llm_adapter._open_local_request",
        lambda _request, timeout: response,
    )

    with pytest.raises(LLMTransportError) as exc_info:
        post_chat_json(
            "http://127.0.0.1:11434/v1",
            "local-only", "test-model", "system", "user", timeout=1,
        )
    assert exc_info.value.failure_code == "provider_empty_response"


def test_remote_chat_keeps_existing_urlopen_transport():
    response = MagicMock()
    response.read.return_value = json.dumps({
        "choices": [{"message": {"content": '{"remote":true}'}}],
    }).encode()
    response.__enter__.return_value = response
    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        content = post_chat_json(
            "https://api.example.test/v1", "remote-key", "test-model",
            "system", "user",
        )
    assert content == '{"remote":true}'
    urlopen.assert_called_once()


def test_ollama_adapter_rejects_non_loopback_base_url():
    with pytest.raises(ValueError, match="loopback"):
        LLMAdapter(base_url="https://api.example.test")
