"""ZhipuLLMAdapter 传输适配器单元测试（阶段5：窄例外）。

不打真实网络：urlopen 打桩。覆盖：
1. HTTPError 响应体可读 → error 含状态码与响应体摘要（fail-closed 返回）
2. HTTPError 响应体读取抛 OSError → 仍 fail-closed 返回，不向上抛
   （历史行为是宽例外 except Exception: pass，被门禁判红，已窄化）
3. URLError → fail-closed 返回
"""
from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest

from director_brain.zhipu_adapter import ZhipuLLMAdapter


@pytest.fixture()
def adapter() -> ZhipuLLMAdapter:
    return ZhipuLLMAdapter(api_key="test-key", model="glm-4-flash", timeout=5)


def _http_error(code: int, *, read_raises: Exception | None = None,
                body: bytes = b'{"error":"bad"}') -> urllib.error.HTTPError:
    err = urllib.error.HTTPError(
        url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
        code=code, msg="error", hdrs=None, fp=None,
    )
    if read_raises is not None:
        err.read = lambda *a, **kw: (_ for _ in ()).throw(read_raises)
    else:
        err.read = lambda *a, **kw: body
    return err


def test_http_error_body_included_in_fail_closed_result(adapter):
    with patch("urllib.request.urlopen",
               side_effect=_http_error(429, body=b'{"error":"rate limited"}')):
        result = adapter.generate_decision("让节奏快一点")
    assert result.decision is None
    assert "SEMANTIC_REASONER_UNAVAILABLE" in result.error
    assert "429" in result.error
    assert "rate limited" in result.error


def test_http_error_body_read_oserror_still_fail_closed(adapter):
    """读错误响应体抛 OSError → 窄例外吞掉、整体仍 fail-closed 返回。"""
    with patch("urllib.request.urlopen",
               side_effect=_http_error(500, read_raises=OSError("connection reset"))):
        result = adapter.generate_decision("让节奏快一点")
    assert result.decision is None
    assert "SEMANTIC_REASONER_UNAVAILABLE" in result.error
    assert "500" in result.error


def test_url_error_fail_closed(adapter):
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("dns failure")):
        result = adapter.generate_decision("让节奏快一点")
    assert result.decision is None
    assert "SEMANTIC_REASONER_UNAVAILABLE" in result.error
