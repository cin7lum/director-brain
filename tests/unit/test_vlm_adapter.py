"""M1.2 VLM 适配器测试。

严格按三窗口分离：测试先行。覆盖 6 个场景：
1. 抽象基类 VLMAdapter 不可直接实例化（抛 TypeError）
2. 工厂函数 get_vlm_adapter 创建 ollama / zhipu 适配器，不支持的 provider 抛 ValueError
3. extract_keyframe 从镜头中点（而非首帧）抽帧，输出文件存在且非空
4. OllamaVLMAdapter 网络调用失败时返回 degraded dict（status=FAILED, degraded=True），不抛异常
5. OllamaVLMAdapter 成功时返回结构化标签（shot_function 在合法枚举内，frame_description 非空）
6. ollama 服务未运行（连接被拒）时不阻塞，5 秒内返回 degraded 结果

所有 HTTP 调用通过 unittest.mock.patch 替换 urllib.request.urlopen，不实际访问外部服务。
唯一例外是场景 6，故意对不存在的端口发起真实连接以验证 fail-closed 路径。
"""
from __future__ import annotations

import json
import io
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from observation_service.vlm_adapter import (
    VLMAdapter,
    OBSERVED,
    FAILED,
    FT_NETWORK,
    FT_RATE_LIMIT,
    FT_REQUEST,
    FT_DECODE,
    FT_PARSE,
    FT_EMPTY,
    FT_UNKNOWN,
)
from observation_service.ollama_vlm_adapter import (
    LocalVLMRuntimeBindingError,
    OllamaVLMAdapter,
    PROJECT_LINK_COMPARISON_NUM_CTX,
    SEMANTIC_NUM_CTX,
    SEMANTIC_GENERATION_PROFILE,
    PROJECT_LINK_COMPARISON_GENERATION_PROFILE,
)
from observation_service.zhipu_vlm_adapter import ZhipuVLMAdapter
from observation_service.vlm_factory import get_vlm_adapter
from observation_service.keyframe import extract_keyframe


# ---------------------------------------------------------------------------
# 合法枚举值（与适配器实现保持一致）
# ---------------------------------------------------------------------------

_VALID_SHOT_FUNCTIONS = {
    "ESTABLISHING", "ACTION", "REACTION", "DETAIL",
    "TRANSITION", "ATMOSPHERIC_EVIDENCE", "SENSORY_INSERT",
}
_VALID_MOTION = {"static", "subtle", "burst"}
_VALID_ROLES = {"hero", "support", "transition", "broll", "discard"}


def _make_dummy_image() -> str:
    """创建一个临时假图片文件（内容随意，因为 HTTP 调用被 mock）。"""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.write(fd, b"\xff\xd8\xff\xe0fake-jpeg-bytes-for-mock-test")
    os.close(fd)
    return path


def _make_test_video(duration_sec: int = 3, size: str = "320x240") -> str:
    """用 ffmpeg testsrc 生成一段测试视频，返回路径。"""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    cmd = [
        "ffmpeg", "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size={size}:rate=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        path, "-y",
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return path


def _json_http_response(payload: dict):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return response


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.com",
        "http://user@127.0.0.1:11434",
        "http://127.0.0.1:11434/proxy",
    ],
)
def test_project_local_adapter_rejects_non_loopback_or_ambiguous_urls(base_url):
    with pytest.raises(LocalVLMRuntimeBindingError):
        OllamaVLMAdapter(
            base_url=base_url,
            model_digest="a" * 64,
            runtime_version="0.32.14",
            enforce_loopback=True,
        )


def test_project_local_adapter_bypasses_ambient_proxy_settings():
    adapter = OllamaVLMAdapter(
        model_digest="a" * 64,
        runtime_version="0.32.14",
        enforce_loopback=True,
    )
    proxy_handlers = [
        handler for handler in adapter._opener.handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    # urllib omits an empty ProxyHandler from the chain, and build_opener then
    # suppresses its ambient-proxy default. No proxy handler means direct I/O.
    assert proxy_handlers == []


def test_project_local_adapter_binds_exact_runtime_and_model_digest():
    digest = "a" * 64
    adapter = OllamaVLMAdapter(
        model="qwen3-vl:4b",
        model_digest=digest,
        runtime_version="0.32.14",
        enforce_loopback=True,
    )
    with patch.object(adapter._opener, "open", side_effect=[
        _json_http_response({"version": "0.32.14"}),
        _json_http_response({"models": [{
            "name": "qwen3-vl:4b", "digest": digest,
        }]}),
    ]) as open_local:
        adapter.verify_runtime_binding()
    assert adapter._runtime_binding_verified is True
    assert open_local.call_count == 2


def test_project_local_adapter_fails_closed_instead_of_changing_generation_profile():
    image = _make_dummy_image()
    try:
        adapter = OllamaVLMAdapter(
            model_digest="a" * 64,
            runtime_version="0.32.14",
            enforce_loopback=True,
        )
        adapter._runtime_binding_verified = True
        with patch.object(
            adapter, "_chat",
            return_value=("", ValueError("unsupported format option")),
        ) as chat:
            result = adapter.analyze_frames([image])
        assert result["status"] == FAILED
        assert result["degraded"] is True
        chat.assert_called_once()
    finally:
        os.remove(image)


@pytest.mark.parametrize(
    "version_payload,tags_payload",
    [
        ({"version": "0.32.15"}, {"models": [{
            "name": "qwen3-vl:4b", "digest": "a" * 64,
        }]}),
        ({"version": "0.32.14"}, {"models": [{
            "name": "qwen3-vl:4b", "digest": "b" * 64,
        }]}),
    ],
)
def test_project_local_adapter_fails_closed_on_runtime_or_digest_drift(
    version_payload, tags_payload
):
    adapter = OllamaVLMAdapter(
        model="qwen3-vl:4b",
        model_digest="a" * 64,
        runtime_version="0.32.14",
        enforce_loopback=True,
    )
    with patch.object(adapter._opener, "open", side_effect=[
        _json_http_response(version_payload),
        _json_http_response(tags_payload),
    ]):
        with pytest.raises(LocalVLMRuntimeBindingError):
            adapter.verify_runtime_binding()


def test_semantic_multiframe_profile_pins_context_size_in_request_and_fingerprint():
    image = _make_dummy_image()
    seen = {}
    adapter = OllamaVLMAdapter()
    response = {
        "function": "SENSORY_INSERT",
        "role": "broll",
        "motion": "subtle",
        "narrative": "transition",
        "emotion": "neutral",
        "action": "sensory",
        "desc": "transport contract fixture",
        "temporal": "",
    }

    def fake_chat(body):
        seen["options"] = dict(body["options"])
        return json.dumps(response), None

    try:
        with patch.object(adapter, "_chat", side_effect=fake_chat):
            result = adapter.analyze_frames([image, image, image])
        assert result["status"] == OBSERVED
        assert seen["options"]["num_ctx"] == SEMANTIC_NUM_CTX == 8192
        assert f"num_ctx={SEMANTIC_NUM_CTX}" in SEMANTIC_GENERATION_PROFILE
    finally:
        os.remove(image)


def test_cross_asset_comparison_profile_pins_six_image_context_size():
    image = _make_dummy_image()
    seen = {}
    adapter = OllamaVLMAdapter()
    response = {
        "assessment": "insufficient_evidence",
        "evidence_for": [],
        "evidence_against": [],
        "limitation": "transport contract fixture",
    }

    def fake_chat(body):
        seen["options"] = dict(body["options"])
        return json.dumps(response), None

    try:
        with patch.object(adapter, "_chat", side_effect=fake_chat):
            result = adapter.compare_cross_asset_frames(
                [image, image, image],
                [image, image, image],
                relation_kind="event_identity",
                left_event_evidence={"scene_description": "source A fixture"},
                right_event_evidence={"scene_description": "source B fixture"},
            )
        assert result["status"] == OBSERVED
        assert seen["options"]["num_ctx"] == PROJECT_LINK_COMPARISON_NUM_CTX == 16384
        assert (
            f"num_ctx={PROJECT_LINK_COMPARISON_NUM_CTX}"
            in PROJECT_LINK_COMPARISON_GENERATION_PROFILE
        )
    finally:
        os.remove(image)


def test_ollama_http_400_preserves_bounded_diagnostic_and_request_failure_type():
    image = _make_dummy_image()
    adapter = OllamaVLMAdapter()
    error_body = json.dumps({
        "error": {
            "code": 400,
            "type": "exceed_context_size_error",
            "message": "request exceeds the available context size",
        }
    }).encode("utf-8")
    error = urllib.error.HTTPError(
        "http://127.0.0.1:11434/api/chat",
        400,
        "Bad Request",
        hdrs=None,
        fp=io.BytesIO(error_body),
    )
    try:
        with patch.object(adapter, "_open", side_effect=error):
            result = adapter.analyze_frames([image, image, image])
        assert result["status"] == FAILED
        assert result["failure_type"] == FT_REQUEST
        assert "exceed_context_size_error" in result["degrade_reason"]
        assert "available context size" in result["degrade_reason"]
    finally:
        os.remove(image)


# ---------------------------------------------------------------------------
# 1. 抽象基类不可直接实例化
# ---------------------------------------------------------------------------

class TestAbstractBaseClass:
    """场景 1：VLMAdapter 含 @abstractmethod，直接实例化应抛 TypeError。"""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            VLMAdapter()


# ---------------------------------------------------------------------------
# 2. 工厂函数
# ---------------------------------------------------------------------------

class TestVLMFactory:
    """场景 2：get_vlm_adapter 按 provider 创建对应实例；未知 provider 抛 ValueError。"""

    def test_create_ollama(self):
        adapter = get_vlm_adapter("ollama")
        assert isinstance(adapter, OllamaVLMAdapter)

    def test_create_zhipu(self):
        adapter = get_vlm_adapter("zhipu", api_key="test-key")
        assert isinstance(adapter, ZhipuVLMAdapter)
        assert adapter.api_key == "test-key"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            get_vlm_adapter("nonexistent_provider")


# ---------------------------------------------------------------------------
# 3. extract_keyframe 从镜头中点抽帧
# ---------------------------------------------------------------------------

class TestExtractKeyframe:
    """场景 3：extract_keyframe 从镜头中点抽帧（不是首帧），输出文件存在且非空。"""

    def test_midpoint_extraction(self):
        video = _make_test_video(duration_sec=3)
        try:
            # 镜头 0~3 秒（3_000_000 us），中点 = 1.5s
            with extract_keyframe(video, 0, 3_000_000) as out_path:
                assert out_path != "", "抽取失败应返回空字符串"
                p = Path(out_path)
                assert p.exists(), f"输出文件不存在: {out_path}"
                assert p.stat().st_size > 0, "输出文件大小为 0"
        finally:
            if os.path.exists(video):
                os.remove(video)

    def test_extract_failure_returns_empty(self):
        """视频路径不存在时应返回空字符串，不抛异常。"""
        with extract_keyframe("/nonexistent/path.mp4", 0, 1_000_000) as result:
            assert result == ""


# ---------------------------------------------------------------------------
# 4. VLM 调用失败返回 degraded
# ---------------------------------------------------------------------------

class TestVLMDegradedOnFailure:
    """场景 4：mock urlopen 抛异常，analyze_frame 应返回 degraded dict，不抛异常。"""

    def test_network_error_returns_degraded(self):
        img = _make_dummy_image()
        try:
            adapter = OllamaVLMAdapter()
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = Exception("connection refused")
                result = adapter.analyze_frame(img)

            assert result["degraded"] is True
            assert result["status"] == FAILED
            assert result["confidence_type"] == "UNAVAILABLE"
        finally:
            os.remove(img)


# ---------------------------------------------------------------------------
# 5. VLM 成功返回结构化标签
# ---------------------------------------------------------------------------

class TestVLMSuccess:
    """场景 5：mock urlopen 返回有效 JSON，analyze_frame 应返回结构化标签。"""

    def test_successful_analysis(self):
        img = _make_dummy_image()
        try:
            # 构造 Ollama API 响应：message.content = 描述 + JSON
            reply_text = (
                "街道上一个人在阳光下行走，光影分明。\n"
                '{"shot_function": "ACTION", '
                '"sensory_wet_heat": 0.3, '
                '"sensory_mood_intensity": 0.7, '
                '"motion_amount": "subtle", '
                '"proposed_role_v2": "hero"}'
            )
            response_bytes = json.dumps({
                "message": {"content": reply_text}
            }).encode("utf-8")

            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = response_bytes

            adapter = OllamaVLMAdapter()
            with patch("urllib.request.urlopen", return_value=mock_resp):
                result = adapter.analyze_frame(img)

            assert result["degraded"] is False
            assert result["status"] == OBSERVED
            assert result["shot_function"] in _VALID_SHOT_FUNCTIONS
            assert result["shot_function"] == "ACTION"
            assert result["motion_amount"] in _VALID_MOTION
            assert result["proposed_role_v2"] in _VALID_ROLES
            assert result["frame_description"], "frame_description 不应为空"
            assert "街道" in result["frame_description"]
            assert result["sensory_wet_heat"] is not None
            assert result["_warnings"] == []
        finally:
            os.remove(img)


# ---------------------------------------------------------------------------
# 5b. 部分字段非法值被 fallback 时：degraded=True, status=OBSERVED, _warnings 记录
# ---------------------------------------------------------------------------

class TestPartialFallbackWarnings:
    """模型输出部分非法值时静默 fallback 不可接受：
    必须 degraded=True（但调用成功，status 仍为 OBSERVED），并在 _warnings 中
    记录被 fallback 的字段名与原始值。
    """

    def test_ollama_invalid_shot_function_warns(self):
        img = _make_dummy_image()
        try:
            reply_text = (
                "画面描述。\n"
                '{"shot_function": "UNKNOWN", '
                '"sensory_wet_heat": 0.3, '
                '"sensory_mood_intensity": 0.7, '
                '"motion_amount": "subtle", '
                '"proposed_role_v2": "hero"}'
            )
            response_bytes = json.dumps({
                "message": {"content": reply_text}
            }).encode("utf-8")
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = response_bytes

            adapter = OllamaVLMAdapter()
            with patch("urllib.request.urlopen", return_value=mock_resp):
                result = adapter.analyze_frame(img)

            assert result["status"] == OBSERVED
            assert result["degraded"] is True
            assert result["shot_function"] == "SENSORY_INSERT"
            joined = " | ".join(result["_warnings"])
            assert "shot_function" in joined
            assert "UNKNOWN" in joined
        finally:
            os.remove(img)

    def test_zhipu_invalid_shot_function_warns(self):
        img = _make_dummy_image()
        try:
            reply_text = (
                "画面描述。\n"
                '{"shot_function": "BOGUS", '
                '"sensory_wet_heat": 0.3, '
                '"sensory_mood_intensity": 0.7, '
                '"motion_amount": "subtle", '
                '"proposed_role_v2": "hero"}'
            )
            response_bytes = json.dumps({
                "choices": [{"message": {"content": reply_text}}]
            }).encode("utf-8")
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = response_bytes

            adapter = ZhipuVLMAdapter(api_key="test-key")
            with patch("urllib.request.urlopen", return_value=mock_resp):
                result = adapter.analyze_frame(img)

            assert result["status"] == OBSERVED
            assert result["degraded"] is True
            assert result["shot_function"] == "SENSORY_INSERT"
            joined = " | ".join(result["_warnings"])
            assert "shot_function" in joined
            assert "BOGUS" in joined
        finally:
            os.remove(img)


# ---------------------------------------------------------------------------
# 6. ollama 服务未运行不阻塞
# ---------------------------------------------------------------------------

class TestNoBlockingWhenOllamaDown:
    """场景 6：不 mock，对不存在的端口发起真实连接，验证 fail-closed 且 5 秒内返回。"""

    def test_returns_degraded_without_blocking(self):
        img = _make_dummy_image()
        try:
            # 指向一个几乎不可能被监听的高位端口
            adapter = OllamaVLMAdapter(
                base_url="http://localhost:59999",
                timeout=5.0,
            )
            start = time.time()
            result = adapter.analyze_frame(img)
            elapsed = time.time() - start

            assert result["degraded"] is True
            assert result["status"] == FAILED
            assert elapsed < 5.0, f"应在 5 秒内返回，实际耗时 {elapsed:.2f}s"
        finally:
            os.remove(img)
