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
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from observation_service.vlm_adapter import (
    VLMAdapter,
    OBSERVED,
    FAILED,
    FT_NETWORK,
    FT_RATE_LIMIT,
    FT_DECODE,
    FT_PARSE,
    FT_EMPTY,
    FT_UNKNOWN,
)
from observation_service.ollama_vlm_adapter import OllamaVLMAdapter
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
