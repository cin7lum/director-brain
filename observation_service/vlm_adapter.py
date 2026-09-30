"""M1.2 VLM 适配器抽象基类与状态词汇表。

任何后端（Ollama / 智谱 / 其他 VLM）必须产出此契约定义的扁平 dict。
observation_service 上层（提案、编辑）不关心是哪个模型产出的结果。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# 状态词汇表
# ---------------------------------------------------------------------------
OBSERVED = "OBSERVED"
FAILED = "FAILED"

# ---------------------------------------------------------------------------
# 失败类型词汇表（status=FAILED 时使用）
# ---------------------------------------------------------------------------
FT_NETWORK = "NETWORK"
FT_RATE_LIMIT = "RATE_LIMIT"
FT_DECODE = "DECODE"
FT_PARSE = "PARSE"
FT_EMPTY = "EMPTY"
FT_UNKNOWN = "UNKNOWN_FAILURE"
FT_UNSUPPORTED = "UNSUPPORTED"


class VLMAdapter(ABC):
    """抽象 VLM 接口。一个方法：分析单帧 -> 结构化 dict。"""

    @abstractmethod
    def analyze_frame(self, image_path: str) -> dict:
        """分析单帧图片，返回扁平 dict。

        返回字段：
            shot_function: ESTABLISHING/ACTION/REACTION/DETAIL/TRANSITION/
                           ATMOSPHERIC_EVIDENCE/SENSORY_INSERT
            sensory_wet_heat: 0-1 浮点数或 None
            sensory_mood_intensity: 0-1 浮点数或 None
            motion_amount: static/subtle/burst
            proposed_role_v2: hero/support/transition/broll/discard
            frame_description: 中文描述字符串
            status: OBSERVED 或 FAILED
            degraded: bool
            confidence_type: SELF_REPORTED 或 UNAVAILABLE
        """
        ...

    def analyze_frames(self, image_paths: list[str]) -> dict:
        """分析多帧（P3 语义观测：镜头内 3 帧一次调用）。

        基类默认实现退化为**首帧单帧分析**——未实现多帧的后端（如 zhipu）
        经此保持协议兼容，调用方按返回值正常消费；多帧深度语义由后端
        自行实现（见 OllamaVLMAdapter.analyze_frames）。
        """
        if not image_paths:
            raise ValueError("analyze_frames 需要至少一帧图片")
        return self.analyze_frame(image_paths[0])
