"""director_brain 共享工具函数。

提供全项目统一的哈希辅助函数，避免各模块重复实现 sha256 / 截断逻辑。
"""
from __future__ import annotations

import hashlib


def short_hash(text: str, length: int = 16) -> str:
    """对文本计算 SHA-256 并截断为前 ``length`` 位十六进制字符串。

    全项目统一使用此函数生成短 ID（brief_id / edl_id / plan_id / shot_id /
    context_id 等），保证算法与截断长度一致。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str) -> str:
    """流式（1MB 块）计算文件 SHA-256，返回完整 64 位十六进制哈希。

    文件不存在或不可读时抛出 OSError，不静默返回空串。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
