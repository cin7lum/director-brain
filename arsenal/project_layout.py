"""Arsenal 项目目录结构。

定义标准项目布局，负责创建与读取项目根目录及各子目录路径。
不涉及媒体内容，仅做目录骨架。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

#: 项目内固定子目录名
_SUBDIRS = (
    "media/originals",
    "media/proxies",
    "media/keyframes",
    "analysis",
    "exports",
)


def create_project(root: str, project_id: str) -> Path:
    """创建标准项目目录结构（幂等）。

    布局::

        <root>/<project_id>/
          project_manifest.json
          media/
            originals/      # 原始素材复制
            proxies/        # 低分辨率代理视频
            keyframes/      # 关键帧截图
          analysis/         # 分析结果缓存
          exports/          # 导出产物

    已存在的目录不报错；``project_manifest.json`` 记录 project_id / 创建时间 / 版本。

    Args:
        root: 项目根父目录。
        project_id: 项目 ID，作为子目录名。

    Returns:
        项目根路径（Path 对象）。
    """
    project_root = Path(root) / project_id
    for sub in _SUBDIRS:
        (project_root / sub).mkdir(parents=True, exist_ok=True)

    manifest = {
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0",
    }
    (project_root / "project_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return project_root


def get_project_paths(root: str, project_id: str) -> dict:
    """返回项目各子目录路径（不创建目录）。

    Args:
        root: 项目根父目录。
        project_id: 项目 ID。

    Returns:
        含以下 key 的 dict，value 均为 Path 对象：
        ``root`` / ``media_originals`` / ``media_proxies`` / ``keyframes`` /
        ``analysis`` / ``exports`` / ``manifest``。
    """
    project_root = Path(root) / project_id
    return {
        "root": project_root,
        "media_originals": project_root / "media" / "originals",
        "media_proxies": project_root / "media" / "proxies",
        "keyframes": project_root / "media" / "keyframes",
        "analysis": project_root / "analysis",
        "exports": project_root / "exports",
        "manifest": project_root / "project_manifest.json",
    }
