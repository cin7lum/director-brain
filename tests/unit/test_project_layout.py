"""Arsenal 项目目录结构单元测试。

验证 create_project / get_project_paths 的目录创建、幂等性与 manifest 写入。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arsenal.project_layout import create_project, get_project_paths


class TestCreateProject:
    """create_project 行为。"""

    def test_creates_all_directories(self, tmp_path: Path) -> None:
        """create_project 后所有子目录都应存在。"""
        project_root = create_project(str(tmp_path), "demo_proj")

        assert project_root == Path(tmp_path) / "demo_proj"
        # media 及子目录
        assert (project_root / "media" / "originals").is_dir()
        assert (project_root / "media" / "proxies").is_dir()
        assert (project_root / "media" / "keyframes").is_dir()
        # 顶层目录
        assert (project_root / "analysis").is_dir()
        assert (project_root / "exports").is_dir()
        # manifest 文件
        assert (project_root / "project_manifest.json").is_file()

    def test_idempotent(self, tmp_path: Path) -> None:
        """重复调用 create_project 不应报错。"""
        create_project(str(tmp_path), "demo_proj")
        # 第二次调用不应抛异常
        project_root = create_project(str(tmp_path), "demo_proj")
        assert project_root.is_dir()
        assert (project_root / "media" / "originals").is_dir()

    def test_writes_manifest(self, tmp_path: Path) -> None:
        """project_manifest.json 应包含 project_id / created_at / version。"""
        project_root = create_project(str(tmp_path), "demo_proj")
        manifest = json.loads(
            (project_root / "project_manifest.json").read_text(encoding="utf-8")
        )

        assert manifest["project_id"] == "demo_proj"
        assert manifest["version"] == "1.0"
        # created_at 应为非空 ISO8601 字符串
        assert isinstance(manifest["created_at"], str)
        assert len(manifest["created_at"]) > 0


class TestGetProjectPaths:
    """get_project_paths 行为（只返回路径，不创建目录）。"""

    def test_returns_correct_paths(self, tmp_path: Path) -> None:
        """返回 dict 应包含所有预期 key，路径拼接正确。"""
        paths = get_project_paths(str(tmp_path), "demo_proj")

        expected_keys = {
            "root",
            "media_originals",
            "media_proxies",
            "keyframes",
            "analysis",
            "exports",
            "manifest",
        }
        assert set(paths.keys()) == expected_keys

        assert paths["root"] == Path(tmp_path) / "demo_proj"
        assert paths["media_originals"] == Path(tmp_path) / "demo_proj" / "media" / "originals"
        assert paths["media_proxies"] == Path(tmp_path) / "demo_proj" / "media" / "proxies"
        assert paths["keyframes"] == Path(tmp_path) / "demo_proj" / "media" / "keyframes"
        assert paths["analysis"] == Path(tmp_path) / "demo_proj" / "analysis"
        assert paths["exports"] == Path(tmp_path) / "demo_proj" / "exports"
        assert paths["manifest"] == Path(tmp_path) / "demo_proj" / "project_manifest.json"

    def test_does_not_create_directories(self, tmp_path: Path) -> None:
        """get_project_paths 不应创建任何目录。"""
        paths = get_project_paths(str(tmp_path), "demo_proj")
        assert not paths["root"].exists()
        assert not paths["media_originals"].exists()
