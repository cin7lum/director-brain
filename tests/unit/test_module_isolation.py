"""P2-c 模块化与插拔验证测试。

02 的两条架构承诺在此固化为可执行断言：
1. **可插拔模块**：director_brain 对 03（arsenal）/04（davinci_execution）/
   05（fql）零 import——拔掉兄弟模块，02 照常工作；
2. **数据契约稳定**：contracts/ 目录的 JSON Schema 与内存生成一致
   （契约漂移必须显式重导出进评审）。

另验证推理器工厂的第三方注册插拔点。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BRAIN_DIR = _PROJECT_ROOT / "director_brain"

#: 02 不得依赖的兄弟模块（03/04/05）——插拔边界
_FORBIDDEN_TOP_MODULES = {"arsenal", "davinci_execution", "fql"}


def _brain_imports() -> set[str]:
    """AST 解析 director_brain 全部 .py 的顶层 import 模块名。"""
    modules: set[str] = set()
    for py in sorted(_BRAIN_DIR.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def test_director_brain_does_not_import_sibling_modules():
    """插拔红线：02 对 03/04/05 零依赖（可独立运行、可整体替换）。"""
    imports = _brain_imports()
    violated = imports & _FORBIDDEN_TOP_MODULES
    assert violated == set(), f"director_brain 违规依赖兄弟模块: {violated}"


def test_contracts_match_committed_schemas():
    """契约漂移闸门：内存 schema 必须与 contracts/ 已提交文件逐字节一致。"""
    sys_path = str(_PROJECT_ROOT)
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    from scripts.export_contracts import CONTRACTS_DIR, build_contracts

    assert CONTRACTS_DIR.is_dir(), "contracts/ 未导出——先跑 scripts/export_contracts.py"
    contracts = build_contracts()
    assert "film_context.schema.json" in contracts
    assert "film_project_manifest.schema.json" in contracts
    context_schema = json.loads(contracts["film_context.schema.json"])["schema"]
    assert "asset_coverage" in context_schema["properties"]
    assert "timeline_scope" in context_schema["properties"]
    manifest_schema = json.loads(
        contracts["film_project_manifest.schema.json"])["schema"]
    assert "assets" in manifest_schema["properties"]
    from director_brain.models.project import ProjectBoundaryBasis
    assert ProjectBoundaryBasis.DATASET_EVENT_ID.value == "dataset_event_id"
    assert ProjectBoundaryBasis.DATASET_RECORDING_ID.value == "dataset_recording_id"
    for fname, text in contracts.items():
        committed = (CONTRACTS_DIR / fname).read_text(encoding="utf-8").rstrip("\n")
        assert committed == text, (
            f"契约漂移: {fname} 与已提交版本不一致——"
            f"请重跑 python scripts/export_contracts.py 并在变更日志中声明契约变更"
        )


def test_contract_manifest_hashes_match():
    from scripts.export_contracts import CONTRACTS_DIR, build_contracts
    manifest = json.loads((CONTRACTS_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    files = build_contracts()
    assert set(manifest["contracts"]) == set(files)
    for fname, text in files.items():
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert manifest["contracts"][fname] == digest


def test_reasoner_factory_accepts_third_party_registration():
    """插拔点：register_reasoner 注册自定义推理器，工厂按名构造。"""
    from director_brain.director_reasoner import (
        DirectorReasoner,
        _REASONER_REGISTRY,
        get_director_reasoner,
        register_reasoner,
    )

    class MyReasoner(DirectorReasoner):
        def generate_plan(self, brief, graph, observations):
            raise NotImplementedError("test stub")

    try:
        register_reasoner("test_plugin", lambda **kw: MyReasoner())
        reasoner = get_director_reasoner("test_plugin")
        assert isinstance(reasoner, MyReasoner)
    finally:
        _REASONER_REGISTRY.pop("test_plugin", None)

    with pytest.raises(ValueError):
        get_director_reasoner("never_registered_strategy")
