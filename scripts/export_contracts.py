"""导出 02 的数据契约（机器可读插拔接口）。

02 作为大项目中的可插拔模块、同时可独立成产品，其与外界的全部约定
落在 6 个 Pydantic 模型上。本脚本把它们的 JSON Schema 导出到
``contracts/``，供 03/04/05 与外部集成方生成类型代码、做契约测试。

契约漂移纪律：``tests/unit/test_module_isolation.py`` 会把内存生成的
schema 与已提交文件比对——**任何契约变更都必须显式重导出并进评审**，
不允许"顺手改个字段"。

用法：
    python scripts/export_contracts.py            # 写 contracts/ 并打印清单
    python scripts/export_contracts.py --stdout   # 只打印不写盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from director_brain import __version__  # noqa: E402
from director_brain.models.director_brief import DirectorBrief  # noqa: E402
from director_brain.models.director_decision import DirectorDecision  # noqa: E402
from director_brain.models.director_plan import DirectorDecisionPlan  # noqa: E402
from director_brain.models.edl import EditorialDecisionList  # noqa: E402
from director_brain.models.film_observation import FilmObservation  # noqa: E402
from director_brain.models.story_graph import StoryGraph  # noqa: E402

#: 边界契约模型：模块名 → (模型类, 在大项目中的角色)
CONTRACT_MODELS: dict[str, tuple[type, str]] = {
    "film_observation": (
        FilmObservation, "上游输入：Film Intelligence 公共感知层的观测"),
    "director_brief": (
        DirectorBrief, "中间产物：用户意图 × 素材事实的编译结果"),
    "story_graph": (
        StoryGraph, "中间产物：素材叙事结构（四幕骨架）"),
    "editorial_decision_list": (
        EditorialDecisionList, "下游输出：交给 03/04 的可执行剪辑清单"),
    "director_decision_plan": (
        DirectorDecisionPlan, "下游输出：导演决策依据（含 rationale/confidence）"),
    "director_decision": (
        DirectorDecision, "链 B 输出：语义导演决策（canonical，交 03 参数化）"),
}

CONTRACTS_DIR = _PROJECT_ROOT / "contracts"


def build_contracts() -> dict[str, str]:
    """在内存中生成全部契约 schema（文件名 → JSON 文本）。"""
    out: dict[str, str] = {}
    for name, (model, role) in CONTRACT_MODELS.items():
        schema = model.model_json_schema()
        doc = {
            "$contract_of": f"director_brain@{__version__}",
            "role": role,
            "schema": schema,
        }
        out[f"{name}.schema.json"] = json.dumps(doc, ensure_ascii=False, indent=2)
    return out


def write_contracts(directory: Path = CONTRACTS_DIR) -> list[str]:
    """写盘并返回生成文件名列表（含 manifest）。"""
    directory.mkdir(parents=True, exist_ok=True)
    files = build_contracts()
    manifest = {
        "director_brain_version": __version__,
        "contracts": {
            fname: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for fname, text in files.items()
        },
    }
    written: list[str] = []
    for fname, text in files.items():
        (directory / fname).write_text(text + "\n", encoding="utf-8")
        written.append(fname)
    manifest_name = "MANIFEST.json"
    (directory / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    written.append(manifest_name)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 02 数据契约 schema")
    parser.add_argument("--stdout", action="store_true", help="只打印不写盘")
    args = parser.parse_args()

    files = build_contracts()
    if args.stdout:
        for fname, text in files.items():
            print(f"===== {fname} =====")
            print(text)
        return 0
    written = write_contracts()
    print(f"契约已导出到 {CONTRACTS_DIR}：")
    for fname in written:
        print(f"  - {fname}")
    print("注意：修改任何边界模型后必须重跑本脚本并提交 diff（契约漂移测试会拦截）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
