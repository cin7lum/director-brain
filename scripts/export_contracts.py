"""导出 02 的数据契约（机器可读插拔接口）。

02 作为大项目中的可插拔模块、同时可独立成产品，其与外界的全部约定
落在各版本化 Pydantic 模型上。本脚本把它们的 JSON Schema 导出到
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
from director_brain.models.director_plan import (  # noqa: E402
    DirectorDecisionPlan,
    ProjectDirectorShadowComparison,
)
from director_brain.models.edl import EditorialDecisionList  # noqa: E402
from director_brain.models.film_context import FilmContextSnapshot  # noqa: E402
from director_brain.models.film_observation import FilmObservation  # noqa: E402
from director_brain.models.project import FilmProjectManifest  # noqa: E402
from director_brain.models.story_graph import StoryGraph  # noqa: E402
from director_brain.models.project_story_graph import (  # noqa: E402
    ProjectStoryGraph,
    ProjectStoryGraphView,
)
from director_brain.models.project_story_link_review import (  # noqa: E402
    ProjectStoryLinkReview,
)
from director_brain.models.project_story_link_comparison import (  # noqa: E402
    ProjectStoryLinkComparison,
)
from director_brain.models.project_story_link_candidates import (  # noqa: E402
    ProjectStoryLinkCandidatePage,
)
from director_brain.models.project_story_mention_review import (  # noqa: E402
    ProjectStoryMentionReview,
)
from director_brain.models.project_story_mention_preview import (  # noqa: E402
    ProjectStoryMentionPreview,
)
from director_brain.models.project_story_link_candidate_preview import (  # noqa: E402
    ProjectStoryLinkCandidatePreview,
)
from director_brain.models.project_director_shadow_comparison import (  # noqa: E402
    ProjectDirectorShadowComparisonRecord,
)

#: 边界契约模型：模块名 → (模型类, 在大项目中的角色)
CONTRACT_MODELS: dict[str, tuple[type, str]] = {
    "film_observation": (
        FilmObservation, "上游输入：Film Intelligence 公共感知层的观测"),
    "film_context": (
        FilmContextSnapshot, "中间产物：项目/素材级可溯源 Film Context"),
    "film_project_manifest": (
        FilmProjectManifest, "上游输入：带边界声明、资产顺序与权限证据的项目清单"),
    "director_brief": (
        DirectorBrief, "中间产物：用户意图 × 素材事实的编译结果"),
    "story_graph": (
        StoryGraph,
        "中间产物：素材时间结构与带来源的本地人物/事件观测；不确认跨素材身份"),
    "project_story_graph": (
        ProjectStoryGraph,
        "中间产物：保留独立素材时间线的项目级结构集合；不含跨素材关系推断"),
    "project_story_graph_view": (
        ProjectStoryGraphView,
        "只读视图：精确项目结构集合与当前调用方声明的跨素材关系快照组合；不执行自动关系推断"),
    "project_story_link_review": (
        ProjectStoryLinkReview,
        "调用方声明的跨素材身份分组与证据绑定语义关系；自动推断保持关闭"),
    "project_story_link_comparison": (
        ProjectStoryLinkComparison,
        "本地 VLM 生成的未审核逐对关系比较；不构成身份确认"),
    "project_story_link_candidate_page": (
        ProjectStoryLinkCandidatePage,
        "基于当前项目 StoryGraph 的全量跨素材候选页；可选本地 shadow 排序但不推断关系"),
    "project_story_mention_review": (
        ProjectStoryMentionReview,
        "调用方对精确素材内人物/事件 mention 的接受、拒绝或内容纠正快照；不改写原始观察"),
    "project_story_mention_preview": (
        ProjectStoryMentionPreview,
        "经本地授权、源哈希复核的临时素材内人物/事件可视预览；不持久化或推断跨素材关系"),
    "project_story_link_candidate_preview": (
        ProjectStoryLinkCandidatePreview,
        "经本地授权、双源哈希复核的临时跨素材候选对可视预览；不持久化或推断关系"),
    "editorial_decision_list": (
        EditorialDecisionList, "下游输出：交给 03/04 的可执行剪辑清单"),
    "director_decision_plan": (
        DirectorDecisionPlan, "下游输出：导演决策依据（含 rationale/confidence）"),
    "project_director_shadow_comparison": (
        ProjectDirectorShadowComparison,
        "SHADOW 输出：未排序的项目导演策略 Plan/EDL 比较，不可确认或执行"),
    "project_director_shadow_comparison_record": (
        ProjectDirectorShadowComparisonRecord,
        "本机持久化的 SHADOW 项目策略比较及精确来源绑定；只支持恢复读取，不可确认或执行"),
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
