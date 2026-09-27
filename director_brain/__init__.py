"""Director Brain（02）公共契约入口。

链上唯一拥有导演权的模块：把「用户想剪成什么」×「素材里有什么事实」
编译为可解释、可追溯、可修订的导演决策。不控制 DaVinci、不渲染、
不给自己参与的成片做最终裁判（评价权在 05/FQL）。

本包对外只承诺以下公共 API（T5 契约）：

- :func:`compile_brief`   —— 观测 + 意图 → DirectorBrief
- :func:`generate_plan`   —— Brief + 故事图 + 观测 → (EDL, DirectorDecisionPlan)
- :func:`propose_revision`—— FQL 反馈 / 目标 → RevisionProposal
- :func:`validate_plan`   —— EDL ↔ Plan 一致性校验（含负向判红能力）
- :func:`repair_plan`     —— 物理修复（镜头集合级问题返回 ABSTAIN）
- :data:`short_hash` / :data:`file_sha256` —— 全项目统一哈希工具

内部模块（带下划线前缀的模块与成员）不属于公共契约，重构可随时改动；
外部代码一律从本入口或子模块公共名导入。
"""
from director_brain.brief_compiler import compile_brief
from director_brain.plan_repair import RepairOutcome, repair_plan
from director_brain.plan_validator import validate_plan
from director_brain.revision_engine import propose_revision
from director_brain.utils import file_sha256, short_hash

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 契约 facade
    "compile_brief",
    "generate_plan",
    "propose_revision",
    "validate_plan",
    "repair_plan",
    "RepairOutcome",
    # 统一工具
    "short_hash",
    "file_sha256",
]


def generate_plan(brief, graph, observations, strategy: str = "heuristic"):
    """从 Brief + 故事图 + 观测产出 ``(edl, plan)``。

    strategy: ``heuristic``（确定性，默认）；``llm`` 为未实现骨架，
    调用会抛 NotImplementedError（见 director_reasoner）。
    """
    from director_brain.director_reasoner import get_director_reasoner

    return get_director_reasoner(strategy).generate_plan(brief, graph, observations)
