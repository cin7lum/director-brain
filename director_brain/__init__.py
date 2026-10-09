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
from director_brain.brief_compiler import compile_brief, compile_project_brief
from director_brain.plan_repair import RepairOutcome, repair_plan
from director_brain.plan_validator import validate_plan
from director_brain.revision_engine import propose_revision
from director_brain.utils import file_sha256, short_hash

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 契约 facade
    "compile_brief",
    "compile_project_brief",
    "generate_plan",
    "generate_project_plan",
    "materialize_project_plan_from_candidate",
    "generate_project_shadow_plan",
    "generate_project_shadow_strategy_options",
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

    strategy: ``heuristic``（确定性，默认）；``llm`` 受
    ``director_strategy_reasoning`` 准入门控。该通路仍为 SHADOW，正式调用
    fail-closed；仅 ``generate_shadow_plan`` 可产生不可确认的本地比较产物。
    """
    from director_brain.director_reasoner import get_director_reasoner

    return get_director_reasoner(strategy).generate_plan(brief, graph, observations)


def generate_project_plan(
    brief, manifest, context, project_graph, observations,
    strategy: str = "heuristic", **kwargs,
):
    """Generate a draft project plan from exact, source-local asset evidence."""
    from director_brain.director_reasoner import get_director_reasoner

    reasoner = get_director_reasoner(strategy)
    method = getattr(reasoner, "generate_project_plan", None)
    if not callable(method):
        raise NotImplementedError(
            f"reasoner strategy {strategy!r} does not implement project planning")
    return method(brief, manifest, context, project_graph, observations, **kwargs)


def materialize_project_plan_from_candidate(
    brief, manifest, context, project_graph, candidate, *, strategy: str = "llm",
    **kwargs,
):
    """Materialize an exact saved project strategy candidate without a provider call."""
    from director_brain.director_reasoner import get_director_reasoner

    reasoner = get_director_reasoner(strategy)
    method = getattr(reasoner, "materialize_project_plan_from_candidate", None)
    if not callable(method):
        raise NotImplementedError(
            f"reasoner strategy {strategy!r} cannot materialize a saved candidate")
    return method(
        brief, manifest, context, project_graph, candidate, **kwargs)


def generate_project_shadow_plan(
    brief, manifest, context, project_graph, observations,
    *, provider: str = "ollama", config: dict | None = None, **kwargs,
):
    """Generate a local, multi-asset comparison Plan/EDL that cannot be confirmed.

    This entry remains SHADOW regardless of provider configuration. The current
    implementation accepts only loopback Ollama and never changes the pathway
    admission state or persists/dispatches the result.
    """
    from director_brain.director_reasoner import LLMDirectorReasoner

    reasoner = LLMDirectorReasoner(provider=provider, config=config)
    return reasoner.generate_project_shadow_plan(
        brief, manifest, context, project_graph, observations, **kwargs)


def generate_project_shadow_strategy_options(
    brief, manifest, context, project_graph, observations,
    *, provider: str = "ollama", config: dict | None = None, **kwargs,
):
    """Generate an unranked local comparison of project strategy hypotheses.

    Every returned strategy includes its own Plan/EDL and exact evidence trace.
    This function returns an in-memory SHADOW value. The project REST endpoint
    persists an immutable local wrapper for restart readback; neither path can
    confirm or dispatch the comparison.
    """
    from director_brain.director_reasoner import LLMDirectorReasoner

    reasoner = LLMDirectorReasoner(provider=provider, config=config)
    return reasoner.generate_project_shadow_strategy_options(
        brief, manifest, context, project_graph, observations, **kwargs)
