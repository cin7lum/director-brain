# Director Brain

将用户创作意图、素材事实与质量反馈转化为可解释、可追溯、可修订的导演决策。

## 项目结构

```
director_brain/      # Brain 内核（9 组件）
  models/            # Pydantic 数据模型
  brief_compiler.py  # 组件1: Intent Intake / Brief Compiler
  context_gateway.py # 组件2: Film Context Gateway
  analysis_cache.py  # 组件3: Analyze Once Reuse Many
  story_graph.py     # 组件4: Story Graph
  director_reasoner.py # 组件5: Director Reasoner
  strategy_confirmation.py # 组件6: Strategy Confirmation
  plan_validator.py  # 组件7: Plan Validator
  revision_loop.py   # 组件8: Revision Loop
  decision_ledger.py # 组件9: Decision Ledger
observation_service/ # 观察服务（独立）
arsenal/             # Arsenal（能力解析）
davinci_execution/   # DaVinci Execution（执行层）
fql/                 # FQL（独立评估）
gen1_adapter/        # GEN-1 V0.1 兼容层
storage/             # 存储层（Repository 抽象）
api/                 # HTTP API（远期，当前用豆包交互）
tests/               # 测试（unit/contract/integration）
scripts/             # 工具脚本
docs/                # 文档
```

## 当前能力边界

- **导演推理**：Heuristic 确定性算法（blur_score × VLM 权重排序 + 四幕选片），非 LLM。
- **LLM 推理**：`LLMDirectorReasoner` 为预留骨架，未实现；`get_director_reasoner("llm")` 会发出 warning，调用 `generate_plan` 抛 `NotImplementedError`。
- **决策可解释性**：`Decision.confidence / alternatives / requires_approval` 与 `Plan.open_questions` 由 Heuristic 真实计算（非常量）。
- **多方案选择**：`select_best(priority="duration")` 选最接近 `target_duration_us` 的方案（不再"最长即最优"）；`generate_variants` 支持 `blur_threshold` 真实变化维度。

## 开发流程

三窗口分离：主控窗口（规划/复验）、开发窗口（编码）、监工窗口（测试/验收）。

每个窗口交付前必须跑 `scripts/ci_check.ps1`。

## 环境

```
pip install -e ".[dev]"
```

可选 HTTP 层：`pip install -e ".[http]"`
