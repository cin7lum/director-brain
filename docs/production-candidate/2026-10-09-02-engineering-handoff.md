# 02 Director Brain — 工程交接

更新时间：2026-10-09（Asia/Shanghai）

## 仓库与检查点

- 权威仓库：`cin7lum/director-brain`；`origin/master` 仍为 `b556abc95add61e4f1d5ff5362f679e3cbcdb5bf`。
- 本地开发分支：`codex/02-production-candidate`。产品候选快照提交：`c0ae875eac950fb63562a20b1437343e3a5863a4`。后续仅在该开发分支继续；本次同步不合并 PR、不改 master。
- 本次没有重跑测试、GitHub CI 或媒体运行。最近保存的全套本地回归记录为 2026-10-08 的 **888 passed / 6 skipped / 8 warnings**，覆盖当时的 dirty worktree，不等同于本提交验证。GitHub master 的历史记录为 **602 passed / 8 skipped**；开发分支 CI 尚未确认。
- 本地工件仍有未跟踪与 ignored 项。已创建主机本地的全仓库备份、逐文件 SHA-256 清单及仅本地资产表；不含于 GitHub 提交的 Evidence、用户卡片、数据集材料、数据库、缓存和环境文件仍留在当前 checkout 与备份中。

## 阶段状态

- **当前阶段：P2 Director Reasoner，IN_PROGRESS。** P1 的当前树项目链已出现一次 `COMPLETE_SOFTWARE_RUNTIME_TRACE / PRODUCT_QUALITY_NOT_PROVEN`：多素材 ingest、Context、StoryGraph/Plan、跨独立 Uvicorn 进程持久化读回。对应 CoMind TRAIN 研究素材及本地候选树，不能证明已获授权真实项目、项目级泛化或导演质量。两项目隔离记录为合成软件测试。正式项目级 DEV/HOLDOUT 与真实项目验收仍 **NOT_PROVEN**。
- **P2 首个未解决断点：** 已有绑定失败回执记录 `shot_count=32`，但 `emotional_trajectory` 只有 2 项；只定位到 cardinality mismatch，未解释 provider 为什么只生成 2 项。2026-10-09 的普通分层 Reasoner 运行以 `LLMStructuredOutputError` 失败、没有产出 Plan/EDL，也没有捕获规范化失败码或 provider-call 数。一次合成 flat-provider smoke 返回候选只证明软件接线，不证明导演质量。不要再扩展 Attempt 05；用已有回执推进 P2 产品修复，保持失败关闭且不保留 prompt、语义文本或原始回复。
- `semantic_reasoner` / `director_strategy_reasoning` 仍为 **SHADOW**；`relation_inference` / `beat_grid` 仍为 **EXPERIMENTAL**。不得因本次提交或测试通过改为 ACTIVE。

## 外部依赖与产品边界

- 05/FQL 提出的 REST lifecycle/export 问题已由 02 owner-side 视为通用 Brain 能力缺口；当前候选包含项目级 Brief/Plan/EDL/Strategy 与 SHADOW comparison 的持久化、版本绑定/读回和幂等行为，并有本地合成软件证据。该证据不等于真实 FQL 联调；05 特定 intake/export、revision loop 仍待 **P4**。历史记录中的 legacy 单素材 lifecycle 不能因项目级路径存在而默认视为全覆盖。
- `dispatch_eligible=false` 及 SHADOW confirmation 不是 04 执行证明。**P3** 仍需绑定真实 04 仓库/包、版本、owner 与 Resolve/MCP 运行资产，并实现正式执行、Readback 和 Execution Receipt。
- 02 拥有导演决策及版本化 Brief、Film Context、StoryGraph、Plan、EDL、StrategyConfirmation、Revision。03 保持 Orchestrator 边界；04 拥有正式 Resolve 执行；05 保持独立 Film Quality 评价。02 内部 evaluator 不替代 05；FFmpeg 只用于 Preview/Verification/Fallback。基础模型与大规模数据集训练不是默认路径。
- 后续顺序保持 **P2 → P3 → P4 → P5**。P5 仍需独立项目边界的开发/holdout、不同片型与真实授权素材、冻结协议后的 owner-approved 数值门、真实盲评/人工偏好、独立质量评价及恢复/可靠性证据。未满足的 holdout、阈值批准和 04 条件只阻塞依赖它们的阶段。

## 证据位置

完整 P1/P2 运行回执与原始 Evidence 保留在本机 `evidence/`，未随本次候选推送；交接时按其中的 `RUN_REVIEW.md`、`IMPLEMENTATION_EVIDENCE.md` 及关联 hash sidecar 核对。最新 P2 记录位于 `evidence/P2_CURRENT_HIERARCHICAL_RUNTIME_RUN/IMPLEMENTATION_EVIDENCE.md`、`evidence/P2_CURRENT_PROMPT_RUNTIME_SMOKE_2026-10-09.md` 与 `evidence/P2_SEGMENT_EMOTION_SCHEMA_ALIGNMENT/ATTEMPT_05_STRUCTURE_ONLY_REVIEW.md`。当前树 P1 trace 位于 `evidence/P1_CURRENT_TREE_FULL_PROJECT_TRACE/attempt-02/RUN_REVIEW.md`。
