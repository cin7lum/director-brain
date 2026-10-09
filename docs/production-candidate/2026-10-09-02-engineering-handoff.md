# 02 Director Brain — 工程交接

更新时间：2026-10-09（Asia/Shanghai）

## 仓库与当前验证

- 权威仓库：`cin7lum/director-brain`。当前开发分支 `codex/02-production-candidate`，本次实查远端开发分支仍为 `bdae0e01ba0626ffd544968a9c55501f2f1dcab4`，`master` 为 `b556abc95add61e4f1d5ff5362f679e3cbcdb5bf`。本地 HEAD 为 `2ca24a1ae0e1fdae6187fb428a4e719ec06cf64b`，是远端分支祖先后的第 6 个提交；本次没有 push。新增提交 `2ca24a1` 仅包含 9 个 P2 源码/测试文件。源码修改前本地 HEAD 为 `101024f81d7c7f838b9326d048f49ba04f279b9b`，其后五个既有提交为文档提交。
- 当前开发工作树全套回归为 **986 passed / 6 skipped / 8 warnings，265.99 秒**，Python 3.12.8。测试在提交前运行；提交后核验提交 patch SHA-256 与运行时 tracked diff 相同（`afc1c0072f965e729f3de4a9ff05c6069378e490bf26e1db089ccd72b75c124a`）。这是本机源码回归，不是 GitHub Actions，也不是产品或艺术质量验收。远端提交 `bdae0e0` 的既有本机全套为 **985/6/8，243.88 秒**；历史证据记录其 GitHub check-runs 为 0。当前运行摘要在忽略目录 `evidence/P2_NASA_LOCAL_DIAGNOSTIC/diagnostic_records/current_worktree_full_suite_2026-10-09.json`。
- 旧记录分开看：2026-10-08 dirty worktree 的最终全套回归为 **885/6/8**；历史 master CI 的 **602/8** 不是当前开发分支结果。
- 历史未跟踪文件、用户卡片、Evidence、数据集材料、数据库、缓存和环境文件继续留在本机；本次只暂存上述 9 个明确文件，未用 `git add -A`，未覆盖或清理其他本地状态。

## 阶段状态

- **P0/P1：沿用既有交接状态，不重做。** 当前 P1 软件运行链有既存 `COMPLETE_SOFTWARE_RUNTIME_TRACE / PRODUCT_QUALITY_NOT_PROVEN` 证据；真实授权项目、独立 holdout 与产品质量仍未证明。
- **当前阶段：P2 Director Reasoner，IN_PROGRESS。** `director_strategy_reasoning` / `semantic_reasoner` 保持 **SHADOW**；`relation_inference` / `beat_grid` 保持 **EXPERIMENTAL**。
- 已有 2026-10-09 CoMind TRAIN 普通分层运行（HEAD `5b4dab0`）：两次 provider call 后失败，`failure_code=segment_schema_invalid`。随后基于相同未核实来源的 attempt 02 只有 freeze、没有终态 `run.json`，状态为 **INCOMPLETE**。不把其视为失败或成功回执，也不再调用该数据。

### 32 镜头短输出故障

- 原始 `attempt-06-hierarchical-diagnostic/failure.json` 仍可访问，6 个 SHA-256 sidecar 均通过核验。它记录一轮 provider call、`phase=project_shadow_reasoner_call`、`failure_code=segment_emotions`：预期 32 个情绪项，实际为 2 个非空字符串，其余位置缺失。请求为 6,680 UTF-8 bytes，低于 24 KiB 请求上限；该运行的 `segment_max_shots=32`、completion cap 为 4,096 tokens。响应 `finish_reason` 与实际 completion 长度未保存。
- 既有两份无 provider 容量审计针对完整 772-observation 请求：输入 43,169 tokens 超过 32,768 context；紧凑完整输出为 10,591 tokens，两套完整序列的探索性布局为 6,052 tokens，均超过 4,096 completion cap。它们证明完整扁平请求存在容量冲突，不能直接解释首个 32 镜头 segment。
- 新的当前 schema 静态探针显示：带 32 个情绪项、双策略、完整来源理由的极简合法 segment 仅 1,709 tokens，低于 4,096。它只能排除“schema 最小形状必然超限”，不能还原历史 response 的 token 数、finish reason 或缺失原因。Attempt 06 冻结的产品源码哈希不匹配当前 Git 提交，也不匹配保留的 pre-sync checkout；当时精确 prompt 源码不可还原。因此历史故障直接根因保持 **NOT_PROVEN**，不称为已证实的截断或模型缺陷。
- 当前代码已有三项针对性约束：segment 用户提示明确要求按本次镜头数返回情绪项和完整策略来源；segment 上限为 16；structured output 的 `finish_reason=length` 被归类为 `provider_output_truncated` 并 fail closed。
- 当前代码下 32 镜头纯合成多素材运行（运行源码 HEAD `82f6af9`）：默认 16 镜头分段，两次 `segment` 加一次 `project_synthesis` 共 3 次本地调用；返回 2 个候选，2 个 Plan 校验通过，2 个 EDL 均绑定合成源身份、区别于启发式基线且彼此不同；0 次非 loopback 尝试。对比保持 `confirmable=false`、质量 `NOT_PROVEN`。该运行只证明合成数据的软件/本地模型链，不证明真实素材判断或艺术质量。
- 本轮 NASA 真实多素材运行（下节）对 85 个语义观察分成 7 个 segment；首轮所有 segment 都通过精确情绪项数校验，然后在 project synthesis 遇到 child strategy selection 校验失败。历史 `segment_emotions` 基数故障由分段路径在这两次当前代码运行中未重现；不能据此声称 provider 普遍可靠。
- 另一个历史 `P2_SEGMENT_EMOTION_SCHEMA_ALIGNMENT/ATTEMPT_05_STRUCTURE_ONLY_REVIEW.md` 已绑定 attempt-06 回执：32 项请求只有 2 个字符串，失败代码 `segment_emotions`；回执未保存该输出为何只有 2 项的原因。现有分段策略是应对完整请求容量问题的代码路径，不篡改历史根因，也不降低情绪完整性契约。
- Attempt 06 的 CoMind TRAIN 项目边界仍为 `declared_unverified`，不能作为新推理输入。真实项目复验需要单独确认本地处理权限的多素材项目或其他已准入数据。不得把合成运行升级为真实项目证据。
- 2026-10-09 公开素材只读筛查的其他候选（NASA LRO launch 多机位、Wikimedia Commons Yosemite 无音轨、Scheldebeker、Wiki Wedding、访谈、意大利旅行剪辑及短列车 POV）仍为 `NOT_ADMITTED`；限制和来源见 `docs/production-candidate/2026-10-09-p2-public-real-video-source-screen.md`。

### NASA SVS 14948 本地 P2 诊断（用户已限定授权）

- 用户允许仅在本机处理 NASA SVS 14948 的两个同事件文件，使用 FFmpeg、固定本地 Qwen3‑VL:4b 抽帧分析和固定本地 Qwen2.5:7b Director Reasoner；禁止云端/第三方上传、训练和人物身份识别。具体文件 SHA-256、大小及运行配置在忽略目录 `evidence/P2_NASA_LOCAL_DIAGNOSTIC/runs/20261009T043240Z/run.json`。
- Attempt `20261009T043240Z`：manifest、StoryGraph 成功；85 次 Qwen3‑VL 请求全部因 `request (8339 tokens) exceeds the available context size (8192 tokens)` 失败，Director Reasoner 在 provider 前以 `evidence_too_poor` 拒绝。没有 Plan/EDL。没有把该失败混称为历史 Reasoner 情绪数组或 Schema 故障。
- 生产关键帧抽取改为保留三个时点、最大宽度 2560 并进入新的 cache profile；单镜头本地探针成功，`test_keyframe.py` 与 `test_context_gateway.py` 共 24 项通过。对应实现已进入 `2ca24a1`。
- Attempt `20261009T045550Z`：Film Context 200、StoryGraph 200；85/85 条 VLM 观察均通过，其中 closeups 33 条、mezzanine 52 条，按源 SHA-256 正确绑定。Director 比较接口 502：`group_child_strategy_unavailable`，失败阶段 `project_synthesis`，累计 8 次 provider 调用；Plan/EDL 未生成。素材权限证据仍为 `declared_unverified`，质量验收仍 `NOT_PROVEN`。
- 根据该回执，`2ca24a1` 将 synthesis Schema 的每个 child selection 的可选 hypothesis ID 限定到对应 child，并在系统提示中要求原样引用。Schema 单测已覆盖，但真实 provider 尚未走到 synthesis 验证这项修复。
- 复用已保存 SQLite 的 Reasoner-only 诊断：`20261009T052350Z` 因复验夹具缺少 VLM profile 身份变量在 provider 前被 409 拒绝，不算模型运行；补齐固定 VLM/runtime 身份后的 `20261009T052529Z` 发起了本地推理，但首个 segment 因 `segment_strategies_identical` 被 fail-closed 拒绝。两次都没有 Plan/EDL。此失败说明当前固定模型的策略结构多样性仍不稳定；未降低校验门槛、未继续重试。
- NASA 授权仅覆盖上述本地诊断，不代表 NASA 许可已核实；两段素材的来源权利状态仍为 `declared_unverified`。本机私人 Insta360 素材没有被读取、抽帧、哈希、模型处理或上传。

### 2026-10-09 P2 回执与目标对齐增补

- 目标与原五份方案一致：02 负责意图、证据绑定的叙事判断与 Plan/EDL；03 路由能力；04 正式 Resolve 执行及读回；05 独立评价与修订。完整目标仍是 Production Candidate，执行路线仍为 P2→P3→P4→P5。当前只评估了执行焦点有无局部偏移：近期失败分类和配套回归有扩张风险，但本轮修复仍围绕 P2 的真实 Reasoner 失败；后续不扩建无关测试或治理机制。
- `20261009T060519Z` 在六个 segment 调用后给出精确失败 `segment_source_rationale_focus_missing`。现有应用层验证正确拒绝了未引用自身焦点镜头的 rationale；原始输出未留存，不能判断模型漏填的直接原因。
- 随后的本地诊断复用已保存观察、不重跑 VLM：`20261009T061258Z` 在 4096 completion / 3067 prompt tokens 截断；`20261009T061626Z` 临时将叶段缩至 8 镜头后仍在 4096 / 2326 截断；`20261009T062006Z` 临时将 completion cap 升至 8192 后仍在 8192 / 2326 截断。三次均为 `finish_reason=length`，没有 Plan/EDL。不能把这些新回执倒推为历史 32 镜头仅 2 项情绪轨迹的根因。
- 一次极小、无媒体的 loopback schema 探针正常停止（51 prompt / 44 completion tokens），但两个 rationale 项未通过本地 required-field 校验。没有保留生成内容；Ollama 对新增跨字段 `anyOf`/`contains` 约束的执行未得到证明。该实验性 schema、8 镜头叶段和 8192 输出上限均已从产品路径撤回；保留原 16 镜头/4096 上限、严格应用层校验、2.16 提示修正和有用的失败归因。没有再对同一 NASA 输入发起推理。
- 当前重点回执：`evidence/P2_NASA_LOCAL_DIAGNOSTIC/runs/20261009T060519Z/run.json`、`20261009T061258Z/run.json`、`20261009T061626Z/run.json`、`20261009T062006Z/run.json`。这些安全摘要仅在本机 `.git/info/exclude` 排除的目录中，不会推送；没有保存原始视频或原始模型响应。修复后 `test_narrative_analyzer.py`、`test_project_story_graph.py` 与 `test_api_endpoints.py` 共 **179 passed, 1 warning**。源码测试通过不代表真实 P2 或艺术质量通过；语义通路保持 SHADOW，P2 Plan/EDL、跨项目泛化和质量均 NOT_PROVEN。

## 产品边界与后续路线

- 02 拥有导演意图理解、Film Context/StoryGraph 消费、解释性策略与 Plan/EDL、版本和修订；03 负责能力路由；04 负责正式 Resolve 执行、Readback 与 Execution Receipt；05 负责独立质量评价。内部 validator、确定性粗剪、SHADOW 候选和全绿源码测试均不等于艺术质量通过。
- 05/FQL 的独立 intake/export 与修订闭环归 P4；不能以 02 内部评分替代。P3 需要真实 04 仓库/包、版本、owner 与 Resolve/MCP 运行资产；P5 需要独立真实项目与 holdout、盲评、人工偏好、独立质量及恢复/可靠性证据。
- 完整阶段顺序保持 **P2 → P3 → P4 → P5**。未准入的语义路径保持 SHADOW；不执行生产 Resolve、NAS、未授权数据上传、不可逆数据操作或受保护主分支合并。

## 证据位置

- P2 32 镜头历史故障：`evidence/P2_PROJECT_DIRECTOR_REASONER_REAL_CONTEXT/attempt-06-hierarchical-diagnostic/`（原回执及 hash sidecar 可核验）。
- 完整项目容量审计：`evidence/P2_PROJECT_DIRECTOR_REASONER_CAPACITY_AUDIT/attempt-01/` 与 `attempt-02/`。
- 32 镜头当前 schema 静态输出容量探针：`evidence/P2_SEGMENT_OUTPUT_CAPACITY_2026-10-09/attempt-01/`。
- 当前 32 镜头合成分层运行：`evidence/P2_CURRENT_32_SHOT_SYNTHETIC_HIERARCHICAL_2026-10-09/attempt-01/`。
- 当前提交本机完整测试回执：`evidence/P2_CURRENT_COMMIT_FULL_SUITE_2026-10-09/attempt-01/`（只含运行元数据与摘要，没有完整 stdout）。P2 修复及根因边界：`docs/plans/2026-10-09-p2-response-contract.md`。旧 P1/P2 Evidence 保存在本机；不得用本文档替代其原始回执。
- 公开真实视频来源筛查及未覆盖类型：`docs/production-candidate/2026-10-09-p2-public-real-video-source-screen.md`。原始网页只读核查；未下载或处理媒体。
