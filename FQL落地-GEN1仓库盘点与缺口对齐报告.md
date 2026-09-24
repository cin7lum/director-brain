# FQL 企业级方案落地：GEN-1 仓库盘点与缺口对齐报告

> 盘点日期：2026-09-24
> 盘点范围：GEN-1 主仓库、FQL 评价仓库、FQL 重建资产
> 对照基准：`05-FQL-企业级方案.md` V1.1

---

## 一、盘点结论：三个核心资产

### 1.1 GEN-1 创作/粗剪执行端（`D:\新建豆包\gen1-roughcut`）

| 维度 | 状态 |
|---|---|
| 定位 | 从素材到成片的自动粗剪引擎（Brain + 执行器） |
| 规模 | 70+ Python 文件，含完整粗剪链路 |
| 测试 | 44 conformance tests 全 pass；大量模块级单测 |
| 技术栈 | Python 3.14 + ffmpeg + opencv + tkinter（单机 GUI） |
| 评审状态 | M6 专家组决策报告已完成（2026-09-19），给出 4 周路线图 |
| 当前阶段 | M5 dry-run 完成，M6 决策完成，待执行 W1-W4 |

**核心模块：**
- 粗剪链路：`media_import` → `shot_discovery` → `deterministic_analysis` → `candidate_pool` → `heuristic_proposal` → `timeline_builder` → `render`
- Critic 闭环：`director_critic`、`critic_loop`、`quality_scores`、`plan_repair`
- VLM 适配层：`vlm_adapter`（接口）、`ollama_vlm_adapter`、`zhipu_vlm_adapter`（默认 StubVLM，真实 VLM 未接入）
- 编辑关系：`edit_relation`、`relation_inference`、`relation_producer`
- 辅助：音乐/音效/字幕/LUT/转场选择器、GUI

**已冻结合同：**
- `AI_PROPOSAL_CONTRACT_V0.1.md`
- `EDIT_EVENT_CONTRACT_V0.1.md`
- `IMPLICIT_PREFERENCE_EXTRACTION_RULES_V0.1.md`
- `vlm_tags_schema_v0.2.md`（M6 新增，未动 V0.1 合同）

**M6 已知工程病：**
- VLM 输出未接进排序（"名存实亡"）
- 缺独立时长分配/节奏层
- confidence 硬编码 0.6 违反合同
- proposal 凑时长截断导致 0.29s 碎镜、9.08s 长镜

---

### 1.2 FQL 独立评价端（`D:\新建豆包\_inspect\film-quality-loop-main`）

| 维度 | 状态 |
|---|---|
| 定位 | 独立于创作端的影片质量评价系统 |
| 版本 | v1.5.0-alpha.hardening.13p3-p3 |
| 成熟度 | ENGINEERING_ALPHA（全部能力 STUB 或 EXPERIMENTAL，**零 VALIDATED/PRODUCTION**） |
| 核心理念 | Deep Hardening——先证明每项能力是否真实成立，不追求功能列表完整 |
| 合同 | 46 个 JSON Schema |
| 入口 | CLI 脚本（score.py、auto_shotlist.py 等），**无 HTTP API** |

**已实现（EXPERIMENTAL）的能力：**

| 能力 | 说明 |
|---|---|
| `shot_engine_hard_cut` | 硬切检测（合成 fixture 验证，缺真实标注语料） |
| `shot_engine_gradual_transition` | 渐变转场检测（默认禁用） |
| `pyscenedetect_adapter` | PySceneDetect 适配器（可选） |
| `timeline_import` | OTIO 时间线导入（缺多轨/转场/嵌套验证） |
| `technical_audio_qc` | 技术音频 QC（响度/峰值/LRA） |
| `editing_intelligence` | 剪辑智能（启发式，未专业验证） |
| `external_evidence_integrity` | SHA-256 证据绑定、stale 证据拒绝 |
| `low_level_visual_observations` | 低层视觉观察（亮度/饱和度/锐度/运动代理） |
| `shot_enrichment_integrity` | 镜头丰富化契约与来源绑定 |
| `l1_audio_repair` / `l1_color_repair` | L1 ffmpeg 级修复 |
| `camera_visual_ground_truth_framework` | 相机视觉 Ground Truth 框架（H09-H13 大量工作） |
| `h13_independent_human_ground_truth_gate` | 人审 Ground Truth 门（工程框架已建，**无真实人审数据**） |
| `project_team_scientific_governance` | 科学治理框架（v1+v2，方向控制、角色独立、预注册） |
| `benchmark_execution_orchestrator` | Benchmark 执行编排（fail-closed） |
| `runtime_environment_fingerprint` | 运行时环境指纹 |

**STUB（未实现）的关键能力：**
- `visual_semantic_judge`（VLM 视觉语义判断）——仅有契约，无内置 provider
- `vqa_dover`——DOVER 推理未集成
- `asr_whisperx`——未实现
- `l2_source_repair` / `l3_project_repair`——未实现
- `human_reference_benchmark`——未实现
- `transnetv2_adapter`——未实现
- 相机 provider registry **当前为空，0 个验证 provider**

---

### 1.3 FQL 重建资产（`D:\新建豆包\FQL_REBUILD_ASSETS`）

9 个分类目录，是历史资产的固化索引：

| 目录 | 内容 |
|---|---|
| 01_CORE_FILM_KNOWLEDGE | 核心电影知识 |
| 02_DIRECTOR_BRAIN_GEN1 | GEN-1 Director Brain 资产（含 aidirector 脚本） |
| 03_BLIND_REVIEW_BENCHMARK | 盲评基准 |
| 04_FILM_SKILL_DISTILLATION | 电影技能蒸馏 |
| 05_MATURE_SOLUTION_RESEARCH | 成熟方案研究（VMAF/EBU/ITU 等） |
| 06_EXECUTION_CAPABILITY_REFERENCE | 执行能力参考（DaVinci MCP/OTIO） |
| 07_OLD_AUTOCLIPSTUDIO_REFERENCE | 旧 AutoClipStudio 参考 |
| 08_CONTRACTS_AND_SCHEMAS | 合同与 Schema |
| 09_FAILURE_CASES_AND_EVIDENCE | 失败案例与证据 |

**专业参考源：** 33 个已索引源（大学教材、NLE 官方文档、学术论文），覆盖剪辑语法、镜头功能、J/L cut、纪录片伦理等。

**已登记的缺失/延期项：**
- General Timeline Repair 仅音频真峰值端到端验证，其余 7 类动作 NOT_VERIFIED
- Q-Align / VBench 授权未确认（RESEARCHED，不得标 Production）
- DaVinci MCP / OTIO / 多 Agent 治理仅为参考，未接入
- 摄影知识、婚礼纪录片、梅雨季视觉语言未形成独立 Skill

---

## 二、与企业级方案的缺口对齐

### 2.1 逐模块对齐表

| 企业级方案模块 | 现有实现 | 成熟度 | 缺口分析 |
|---|---|---|---|
| **Artifact Intake**（hash 绑定、媒体校验） | FQL `external_evidence_integrity` | EXPERIMENTAL | 技术基本可用；缺与 GEN-1 render 输出的对接约定 |
| **L0 Production QA**（ffprobe + EDL 读回 + cut boundary + 交付 profile） | FQL `technical_audio_qc` + `shot_engine_hard_cut` + `timeline_import`（分散） | EXPERIMENTAL | **缺统一 L0 编排层**；缺 EDL 与实际渲染的读回核对；缺交付 profile 检查；缺 render hash 与 receipt 匹配；cut boundary 自检可借鉴 video-use 但未整合 |
| **Measurement Adapters**（VMAF / EBU R128 / 黑帧） | FQL `technical_audio_qc`（音频响度/峰值/LRA） | EXPERIMENTAL | **VMAF 未集成**；黑帧检测需确认是否在 shot_engine 中；缺统一适配器抽象和工具版本/参数记录 |
| **Observation/Finding Engine**（多模态候选 + 时间定位） | FQL `low_level_visual_observations` + `shot_enrichment_integrity` | EXPERIMENTAL | `visual_semantic_judge`(VLM) 是 **STUB**，无真正多模态模型集成；**无企业级 Finding 数据模型**（category/severity/time_range/claim_kind/evidence_refs）；现有观察是镜头级而非 finding 级 |
| **Profile 体系**（片型化门槛、版本化、批准流程） | FQL `genres/` 目录 + `--genre` 参数 | 部分 | **无企业级 FQLProfile 数据模型**（profile_id/version/technical_thresholds/intent_dimensions/sampling_plan/human_review_requirements/approval_owner）；genres 只是评分参数，不是治理化 Profile |
| **Human Review + Pairwise 盲评** | FQL `h13_independent_human_ground_truth_gate`（仅相机/视觉维度） | EXPERIMENTAL（工程框架） | H13 是针对**相机景别 Ground Truth** 的人审门，不是全片质量盲评；**无 PairwiseComparison 系统**；无 A/B 版本随机化比较；无盲评协议/评审群体管理；**无真实人审数据** |
| **Revision Verification + Regression Gate** | FQL `quality_loop`（L1 ffmpeg 修复循环） | EXPERIMENTAL | 现有是 ffmpeg 级 L1 修复，**无企业级 RevisionVerification**（目标区间/控制区间/回归 findings/人类偏好变化）；无与 Brain revision 链路的对接；无控制区回归检查 |
| **数据模型 + API 合同** | FQL 46 个 JSON Schema | 存在 | Schema 覆盖评分/镜头/benchmark/治理，但**无企业级定义的 ProductionQAAssessment、FQLAssessment、Finding、PairwiseComparison、RevisionVerification 模型**；**无 REST API**（CLI only），无 POST /v1/production-qa 等接口 |
| **与 Brain/Resolve 集成** | FQL `timeline_import`（OTIO） | EXPERIMENTAL | **无 DaVinci Resolve 直接集成**；无 render receipt 格式约定；无 Director Brief 接口；**与 GEN-1 目前完全没有对接**；OTIO 导入缺多轨/转场/嵌套验证 |
| **GEN-1 Critic/Repair 迁移** | GEN-1 `director_critic` + `critic_loop` + `plan_repair` + `quality_scores` | 存在 | 需逐项盘点接口、模型依赖、证据链、失败模式；GEN-1 Critic 五维分与 FQL 评价维度需对齐；GEN-1 repair 是提案级，FQL L1 是 ffmpeg 执行级，需明确分工 |

### 2.2 缺口严重度分级

| 严重度 | 缺口 | 说明 |
|---|---|---|
| **P0 阻塞级** | GEN-1 与 FQL 无对接 | 两个仓库独立运行，成片无法自动进入评价；render receipt/EDL 格式未约定 |
| **P0 阻塞级** | 无统一 L0 Production QA 编排 | 现有能力分散在 audio_qc/shot_engine/timeline_import，没有统一的 L0 任务编排和 PASS/FAIL/INCOMPLETE 判定 |
| **P1 关键级** | 无 Finding 数据模型 | 企业级方案的核心输出是带时间定位/严重度/证据的 finding，现有只有镜头级观察和评分 |
| **P1 关键级** | 无 HTTP API 服务化 | 现有是 CLI 脚本，企业级方案要求 REST API；无法被 Brain/Resolve/其他系统调用 |
| **P1 关键级** | VLM 视觉语义判断未集成 | FQL 的 visual_semantic_judge 是 STUB；GEN-1 的 VLM 也未真实接入；多模态观察候选无法产生 |
| **P2 重要级** | Profile 体系未治理化 | genres/ 只是评分参数，缺版本化、门槛预注册、批准流程 |
| **P2 重要级** | 无全片质量盲评/Pairwise | H13 仅覆盖相机景别 Ground Truth，缺全片 A/B 盲评系统 |
| **P2 重要级** | 无 Revision Verification/回归 | 现有 L1 修复循环无目标区/控制区概念，无法验证修改收益和拦截回归 |
| **P3 改进级** | VMAF 未集成 | 可后续补充，不阻塞 L0 |
| **P3 改进级** | 黑帧/损坏帧检测需确认 | 可能已在 shot_engine 中，需核实 |

---

## 三、关键发现

### 3.1 企业级方案的前提判断需要修正

企业级方案第 2 章称"当前对话提到 FQL 已有能力/打分，但没有提供 FQL 仓库、合同或验收报告。现有实现能力与生产状态均为 NOT_DETERMINED。"

**实际情况：FQL 仓库存在**（`film-quality-loop-main`），且有：
- 46 个 JSON Schema 合同
- 完整的评分/镜头检测/修复循环 CLI
- 大量 Deep Hardening 证据链工作
- 科学治理框架

但仓库处于 `_inspect/` 目录下，且全部能力为 EXPERIMENTAL/STUB，**确实没有通过企业级验收**。方案的"NOT_DETERMINED"结论在"生产状态"层面成立，但在"代码存在性"层面不成立。

### 3.2 FQL 仓库的战略方向与企业级方案高度一致

FQL 仓库的 Deep Hardening 理念（证据绑定、fail-closed、角色独立、预注册、不把接口存在当能力完成）与企业级方案的核心原则（证据化、可复现、评价独立、不冒充质量）**完全一致**。这意味着可以在现有仓库基础上演进，不需要从零新建。

### 3.3 最大的缺口是"集成层"而非"底层技术"

底层测量（ffprobe/音频/镜头检测）、证据绑定、修复循环都已有实现。真正缺的是：
1. **把分散能力编排成统一 L0/Film Quality 通道的编排层**
2. **GEN-1 创作端与 FQL 评价端的对接层**（render receipt、EDL 交换、Director Brief 传递）
3. **服务化层**（CLI → REST API）
4. **Finding/Profile/盲评/回归等企业级数据模型**

### 3.4 GEN-1 的 M6 路线与 FQL 落地可以并行但需协调

GEN-1 M6 已规划 W1-W4（仓库清理 → CLI → 修 heuristic → 真实素材 → 错误恢复 → VLM 适配器 → 偏好落盘）。FQL 落地不应阻塞 GEN-1 的 M6 执行，但需要在 GEN-1 的 export 环节预留 FQL 评价入口（render receipt 输出）。

---

## 四、开发方向建议

### 4.1 总体策略：在现有 FQL 仓库基础上演进，不新建

**理由：**
- FQL 仓库已有 46 个 Schema、完整 CLI、证据链框架、科学治理
- 核心理念与企业级方案一致
- 从零新建会浪费已有工程基础，且难以达到同等的证据严谨度

**做法：** 将 `film-quality-loop-main` 从 `_inspect/` 迁出为正式工作仓库，按企业级方案要求增量补全。

### 4.2 优先级：L0 先行 + 对接 GEN-1

**第一优先级：L0 Production QA 统一编排 + GEN-1 对接**
- 把 FQL 现有的 audio_qc/shot_engine/timeline_import 整合成统一 L0 通道
- 定义 render receipt 格式，让 GEN-1 export 时输出
- 实现 EDL 读回核对、hash 匹配、交付 profile 检查
- L0 先以影子模式运行（只报告不做 Gate）

**第二优先级：Finding 数据模型 + 服务化**
- 实现企业级 Finding 模型（category/severity/time_range/claim_kind/evidence_refs）
- 将现有镜头级观察升级为 finding 级输出
- 封装 REST API（POST /v1/production-qa 等）

**第三优先级：Film Quality 测量 + VLM 观察候选**
- 集成 VMAF
- 接入 1 个多模态模型（推荐 Doubao-Seed-1.6-vision，与 GEN-1 M6 选型一致）产观察候选
- 候选 finding 只标 MODEL_OBSERVATION，不进 Gate

**第四优先级：Profile 体系 + 人审/盲评 + 回归闭环**
- 从 genres/ 演进为治理化 Profile
- 建立全片质量 Pairwise 盲评（可借鉴 H13 框架但扩展到全片维度）
- 实现 Revision Verification + 控制区回归

---

## 五、分阶段开发路线

### 阶段 0：仓库就位与基线确认（1 周）

| 任务 | 交付 | 验证 |
|---|---|---|
| 将 film-quality-loop-main 迁出 _inspect，建立正式仓库 | 正式 FQL 仓库 + git 初始化 | git log 可追溯 |
| 跑通 FQL 现有 CLI（preflight → score → repair_plan → user_report） | 基线运行记录 | 用 Sintel trailer 跑通全链路 |
| 跑通 GEN-1 44 conformance tests | 测试基线 | 44 passed |
| 确认 GEN-1 export 产物结构（mp4 + sidecar + proposal） | 产物清单文档 | 与 GEN-1 代码核对 |

### 阶段 1：L0 Production QA 统一编排 + GEN-1 对接（3-4 周）

| 任务 | 交付 | 验证 |
|---|---|---|
| 定义 render receipt JSON 格式（render_hash、edl_hash、时长、帧率、音轨、交付 profile） | receipt schema + GEN-1 输出适配 | GEN-1 export 自动产出 receipt |
| 实现 L0 编排器：Artifact Intake → hash 校验 → ffprobe 测量 → EDL 读回核对 → cut boundary 检查 → PASS/FAIL/INCOMPLETE | L0 统一入口脚本/模块 | 用 GEN-1 产出的成片跑 L0，报告覆盖与限制 |
| 实现交付 profile 检查（帧率/音轨/画幅/时长与预期相符） | profile 检查模块 | 构造不符样本验证 FAIL |
| L0 影子模式接入 GEN-1 export 后自动触发 | 对接脚本 | GEN-1 export → 自动 L0 报告 |
| 集成 VMAF（有参考源时） | VMAF 适配器 | 用已知压缩样本验证 |

**放行条件：** L0 能对 GEN-1 产出的成片稳定报告覆盖范围与限制；误报率可接受；不产出 Film Quality PASS。

### 阶段 2：Finding 模型 + 服务化（3-4 周）

| 任务 | 交付 | 验证 |
|---|---|---|
| 实现企业级 Finding 数据模型（finding_id/category/severity/time_range/claim_kind/evidence_refs/confidence/review_status） | Finding schema + 生成逻辑 | 现有观察可转换为 finding |
| 实现 ProductionQAAssessment 和 FQLAssessment 模型 | Assessment schema | L0 结果可序列化为 Assessment |
| 封装 REST API（POST /v1/production-qa、GET /v1/production-qa/{id}、POST /v1/assessments 等） | HTTP 服务 | API 契约测试通过 |
| 实现 Finding 审核接口（POST /v1/findings/{id}:review） | 审核流程 | 可确认/驳回/修正 finding |
| 报告生成（L0 Report 与 Film Quality Report 分离） | 报告模板 | 两份报告独立输出 |

**放行条件：** API 契约测试通过；L0 和 Film Quality 报告独立；finding 带时间定位和证据来源。

### 阶段 3：Film Quality 测量 + VLM 观察候选（4-6 周）

| 任务 | 交付 | 验证 |
|---|---|---|
| 接入 Doubao-Seed-1.6-vision 产镜头级观察候选（与 GEN-1 M6 选型一致） | VLM 适配器 + 候选 finding 生成 | 候选标 MODEL_OBSERVATION，不进 Gate |
| 完善 Measurement Adapters（黑帧/损坏帧/音频连续性/对白可懂度） | 测量适配器集合 | 各测量有合成 fixture 验证 |
| 实现 Profile v0.1（先覆盖 1-2 种片型，如短视频/产品广告） | Profile 定义 + 门槛预注册 | 门槛可版本化、可批准 |
| Observation/Finding Engine 编排（测量 + VLM 候选 → 统一 finding 流） | Finding Engine | finding 流可追溯来源 |

**放行条件：** 测量可复现；VLM 候选定位误差可量化；Profile v0.1 覆盖 1-2 种片型；候选确认率有基线数据。

### 阶段 4：人审 + 盲评 + 回归闭环（6-8 周）

| 任务 | 交付 | 验证 |
|---|---|---|
| 建立 Pairwise 盲评系统（随机化、左右交换、版本 hash 绑定、盲化） | Pairwise 模块 | 盲评协议测试通过 |
| Human Review 工作流（评审群体、意见理由、冲突裁决） | 人审流程 | 可管理评审任务 |
| Revision Verification（目标区/控制区/回归 findings/人类偏好变化） | 修订验证模块 | 目标修复 + 控制区回归可验证 |
| 与 GEN-1 Brain revision 链路打通 | 对接协议 | finding → Brain revision → Resolve receipt → FQL 重测 |
| 治理、审计、回滚、模型更新流程 | 运维文档 | 生产准入检查通过 |

**放行条件：** 盲评一致性达标；回归可拦截；误报/漏报监控运行；责任人批准。

---

## 六、立即行动项（本周可启动）

1. **将 `film-quality-loop-main` 迁出 `_inspect/`**，建立正式 FQL 工作仓库（1 天）
2. **跑通 FQL 基线**：用 Sintel trailer 跑 preflight → score → repair_plan → user_report，记录基线（1 天）
3. **定义 render receipt 格式**：与 GEN-1 的 export 产物对齐，确定 receipt 字段（2 天）
4. **GEN-1 侧适配**：在 GEN-1 export 时输出 render receipt（不阻塞 M6 W1-W3，可在 W3-W4 加入）（2 天）
5. **L0 编排器设计**：基于 FQL 现有模块设计统一 L0 入口，明确哪些检查是确定性的、哪些是 SELF_QA（3 天）

---

## 七、风险与注意事项

| 风险 | 影响 | 缓解 |
|---|---|---|
| FQL 仓库从 _inspect 迁出后与历史引用断裂 | 文档/链接失效 | 保留 _inspect 副本或建立软链接；更新所有引用路径 |
| GEN-1 M6 执行与 FQL 对接冲突 | 两边进度互相阻塞 | FQL 对接只要求 GEN-1 输出 receipt，不改变 M6 核心任务；receipt 输出可在 M6 W3 后加入 |
| FQL 现有 EXPERIMENTAL 能力直接用于生产 | 误报/不可靠 | 严格按 STUB→EXPERIMENTAL→VALIDATED→PRODUCTION 晋级；L0 先影子模式 |
| VLM 选型与 GEN-1 不一致 | 维护两套适配器 | 统一用 Doubao-Seed-1.6-vision（GEN-1 M6 已选定），FQL 复用同一适配器层 |
| 人审资源不到位 | 阶段 4 无法闭环 | 阶段 1-3 不依赖人审；人审从抽样开始，用工具降低评审负担 |
| 现有 46 个 Schema 与企业级数据模型不一致 | 迁移成本 | 企业级模型（Finding/Assessment/Profile）作为新 Schema 增量加入，不破坏现有 Schema |

---

## 八、结论

1. **FQL 仓库实际存在**且有相当工程基础（46 Schema、完整 CLI、证据链框架、科学治理），企业级方案的"NOT_DETERMINED"需修正为"代码存在但全部 EXPERIMENTAL/STUB，未通过企业级验收"。
2. **应在现有 FQL 仓库基础上演进，不新建**。核心理念与企业级方案一致，从零建会浪费已有基础。
3. **最大缺口是集成层**（L0 统一编排、GEN-1 对接、服务化、Finding 模型），而非底层技术。
4. **推荐路线**：仓库就位（1周）→ L0 + GEN-1 对接（3-4周）→ Finding 模型 + 服务化（3-4周）→ Film Quality 测量 + VLM 候选（4-6周）→ 人审 + 盲评 + 回归闭环（6-8周）。
5. **本周可启动**：仓库迁出、基线跑通、render receipt 格式定义。
