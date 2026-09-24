# GEN-1 拆解 × Director Brain 企业级方案 — 搭建方案

> 生成日期：2026-09-24
> 前提：用户明确要求"把 GEN-1 拆解，按方案来搭建"，而非在 GEN-1 上增量演进
> 基于：GEN-1 仓库实查（60+ 模块、4 份合同、90+ 文档、conformance 44 项）+ 企业级方案 V1.1

---

## 一、需要先确认的 4 个关键决策

以下决策直接决定项目结构和技术栈，不确认无法开工。我给出推荐值，但不自行默认。

### 决策 1：技术栈 — Python stdlib 还是引入框架？

| 选项 | 说明 | 代价 |
|---|---|---|
| **A. 沿用 GEN-1 纯 Python stdlib** | HTTP 用 urllib，无 Web 框架，接口用手写 JSON 校验 | 开发慢，但零依赖、可移植、与 GEN-1 纪律一致 |
| **B. 轻量框架（FastAPI + Pydantic）** | OpenAPI 自动生成、schema 校验、异步支持 | 引入依赖，但企业级方案要求 OpenAPI 接口，框架天然适配 |
| **C. OpenAI Agents SDK** | 企业级方案提到的候选，但 GEN-1 明确"不引 openai SDK" | 协议兼容≠功能兼容，且与 GEN-1 纪律冲突 |

**推荐 B**。理由：企业级方案明确要求"版本化 JSON Schema / OpenAPI"接口，FastAPI + Pydantic 天然生成，手写 urllib 维护成本高。GEN-1 的 stdlib 纪律是个人自用项目的约束，企业级方案定位不同。

### 决策 2：四系统分离 — 真拆服务还是单仓多模块？

| 选项 | 说明 |
|---|---|
| **A. 单仓多模块** | 一个代码库，brain/arsenal/davinci/fql 分目录，通过内部函数调用，接口合同先定义好 |
| **B. 真拆 4 个独立服务** | 4 个进程/容器，HTTP 通信，各自独立部署 |
| **C. 先单仓后拆** | V1 单仓多模块，接口按 HTTP 合同设计，远期拆服务 |

**推荐 C**。理由：企业级方案说"四系统共享版本化事实合同，各自拥有独立状态"，但 V1 阶段真拆 4 个服务会大幅增加联调成本。先单仓、接口按 HTTP 合同设计，后续拆服务时只需把函数调用换成 HTTP 调用。

### 决策 3：存储 — SQLite + 本地文件还是 PostgreSQL + S3？

| 选项 | 说明 |
|---|---|
| **A. SQLite + 本地文件系统** | 零运维，适合开发和个人使用 |
| **B. PostgreSQL + S3 兼容存储** | 企业级方案提到的"关系型数据和对象存储"，适合生产 |
| **C. 先 SQLite，数据层抽象，远期切 PostgreSQL** | 开发期用 SQLite，Repository 模式隔离，生产切 PG |

**推荐 C**。理由：V1 开发期 SQLite 足够，Repository 抽象保证可迁移。对象存储先用本地目录模拟 S3 接口。

### 决策 4：GEN-1 FROZEN V0.1 合同 — 兼容还是重定义？

| 选项 | 说明 |
|---|---|
| **A. 新系统用企业级方案数据模型，V0.1 合同仅作迁移参考** | 不兼容，迁移时做格式转换 |
| **B. 新系统兼容 V0.1 合同，企业级模型作为扩展** | 兼容，但 V0.1 的扁平 slots 结构与企业级 EDL/StoryGraph 有范式差异 |
| **C. 新系统用企业级模型，提供 V0.1 导入导出适配器** | 内部用新模型，边界处做转换 |

**推荐 C**。理由：企业级方案的数据模型（FilmContextSnapshot、StoryGraph、EditorialDecisionList、DirectorDecisionPlan）比 V0.1 丰富得多，强行兼容会束缚设计。但 GEN-1 已有大量 proposal 数据，需要导入适配器来做影子评估和迁移。

---

## 二、GEN-1 模块拆解映射

按企业级方案的系统边界，将 GEN-1 的 60+ 模块分为五类：**Brain 内核（迁移/改写）、观察服务（迁移）、Arsenal（拆出）、DaVinci Execution（拆出）、FQL（拆出）、丢弃/技术债**。

### 2.1 Brain 内核 — 迁移并改写（9 组件对应）

| 企业级组件 | GEN-1 来源模块 | 迁移方式 | 改写要点 |
|---|---|---|---|
| **1. Brief Compiler** | `creative_brief.py` | 迁移+扩展 | GEN-1 有 14 字段，企业级方案需补 themes/relationships/emotional_arc/visual_language/editing_language/sound_language/privacy_constraints/approval_state。校验逻辑可复用。 |
| **2. Film Context Gateway** | `candidate_pool.py`、`shot_index.py`、`analysis_pipeline.py` | 改写 | GEN-1 是全量拍平，需重构为四层（PROJECT/ASSET/SCENE/EVIDENCE）+ 按需展开路由。`shot_index.build_call_sheet` 可作为 ASSET 层的基础。 |
| **3. Analyze Once Reuse Many** | 无直接对应，analysis sidecar 散落各处 | 新建 | GEN-1 有 `analysis_<sid>.json`、`vlm_tags_<sid>.json` 落盘，但无统一缓存键。需新建缓存层，定义 analysis_fingerprint。 |
| **4. Story Graph** | `relation_inference.py`、`relation_producer.py`、`edit_relation.py` | 迁移+扩展 | GEN-1 只有镜头间 REACTION/CONTRAST/MONTAGE 三种关系。需扩展 FilmEntity（人物/地点/事件）和完整图谱。推断逻辑可复用。 |
| **5. Director Reasoner** | `director_planner.py`、`narrative_arc.py`、`director_playbook.md` | 迁移+改写 | 核心逻辑保留（Call Sheet → LLM → EditPlan → arc → repair）。需改写为多方案生成，每个方案带候选/不确定性/证据引用。prompt v0.5_arc 可作为起点。 |
| **6. Strategy Confirmation** | `creative_brief.py` 的 `human_control_level` | 新建 | GEN-1 只有三级 HITL 枚举，无 hash 绑定。需新建确认流程，绑定 plan_hash/edl_hash。 |
| **7. Plan Validator** | `proposal_validator.py`、`plan_repair.py` 的物理规则部分 | 迁移+分列 | GEN-1 validator 主要是物理规则。需分列 hard rules（时间码/帧范围/源文件/媒体可用/限制性条款）和 artistic choices（镜头 hold/叙事顺序/风格/情绪节奏）。 |
| **8. Revision Loop** | `critic_loop.py` 的输入处理部分、`timeline_rewriter.py` 的修订逻辑 | 迁移+改写 | GEN-1 critic 是自评重剪，企业级方案是 FQL finding → 最小 EDL 修订提案。需将"自评"改为"接收 FQL 外部 finding → 提案"。重写逻辑可复用。 |
| **9. Decision Ledger** | `director_decisions_store.py`、`evidence_package.py`、`decision_bundle.py` | 迁移+补全 | GEN-1 有落盘但字段不全。需补 context 快照引用、analysis_fingerprint、候选列表、用户确认、EDL hash、规则检查、版本关系。 |

### 2.2 观察服务 — 迁移（独立于 Brain）

| GEN-1 模块 | 职责 | 迁移方式 |
|---|---|---|
| `shot_discovery.py` | ffmpeg 场景切分 | 直接迁移 |
| `deterministic_analysis.py` | OpenCV 8 项技术指标（blur/shake/exposure/color 等） | 直接迁移，输出标记为 MEASURED 证据 |
| `vlm_adapter.py` / `ollama_vlm_adapter.py` / `zhipu_vlm_adapter.py` | VLM 单帧标签 | 迁移，需修 B3（喂镜头关键帧而非首帧） |
| `asr.py` | faster-whisper 语音转写 | 直接迁移 |
| `shot_function_normalize.py` | 镜头功能归一化 | 迁移 |
| `canonical_vocabulary.py` / `canonical_vocabulary.json` | 规范词汇表 | 迁移 |
| `video_features.py` | 视频特征 | 迁移 |
| `media_import.py` | 媒体导入 | 迁移 |
| `analysis_pipeline.py` | 分析管道编排 | 改写为观察服务的入口 |

**注意**：企业级方案明确说"Brain 负责解释 FilmObservation，但不拥有原始观测的真实性。观测服务负责记录模型、版本、时间戳、媒体指纹和置信度。" 所以这些模块应拆为独立的观察服务，Brain 通过 FilmObservation 接口消费。

### 2.3 Arsenal — 拆出（能力解析，Brain 不包含）

| GEN-1 模块 | 职责 |
|---|---|
| `hub_arbitrator.py` | Provider 仲裁（ollama/智谱/JEV） |
| `music_selector.py` | 音乐选择 |
| `sfx_events.py` | 音效事件 |
| `lut_selector.py` | LUT 选择 |
| `transition_selector.py` | 转场选择 |
| `font_selector.py` | 字体选择 |
| `jev_adapter.py` | JEV/TypeSafe 适配器 |

**企业级方案原文**："Arsenal 解析能力，DaVinci 执行，FQL 独立评估。" Brain 只输出 EDL 和约束，具体用什么 LUT/转场/音乐由 Arsenal 解析。

### 2.4 DaVinci Execution — 拆出（执行层，Brain 不包含）

| GEN-1 模块 | 职责 |
|---|---|
| `timeline_builder.py` | 时间线构建 |
| `timeline_rewriter.py` | 时间线重写 |
| `render.py` | ffmpeg 渲染 |
| `subtitle.py` | 字幕烧录 |
| `editor_api.py`（执行部分） | 编辑器 API |
| `editor_session.py`（执行部分） | 编辑会话 |
| `app_service.py`（执行部分） | 应用服务 |

**注意**：`editor_api.py` / `editor_session.py` / `app_service.py` 混合了 Brain 决策和 Execution 执行，拆解时需要按职责切分。决策部分（proposal 生成、验证）归 Brain，执行部分（时间线操作、渲染、导出）归 DaVinci Execution。

### 2.5 FQL — 拆出（独立评估，Brain 不包含）

| GEN-1 模块 | 职责 |
|---|---|
| `critic_loop.py` | Critic 循环 |
| `director_critic.py` | 导演批评 |
| `quality_scores.py` | 质量评分（F3 密度 + structural gates + motion proxy） |
| `eval_director.py` | 导演评估 |

**企业级方案原文**："FQL 独立评估"、"将 FQL 的可定位问题转化为边界清楚的修订提案"。Brain 接收 FQL finding 但不自己做 FQL。

### 2.6 丢弃 / 技术债 / 工具

| GEN-1 模块 | 处理方式 | 理由 |
|---|---|---|
| `plan_repair.py` 的 rule 10/11/12 | **丢弃** | AGENTS.md 明确登记为技术债，Critic 闭环替代后删除。rule 1-9 的物理过滤可迁移到 Plan Validator。 |
| `heuristic_proposal.py` | **保留为 baseline provider** | 企业级方案要求"现有确定性 heuristic fallback 作为 baseline provider 与新模型并行影子评估"。不丢弃，作为影子评估基线。 |
| `gui.py` | **丢弃** | 企业级方案无 GUI，V1 是 API 服务。 |
| `fix_candidates.py` | **迁移为工具** | 候选修复逻辑可复用。 |
| `clip_window.py` | **迁移为工具函数** | 中段取片纯函数，通用。 |
| `init_manifest.py` | **迁移为工具** | 清单初始化。 |
| `project_layout.py` | **重写** | 新项目目录结构不同。 |
| `cli.py` | **重写** | 新 CLI 按新接口设计。 |
| `conformance_harness/` | **参考，不直接迁移** | 44 项测试针对 V0.1 合同，新系统有新合同。测试方法论可复用，用例需重写。 |
| 大量 `cli_*/` 运行产物目录 | **不迁移** | 运行产物，不属于代码。 |

### 2.7 拆解汇总

```
GEN-1 (60+ 模块, 单体)
├── Brain 内核（9 组件）→ 迁移+改写，约 15 个模块
├── 观察服务 → 直接迁移，约 10 个模块
├── Arsenal → 拆出，约 7 个模块
├── DaVinci Execution → 拆出，约 6 个模块
├── FQL → 拆出，约 4 个模块
├── 工具函数 → 迁移，约 4 个模块
├── baseline provider → 保留（heuristic_proposal），1 个模块
└── 丢弃/重写 → plan_repair 技术债、gui、cli、project_layout 等
```

---

## 三、新项目目录结构

基于决策 2（单仓多模块，先单仓后拆）和决策 3（SQLite + Repository 抽象）：

```
D:\新建豆包\AI-Director\
├── README.md                          # 项目说明
├── pyproject.toml                     # 依赖管理（FastAPI + Pydantic + pytest）
├── .env.example                       # 环境变量模板
│
├── director_brain/                    # ★ Brain 内核（核心交付物）
│   ├── __init__.py
│   ├── config.py                      # 配置管理
│   ├── models/                        # 数据模型（Pydantic，对应企业级方案第5节）
│   │   ├── __init__.py
│   │   ├── director_brief.py          # DirectorBrief
│   │   ├── film_context.py            # FilmContextSnapshot
│   │   ├── film_observation.py        # FilmObservation + claim_kind 6级
│   │   ├── film_entity.py             # FilmEntity + StoryRelation
│   │   ├── story_graph.py             # StoryGraph
│   │   ├── edl.py                     # EditorialDecisionList + EditItem
│   │   ├── director_plan.py           # DirectorDecisionPlan
│   │   └── revision.py                # RevisionProposal
│   ├── brief_compiler.py              # 组件1: Intent Intake / Brief Compiler
│   ├── context_gateway.py             # 组件2: Film Context Gateway（四层渐进披露）
│   ├── analysis_cache.py              # 组件3: Analyze Once Reuse Many
│   ├── story_graph.py                 # 组件4: Story Graph 管理
│   ├── director_reasoner.py           # 组件5: Director Reasoner（多方案生成）
│   ├── strategy_confirmation.py       # 组件6: Strategy Confirmation（hash绑定）
│   ├── plan_validator.py              # 组件7: Plan Validator（hard/artistic分列）
│   ├── revision_loop.py               # 组件8: Revision Loop（接收FQL finding）
│   ├── decision_ledger.py             # 组件9: Decision Ledger
│   └── prompts/                       # prompt 版本管理
│       ├── director_prompt_v0.1.md    # 从 GEN-1 director_playbook.md 迁移
│       └── ...
│
├── observation_service/               # 观察服务（独立，Brain 通过接口消费）
│   ├── __init__.py
│   ├── shot_discovery.py              # 从 GEN-1 迁移
│   ├── deterministic_analysis.py      # 从 GEN-1 迁移
│   ├── vlm_adapter.py                 # 从 GEN-1 迁移（含 ollama/zhipu）
│   ├── asr.py                         # 从 GEN-1 迁移
│   └── pipeline.py                    # 分析管道编排
│
├── arsenal/                           # Arsenal（能力解析）
│   ├── __init__.py
│   ├── provider_arbiter.py            # 从 hub_arbitrator.py 迁移
│   ├── music_selector.py              # 从 GEN-1 迁移
│   ├── lut_selector.py                # 从 GEN-1 迁移
│   ├── transition_selector.py         # 从 GEN-1 迁移
│   └── ...
│
├── davinci_execution/                 # DaVinci Execution（执行层）
│   ├── __init__.py
│   ├── timeline_builder.py            # 从 GEN-1 迁移
│   ├── render.py                      # 从 GEN-1 迁移
│   └── ...
│
├── fql/                               # FQL（独立评估）
│   ├── __init__.py
│   ├── quality_scores.py              # 从 GEN-1 迁移
│   ├── critic.py                      # 从 critic_loop/director_critic 迁移
│   └── ...
│
├── gen1_adapter/                      # GEN-1 V0.1 兼容层（决策4：导入导出适配器）
│   ├── __init__.py
│   ├── v01_importer.py                # V0.1 proposal → 新 EDL
│   ├── v01_exporter.py                # 新 EDL → V0.1 proposal（影子评估用）
│   └── heuristic_baseline.py          # 从 heuristic_proposal.py 迁移
│
├── storage/                           # 存储层（Repository 抽象）
│   ├── __init__.py
│   ├── repository.py                  # 抽象接口
│   ├── sqlite_repository.py           # SQLite 实现
│   └── object_store.py                # 对象存储（本地目录模拟 S3）
│
├── api/                               # HTTP API（FastAPI，对应企业级方案第6节）
│   ├── __init__.py
│   ├── main.py                        # FastAPI app 入口
│   ├── film_context.py                # POST /v1/film-context:snapshot, GET ...
│   ├── briefs.py                      # POST /v1/briefs:compile
│   ├── director_plans.py              # POST /v1/director-plans:generate, :confirm-strategy, :validate
│   ├── revisions.py                   # POST /v1/revisions:propose
│   └── decision_ledger.py             # GET /v1/projects/{id}/decision-ledger
│
├── tests/                             # 测试
│   ├── unit/                          # 单元测试
│   ├── contract/                      # 合同测试（替代 GEN-1 conformance_harness）
│   └── integration/                   # 集成测试
│
├── docs/                              # 文档
│   ├── architecture.md                # 架构说明
│   ├── api.md                         # API 文档（FastAPI 自动生成亦可）
│   ├── migration_from_gen1.md         # GEN-1 迁移指南
│   └── ...
│
└── scripts/                           # 工具脚本
    ├── verify_env.py                  # 环境检查（从 GEN-1 verify_env.ps1 改写）
    └── import_gen1_project.py         # 导入 GEN-1 项目数据
```

---

## 四、搭建顺序（5 个里程碑）

### M0：项目骨架（1 周）

**目标**：空项目能跑起来，接口合同定义完毕。

| 项 | 内容 | 验证 |
|---|---|---|
| M0.1 | 初始化项目：pyproject.toml、目录结构、README | `pip install -e .` 成功 |
| M0.2 | 定义全部数据模型（Pydantic）：DirectorBrief、FilmContextSnapshot、FilmObservation、FilmEntity、StoryGraph、EditorialDecisionList（含 EditItem）、DirectorDecisionPlan、RevisionProposal | `pytest tests/unit/test_models.py` 通过 |
| M0.3 | 定义 API 接口骨架（FastAPI），8 个端点全部注册，返回 stub | `uvicorn api.main:app` 启动，`/docs` 可见全部端点 |
| M0.4 | 存储层：SQLite + Repository 抽象，对象存储本地模拟 | 基本 CRUD 可运行 |
| M0.5 | GEN-1 V0.1 导入适配器骨架：能读取 V0.1 proposal JSON 并转为内部模型 | 导入 GEN-1 一个 proposal 文件成功 |

**放行条件**：项目可启动、模型可校验、API 文档可见、V0.1 导入可跑通。

### M1：观察服务 + 缓存层（2 周）

**目标**：能分析素材，产出 FilmObservation，缓存可复用。

| 项 | 内容 | 验证 |
|---|---|---|
| M1.1 | 迁移 `shot_discovery.py` + `deterministic_analysis.py`，输出标记为 MEASURED 的 FilmObservation | 对一个测试视频跑通，产出镜头切分和技术指标 |
| M1.2 | 迁移 VLM 适配器（ollama + zhipu），修 B3（喂镜头关键帧），输出 MODEL_OBSERVATION | VLM 标签与镜头内容对应 |
| M1.3 | 迁移 `asr.py`，输出带时间戳的转写 | 对白型素材转写正确 |
| M1.4 | 实现 `analysis_cache.py`：analysis_fingerprint = content_hash + provider/model/version + prompt/schema + sampling + timebase | 同 hash 命中缓存不重跑；source 变化后失效 |
| M1.5 | Film Context Gateway 的 ASSET 层：资产索引（hash、时长、帧率、音轨、转写缓存、镜头索引） | `GET /v1/film-context/{id}?layer=ASSET` 返回正确 |

**放行条件**：素材分析全链路跑通、缓存命中率可监控、ASSET 层可查询。

### M2：Brain 核心决策链（3 周）

**目标**：Brief → Context → Story Graph → Director Reasoner → EDL → Validator 全链路跑通。

| 项 | 内容 | 验证 |
|---|---|---|
| M2.1 | Brief Compiler：从 `creative_brief.py` 迁移，扩展企业级方案字段，冲突检测 | `POST /v1/briefs:compile` 输出 Brief 草案 + 冲突项 |
| M2.2 | Story Graph：从 `relation_inference.py` 迁移 Tier-A 关系，扩展 FilmEntity（人物/地点/事件） | 图谱可查询，推断边带 inference_status 和证据 |
| M2.3 | Film Context Gateway 的 PROJECT + SCENE 层 + Context Routing Policy | 按需展开，L0 不传入完整媒体 |
| M2.4 | Director Reasoner：从 `director_planner.py` 迁移，改写为多方案生成（2-3 个可比较方案），每个带目的/证据/候选/不确定性 | `POST /v1/director-plans:generate` 输出多方案 + EDL 草案 |
| M2.5 | Plan Validator：从 `proposal_validator.py` 迁移，分列 hard rules 和 artistic choices | `POST /v1/director-plans/{id}:validate` 输出分列结果 |
| M2.6 | EDL 的 EditItem schema 定义（企业级方案缺口） | EditItem 含 source_asset、in_frame、out_frame、transition、effect_refs |

**放行条件**：输入 Brief + 素材，输出多方案 EDL，hard rules 可校验，artistic choices 可解释。

### M3：确认 + 修订 + 审计（2 周）

**目标**：策略确认、修订循环、决策账本全链路。

| 项 | 内容 | 验证 |
|---|---|---|
| M3.1 | Strategy Confirmation：hash 绑定，变更使确认失效 | `POST /v1/director-plans/{id}:confirm-strategy` 绑定 plan_hash/edl_hash；变更后确认失效 |
| M3.2 | Revision Loop：接收 FQL finding IDs → 最小 EDL 修订提案 → 确认 → 回归 | `POST /v1/revisions:propose` 输出修订提案，不直接执行 |
| M3.3 | Decision Ledger：追加保存 context 快照、analysis_fingerprint、候选、用户确认、EDL hash、规则检查、版本关系 | `GET /v1/projects/{id}/decision-ledger` 可追溯完整版本链 |
| M3.4 | GEN-1 heuristic baseline 影子评估：新方案与 heuristic 并行跑，结果带来源和版本 | 同输入产出两份 EDL，可对比 |
| M3.5 | EVIDENCE 层：按需原始证据加载（filmstrip、波形、邻接镜头），仅歧义/高风险时 | 无目标时不做全帧倾倒 |

**放行条件**：策略确认流程跑通、修订提案可生成、决策可追溯、影子评估可对比。

### M4：集成 + 验收（2-3 周）

**目标**：与 Arsenal / DaVinci Execution / FQL 联调，三级验收。

| 项 | 内容 | 验证 |
|---|---|---|
| M4.1 | Arsenal 迁移：provider 仲裁 + 音乐/LUT/转场/字体选择 | EDL 输出后 Arsenal 可解析具体能力 |
| M4.2 | DaVinci Execution 迁移：时间线构建 + 渲染 | 确认后的 EDL 可编译为 ChangeSet 并渲染 |
| M4.3 | FQL 迁移：质量评分 + Critic，finding 可回传 Revision Loop | FQL 产出 finding，Brain 生成修订提案 |
| M4.4 | L1 验收：合同、确定性规则、流程重放 | 全部通过 |
| M4.5 | L2 验收：3 个独立项目，代表性素材 + 人工标注对照 | 关键指标达标 |
| M4.6 | L3 验收：5 个独立项目，真实剪辑复核 + 盲评 | 盲评达阈值 |

**放行条件**：四系统联调通过、三级验收达标、安全/隐私评审通过。

---

## 五、GEN-1 资产迁移清单

### 5.1 代码迁移优先级

| 优先级 | 模块 | 目标里程碑 | 迁移方式 |
|---|---|---|---|
| P0 | `creative_brief.py` | M2.1 | 迁移+扩展字段 |
| P0 | `director_planner.py` | M2.4 | 迁移+改写多方案 |
| P0 | `narrative_arc.py` | M2.4 | 直接迁移 |
| P0 | `proposal_validator.py` | M2.5 | 迁移+分列 |
| P0 | `relation_inference.py` | M2.2 | 迁移+扩展实体 |
| P0 | `shot_discovery.py` | M1.1 | 直接迁移 |
| P0 | `deterministic_analysis.py` | M1.1 | 直接迁移 |
| P0 | `vlm_adapter.py` 系列 | M1.2 | 迁移+修 B3 |
| P0 | `asr.py` | M1.3 | 直接迁移 |
| P0 | `heuristic_proposal.py` | M3.4 | 迁移为 baseline |
| P1 | `shot_index.py` | M1.5 | 迁移为 ASSET 层 |
| P1 | `candidate_pool.py` | M1.5 | 改写为 Context 层 |
| P1 | `clip_window.py` | M2.4 | 迁移为工具 |
| P1 | `text_llm_adapter.py` | M2.4 | 迁移为 LLM 适配 |
| P1 | `director_decisions_store.py` | M3.3 | 迁移+补全字段 |
| P1 | `evidence_package.py` | M3.3 | 迁移 |
| P1 | `decision_bundle.py` | M2.4 | 迁移为 DirectorDecisionPlan |
| P1 | `semantics_auditor.py` | M0.2 | 迁移为输入校验 |
| P2 | `hub_arbitrator.py` | M4.1 | 迁移为 Arsenal |
| P2 | `music_selector.py` 等 | M4.1 | 迁移为 Arsenal |
| P2 | `timeline_builder.py` | M4.2 | 迁移为 Execution |
| P2 | `render.py` | M4.2 | 迁移为 Execution |
| P2 | `quality_scores.py` | M4.3 | 迁移为 FQL |
| P2 | `critic_loop.py` | M4.3 | 迁移为 FQL |
| 丢弃 | `plan_repair.py` rule10/11/12 | — | 技术债，rule1-9 物理规则并入 Plan Validator |
| 丢弃 | `gui.py` | — | 企业级方案无 GUI |
| 重写 | `cli.py`、`project_layout.py` | M0 | 新项目结构不同 |

### 5.2 数据迁移

| 资产 | 迁移方式 |
|---|---|
| GEN-1 proposal JSON（`proposals/*.json`） | 通过 `gen1_adapter/v01_importer.py` 转为内部模型，用于影子评估和测试基线 |
| GEN-1 analysis sidecar（`analysis_*.json`、`vlm_tags_*.json`） | 导入时重新计算 analysis_fingerprint，标记为 legacy 来源 |
| GEN-1 事件日志（`events/events.jsonl`） | 仅作审计参考，不导入新系统 |
| GEN-1 4 版 AB 成片 | 作为 L2/L3 验收的对照基线，不导入系统 |
| GEN-1 测试素材（m5_real/） | 直接复用为新系统测试集 |

### 5.3 文档迁移

| 文档 | 处理 |
|---|---|
| `contracts/AI_PROPOSAL_CONTRACT_V0.1.md` 等 4 份 | 归档为 `docs/gen1_v01_contracts/`，作为迁移参考 |
| `docs/DIRECTOR_BRAIN_DESIGN.md` | 参考，核心设计理念吸收到新架构 |
| `docs/CAPABILITY_SCORECARD.md` | 能力基线数据迁移到新系统的验收基准 |
| `docs/FAILURE_LOG.md` | 迁移为新系统的"已证伪的路"清单 |
| `docs/HANDOFF.md` | 归档，环境搭建部分改写为新 `scripts/verify_env.py` |
| `AGENTS.md` | 纪律性内容（双轨 DoD、零漂移、能力线）吸收到新系统的开发规范 |

---

## 六、风险与应对

| 风险 | 早期信号 | 应对 |
|---|---|---|
| **拆解时职责切分不清** | editor_api/app_service 混合决策和执行，拆解时遗漏 | 按"是否产生 EDL/决策"和"是否操作时间线/渲染"切分；产生决策归 Brain，操作执行归 Execution |
| **GEN-1 模块间隐式依赖** | 迁移后模块单独运行失败，依赖隐藏在全局状态 | 迁移时先画依赖图，每个模块迁移后立即跑单元测试 |
| **V0.1 合同与新模型范式差异** | 导入适配器丢失信息（如 V0.1 扁平 slots 无法表达 StoryGraph 推断边） | 导入时标记 INFERRED/NOT_DETERMINED，不补造信息；丢失字段显式记录 |
| **多方案生成质量不稳定** | LLM 产出的方案差异度不足或包含非法 shot_id | Plan Validator 过滤 + heuristic 回退；非法方案丢弃，不发布半成品 |
| **缓存键设计过严导致命中率低** | Analyze Once 实际重复计算率 >50% | 分强键/弱键，弱键变化时增量重分析 |
| **四系统联调周期长** | M4 联调时接口不一致 | M0 就定义全部接口合同（OpenAPI），各系统按合同开发，联调前做合同测试 |
| **plan_repair 技术债丢弃后节奏能力退化** | 新系统没有 rule10/11/12 的节奏后处理，成片节奏变差 | Critic 闭环 + Director Reasoner 的叙事弧设计替代；过渡期可保留 rule10 的 hook 提升作为 artistic choice |

---

## 七、立即行动项（确认决策后）

1. **确认 4 个关键决策**（技术栈/单仓还是拆服务/存储/合同兼容），确认后立即开工 M0。
2. **M0 第 1 天**：初始化项目骨架，`pip install fastapi uvicorn pydantic pytest`，建目录结构。
3. **M0 第 2-3 天**：定义全部 Pydantic 数据模型，这是整个系统的地基。
4. **M0 第 4-5 天**：FastAPI 接口骨架 + SQLite 存储层。
5. **M0 第 6-7 天**：V0.1 导入适配器，能读取 GEN-1 proposal 并转为内部模型。

---

*本方案基于 GEN-1 仓库实查和企业级方案 V1.1 生成。4 个关键决策需用户确认后开工。*
