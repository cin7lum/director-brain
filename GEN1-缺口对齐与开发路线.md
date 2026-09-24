# GEN-1 仓库检查 × Director Brain 企业级方案 — 缺口对齐与开发路线

> 生成日期：2026-09-24
> 检查范围：`D:\新建豆包\gen1-roughcut` 全仓库（源码 60+ 模块、合同 4 份、文档 90+、conformance 44 项）
> 对照方案：`E:\02-Director-Brain-企业级方案.md` V1.1

---

## 一、GEN-1 仓库真实状态

### 1.1 项目定位与当前阶段

- **定位**：个人自用智能导演系统，核心是 Director Brain。北极星："喂新素材、不改代码，自动产出'普通人敢发朋友圈'的成片"。
- **当前阶段**：科目三（全功能集成路考 A/B），4 片已出（`scratch/kt3_{A,B}_{0918,4421}.mp4`）待看片反馈。科目四（盲测）未开始。
- **Git 头**：`09ed3a2`（master）。

### 1.2 环境验证结果（verify_env.ps1 实跑）

| 项 | 结果 |
|---|---|
| Python 3.14.7 + cv2/numpy/PIL | PASS |
| faster-whisper + ctranslate2 + CUDA(1设备) | PASS |
| ffmpeg 9.0.1 + ffprobe | PASS |
| ollama 在 PATH | PASS |
| ollama 服务运行 | **FAIL**（localhost:11434 无法连接） |
| ASR 模型权重 large-v3-turbo 1.51GB | PASS |
| JEV key (TYPESAFE_API_KEY) | WARN（未配置，LUT/转场/音乐走纯函数兜底） |
| 智谱 key (ZHIPU_API_KEY) | 未检出（脚本未单独报，.env 可能有） |
| conformance 测试 | **44 passed** |
| Sintel + VID0819 基线素材 | PASS |

**结论**：13 PASS / 1 FAIL / 1 WARN。唯一 FAIL 是 ollama 服务未启动，不影响代码审查和 conformance。

### 1.3 核心模块盘点（60+ 模块，按职责分组）

| 职责域 | 已实现模块 | 成熟度 |
|---|---|---|
| **导演决策** | `director_planner.py`(prompt v0.5_arc)、`creative_brief.py`(v1合同)、`narrative_arc.py`(4幕)、`relation_inference.py`(Tier-A)、`relation_producer.py`、`plan_repair.py`(Rule1-12)、`proposal_validator.py`、`director_playbook.md` | **高** — 全链路已串联 |
| **视觉理解** | `shot_discovery.py`、`deterministic_analysis.py`(8项指标)、`vlm_adapter.py`、`ollama_vlm_adapter.py`、`zhipu_vlm_adapter.py`、`shot_function_normalize.py`、`canonical_vocabulary.py` | 中高 — VLM 已接但喂帧有 bug |
| **候选与索引** | `candidate_pool.py`、`shot_index.py`(Call Sheet)、`clip_window.py`、`fix_candidates.py` | 高 |
| **Critic 闭环** | `critic_loop.py`、`director_critic.py`、`quality_scores.py`(F3+gates+motion proxy)、`eval_director.py` | 中 — 纯规则版已上线，VLM 版待接 |
| **基线** | `heuristic_proposal.py`(blur×VLM权重) | 高 — 永久基线，零漂移约束 |
| **后期元素** | `music_selector.py`、`sfx_events.py`、`lut_selector.py`、`transition_selector.py`、`font_selector.py`、`jev_adapter.py` | 中 — JEV 未配置时走纯函数 |
| **时间线与渲染** | `timeline_builder.py`、`timeline_rewriter.py`、`render.py`、`subtitle.py`、`asr.py` | 高 |
| **编辑器与服务** | `editor_api.py`、`editor_session.py`、`app_service.py`、`cli.py`、`hub_arbitrator.py`(ollama/智谱/JEV仲裁) | 高 |
| **证据与审计** | `evidence_package.py`、`decision_bundle.py`、`director_decisions_store.py`、`semantics_auditor.py` | 中高 |
| **模型适配** | `text_llm_adapter.py`(glm-4-flash, timeout30, 重试)、`ollama_vlm_adapter.py`、`zhipu_vlm_adapter.py` | 高 |

### 1.4 合同状态（FROZEN V0.1）

| 合同 | 状态 | 关键内容 |
|---|---|---|
| `AI_PROPOSAL_CONTRACT_V0.1.md` | FROZEN，44-test 通过 | proposal_id/rough_cut_id 分离；slots 含 source_shot_id/proposed_in/out_us/reason/confidence；不可变 |
| `EDIT_EVENT_CONTRACT_V0.1.md` | FROZEN | 编辑事件；V0.2 延期项：TRANSITION/MUSIC/COLOR/SPEED/AUDIO_LEVEL |
| `IMPLICIT_PREFERENCE_EXTRACTION_RULES_V0.1.md` | FROZEN | L2/L3 偏好提取规则 |
| `vlm_tags_schema_v0.2.md` | 非 FROZEN（sidecar） | VLM 标签 schema |

### 1.5 已知技术债与 bug

**AGENTS.md 明确登记的技术债**：
- `plan_repair` rule11/12/13 是待偿还技术债，Critic 闭环能替代后删除。
  - 实查代码：rule 10（节奏后处理：hook提升/尾部长镜裁剪/快速剪辑/rhythm CV）、rule 11（打断同parent长连续，插入filler）、rule 12（移动已选slot进入连续区）。**代码中无 rule 13**，可能已合并或移除。
- 不新增硬编码审美规则。

**GEN1_DIRECTOR_ALIGNMENT_REPORT.md 登记的 5 个 bug（B1-B5）**：
| # | bug | 位置 | 状态 |
|---|---|---|---|
| B1 | replace 路径 slot_id 永远写死 `"slot_01"` | editor_api.py L61-72 / editor_session.py L39-51 | 待修 |
| B2 | EXPORT 事件 file_hash 硬编码 `"sha256:fixture"` | commands.py L419 | 待修 |
| B3 | VLM 喂素材首帧而非镜头关键帧 | analysis_pipeline.py L33-34 | 待修 |
| B4 | render 切原始全分辨率而非 480p proxy | render.py L43 | 待修 |
| B5 | L2/L3 偏好信号从不落盘 | project_layout.py L24 建了目录但无写入 | 待修 |

---

## 二、企业级方案 vs GEN-1 缺口对齐

### 2.1 九组件对齐表

| # | 企业级方案组件 | GEN-1 对应实现 | 对齐度 | 缺口说明 |
|---|---|---|---|---|
| 1 | **Intent Intake / Brief Compiler** | `creative_brief.py` v1：14 字段（goal/audience/platform/target_duration/genre/style/tone/narrative_focus/must_include/must_avoid/music_preference/subtitle_preference/technical_constraints/human_control_level），含校验和落盘 | ✅ **已有基础** | 企业级方案的 DirectorBrief 多了 themes/relationships/emotional_arc/visual_language/editing_language/sound_language/privacy_constraints/approval_state。GEN-1 的 brief 是精简版，可增量扩展。 |
| 2 | **Film Context Gateway（四层渐进披露）** | `candidate_pool.py` 把所有镜头拍平成 dict；`shot_index.py` 构建 Call Sheet；`analysis_pipeline.py` 做分析。**无分层**，一次性全量喂入 | ⚠️ **部分** | 缺 PROJECT/ASSET/SCENE/EVIDENCE 四层划分和按需展开路由。当前 ≤20 镜头时全量喂入可行，镜头量增长后需要分层。 |
| 3 | **Analyze Once Reuse Many（缓存）** | analysis sidecar 文件落盘（`analysis_<sid>.json`、`vlm_tags_<sid>.json`），但**无统一缓存键和失效策略** | ⚠️ **部分** | 缺 analysis_fingerprint（content_hash + provider/model/version + prompt/schema + sampling + timebase）和自动失效。当前靠文件存在性判断，source 变化后不会自动重跑。 |
| 4 | **Story Graph** | `relation_inference.py` Tier-A：REACTION/CONTRAST/MONTAGE 三种关系；`relation_producer.py`；`edit_relation.py`。**只有镜头间关系，无人物/地点/事件实体** | ⚠️ **部分** | 缺 FilmEntity（人物/地点/事件）和完整 StoryGraph（节点+边+推断状态）。当前关系推断是纯函数从候选字段算，不是图谱查询。 |
| 5 | **Director Reasoner** | `director_planner.py`：Call Sheet → 文本 LLM (glm-4-flash) → EditPlan → `narrative_arc` 4幕分配 → `relation_inference` → `plan_repair` 确定性修复 → `proposal_validator`。prompt v0.5_arc，含 style hints、brief 接入、director playbook | ✅ **已有核心** | 企业级方案要求"生成少量可比较方案"，GEN-1 当前只生成 1 个方案。多方案对比是增量能力。 |
| 6 | **Strategy Confirmation（hash 绑定）** | `creative_brief.py` 的 `human_control_level` 枚举：FULL_AUTO / CONFIRM_IF_LOW_CONFIDENCE / HUMAN_IN_LOOP。**有 HITL 级别，但无 plan_hash/edl_hash 绑定确认流程** | ⚠️ **部分** | 缺 hash 绑定和"变更使确认失效"机制。当前 HITL 是粗粒度级别，不是版本化确认。 |
| 7 | **Plan Validator** | `proposal_validator.py`：校验 source_shot_id 存在、帧范围合法、去重、时长容差。fail-closed | ✅ **已有** | 企业级方案要求 hard rules 与 artistic choices 分列，GEN-1 当前 validator 主要是物理规则，artistic 校验较弱。 |
| 8 | **Revision Loop** | `critic_loop.py` + `director_critic.py`：自评 → 发现问题 → `timeline_rewriter.py` 重剪。`plan_repair.py` 做确定性修复。**有闭环雏形** | ✅ **已有基础** | 企业级方案要求 FQL finding → 最小 EDL 修订提案 → 确认 → 回归。GEN-1 当前 critic 是纯规则版，VLM 版待接，修订提案的可解释性较弱。 |
| 9 | **Decision Ledger** | `director_decisions_store.py` + `evidence_package.py` + `decision_bundle.py`：决策记录、证据包、决策束落盘 | ✅ **已有基础** | 企业级方案要求追加保存 context 快照、analysis_fingerprint、候选、用户确认、EDL hash、规则检查、版本关系。GEN-1 有落盘但字段不全，缺版本链追溯。 |

### 2.2 对齐度汇总

```
✅ 已有基础/核心：  Brief Compiler、Director Reasoner、Plan Validator、Revision Loop、Decision Ledger  (5/9)
⚠️ 部分实现：       Film Context Gateway、Analyze Once Reuse Many、Story Graph、Strategy Confirmation  (4/9)
❌ 完全缺失：       无
```

**关键发现：企业级方案描述的 9 个组件中，没有一个是 GEN-1 完全空白的。5 个已有可工作的基础，4 个有部分实现。企业级方案更像是 GEN-1 的"理想化目标态"，而不是从零开始的新系统。**

### 2.3 企业级方案与 GEN-1 的范式差异

| 维度 | 企业级方案 | GEN-1 现状 | 差异本质 |
|---|---|---|---|
| 系统形态 | 四系统分离（Brain/Arsenal/DaVinci/FQL），版本化合同通信 | 单体 Python 进程，全链路内聚 | 架构解耦程度 |
| 定位 | 企业级，多租户，权限隔离 | 个人自用，单机 | 部署规模 |
| 上下文管理 | 四层渐进披露，按需加载 | 全量拍平，≤20 镜头 | 规模适配 |
| 方案生成 | 少量可比较方案 + 候选 + 不确定性 | 单方案 + 确定性修复 | 决策多样性 |
| 证据分级 | 6 级（OBSERVED/MEASURED/HUMAN_CONFIRMED/INFERRED/USER_ASSERTED/NOT_DETERMINED） | confidence_type 5 级（SELF_REPORTED/MODEL_PROBABILITY/HEURISTIC/CALIBRATED/UNAVAILABLE） | 粒度不同，可映射 |
| 策略确认 | hash 绑定，变更失效 | human_control_level 三级 | 确认粒度 |
| 技术栈 | 未指定，倾向成熟服务 + 局部自研 | 单机纯 Python stdlib，ollama 本地 + 智谱云端 | 工程约束 |

---

## 三、开发方向选择

### 3.1 两个路径

**路径 A：在 GEN-1 基础上增量演进**
- 不新建项目，在 `gen1-roughcut` 上继续迭代
- 把企业级方案的概念逐层吸收（证据分级、缓存指纹、策略确认、多方案）
- 优先解决科目三/四的实际问题
- 远期再考虑架构解耦

**路径 B：新建 Director Brain 项目，GEN-1 作为遗留资产迁移**
- 在 `D:\新建豆包\AI-Director` 下按企业级方案架构从头实现
- GEN-1 模块作为参考/迁移来源
- 一次性达到企业级目标态

### 3.2 选择路径 A 的理由

1. **GEN-1 已有 60+ 可工作模块、44 项 conformance 测试、完整出片链路、零漂移基线**。新建项目意味着重写这些，且会丢失测试基线。
2. **企业级方案自身说"V1 复用现有 Agent Harness、结构化输出和适配器"**，设计上就是增量演进，不是推倒重来。
3. **企业级方案 9 组件中 5 个已有基础、4 个部分实现**，没有从零开始的必要。
4. **当前最紧迫的是科目三看片反馈和科目四盲测**，不是架构重建。架构解耦应在能力验证之后。
5. **GEN-1 的 AGENTS.md 有严格的双轨 DoD（卫生线 + 能力线）和零漂移约束**，这是质量保障，新建项目会丢失这些纪律。
6. **企业级方案的"四系统分离"在个人自用场景下是过度设计**。Brain/Arsenal/DaVinci/FQL 分离的收益在多团队、多租户时才显现。

### 3.3 路径 A 的边界

- **不做**：不新建独立项目、不重写已有模块、不破坏 FROZEN V0.1 合同、不丢零漂移基线。
- **做**：在 GEN-1 上增量吸收企业级方案的高价值概念，优先补 4 个"部分实现"组件的缺口。
- **远期**：当 GEN-1 能力线稳定、镜头量超过单进程舒适区、或有多用户需求时，再按企业级方案做架构解耦。

---

## 四、分阶段开发路线

### 阶段 0：当前收敛（1-2 周）— 科目三/四闭环

**目标**：把已出的 4 片看完，能力线有真实数据，修完已知 bug。

| 项 | 内容 | 验证方式 |
|---|---|---|
| 0.1 | 启动 ollama 服务，确认 qwen2.5:14b + qwen3-vl 可用 | `ollama list` |
| 0.2 | 看片 KT3 A/B 4 片，登记 CAPABILITY_SCORECARD | docs/CAPABILITY_SCORECARD.md 更新 |
| 0.3 | 修 B1（slot_id 写死）+ B2（EXPORT 假哈希） | conformance 44 passed + 导出事件 slot_id 正确 |
| 0.4 | 修 B3（VLM 喂镜头关键帧而非首帧） | VLM 标签与镜头内容对应 |
| 0.5 | 修 B4（render 用 proxy 而非原文件） | 导出耗时下降，画质肉眼无差 |
| 0.6 | 修 B5（L2/L3 偏好落盘） | 导出后 preferences/ 目录有文件 |
| 0.7 | 科目四盲测设计与执行 | 盲评结果登记 |

**放行条件**：4 片看片反馈完成、5 个 bug 全修、conformance 44 passed、stub 零漂移 hash 不变、科目四盲测有初步结果。

### 阶段 1：证据与缓存加固（2-3 周）— 补企业级方案缺口

**目标**：把企业级方案的"证据分级"和"缓存复用"吸收到 GEN-1，提升可追溯性。

| 项 | 内容 | 对应企业级组件 | 验证方式 |
|---|---|---|---|
| 1.1 | 统一证据分级：将 confidence_type 扩展为 6 级（MEASURED/MODEL_OBSERVATION/HUMAN_CONFIRMED/INFERRED/USER_ASSERTED/NOT_DETERMINED），与方案第 7 节对齐 | 证据类型 | FilmObservation.claim_kind 与证据类型一致 |
| 1.2 | 给 analysis sidecar 加 analysis_fingerprint：content_hash + provider/model/version + prompt/schema + sampling + timebase | Analyze Once Reuse Many | 同 hash 命中缓存不重跑；source 变化后失效 |
| 1.3 | 缓存失效策略：source_content_hash 变化、provider/model 变化、schema 版本变化 → 自动失效重跑 | Analyze Once Reuse Many | 变更后重跑，不变更时命中 |
| 1.4 | Decision Ledger 补全字段：context 快照引用、analysis_fingerprint、候选列表、版本关系 | Decision Ledger | director_decisions/<id>.json 含完整字段 |
| 1.5 | Story Graph 扩展：在 relation_inference 基础上增加人物/地点实体抽取（从 VLM 标签和 ASR 转写中提取） | Story Graph | FilmEntity + StoryRelation 可查询 |

**放行条件**：证据分级全链路一致、缓存命中率可监控、决策记录可追溯版本链、人物/地点实体可从素材中抽取。

### 阶段 2：决策能力升级（3-4 周）— 多方案 + 策略确认

**目标**：从单方案升级为可比较方案，补策略确认 hash 绑定。

| 项 | 内容 | 对应企业级组件 | 验证方式 |
|---|---|---|---|
| 2.1 | Director Reasoner 生成 2-3 个可比较方案（不同叙事弧/节奏/镜头选择），每个带目的、证据、候选、不确定性 | Director Reasoner | 同输入产出多方案，差异可解释 |
| 2.2 | 方案间对比展示：结构、片长、关键取舍、EDL 摘要 | Strategy Confirmation | 用户可看到方案差异 |
| 2.3 | Strategy Confirmation：用户批准特定 plan_hash，变更后确认失效 | Strategy Confirmation | hash 绑定；未确认计划不可执行；变更后需重新确认 |
| 2.4 | Plan Validator 分列 hard rules 与 artistic choices | Plan Validator | 物理规则确定性校验；艺术选择可解释不强制 |
| 2.5 | Critic VLM 版接入：从导出 mp4 抽帧让 VLM 判视觉语言一致性 | Revision Loop | F1 VISUAL_LANGUAGE_COHERENCE 可运行 |

**放行条件**：多方案生成稳定、策略确认流程跑通、hard/artistic 分列清晰、Critic VLM 版在测试集上有分数。

### 阶段 3：上下文分层（2-3 周）— 规模适配

**目标**：补 Film Context Gateway 四层渐进披露，支撑镜头量增长。

| 项 | 内容 | 对应企业级组件 | 验证方式 |
|---|---|---|---|
| 3.1 | 定义 Context Routing Policy：decision_type + confidence + evidence_coverage → 需要展开的层 | Film Context Gateway | 路由规则可复现 |
| 3.2 | PROJECT 层：项目摘要（brief、约束、计划版本、素材覆盖） | Film Context Gateway | L0 只读时不传入完整媒体 |
| 3.3 | ASSET 层：媒体索引（hash、时长、帧率、音轨、转写缓存、镜头索引） | Film Context Gateway | 按资产查询不加载帧 |
| 3.4 | SCENE 层：相关片段上下文（时间窗、观察、候选镜头、故事图证据） | Film Context Gateway | 针对当前事件展开 |
| 3.5 | EVIDENCE 层：按需原始证据（filmstrip、波形、邻接镜头），仅歧义/高风险时加载 | Film Context Gateway | 无目标时不做全帧倾倒 |

**放行条件**：四层可独立查询、路由策略可复现、大素材集（>50 镜头）下上下文 token 量可控。

### 阶段 4：架构解耦评估（远期，按需）

**触发条件**（满足任一才启动）：
- 单进程内存/性能成为瓶颈（镜头量 >200 或并发项目 >3）
- 多用户/多租户需求出现
- DaVinci Execution 需要独立部署
- FQL 需要独立服务化

**解耦方向**：按企业级方案四系统分离，Brain 抽为独立服务，通过版本化 JSON Schema/OpenAPI 与 Arsenal/DaVinci/FQL 通信。

---

## 五、立即行动项（本周）

按优先级排序：

1. **启动 ollama 服务**，确认本地模型可用（`ollama serve`，然后 `ollama list` 确认 qwen2.5:14b + qwen3-vl）。
2. **看片 KT3 A/B 4 片**，在 `docs/CAPABILITY_SCORECARD.md` 登记真实反馈。这是能力线的唯一权威数据来源。
3. **修 B1 + B2**（slot_id 写死 + EXPORT 假哈希）。这两个不修，L2 偏好信号全是脏的，后续所有分析不可信。
4. **修 B3**（VLM 喂镜头关键帧）。这直接影响导演规划的输入质量。
5. **确认 .env 中 ZHIPU_API_KEY 是否配置**。director_planner 依赖 glm-4-flash，如果 key 缺失则 LLM 路径不可用。

---

## 六、风险与注意事项

1. **不要在阶段 0 之前动架构**。科目三/四的真实反馈是所有后续决策的依据，没有数据就谈架构是空中楼阁。
2. **FROZEN V0.1 合同不可动**。所有新能力走 sidecar 或新文件，不改合同语义。企业级方案的新数据模型（FilmContextSnapshot、StoryGraph 等）先作为 sidecar 验证，成熟后再考虑 V0.2。
3. **零漂移基线不可破**。stub 模式下 Sintel hash `efab44ef...` 和 VID hash `1b8dd9b9...` 必须逐字节一致。任何改动先跑 stub 验证。
4. **heuristic 是永久基线**。LLM 路径失败时必须回退 heuristic，不可让 LLM 成为单点。
5. **plan_repair rule10/11/12 是技术债**。Critic 闭环成熟后应删除，不要在这些规则上继续叠加逻辑。
6. **企业级方案的"四系统分离"在当前阶段不适用**。个人自用场景下单体更高效，解耦是远期选项。

---

*本报告基于 GEN-1 仓库实读（源码、合同、文档、环境实跑）与企业级方案 V1.1 对照生成。所有模块名、行号、hash 均来自实际检查，未做推测。*
