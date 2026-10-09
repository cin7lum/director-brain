# 02 · Director Brain（导演脑）

[![CI](https://github.com/cin7lum/director-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/cin7lum/director-brain/actions/workflows/ci.yml)

将「用户想剪成什么」（自然语言意图）×「素材里有什么」（多模态观测）转化为**可解释、可追溯、可修订**的剪辑决策，并渲染成片。AI 剪辑流水线（02 导演脑 → 03 军火库 → 04 剪映/Resolve 执行 → 05 质检）中唯一拥有"导演权"的模块，同时是可插拔大模块与独立产品（CLI / REST API）。

**核心立场**：证据不足就拒绝导演（fail-closed）；每条决策可审计；全绿不等于能用——一切能力以真实实测工件为验收标准。

## 能力总览

| 能力 | 说明 |
|---|---|
| 一句话出片 | 素材 + 意图 + 目标时长 → 真实 mp4（确定性链 A 生产） |
| 语义驱动选片 | 多帧 VLM 看懂镜头内容/情绪/叙事角色/重要性，驱动幕分配与排序（`--semantic`） |
| 镜头卡系统 | 12 张导演风格卡 + 参考片学习（"照这感觉剪"→显式参数卡）；`--card` |
| 人物实体解析 | 跨镜头"同一个人"外观聚类（置信分级诚实）+ 叙事连续性加权 |
| 语音驱动 | 对白覆盖率参与镜头价值（`--voice-led`）；J/L-cut 音频时间轴渲染（`--j-cut`/`--l-cut`） |
| 节拍网格 | BGM 节拍检测与切点吸附（librosa；`--bgm`） |
| 多方案对比 | `--variants N`：多变体评分择优，全量落账本 |
| 生产转场 | 幕边界叠化 + 生成端 ΣD 预补偿（ffprobe 实测 33ms 精度） |
| 治理 | Plan 状态机 16 态 + 策略确认 hash 绑定 + 渲染闸门（未确认不渲染）+ 决策账本 + fail-closed 族 |
| 评估 | L1 客观计分卡 / L2 素材矩阵（dev/holdout 防过拟合）/ 冻结量表评审 / 切点自然度 / 重要性 F1 |

## 快速开始

```bash
pip install -e ".[dev]"
python -m pytest tests -q          # 记录本次实际输出；环境会影响 skip 数
python scripts/doctor.py           # 环境自检

# 一键演示（素材 + 意图 → 成片 + 计分卡 + 评审）
python scripts/demo.py -i 素材.mp4 --intent "快剪风格，避免模糊镜头" -t 8

# 全功能（语义选片 + 多方案 + 镜头卡 + 转场）
python scripts/demo.py -i 素材.mp4 -t 8 --semantic --variants 3 --card fast_cut --transitions
```

配置走 `.env`（模板见 `.env.example`；**密钥永不入库**）。LLM 决策链需要火山方舟 key（`ARK_API_KEY` + `ARK_MODEL` 钉扎准入快照 ID）。
项目级 API 需要配置 `DIRECTOR_BRAIN_PROJECT_API_TOKEN`，并只接受 loopback 客户端与 `Authorization: Bearer` 凭证。涉及本地媒体读取、哈希、探测或分析的请求还须配置 `DIRECTOR_BRAIN_ALLOWED_MEDIA_ROOT`，且拒绝媒体根目录之外的路径。素材级权利依据与引用必须逐项登记，但系统将其标记为 `declared_unverified`，不独立判定其法律充分性。只有明确声明 `local_processing_allowed` 的素材才会进入本地路径解析、哈希、探测与分析；未核验素材以及仅允许第三方处理的素材，只能以 `dataset://`、`source://` 或 `asset://` 不透明引用登记，不会被本地读取。第三方处理通路尚未接通，声明第三方许可不会触发云端调用；上下文会逐素材标记为 `not_authorized`，并保留已获准本地处理素材的部分结果。

多素材项目流程按顺序调用 `PUT /v1/projects/{project_id}/film-manifest`、`POST /v1/projects/{project_id}/film-context:snapshot`、`POST /v1/projects/{project_id}/story-graph:build`；用 `GET /v1/projects/{project_id}/story-graph` 读回当前图，或传 `?revision=N` 读取历史版本。需要生成草案时，调用 `POST /v1/projects/{project_id}/director-plans:generate`，提供当前 `manifest_revision`、目标时长和用户意图。Reasoner 保留素材独立时间轴与项目素材身份，不推断跨素材人物/事件关系；项目边界仍为 `declared_unverified`，接口响应将质量验收标为 `not_proven`。未覆盖或未授权素材作为待确认项保留，不会被合成素材替代。没有确定性技术观测的素材不会生成空的四幕图。项目接口都受 loopback 和 Bearer token 约束；上下文读取或分析本地媒体还需要设置媒体根目录。

本地分析可能持续较久时，可调用 `POST /v1/projects/{project_id}/film-context:jobs` 并提供 `Idempotency-Key`；用 `GET /v1/projects/{project_id}/film-context-jobs/{job_id}` 查询持久状态，用 `/events` 读取生命周期记录，也可 POST 到 `{job_id}:cancel` 请求协作式取消。任务、事件、分析缓存和 Context 共用配置的 SQLite 数据库；本机 Huey SQLite 队列在服务重启后恢复未完成任务，并从已提交的逐镜头缓存继续。取消会在当前分析单元结束后生效。`succeeded` 仅表示本次 Context 已持久化，不代表导演质量验收通过。

对已生成的项目 StoryGraph，可调用 `GET /v1/projects/{project_id}/story-link-candidates` 读取未排序的跨素材 Person/Event 候选。默认不带锚点时保留完整候选集与稳定分页；传入 `selected_anchor_story_graph_node_id` 后，只返回该当前 mention 与其他素材的完整候选集合，适合从一个已知人物/事件开始核对。该范围选择不做排序或身份推断，不自动建立关系；无效或已失效锚点会返回 `422`。候选页明确返回 `source_hash_validation_state=not_revalidated_by_read`：分页只读取持久化 Manifest/Context/StoryGraph，不重读大体积源媒体；发起比较、保存关系审阅或生成/确认 Plan 时，服务端仍会复核注册的源文件哈希。每条结果仍须由调用方核对证据，再通过独立的 comparison/review 生命周期处理。

调用方可用候选中的素材 ID、StoryGraph 节点 ID 和关系类型，请求 `GET /v1/projects/{project_id}/story-graph/{story_graph_id}/mentions/{story_graph_node_id}/preview?manifest_revision=N&relation_kind=person_identity|event_identity` 获取该精确 mention 的三张本地预览帧。此接口仅允许 loopback 与 Bearer token；还会检查本地媒体根目录、当前素材权限声明和读取前后源哈希，并将预览抽帧限制为 FFmpeg 的本地 `file` 协议。最多返回三张 1280 像素以内的 JPEG（JSON Base64、每帧不超过 1.5 MB），响应为 `no-store`，不写入数据库、不调用模型，也不泄露本机路径。它只提供人工查看所需的像素证据，不代表人工已经审核、身份正确或降低了审核工作量。

项目导演草案会以请求身份绑定的 ID 原子保存 Brief、Plan 与 EDL，可通过 `GET /v1/projects/{project_id}/director-plans` 查找版本，再用 `GET /v1/projects/{project_id}/director-plans/{plan_id}` 读回；同一输入的重复生成复用既有草案，内容漂移会失败关闭。读取按 project ID 隔离，并标识草案属于当前或历史 manifest revision。该持久化只覆盖未确认草案，不代表策略确认、Resolve 执行或影片质量验收已完成。

## 架构

```
director_brain/          # 内核：brief → 故事图 → 选片内核 → 校验/修复 → 状态机
  contracts/             # 9 份边界模型 JSON Schema（机器可读，跨系统集成）
  cards/                 # 镜头卡库（内置 + user 卡，schema 校验即插拔）
observation_service/     # 观测：镜头发现(scenedetect)/确定性分析/VLM/ASR/节拍网格
execution/               # 预览渲染器（ffmpeg 单命令 filter_complex，NVENC）
api/                     # REST：brief/plan/confirm/validate/revision/ledger/film-context/project-manifest/project-story-graph
scripts/                 # roughcut / demo / 评估栈（l1_metrics/l2_matrix/judge）
tests/                   # 单元/集成/契约
storage/                 # 决策账本存储（sqlite repository）
```

双链设计：链 A（确定性启发式，生产）；链 B（已准入 LLM 的语义决策，影子对账治理中）。
信号通路一律四态管理（EXPERIMENTAL/SHADOW/ACTIVE/RETIRED），决策消费强制过闸门
（`ensure_decision_use_allowed`，fail-closed）。

## 历史实验记录（不代表 Production Candidate 准入）

- 仓库历史 L2 文件记录 120 个任务来自 40 个素材，按素材划分 dev/holdout；它没有独立拍摄项目 ID、完整源哈希和授权收据，不能证明项目级隔离或当前产品质量。
- 历史模型准入的 32 例、切点自然度和重要性 F1 属于各自冻结版本/范围的实验记录，不能替代多素材项目、独立 05 质量评估或 Production Candidate 盲评。
- 当前阶段状态、证据和未验证项以 `docs/production-candidate/` 的带日期记录为准；不对历史阈值或 PASS 标签作追溯性改写。

## 文档

- `02项目交接文档.md` — 能力清单/进度总账/红线/待决
- `导演脑-差距与开发计划.md` — D 系列差距计划与准入证据包
- `镜头卡目录.md` — 卡库总览与选择指引
- `成片质量评估方案-L1L2L3.md` / `开发路线与进度.md` — 评估体系与运行章程

## 红线（改动前必读）

1. 不改 `gate/`（金门禁，裁判不可篡改）
2. 不下调模型准入阈值、不为过基准改 prompt/用例/金标
3. 不删测试放水、不用 `except: pass`
4. 决策函数消费信号通路前必须过 `ensure_decision_use_allowed`（fail-closed）
5. 全绿 ≠ 能用：每个"完成"必须有真实实测工件

## License

MIT（见 LICENSE）。镜头卡 schema 蓝本致谢
[video-shotcraft](https://github.com/Vincentwei1021/video-shotcraft)（Apache-2.0）。
