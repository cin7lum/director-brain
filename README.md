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
python -m pytest tests -q          # 612 passed, 3 skipped（环境条件跳过）
python scripts/doctor.py           # 环境自检

# 一键演示（素材 + 意图 → 成片 + 计分卡 + 评审）
python scripts/demo.py -i 素材.mp4 --intent "快剪风格，避免模糊镜头" -t 8

# 全功能（语义选片 + 多方案 + 镜头卡 + 转场）
python scripts/demo.py -i 素材.mp4 -t 8 --semantic --variants 3 --card fast_cut --transitions
```

配置走 `.env`（模板见 `.env.example`；**密钥永不入库**）。LLM 决策链需要火山方舟 key（`ARK_API_KEY` + `ARK_MODEL` 钉扎准入快照 ID）。

## 架构

```
director_brain/          # 内核：brief → 故事图 → 选片内核 → 校验/修复 → 状态机
  contracts/             # 6 份边界模型 JSON Schema（机器可读，跨系统集成）
  cards/                 # 镜头卡库（内置 + user 卡，schema 校验即插拔）
observation_service/     # 观测：镜头发现(scenedetect)/确定性分析/VLM/ASR/节拍网格
execution/               # 预览渲染器（ffmpeg 单命令 filter_complex，NVENC）
api/                     # REST：brief/plan/confirm/validate/revision/ledger/film-context
scripts/                 # roughcut / demo / 评估栈（l1_metrics/l2_matrix/judge）
tests/                   # 612 项（单元/集成/契约）
storage/                 # 决策账本存储（sqlite repository）
```

双链设计：链 A（确定性启发式，生产）；链 B（已准入 LLM 的语义决策，影子对账治理中）。
信号通路一律四态管理（EXPERIMENTAL/SHADOW/ACTIVE/RETIRED），决策消费强制过闸门
（`ensure_decision_use_allowed`，fail-closed）。

## 验收基线

- 120 任务正式矩阵（40 素材 × 3 意图，dev/holdout 隔离）：**出片率 80%、硬伤率 0/96、拒绝正确率 96%、错误 0、dev/holdout PASS**
- 模型准入：32 例冻结基准全门（评测器治理链含反证批次与冻结哈希）
- 切点自然度 75%（PySceneDetect 权威边界同源）；重要性 F1 precision 1.0

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
