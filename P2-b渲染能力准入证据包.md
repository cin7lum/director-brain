# P2-b 渲染能力准入证据包（转场 / 音频 / 字幕）· 2026-09-27

执行：GLM ｜ 依据：前置工程准入规则（自研前必须交证据包）+ 链路审计发现 5.1/5.3
（渲染器只认 3 个时间字段，EDL 表现字段全部休眠）
调研：专家组市调（FFmpeg 8.0.3 官方滤镜、moviepy 2.2.1、PyAV 16、faster-whisper）
结论速览：**三类能力全部落在"采用/配置级"——零新依赖，全部用 ffmpeg 官方内置滤镜，
免自研准入**。落地顺序：字幕 → 音频 → 转场（改动半径从小到大，成片时间线机制先行）。

---

## 能力 1 · 字幕（第 1 期，改动半径最小）

**检索**：`subtitles` 滤镜（libass 烧录，FFmpeg 1.1 起）、`mov_text` 软字幕（MP4 标准）、
SRT 公开格式、faster-whisper 词级时间戳（已在本项目 ASR 观测中就绪）。

**候选对比**：
| 方案 | 可开关 | 性能 | 兼容性 | 判定 |
|---|---|---|---|---|
| mov_text 软字幕 | 可 | 零重编码 | 桌面好/网页参差 | **默认档** |
| subtitles 滤镜烧录 | 不可 | 本项目反正重编码，边际≈0 | 100% | 导出档开关 |
| .srt 旁挂 | 可（外部播放器） | 零 | 视播放器 | 调试产物 |

**差距分析**：ASR 观测已存在；缺口 = ① segments→SRT 序列化（薄封装 ~15 行，
公开格式非自研）② **成片时间线映射**（转场引入后成片时间 ≠ 素材时间——由导演层
在 EDL 汇总时产出"成片时间线 SRT"根治，渲染器对时间戳零假设）③ 输出端加流/滤镜。

**结论：采用（配置级）+ SRT 薄序列化。** 落地：`subtitle_refs =
[{"path", "lang": "chi", "hardsub": false}]`；渲染器软字幕 `-c:s mov_text`，
硬字幕末端 `subtitles=filename=...:force_style='Fontname=Microsoft YaHei,...'`。
风险：时间线错位（已用导演层归一机制根治）；Windows libass 字体需固定字体名。

## 能力 2 · 音频（第 2 期）

**检索**：`amix`（多轨混音，weights 权重）、`sidechaincompress`（侧链 ducking，
FFmpeg 2.8 起）、`volume/afade/aloop/loudnorm` 全内置；FFmpeg 8.0 甚至官方加了
whisper 滤镜（佐证音频走 ffmpeg 主线正确）。pydub 停滞、moviepy 慢 5-20 倍——均不采用。

**候选对比**：
| 方案 | ducking | 改动量 | 判定 |
|---|---|---|---|
| amix 固定权重（`weights="1 0.35"`） | 无（音乐恒定小音量） | 极小 | **一期采用** |
| sidechaincompress+amix | 有（threshold 0.03/ratio 8/attack 50ms/release 500ms） | 小 | 二期 |
| moviepy/pydub/sox | — | 大 | 不采用 |

**差距分析**：现音频链 = atrim→concat 单链；缺口 = 配乐第二输入 + 混音图
（渲染器 ~40 行音频图生成函数）+ ducking 参数试听调优（唯一非代码工作量）。

**结论：采用（配置级）。** 落地：`audio_refs = [{"path", "role": "music",
"gain_db": -8, "duck": true, "fade_in": 1.0, "fade_out": 2.0}]`；
一期只做 amix 权重（效果稳定），ducking 二期。
风险：sidechaincompress 的 threshold 是线性值非 dB（0.125≈-18dBFS）易踩坑；
持续环境音会把音乐永久压低——duck 触发需响度门限。

## 能力 3 · 转场（第 3 期，改动最大）

**检索**：`xfade`（FFmpeg 4.3 起，8.0.3 共 **57 种**转场含 custom 表达式）+
`acrossfade`（音频侧，曲线族可配）；NVENC 兼容（xfade 是 CPU 滤镜，CPU 滤镜链 +
h264_nvenc 编码是标准流水线）；替代方案 moviepy（慢 5-20 倍、官方自认 >100 源内存
问题）、PyAV（等于重造渲染器=自研级）、concat+黑场白闪（合法降级路径）——均不采用。

**语义差异（关键数学）**：concat 无重叠（总时长=Σ段）；xfade 重叠混合
（总时长 = Σd − ΣD）；offset_k = Σ_{i≤k} d_i − Σ_{j<k} D_j − D_k，
offset ≤ 段1时长 − duration 否则尾帧冻结。两输入须同分辨率/帧率/时间基
（本项目单源 trim 天然满足）。

**结论：采用（配置级）——链式 xfade 的 offset 递推是渲染器内 ~20 行数学，
滤镜/编码零缺口。** 落地：
`EditItem.transition = {"type": "cut"|"xfade", "name": "fade|dissolve|...",
"duration": 0.5, "audio_duration": 0.5}`（挂在出点侧=本镜头与下一镜头之间；
audio_duration 预留 J/L-cut）；全 cut 时走旧 concat 路径（渐进增强非破坏）。
**职责**：导演层 policy 决定何时何处加什么转场（审美决策，默认全 cut——
策略将来挂情绪转折边/VLM 语义）；渲染器只解释执行，未知转场名报错回退 cut。
风险：offset 算错→尾帧冻结（数学公式可测试兜底）；多源场景需前置格式归一。

---

## 统一落地顺序与理由

**字幕 → 音频 → 转场**：① 字幕不动 concat 主链（输出端加流），且"成片时间线"
机制是转场的地基；② 音频是旁路；③ 转场重构视频主结构并改变总时长语义，放最后，
且届时字幕已走成片时间线自动跟随不返工。每期旧路径保持可用（字段空 = 现行为），
EDL 渐进增强而非破坏，渲染器始终可回退。

**审批状态**：本证据包结论为"采用/配置级"（ffmpeg 官方滤镜，零新依赖、零自研），
按准入规则免完整准入，可直接进入实现排期。实现排期待项目所有者确认优先级。
