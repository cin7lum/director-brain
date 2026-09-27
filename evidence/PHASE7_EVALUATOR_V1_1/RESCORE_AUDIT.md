# 评测器 v1.1 重算与反证审计（RESCORE AUDIT）

**日期**: 2026-09-28 ｜ **执行**: ZCode（受用户授权按专家组裁定执行）
**裁定依据**: 三路专家组独立裁定（评测方法学 / 系统治理 / 模型路径），一致结论：Path A（评测器构念修正）合法且必要；Path B（换模型绕过）逻辑不成立——评测器缺陷对所有正确使用 schema 的模型一视同仁扣分；Path C（静默豁免）拒绝。

## 一、重算矩阵（全部存量输出集 × 双版本评测器）

| 输出集 | v1.0 负向召回 | v1.1 负向召回 | 判定变化 |
|---|---|---|---|
| doubao-seed-2-1-lite（14125c6 存量） | 87.5%（14/16） | **100%（16/16）** | REJECT → PASS* |
| doubao-seed-2-1-pro（14125c6 存量） | 87.5%（14/16） | **100%（16/16）** | REJECT → PASS* |
| doubao-seed-2-0-lite（反证组） | 68.75%（11/16） | 81.25%（13/16） | **仍 REJECT** ✓ |
| POC Doubao（冻结前轮） | 93.75%（15/16） | 100%（16/16） | 参照（当时 NOT callable） |
| qwen3-vl:latest（残缺存量 18/32 例） | 40% | 40% | 零翻转，**仍 REJECT** ✓ |
| qwen2.5:14b（本地全量重跑） | 历史聚合 62.5% | 见 §三 | **仍 REJECT** ✓ |

\* 存量输出的 PASS 仅为审计结论；**准入分数以 §四 全新 API 确认跑为准**（专家组硬条件 5）。

注：commit 14125c6 提交说明中 "21/24" 为分母笔误，实际负向实例总数为 16（14/16 = 87.5%，百分比相同）。以本审计数据为准。

## 二、逐案翻转审计（全部输出集合计仅两个翻转点）

| 案例 | 金标负向标签 | v1.0 判定 | v1.1 判定 | 通道 | 依据 |
|---|---|---|---|---|---|
| SD-28 | conflicting_constraints | MISS | HIT | status_enum | 冻结前 KNOWN_ISSUES.md 第 4 条立案："表达差异，非语义漏检"；模型 status=CONFLICTING_CONSTRAINTS（schema 枚举） |
| SD-22 | preserve_reaction_shot | MISS | HIT | target_guarded | 模型 target 含 reaction_shot 令牌 + must_preserve=[shot_identity]、must_avoid=[shot_substitution]（承诺非空，守卫通过） |

其余全部 30 个负向实例在两版本下判定一致。SD-11（金标 extend_visible_duration vs 模型 adjust_pacing）属 positive 侧真实语义近邻分歧，v1.1 不涉及、不翻转。

## 三、反证测试（差分效度：修尺子不能把落榜生送过关）

1. **doubao-seed-2-0-lite**（全量 32 例存量，14125c6 轮 REJECT）：v1.1 下 81.25% < 90% → **仍拒绝** ✓（与专家组离线预演 68.8%→81.2% 完全一致）
2. **qwen3-vl:latest**（残缺存量，无 SD-22/SD-28）：v1.0/v1.1 同为 40% → 零翻转 ✓
3. **qwen2.5:14b**（历史 REJECT_PRIMARY，负向 62.5%，原始输出未存档）→ 本地 Ollama 生产适配器（LLMAdapter，schema 约束生成）全量 32 例重跑：32/32 成功，中位延迟 4537ms。**v1.1 下负向 62.5%，与历史聚合分分毫不差**（positive 81.0% ✗、status 65.6% ✗）——其失败为真实语义失败，评测器修正对其零虚增，差分效度最强形态 ✓
   - 运行日志: `falsify_qwen25_14b_run.log` ｜ 输出: `experiments/director_semantic_poc/model_outputs_qwen25_14b_v11_check.jsonl`
   - v1.1 评分: `results_falsify_qwen25_14b_v1.1.json` + `eval_falsify_qwen25_14b_v1.1.log`

## 四、全新 API 确认跑（准入分数，专家组硬条件 5）

**doubao-seed-2-1-lite-260915**（快照 ID 钉扎）：32/32 调用成功，中位延迟 3365ms，总耗时 142s，成本约 0.1 元。冻结 prompt 原样传输。

| 指标 | 阈值 | 实测（评测器 v1.1） | 判定 |
|---|---|---|---|
| schema_valid | ≥95% | 100% | ✓ |
| positive_intent | ≥95% | 95.238%（原始分，20/21；缺 SD-21） | ✓ 贴线通过，未做任何取整 |
| **negative_constraint** | **≥90%** | **100%（16/16）** | **✓ 闸门指标** |
| direction | ≥95% | 100% | ✓ |
| ambiguity | ≥95% | 100% | ✓ |
| status_match | ≥90% | 100% | ✓ |
| exact_param_hallucination | =0 | 0 | ✓ |
| tool_leakage | =0 | 0 | ✓ |
| vertical_slice (SD-01) | PASS | PASS | ✓ |

翻转明细：SD-28 经 status_enum 通道；SD-22 本轮直接命中字段数组（模型轮间非确定性，两通道均合法）。正召回缺口由存量轮的 SD-11 轮换为 SD-21（单例波动，同一分数 95.238%）。

**doubao-seed-2-1-pro-260915** 确认跑：32/32 成功，中位延迟 5576ms，总耗时 217s。v1.1 下 schema 100% / positive 95.238%（原始分）/ **negative 100%** / direction 100% / ambiguity 100% / status 93.8% / 幻觉 0 / 泄漏 0 / VS PASS —— 热备型号同等验证通过（`results_confirm_21pro_v1.1.json`）。

一次性规则：确认跑结果如实接受。若未来复检跌破阈值，按吊销标准处理（MODEL_ADMISSION.md §吊销），**不出现评测器 v1.2**。

## 五、文件清单

- `FROZEN_EVALUATOR_SHA256.md` — 评测器双版本哈希封存 + 冻结工件复核
- `results_{21lite,21pro,20lite,poc,qwen3vl}_v{1.0,1.1}.json` + 对应 `eval_*.log` — 重算矩阵原始产物
- `results_confirm_21lite_v1.1.json` / `confirm_run_21lite.log` — 准入确认跑（lite）
- `results_confirm_21pro_v1.1.json` / `confirm_run_21pro.log` — 热备确认跑（pro）
- `results_falsify_qwen25_14b_v1.1.json` / `falsify_qwen25_14b_run.log` — 强化反证
- `experiments/director_semantic_poc/model_outputs_confirm_2_1_{lite,pro}.jsonl`、`model_outputs_qwen25_14b_v11_check.jsonl` — 全新原始输出
- `experiments/director_semantic_poc/run_benchmark_local.py` — 本地通道 runner（只做传输）
