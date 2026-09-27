# Model Admission Decision

## Final result: NO_MODEL_ADMITTED

No candidate model meets all admission thresholds through the formal production path.

## Threshold comparison

| Metric | Threshold | Doubao (POC) | qwen2.5:14b | qwen3-vl |
|--------|-----------|-------------|-------------|----------|
| schema_valid | >=95% | 100% ✓ | 100% ✓ | partial ✗ |
| positive_intent | >=95% | 100% ✓ | 85.7% ✗ | partial ✗ |
| negative_constraint | >=90% | 93.8% ✓ | 62.5% ✗ | partial ✗ |
| direction | >=95% | 100% ✓ | 100% ✓ | partial ✗ |
| ambiguity | >=95% | 100% ✓ | 100% ✓ | partial ✗ |
| status_match | >=90% | 100% ✓ | 65.6% ✗ | partial ✗ |
| hallucination | =0 | 0 ✓ | 0 ✓ | 0 ✓ |
| tool_leakage | =0 | 0 ✓ | 0 ✓ | >0 ✗ |
| Vertical Slice | PASS | PASS ✓ | PASS ✓ | FAIL ✗ |
| Adapter callable | YES | NO ✗ | YES ✓ | YES ✓ |

## Classification
- **Doubao**: POC_REFERENCE_ONLY — meets thresholds but not callable
- **qwen2.5:14b**: REJECT_PRIMARY — retained as LOCAL_BASELINE_CANDIDATE
- **qwen3-vl**: REJECT_PRIMARY — hard gate failure (VS + empty content + leakage)

## Paths forward (for Chief decision)
1. **Provide Doubao/Ark API key** → thin OpenAI-compatible adapter → re-run admission
2. **Install stronger local model** (qwen2.5:32b / qwen3:32b) if hardware permits
3. **Accept qwen2.5:14b as interim** with documented semantic limitations
   (NOT recommended — negative constraint 62.5% is a safety concern)
4. **Chief provides external model endpoint** (Zhipu GLM, etc.) with API key

## NOT done (out of scope)
- No Model Router
- No fallback/voting
- No prompt tuning per model
- No benchmark modification
- No schema modification

---

# ADMITTED: doubao-seed-2-1-lite-260915（2026-09-28 更新）

**本节为 2026-09-28 准入决定，上节历史记录原样保留。**

## 结果：ADMITTED_PRIMARY = doubao-seed-2-1-lite-260915（快照 ID 钉扎，禁止别名）

原 NO_MODEL_ADMITTED 唯一缺口（negative_constraint 87.5% < 90%）经三路专家组独立裁定为**评测器构念缺陷**（SD-28 冻结前已立案的表达差异 + SD-22 匹配面不含 target），非模型能力缺陷。评测器升 v1.1（匹配面修正、阈值/金标/prompt/案例零改动、双版本分数存证、反证测试通过），并以**全新 API 确认跑**（32/32 成功，约 0.1 元）取得准入分数：

| 指标 | 阈值 | doubao-seed-2-1-lite-260915（v1.1 全新确认跑） | doubao-seed-2-1-pro-260915（v1.1 全新确认跑） | qwen2.5:14b（本地 v1.1 反证） |
|---|---|---|---|---|
| schema_valid | >=95% | 100% ✓ | 100% ✓ | 100% ✓ |
| positive_intent | >=95% | 95.238% ✓（原始分） | 95.238% ✓（原始分） | 81.0% ✗ |
| negative_constraint | >=90% | **100% ✓** | **100% ✓** | **62.5% ✗（与历史聚合分一致）** |
| direction | >=95% | 100% ✓ | 100% ✓ | 100% ✓ |
| ambiguity | >=95% | 100% ✓ | 100% ✓ | 100% ✓ |
| status_match | >=90% | 100% ✓ | 93.8% ✓ | 65.6% ✗ |
| hallucination | =0 | 0 ✓ | 0 ✓ | 0 ✓ |
| tool_leakage | =0 | 0 ✓ | 0 ✓ | 0 ✓ |
| Vertical Slice | PASS | PASS ✓ | PASS ✓ | PASS ✓ |

- **doubao-seed-2-1-pro-260915**: ADMITTED_HOT_STANDBY（与 lite 同族热备，lite 限流/下线时按降级链切换，切换前须重跑冻结基准）
- **qwen2.5:14b**: REJECT 再次确认——v1.1 下本地全量重跑 32 例，负向 62.5% 与历史分毫不差，证明其失败为真实语义失败，评测器修正未对其虚增
- 证据: `evidence/PHASE7_EVALUATOR_V1_1/`（RESCORE_AUDIT.md + FROZEN_EVALUATOR_SHA256.md + 全部原始输出）

## 准入绑定（冻结三元组）

本准入仅对以下三元组有效，任一变更须重走完整准入流程：
1. 冻结 prompt（prompt.md，SHA256 见 FROZEN_EVALUATOR_SHA256.md）
2. 冻结 32 例基准 + 金标（SHA256 `834239E9…` / `544E31A3…`）
3. 评测器 v1.1（SHA256 `A42B33C0…`；一次性修正，不存在 v1.2 迭代路径）

## 运行状态与补偿性控制

1. **链 B 先 SHADOW 后 ACTIVE**：语义通路按 pathway_protocol 以 SHADOW 运行，L2 真实素材矩阵连续达标 + L3 盲评通过后方可置 ACTIVE；**置 ACTIVE 须用户本人确认**，不得由开发顺手改注册表。
2. **负向约束双轨执法不放松**：P1-a 确定性过滤、校验器 Rule 9、渲染闸门保持现状。已知风险登记：非技术词表的语义类 must_avoid 目前无独立确定性执法（intent_constraints.py 归入 unverifiable），模型是唯一防线——SHADOW 期内对此类约束人工抽检。
3. **L1 计分卡为每片必过的 CI 闸门**（l1_metrics.py，硬伤 exit 1），长期不豁免。
4. **账本留痕**：每次成片在决策账本记录 model 快照 ID、prompt 版本、评测器版本，形成可复算链。
5. **已知局限**：SD-11/SD-21 类语义近邻分歧（单例轮间波动，95.238% 贴线）列入已知局限清单。

## 吊销标准（触发任一即回退 SHADOW 或 RETIRED）

1. L1 CI 闸门出现归因于语义决策的硬伤，短窗口反复（如 3 次内 2 次）且非素材原因；
2. 真实素材上技术词表之外的负向约束漏检率实测 >10%；
3. 生产路径出现任何一次幻觉（exact_value 捏造）或工具/API 泄漏；
4. 定期复跑冻结基准（每月或模型方版本变更时）任一阈值跌破——API 模型存在供应商静默更新漂移面；
5. L3 盲评显著劣于 SHADOW 期预期，或用户端质量投诉集中出现；
6. 发现任何一次通路协议被绕过（fail-closed 被静默关闭、SHADOW 信号驱动了决策）。

## 供给风险触发器（用户动作条件）

降级链：`doubao-seed-2-1-lite-260915 → doubao-seed-2-1-pro-260915 → DeepSeek V4.1-Flash / GLM-5.3-Flash（需新 key + 重跑准入）`。出现以下情形之一时请用户注册备选 key：
- **T1**（立即动作）：方舟出现 lite/pro 下线公告（约 3 个月迁移窗口，注册有等待期，公告即行动）；
- **T2**：单价大涨（如输入价涨超 3 倍）或个人配额收紧；
- **T3**：生产质量回退且新快照重跑基准不达标；
- **T4**：若当年 lite/pro 未过闸（本情景已消除）。
