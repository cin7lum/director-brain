# Frozen Evaluator Identity（评测器 v1.1 封存）

**日期**: 2026-09-28 ｜ **裁定**: 专家组三路（评测方法学/系统治理/模型路径）一致有条件支持，条件全部履行（见 RESCORE_AUDIT.md）

## 冻结不变的工件（复核于 2026-09-28）

| 工件 | SHA256 | 与封存记录 |
|---|---|---|
| benchmark.jsonl（32 例） | `834239E94D634C4FC85CFE705409148C1DB4CDC6C902574B529E78AAED0FCACF` | 一致 ✓ |
| human_reference.jsonl（金标） | `544E31A37B13AB08F11B4CB74647AF28BD535985E37426943A6989EBDA0E04D9` | 一致 ✓ |
| prompt.md（冻结 SYSTEM_PROMPT v1.1） | `E7819469889EE646…`（本轮起记录） | 未改动 |
| 阈值表（7 项 + 幻觉/泄漏 =0） | MODEL_ADMISSION.md 所载 | 未改动 |

**本次修正只动了评测器的匹配面，没有动考卷、金标、prompt、阈值中的任何一个。**

## 评测器身份

| 版本 | SHA256 | 说明 |
|---|---|---|
| v1.0（冻结轮原版） | `0C69A71E2A04C6F7405643C09ABBFAA2267C54DA72C2B1B7AC667BF3B9661110` | `git show 0f4ffd7:experiments/director_semantic_poc/evaluate.py` |
| v1.1（构念修正版） | `A42B33C0FAE50703974E016E4FDD81E8371F3264D4316423E6A55CF34C71BC2F` | 当前工作区 `experiments/director_semantic_poc/evaluate.py` |

v1.0 可随时用 `--evaluator-version 1.0` 复现原分数；results JSON 内含 `evaluator_version` 字段，分数必须绑定评测器版本引用。

## v1.1 修正定义（case 无关，无案例特判）

语义标准不变："模型是否表达了这条负向约束"。在 v1.0 的三数组子串匹配之外，新增两个由**冻结 prompt schema 自身定义**的强类型证据通道：

1. **status_enum 通道**：金标负向标签（去 avoid_/preserve_ 前缀后）与模型 status 枚举值全等。依据：金标 `conflicting_constraints` 本身就是 schema 的 status 枚举字面量；冻结前 `DIRECTOR_BRAIN_SEMANTIC_DECISION_POC/KNOWN_ISSUES.md` 第 4 条已立案此表达差异（先于任何候选轮）。
2. **target_guarded 通道（带矛盾守卫）**：金标标签以令牌形式出现在模型 target 列表中，**且**模型做出了对应承诺（preserve_* → must_preserve 非空；avoid_* → must_avoid 非空）。裸提及 target 而无任何承诺不给分。

永久排除面：creative_intent / evidence / user_terminology 等自由文本字段（不可反证，防滑坡）。评测器任何后续变更必须重走全套流程（反证 + 双版本 + 逐案审计 + 新鲜确认跑），**一次性修正，禁止迭代到过线**。

## 敏感性声明（专家组硬条件 7）

仅加 status_enum 通道的最小修正即可使 2.1 双模型达 15/16 = 93.8% ≥ 90% 过闸（SD-22 经 target_guarded 通道后为 16/16）。闸门结论对最窄修正稳健，不存在"扩面恰好够过线"的定制嫌疑。
