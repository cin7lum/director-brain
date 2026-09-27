# Phase-7 Admission Round · Volcano Ark (Doubao) — 2026-09-27

执行：GLM ｜ 触发：项目所有者提供 ARK API key ｜ 基线：c72f0d1 冻结基准
（32 例 / human_reference 金标 / 冻结阈值 schema≥95 positive≥95 negative≥90
direction≥95 ambiguity≥95 status≥90 hallucination=0 leakage=0 / VS 硬例 PASS）
红线遵守：未改 prompt（SYSTEM_PROMPT v1.1）、未改用例、未改金标、未下调阈值。
传输层适配（与 zhipu 同族，已记录）：json_object + schema 内嵌 system prompt +
`thinking:{type:disabled}`（seed 系思考模式与 json_object 冲突致 ~50% 空内容）+
markdown 围栏剥离（~20% 用例）+ 传输级重试（间歇性空内容）。

## 裁决：`NO_MODEL_ADMITTED`（维持）——但缺口收窄至一条

| 模型 | schema | positive | negative | direction | ambiguity | status | 幻觉 | 泄漏 | VS | 延迟中位 | 裁决 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| doubao-seed-2-0-lite-260428 | 100% | 85.7% | 68.8% | 100% | 100% | 96.9% | 0 | 1 | PASS | 5.4s | REJECT（4 项未达标） |
| doubao-seed-2-1-lite-260915 | 100% | 95.2% | **87.5%** | 100% | 100% | 93.8% | 0 | 0 | PASS | 3.5s | REJECT（1 项未达标） |
| doubao-seed-2-1-pro-260915 | 100% | 95.2% | **87.5%** | 100% | 100% | 96.9% | 0 | 0 | PASS | 5.5s | REJECT（1 项未达标） |

对照历史：qwen2.5:14b 负约束 62.5% / glm-4-flash 56.2% → **doubao-seed-2-1 达 87.5%**，
距 90% 阈值仅差 1 条约束（21/24，需 22/24）。

## 缺口解剖（两代模型同缺，逐条）

| # | 缺失项 | 性质 | 备注 |
|---|---|---|---|
| 1 | SD-28 `conflicting_constraints` | **已知评测器表示差异孤立例** | 初代豆包 POC 评测（pre-freeze）即标注 ISOLATED_CASE：该金标键实为状态语义（应报 CONFLICTING_CONSTRAINTS）却挂在负约束召回指标下 |
| 2 | SD-22 `preserve_reaction_shot` | 真实缺失 | 模型未输出该 preserve 项（两代同缺） |
| 3 | SD-11 正向缺失 `extend_visible_duration`→实际 `adjust_pacing` | 真实缺失（影响 positive 85.2→95.2 之间） | 语义近邻但词不匹配精确子串判定 |

## 待主控/所有者裁定的治理问题（执行器无权决定）

1. **SD-28 孤立例先例**：初代 POC 评测对同一用例同一问题已标注"评测器表示差异"。
   若沿用该先例将其从负约束召回分母剔除（21/23 = 91.3% ≥ 90%），
   doubao-seed-2-1 即**全指标达标**。这是金标口径裁定，不是执行器可做的下调。
2. 评测器精确子串匹配对语义近邻（adjust_pacing vs extend_visible_duration）
   系统性低估——KNOWN_ISSUES ⑧ 已记录，同样属冻结评测器的已知局限。

## 传输层发现（对所有后续候选模型有效）

- seed 系思考模式 + json_object → 间歇性空内容（实测 0-62% 浮动）；`thinking:disabled` 后单例稳定但批量仍间歇 → 传输级重试（3 次/2s/6s）+ 1s 间隔后 32/32 成功
- json_object 模式下 ~20% 输出带 markdown 围栏 → 适配器剥离
- 成本实测：3 轮 32 例 ≈ 0.3 元（lite/pro 混合）

## 输出物

- `model_outputs_doubao_seed_2_0_lite.jsonl` / `_2_1_lite.jsonl` / `_2_1_pro.jsonl`（全量原始输出，含失败例 _error/_raw）
- `eval_lite.log` / `eval_21lite.log` / `eval_21pro.log` + eval_results.json（每次评分覆盖，最终为 pro 轮）
- 运行器：`experiments/director_semantic_poc/run_benchmark.py`（可对任意 Ark 模型复跑）
