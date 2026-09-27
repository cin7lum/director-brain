# 阶段 7.5 · 链 B SHADOW 影子接入 — 真实素材验证证据

**日期**: 2026-09-28 ｜ **模型**: doubao-seed-2-1-lite-260915（阶段 7 准入快照）
**代码**: director_brain/semantic_shadow.py + pathway_protocol 注册（semantic_reasoner=SHADOW）+ scripts/roughcut.py 接线

## 验证跑（100CANON 真实相机素材）

| 跑 | 素材/目标 | 意图 | 链 A 结果 | 链 B 影子结果 | L1 计分卡 |
|---|---|---|---|---|---|
| tech_4198 | MVI_4198 (9.76s→3s) | 快剪+不要模糊+不要过暗 | 出片 PASS | NEEDS_CONTEXT / adjust_pacing；未覆盖 blurry_visuals, underexposed_shots | **PASS**（偏差 8.0%） |
| semantic_4215 | MVI_4215 (6.92s→5s) | 保留动作完整+不要切碎+慢节奏 | 修复后出片 PASS | NEEDS_CONTEXT / adjust_pacing；未覆盖 duration_extension, shot_identity, source_media | **PASS**（偏差 8.0%） |

账本核验：`decision_ledger` 出现 `semantic_shadow_completed`×2，detail 含 model 快照 ID / prompt v1.1 / 通路状态（治理条件"三版本留痕"满足）。
sidecar：`*.shadow.json` 与 `.edl.json/.plan.json` 同族落盘。mp4 不入库（与 L2 outputs 同惯例），可由日志+sidecar 复现。

## 影子期发现（转后续工作项）

1. **双语词表不通**（新发现）：链 B 输出英文受控词表（blurry_visuals/underexposed_shots/duration_extension），链 A 编码用户原词（模糊/过暗）。对账按字面子串如实报"未覆盖"——**不加双语映射表**（关键词映射=红线禁止的兜底语义修补）。处置方向：归入"语义类 must_avoid 无确定性执法"治理项，ACTIVE 决策前须给出机制级方案（如 validator 增加受控词表规则层），不得在影子对账器里打补丁。
2. **状态分歧模式**：两跑链 B 均判 NEEDS_CONTEXT（要求节奏上下文），链 A 照常出片。属保守性分歧（链 B 更谨慎），无安全风险；持续观测。
3. **路径覆盖**：影子覆盖"正常验证后"与"修复放弃"两条退出路径（真素材第一跑即触发修复放弃路径并成功对账）；**EvidenceTooPoorError 路径（plan 尚未生成）暂不覆盖**——该路径无 plan_id/账本句柄，列为 7.5 已知边界，后续阶段补。

## 防御性设计自证

- 适配器抛意外异常按传输失败处理（响亮降级），影子故障绝不阻断主链（测试 test_adapter_unexpected_exception_never_blocks_chain）。
- 无 ARK_API_KEY → 响亮跳过；ARK_MODEL 未钉扎 → 拒绝构造（防回退被拒模型）。
- 消费闸门：`ensure_decision_use_allowed("semantic_reasoner")` 非 ACTIVE 即抛——影子输出在协议层无法驱动决策。

## 文件清单

- `run_tech_4198.log` / `run_semantic_4215.log` — 端到端运行日志（含影子对账打印）
- `shadow_*.mp4.{edl,plan,shadow}.json` — 三族 sidecar
- `l1_tech_4198.log` / `l1_semantic_4215.log` — L1 计分卡（双 PASS）
