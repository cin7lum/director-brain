# M2 决策链 P0 修复报告

## 概览

- **基线**：216 passed, 1 skipped
- **修复后**：221 passed, 1 skipped（+5 新增测试，0 退步）
- **真实端到端**：`ALL P0 FIXES VERIFIED`

---

## P0-1：target_duration 概念错误 — 已修复

**问题**：`total_duration = tech_obs[-1].end_frame`（源视频总时长 52.2s）被错误赋给 `DirectorBrief.target_duration`，覆盖了 15s 默认目标。

**修复方式**：
1. `director_brain/models/director_brief.py`：在 `target_duration` 之后新增 `source_duration_us: int | None = None`。
2. `director_brain/brief_compiler.py`：
   - `compile_brief` 新增可选参数 `target_duration_us: int | None = None`。
   - 未传入时用 `DEFAULT_TARGET_DURATION_US`（15s）作为 `target_duration`。
   - 源视频时长（`tech_obs[-1].end_frame`）单独存入 `source_duration_us`，**不再**覆盖 target。
3. 测试更新：
   - `test_brief_compiler.py` 第 108 行：`target_duration == 15_000_000` + `source_duration_us == 9_000_000`
   - 第 127 行：同上模式
   - 新增 `test_explicit_target_duration_us_overrides_default`：传 30s 时 target=30s

**验收**：真实端到端中 `brief.target_duration=15000000`，`source_duration_us=52208333`（52.2s 源素材）。

---

## P0-2：plan_repair 无时长修复能力 — 已修复

**问题**：rule1-9 只处理结构问题，不处理总时长偏离目标范围。

**修复方式**（`director_brain/plan_repair.py`）：
- 执行顺序调整为：rule1-7 → **rule10（时长修复）** → rule8（重算时长+hashes）→ rule9（重新验证）。
- rule10 逻辑：
  1. `from director_brain.plan_validator import _parse_target_duration` 解析 target。
  2. **时长不足**（current < 0.9×target）：
     - 第一步：逐个延长现有片段（优先 rationale 含 `heuristic:` 的），延长量受源镜头 `end_frame` 和 `MAX_CLIP_US` 及下一片段 `in_frame` 约束。
     - 第二步：仍不足时，从未选中的 deterministic_technical 观测中按 blur_score 降序、exposure_ok 优先追加到 EDL，追加后排序并修复重叠。
  3. **时长过长**（current > 1.1×target）：优先缩短非 heuristic: 的最长片段，不低于 MIN_CLIP_US。
  4. 每次延长/追加/截断都在 `plan.open_questions` 记录操作日志。
- 新增测试：`test_rule10_extends_and_appends_when_too_short`、`test_rule10_truncates_when_too_long`。

**验收**：真实端到端中 `validate_plan(redl, rplan, all_obs)` 返回 `True, []`。

---

## P0-3：graph 死参数，四幕结构未被消费 — 已修复

### relation_inference.py
- 从 `graph.nodes` 中四幕节点的 `attributes["shot_ids"]` 建立 `shot_id → act_node_id` 映射。
- 同幕边（from/to 同幕）：confidence += 0.1（上限 0.95）；跨幕边：confidence -= 0.05（下限 0.3）。
- `evidence_refs` 追加幕节点 node_id（如 `act_hook`）。
- 使用 `INFERENCE_STATUS_INFERRED` 常量替代硬编码字符串。
- 测试：`test_all_edges_have_inferred_status_and_evidence` 断言改为 `>= 2`；新增 `test_same_act_edge_gets_confidence_boost_and_act_evidence`。

### director_reasoner.py
- 从 `graph.nodes` 获取每幕 `attributes["shot_ids"]`。
- **每幕单独运行 HeuristicBaseline.generate**（按幕比例分配 target：hook 15%、develop 35%、peak 30%、resolve 20%，每幕至少 MIN_CLIP_US）。
- 某幕选不出时，取该幕 blur_score 最高的完整镜头作为兜底。
- EDL 按幕时间顺序排列（hook→develop→peak→resolve），同幕内按 in_frame 升序。
- `Decision.rationale` 记录 `act=xxx, heuristic:blur=...`。
- `EditItem.shot_function` 基于幕赋值：hook→opening、develop→pacing、peak→peak、resolve→closing。
- 新增测试：`test_four_act_selection_each_act_has_decision_and_act_order`。

**验收**：真实端到端中 6 个 decision 全部带 `act=` 标记，覆盖 hook/develop/peak/resolve 四幕。

---

## P1 处理情况

| # | 问题 | 状态 |
|---|------|------|
| 1 | StoryNodeType 缺 ACT | ✅ 已加 `ACT = "act"`，story_graph_builder 改用 `StoryNodeType.ACT`，测试同步更新 |
| 2 | shot_function 全是 "heuristic_selected" | ✅ 在 P0-3 director_reasoner 中修复为 opening/pacing/peak/closing |
| 3 | _parse_target_duration 复用 | ✅ plan_repair 直接 import，未重复实现 |
| 4 | inference_status 自由字符串 | ✅ 新增 `INFERENCE_STATUS_STRUCTURAL` / `INFERENCE_STATUS_INFERRED` 常量，relation_inference 和 story_graph_builder 已使用 |
| 5 | 重复代码提取到 _utils.py | ⏭ 按任务要求跳过，仅记录 |

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `director_brain/models/director_brief.py` | 新增 `source_duration_us` 字段 |
| `director_brain/brief_compiler.py` | 新增 `target_duration_us` 参数；分离 source/target 时长 |
| `director_brain/models/story_graph.py` | 新增 `StoryNodeType.ACT` + inference_status 常量 |
| `director_brain/story_graph_builder.py` | 改用 `StoryNodeType.ACT` + `INFERENCE_STATUS_STRUCTURAL` |
| `director_brain/relation_inference.py` | 消费 graph：同幕/跨幕置信度调整 + 幕节点证据 |
| `director_brain/director_reasoner.py` | 四幕单独选片 + act= 标记 + shot_function 映射 |
| `director_brain/plan_repair.py` | 新增 rule10 时长修复（延长/追加/截断） |
| `tests/unit/test_brief_compiler.py` | 更新断言 + 新增显式 target 测试 |
| `tests/unit/test_story_graph_builder.py` | SHOT → ACT 断言 |
| `tests/unit/test_relation_inference.py` | evidence_refs 断言更新 + 新增同幕边测试 |
| `tests/unit/test_director_reasoner.py` | 新增四幕选片测试 |
| `tests/unit/test_plan_repair.py` | 新增 rule10 时长修复测试 ×2 |

---

## 真实端到端完整输出

```
=== Step 1: analyze_media ===
deterministic_technical obs: 15
=== Step 2: transcribe (ASR) ===
asr obs: 2
=== Step 3: compile_brief ===
  target_duration=15000000
  source_duration_us=52208333
=== Step 4: build_story_graph ===
  nodes=4, edges=3
=== Step 5: infer_relations ===
  relations=15
=== Step 6: generate_plan ===
  edits=6, expected_duration=15083331
    shot_d4142460dc2c0ecb: 4875000-7125000 (2250000us) fn=opening
    shot_26c618400a859c3d: 16500000-17500000 (1000000us) fn=pacing
    shot_783a01ae193b1b4d: 23791667-25874999 (2083332us) fn=pacing
    shot_96e5aab5c71956f9: 25875000-27416666 (1541666us) fn=pacing
    shot_816f9552afc6c24d: 32000000-36000000 (4000000us) fn=peak
    shot_f19c0cc954659923: 48000000-52208333 (4208333us) fn=closing
=== Step 7: validate_plan (pre-repair) ===
  ok=True, errors=[]
=== Step 8: repair_plan ===
  repaired edits=6, expected_duration=15083331
=== Step 9: validate_plan (post-repair) ===
  ok=True, errors=[]
=== Assertions ===
  P0-1 OK: target_duration=15000000
  P0-2 OK: validate_plan after repair = True
  P0-3 OK: 6 decisions have act= label
    act=hook, heuristic:blur=180.87
    act=develop, heuristic:blur=101.77
    act=develop, heuristic:blur=270.58
    act=develop, heuristic:blur=279.11
    act=peak, heuristic:blur=323.72
    act=resolve, heuristic:blur=2508.85
ALL P0 FIXES VERIFIED
```

## 全量测试结果

```
221 passed, 1 skipped in 12.73s
```

（基线 216 passed → 221 passed，新增 5 个测试：1 个 brief_compiler 显式 target、1 个 relation_inference 同幕边、1 个 director_reasoner 四幕选片、2 个 plan_repair rule10。）

---

## 修复过程中发现的新问题（记录，未扩大修复范围）

1. **rule5 与 rule10 的交互边界**：rule5（MAX_CLIP_US=6s 截断）在 rule10 之前执行。如果源镜头数量少且都很长，rule5 会把每段压到 6s，总时长可能低于目标下限；此时 rule10 的延长步因已达 MAX_CLIP_US 上限无法延长，追加步因无未选中镜头也无法补足。真实素材（15 个镜头）下不触发，但镜头极少的边界场景可能仍需人工介入。
2. **端到端 pre-repair 即已 valid**：四幕分配 + HeuristicBaseline 按比例分配 target 后，初始 EDL 时长 15.08s 已在 [13.5s, 16.5s] 范围内，rule10 本次为 no-op。rule10 作为安全网在单元测试中验证了延长/截断路径有效。
3. **_parse_claim/_num/_short_hash 重复代码**：按任务要求未提取到 `_utils.py`，留待后续。
