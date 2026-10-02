"""P0 端到端粗剪 CLI：一条命令从原始素材产出粗剪视频。

流程：analyze_media + transcribe → compile_brief → build_story_graph →
infer_relations → reasoner.generate_plan → validate_plan → (repair_plan) →
render_edl。

用法：
    python scripts/roughcut.py --input <video> --output <output.mp4> [--target-duration 15] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 确保项目根在 sys.path 中（脚本从 scripts/ 运行时）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from director_brain.audit_trail import log_decision
from director_brain.brief_compiler import compile_brief
from director_brain.config import load_settings
from director_brain.director_reasoner import EvidenceTooPoorError, get_director_reasoner
from director_brain.pathway_protocol import describe as describe_pathways
from director_brain.pathway_protocol import get_pathway_status
from director_brain.plan_repair import repair_plan
from director_brain.plan_state import (
    PlanState,
    confirm_strategy as _bind_strategy_confirmation,
    is_confirmation_valid,
    transition_plan,
)
from director_brain.plan_validator import validate_plan
from director_brain.relation_inference import infer_relations
from director_brain.semantic_shadow import ShadowReport, run_shadow_semantic
from director_brain.story_graph_builder import build_story_graph
from execution.renderer import render_edl
from observation_service.pipeline import analyze_media
from observation_service.asr import transcribe
from observation_service.media_info import probe_audio_stream

_DEFAULT_TARGET_DURATION = 15  # 秒


def _open_ledger():
    """打开决策账本（审计留痕）。fail-soft：打开失败打警告并返回 None
    （账本故障不阻断出片，但每次都会响亮提示——审计降级也必须可见）。"""
    try:
        settings = load_settings()
        if settings.storage_backend != "sqlite":
            print(
                f"      警告：账本未启用（storage_backend={settings.storage_backend}，"
                f"当前仅支持 sqlite）——本次运行不留审计记录"
            )
            return None
        db_path = settings.sqlite_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        from storage.sqlite_repository import SqliteRepository

        return SqliteRepository(db_path)
    except Exception as exc:  # noqa: BLE001
        print(f"      警告：账本打开失败（审计留痕降级）：{type(exc).__name__}: {exc}")
        return None


def _safe_log(ledger, decision_id: str, action: str, detail: dict) -> None:
    """账本写入的 fail-soft 包装：写入失败只警告不阻断主流程。"""
    if ledger is None:
        return
    try:
        log_decision(ledger, decision_id, action, detail)
    except Exception as exc:  # noqa: BLE001
        print(f"      警告：账本写入失败（{action}）：{type(exc).__name__}: {exc}")


def _act_of(edit) -> str:
    """读片段所属幕：优先结构化字段 act（P1-b），旧数据回退 rationale 文本。"""
    act = getattr(edit, "act", None)
    if act:
        return act
    if edit.rationale:
        for token in edit.rationale.split(","):
            token = token.strip()
            if token.startswith("act="):
                return token[4:]
    return "unknown"


def _print_edl_summary(edl, plan, validation_result, relations, pathway_report):
    """打印 EDL 摘要：镜头数、总时长、四幕分配、验证状态、通路状态。"""
    edits = edl.ordered_edits
    total_us = sum(e.out_frame - e.in_frame for e in edits)
    total_s = total_us / 1_000_000

    # 四幕分配：结构化字段（P1-b），旧数据回退 rationale 解析
    act_counts: dict[str, int] = {}
    for e in edits:
        act = _act_of(e)
        act_counts[act] = act_counts.get(act, 0) + 1

    is_valid, errors = validation_result

    degraded = getattr(plan, "degraded", False)
    events = getattr(plan, "degradation_events", [])

    print("=" * 60)
    print("EDL 摘要")
    print("=" * 60)
    print(f"  镜头数:     {len(edits)}")
    print(f"  总时长:     {total_s:.2f}s ({total_us}us)")
    print(f"  四幕分配:   {act_counts}")
    print(f"  关系边数:   {len(relations)}（影子信号，逐条见上方）")
    print(f"  验证状态:   {'PASS' if is_valid else 'FAIL'}")
    print(f"  降级:       {'是（' + str(len(events)) + ' 项，见下）' if degraded else '否'}")
    for ev in events:
        print(f"    - {ev}")
    if errors:
        for err in errors:
            print(f"    - {err}")
    if pathway_report:
        print(f"  {pathway_report.replace(chr(10), chr(10) + '  ')}")
    print(f"  EDL ID:     {edl.edl_id}")
    print(f"  Plan ID:    {plan.plan_id}")
    print("=" * 60)


def _print_shadow_summary(report: ShadowReport) -> None:
    """打印链 B 影子对账结果（全量上报，不驱动成片）。"""
    if report.status == "completed":
        print(
            f"      影子语义决策（{report.pathway_status}）: 模型 {report.model}"
            f" ｜ 延迟 {report.latency_ms}ms ｜ 重试 {report.attempts} 次"
        )
        print(
            f"        链 B 状态: {report.chain_b.get('status')}"
            f" ｜ 正向意图: {report.positive_intents or '无'}"
        )
        print(
            f"        负向对账: 已覆盖 {len(report.covered_negatives)} 条"
            f"（链 A 确定性执法）；未覆盖 {len(report.uncovered_negatives)} 条"
            f"（无确定性执法，人工抽检项）"
        )
        for token in report.uncovered_negatives:
            print(f"          - 未覆盖: {token}")
        if report.status_divergence:
            print(f"        状态分歧: {report.status_divergence}")
    elif report.status == "failed":
        print(f"      影子语义决策失败（响亮降级，主链不受影响）: {report.reason}")
    else:
        print(f"      影子语义决策跳过: {report.reason}")


def run_roughcut(
    input_path: str,
    output_path: str,
    target_duration: int = _DEFAULT_TARGET_DURATION,
    dry_run: bool = False,
    intent_text: str | None = None,
    semantic: bool = False,
    confirm_strategy: bool = False,
    variants: int = 1,
    transitions: bool = False,
    card_id: str | None = None,
) -> int:
    """执行端到端粗剪流程。返回 0 成功，非 0 失败。

    intent_text: 用户创作意图的自然语言原文，透传给 brief_compiler
    （规则抽取 must_include / must_avoid 约束）；None 表示未提供。
    semantic: 语义驱动选片（候选①生产入口）——多帧 VLM 语义观测 +
    跨镜头叙事弧进入选片内核。**硬前置**：vlm_semantic 通路必须
    ACTIVE（治理决策，set_pathway_status 留痕）；非 ACTIVE 时
    fail-closed 拒绝启动（禁止半消费——影子信号只记录不驱动）。
    confirm_strategy: 策略确认（候选③执法点）——渲染属写操作，红线
    "未 STRATEGY_CONFIRMED 不得渲染"由此生效：验证 PASS 后 plan 停在
    READY_FOR_STRATEGY_CONFIRMATION，传 True 才绑定 plan_hash+edl_hash
    推进到 DISPATCH_ELIGIBLE 并放行渲染；不传则只出 EDL/plan 工件
    （exit 2 = 策略未确认）。dry_run 不受影响。
    """
    # ---- 输入检查 ----
    if not os.path.isfile(input_path):
        print(f"错误：输入视频不存在: {input_path}", file=sys.stderr)
        return 1

    project_id = Path(input_path).stem
    target_duration_us = target_duration * 1_000_000

    try:
        # ---- 1. 观测 ----
        print(f"[1/7] 分析媒体: {input_path}")
        tech_obs = analyze_media(input_path)
        print(f"      技术观测: {len(tech_obs)} 条")

        print("[2/7] 语音转写...")
        track = probe_audio_stream(input_path)
        speech_obs = transcribe(input_path)
        if not track.ok:
            print(
                f"      语音观测: {len(speech_obs)} 条"
                f"（ASR 归因: 音轨探测失败——{track.reason}）"
            )
        elif not track.has_audio:
            print(
                f"      语音观测: {len(speech_obs)} 条"
                f"（ASR 归因: 视频无音轨 → 0 条为预期，非通路故障）"
            )
        elif speech_obs:
            print(f"      语音观测: {len(speech_obs)} 条（ASR 归因: 有音轨且有转写）")
        else:
            print(
                "      语音观测: 0 条"
                "（ASR 归因: 有音轨但无有效转写——静音/无语音，详见服务日志）"
            )

        all_obs = tech_obs + speech_obs

        # ---- 1.5 语义观测（候选①生产入口；通路 ACTIVE 硬前置）----
        narrative = None
        entities = None
        if semantic:
            from director_brain.pathway_protocol import PathwayStatus
            pw = get_pathway_status("vlm_semantic")
            if pw is not PathwayStatus.ACTIVE:
                print(
                    f"[fail-closed] --semantic 要求 vlm_semantic 通路 ACTIVE，"
                    f"当前 {pw.value}——影子期信号只记录不驱动（禁止半消费）。"
                    f"转正属治理决策：以影子期证据经主控批准后 "
                    f"set_pathway_status('vlm_semantic', ACTIVE)。"
                )
                return 1
            print("[1.5] 多帧语义观测（vlm_semantic ACTIVE）...")
            shots = [
                {"shot_id": o.media_asset_id, "source_in_us": o.start_frame,
                 "source_out_us": o.end_frame, "source_media_hash": o.media_hash}
                for o in sorted(tech_obs, key=lambda x: x.start_frame)
            ]
            from observation_service.vlm_observation import batch_vlm_observations
            from director_brain.models.film_observation import ClaimKind
            vlm_obs = batch_vlm_observations(input_path, shots)
            all_obs = all_obs + vlm_obs
            ok_obs = [o for o in vlm_obs
                      if o.claim_kind is ClaimKind.MODEL_OBSERVATION]
            print(f"      语义观测: {len(ok_obs)}/{len(vlm_obs)} 镜头有效")

            from director_brain.config import read_env_key
            ark_key = read_env_key("ARK_API_KEY") or ""
            if ark_key:
                print("[1.6] 跨镜头叙事弧（P3-3）...")
                from director_brain.narrative_analyzer import analyze_narrative
                ordered = sorted(tech_obs, key=lambda x: x.start_frame)
                shot_ids = [o.media_asset_id for o in ordered]
                sems = []
                claims = {
                    o.media_asset_id: json.loads(o.claim)
                    for o in ok_obs
                }
                for sid in shot_ids:
                    c = claims.get(sid)
                    if c:
                        sems.append({
                            "scene_description": c.get("scene_description", ""),
                            "action_type": c.get("action_type"),
                            "emotional_tone": c.get("emotional_tone"),
                            "narrative_role": c.get("narrative_role"),
                            "importance": c.get("importance"),
                        })
                from director_brain.entity_resolver import resolve_entities
                entities = resolve_entities(vlm_obs)
                n_ent = len(entities.entities)
                print(f"      实体解析: {n_ent} 个人物实体"
                      f"（{sum(1 for v in entities.assignments.values() for _ in v)} 次归属）")
                try:
                    narrative = analyze_narrative(
                        sems, shot_ids=shot_ids, api_key=ark_key)
                    print(f"      叙事弧: {narrative.get('story_arc', '')[:60]}...")
                    print(f"      幕边界: {len(narrative.get('act_boundaries_resolved', []))} 段"
                          f" ｜ 推荐顺序: "
                          f"{'有' if narrative.get('suggested_order_resolved') else '无'}")
                except Exception as exc:  # noqa: BLE001
                    # 响亮降级：叙事层失败不阻断（语义逐镜头观测仍生效）
                    print(f"      叙事分析失败（响亮跳过）: "
                          f"{type(exc).__name__}: {str(exc)[:80]}")
            else:
                print("      叙事分析跳过（无 ARK_API_KEY——逐镜头语义仍生效）")

        # ---- 2. Brief ----
        print("[3/7] 编译 Brief...")
        brief = compile_brief(
            project_id=project_id,
            video_path=input_path,
            observations=all_obs,
            target_duration_us=target_duration_us,
            intent_text=intent_text,
        )
        print(f"      Brief ID: {brief.brief_id}, 目标时长: {brief.target_duration / 1e6:.1f}s")
        if intent_text:
            print(f"      用户意图: 已接收（{len(intent_text)} 字，规则抽取入 Brief 约束）")

        # ---- 慢节奏物理可行性告知（规划层前置，避免跑到修复才拒绝）----
        if (brief.editing_language == "slow_paced" and tech_obs):
            durs = sorted(o.end_frame - o.start_frame for o in tech_obs)
            median_us = durs[len(durs) // 2]
            if median_us < 1_500_000:
                print(
                    f"      [告知] 慢剪意图 + 中位镜头 {median_us/1e6:.2f}s "
                    f"（慢剪下界 1.5s）：素材物理受限，出片率会显著下降或拒绝——"
                    f"建议改用快剪/均衡或提供更长镜头素材")

        # ---- D1 镜头卡：显式 --card 选择（自动匹配待矩阵验证后默认开启）----
        card = None
        if card_id:
            from director_brain.shot_cards import select_card
            card = select_card(brief, len(tech_obs), card_id=card_id)
            print(f"      镜头卡: {card.name}（{card.card_id}@{card.version}）"
                  f" energy={card.energy}")

        # ---- 3. Story Graph ----
        print("[4/7] 构建故事图...")
        graph = build_story_graph(brief, all_obs)
        print(f"      图节点: {len(graph.nodes)}, 边: {len(graph.edges)}")

        # ---- 4. 关系推断（影子信号：全量上报，不并入图、不进决策）----
        print("[5/7] 推断关系...")
        relations = infer_relations(all_obs, graph)
        relation_status = get_pathway_status("relation_inference")
        print(
            f"      推断关系边: {len(relations)} 条"
            f"（通路: {relation_status.value}——影子信号，不进决策）"
        )
        for edge in relations:
            print(
                f"        - {edge.from_node} → {edge.to_node} "
                f"[{edge.edge_type.value}] conf={edge.confidence}"
            )

        # ---- 5. 生成计划（--variants>1 时多方案对比择优）----
        ledger = _open_ledger()
        print("[6/7] 生成导演计划...")
        reasoner = get_director_reasoner("heuristic")
        try:
            if variants > 1:
                # 成品级扫荡：strategy_selector（此前休眠）接线为生产入口——
                # 围绕目标时长 ±10% 步进 × 不同 blur 阈值生成 N 个变体，
                # 评分卡择优（对齐 OpusClip"多候选选最优"实践），全量落账本
                from director_brain.strategy_selector import (
                    compare_plans,
                    generate_variants,
                    select_best,
                )
                print(f"      多方案对比: {variants} 个变体...")
                configs = [
                    {
                        "target_duration_us": int(
                            target_duration_us
                            * (1.0 + (i - (variants - 1) / 2) * 0.1)
                        ),
                        "blur_threshold": 10.0 + i * 2.5,
                    }
                    for i in range(variants)
                ]
                variant_pairs = generate_variants(
                    brief, graph, all_obs, configs, narrative=narrative)
                scorecards = compare_plans(variant_pairs, all_obs)
                best = select_best(
                    scorecards, priority="duration",
                    target_duration_us=target_duration_us)
                edl, plan = variant_pairs[best]
                _safe_log(ledger, plan.plan_id, "variants_compared", {
                    "variant_count": variants,
                    "configs": configs,
                    "scorecards": scorecards,
                    "chosen_index": best,
                    "priority": "duration",
                })
                for i, sc in enumerate(scorecards):
                    mark = " ← 选优" if i == best else ""
                    print(f"      变体{i + 1}: {sc['shot_count']} 镜头 "
                          f"{sc['duration_us'] / 1e6:.2f}s "
                          f"avg_blur={sc.get('avg_blur', 0):.1f}{mark}")
            else:
                edl, plan = reasoner.generate_plan(
                    brief, graph, all_obs, narrative=narrative,
                    transition_policy=("dissolve_act_boundary"
                                       if transitions else "none"),
                    card=card, entities=entities)
        except EvidenceTooPoorError as exc:
            # T2 fail-closed：技术证据不足，拒绝导演（不注水选片）
            print(f"      导演放弃（evidence_too_poor）: {exc}")
            return 1
        # 候选③：状态机接线（DRAFT → CONTEXT_READY → VALIDATING）
        transition_plan(plan, PlanState.CONTEXT_READY)
        transition_plan(plan, PlanState.VALIDATING)
        _safe_log(ledger, plan.plan_id, "plan_generated", {
            "edl_id": edl.edl_id,
            "shot_count": len(edl.ordered_edits),
            "degraded": plan.degraded,
            "degradation_events": plan.degradation_events,
            "open_questions": plan.open_questions,
            "constraints": plan.constraints,
            "state": plan.state,
        })
        if entities is not None and entities.entities:
            _safe_log(ledger, plan.plan_id, "entities_resolved", entities.to_dict())

        # ---- 链 B 影子语义决策辅助（阶段 7.5，SHADOW）----
        # 与链 A 并行产出 DirectorDecision 并全量对账上报；影子输出只进
        # 账本/报告，绝不驱动选片/修复/渲染（消费侧由 pathway_protocol
        # fail-closed 把关）。任何失败响亮降级，不阻断主链。
        # 覆盖两条退出路径：修复放弃（chain A 拒绝时链 B 的独立读法正是
        # 最有价值的对账信号）+ 正常验证后——禁止半消费。
        def _run_semantic_shadow(chain_a_proceeded: bool) -> None:
            if not intent_text:
                return
            print("[shadow] 链 B 语义决策（影子对账，不驱动成片）...")
            report = run_shadow_semantic(
                intent_text,
                chain_a_constraints=plan.constraints,
                chain_a_valid=chain_a_proceeded,
                decision_id=f"shadow_{plan.plan_id}",
                ledger=ledger,
                output_path=output_path,
            )
            _print_shadow_summary(report)

        # ---- 6. 验证 + 修复 ----
        print("[7/7] 验证计划...")
        validation_result = validate_plan(edl, plan, all_obs)
        is_valid, errors = validation_result
        repaired = False
        _safe_log(ledger, plan.plan_id, "plan_validated",
                  {"stage": "pre_repair", "valid": is_valid, "errors": errors})
        if not is_valid:
            print(f"      验证未通过 ({len(errors)} 个错误)，执行修复...")
            outcome = repair_plan(edl, plan, all_obs)
            _safe_log(ledger, plan.plan_id, "plan_repaired", {
                "status": "abstain" if outcome.requires_director else "ok",
                "reason_code": outcome.reason_code,
                "adjustments": outcome.adjustments,
            })
            if outcome.requires_director:
                # Plan=导演依据：修复器无权增删镜头，物理修复不可行时 fail-closed 上抛
                print(f"      修复放弃（repair_requires_director）: {outcome.reason_code}")
                print(f"      原因: {outcome.reason}")
                transition_plan(plan, PlanState.FAILED_VALIDATION)
                _run_semantic_shadow(chain_a_proceeded=False)
                _print_edl_summary(edl, plan, validation_result, relations, describe_pathways())
                return 1
            edl, plan = outcome.edl, outcome.plan
            repaired = True
            if outcome.adjustments:
                print(f"      物理调整 {len(outcome.adjustments)} 处（已留痕 plan.open_questions）")
            validation_result = validate_plan(edl, plan, all_obs)
            is_valid, errors = validation_result
            _safe_log(ledger, plan.plan_id, "plan_validated",
                      {"stage": "post_repair", "valid": is_valid, "errors": errors})
            print(f"      修复后验证: {'PASS' if is_valid else 'FAIL'}")

        # P1-c：验证状态回写——plan.validation_status 不再停留在 pending
        if not repaired or is_valid:
            plan.validation_status = "valid" if is_valid else "invalid"

        # ---- 候选③：终态回写状态机（VALIDATING → 确认就绪 / 验证失败）----
        confirmation = None
        if is_valid:
            if plan.state == PlanState.DRAFT.value:
                # 修复器重建的 plan 从 DRAFT 重新进入管线（真实 repair_plan
                # 经 model_copy 保留 state，此处兼容重建实现）
                transition_plan(plan, PlanState.CONTEXT_READY)
                transition_plan(plan, PlanState.VALIDATING)
            transition_plan(plan, PlanState.READY_FOR_STRATEGY_CONFIRMATION)
            if confirm_strategy:
                # 用户显式批准（CLI flag = 确认动作）：绑定 plan_hash+edl_hash，
                # 任何内容变化使确认失效（渲染前还会复验）
                confirmation = _bind_strategy_confirmation(
                    plan, edl, confirmed_by="cli_user",
                    output_target="preview",
                )
                transition_plan(plan, PlanState.STRATEGY_CONFIRMED)
                transition_plan(plan, PlanState.DISPATCH_ELIGIBLE)
                _safe_log(ledger, plan.plan_id, "strategy_confirmed", {
                    "plan_hash": confirmation.plan_hash,
                    "edl_hash": confirmation.edl_hash,
                    "confirmed_by": confirmation.confirmed_by,
                    "output_target": confirmation.output_target,
                    "state": plan.state,
                })
                print(f"      策略已确认: plan_hash={confirmation.plan_hash} "
                      f"edl_hash={confirmation.edl_hash} → {plan.state}")
        else:
            transition_plan(plan, PlanState.FAILED_VALIDATION)

        # ---- 6.5 链 B 影子语义决策：正常验证路径（修复放弃路径已在本函数
        # 内提前覆盖）——禁止半消费 ----
        _run_semantic_shadow(chain_a_proceeded=is_valid)

        # ---- 打印摘要 ----
        _print_edl_summary(edl, plan, validation_result, relations, describe_pathways())

        # ---- 7. 渲染闸门（P1-c：FAIL 不出片；候选③：未确认不渲染）----
        if not is_valid:
            print("      [fail-closed] 最终验证未通过，拒绝渲染不合格成片。")
            return 1
        if dry_run:
            # dry-run = 只产工件不渲染（方案：未确认可生成本地草案）
            print("[dry-run] 跳过渲染，不创建输出文件。")
            return 0
        if plan.state != PlanState.DISPATCH_ELIGIBLE.value:
            print(
                "      [fail-closed] 策略未确认（红线：未 STRATEGY_CONFIRMED "
                "不得渲染）——已产出 EDL/plan 工件；加 --confirm-strategy 授权"
                "渲染本 plan（plan_hash 绑定，内容变化即失效）。"
            )
            return 2
        if confirmation is not None and not is_confirmation_valid(confirmation, plan, edl):
            print("      [fail-closed] 确认后 plan/EDL 发生变化（hash 不符），拒绝渲染。")
            return 1

        # ---- 8. 渲染 ----
        print(f"渲染中 -> {output_path}")
        result_path = render_edl(edl, input_path, output_path)
        file_size = os.path.getsize(result_path)
        print(f"渲染完成: {result_path} ({file_size / 1024:.1f} KB)")
        # P1 配套：剪辑清单 sidecar（L1 指标脚本 / 审计 / 复算的数据源）
        try:
            edl_json = f"{output_path}.edl.json"
            plan_json = f"{output_path}.plan.json"
            Path(edl_json).write_text(edl.model_dump_json(indent=2), encoding="utf-8")
            Path(plan_json).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            print(f"      剪辑清单: {edl_json} + {plan_json}")
        except Exception as exc:  # noqa: BLE001
            print(f"      警告：剪辑清单写入失败（L1 指标将缺少 EDL 维度）：{exc}")
        _safe_log(ledger, plan.plan_id, "render_completed", {
            "output_path": result_path,
            "file_size_bytes": file_size,
        })
        return 0

    except FileNotFoundError as exc:
        print(f"错误：文件未找到: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"错误：渲染失败: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"错误：未预期的异常: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main():
    # log_level 配置接线（成品级扫荡：此前 LOG_LEVEL 配置零消费）
    try:
        import logging
        logging.basicConfig(level=load_settings().log_level)
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        description="端到端粗剪：从原始素材产出粗剪视频",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入视频路径",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出视频路径 (.mp4)",
    )
    parser.add_argument(
        "--target-duration",
        type=int,
        default=_DEFAULT_TARGET_DURATION,
        help=f"目标成片时长（秒），默认 {_DEFAULT_TARGET_DURATION}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成 EDL 并打印摘要，不渲染视频",
    )
    parser.add_argument(
        "--intent",
        type=str,
        default=None,
        help="用户创作意图（自然语言），编译进 Brief 约束（P0-4 入口）",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="语义驱动选片（候选①）：多帧语义观测+叙事弧进入选片内核；"
             "硬前置 vlm_semantic 通路 ACTIVE，否则 fail-closed 拒绝",
    )
    parser.add_argument(
        "--card",
        type=str,
        default=None,
        help="镜头卡（D1 美学策略载体）：如 fast_cut/slow_paced/balanced；"
             "未知卡响亮失败，素材条件不满足响亮失败",
    )
    parser.add_argument(
        "--transitions",
        action="store_true",
        help="幕切换处自动 dissolve（导演层 artistic choice；ΣD 感知验证）",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="多方案对比：生成 N 个变体（时长±10%%步进×不同模糊阈值）评分择优",
    )
    parser.add_argument(
        "--confirm-strategy",
        action="store_true",
        help="策略确认（候选③）：绑定 plan_hash+edl_hash 并授权渲染"
             "（红线：未确认不渲染）；不传只出 EDL/plan 工件",
    )
    args = parser.parse_args()
    sys.exit(run_roughcut(
        input_path=args.input,
        output_path=args.output,
        target_duration=args.target_duration,
        dry_run=args.dry_run,
        intent_text=args.intent,
        semantic=args.semantic,
        confirm_strategy=args.confirm_strategy,
        variants=args.variants,
        transitions=args.transitions,
        card_id=args.card,
    ))


if __name__ == "__main__":
    main()
