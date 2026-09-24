"""P0 端到端粗剪 CLI：一条命令从原始素材产出粗剪视频。

流程：analyze_media + transcribe → compile_brief → build_story_graph →
infer_relations → reasoner.generate_plan → validate_plan → (repair_plan) →
render_edl。

用法：
    python scripts/roughcut.py --input <video> --output <output.mp4> [--target-duration 15] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 确保项目根在 sys.path 中（脚本从 scripts/ 运行时）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from director_brain.brief_compiler import compile_brief
from director_brain.director_reasoner import get_director_reasoner
from director_brain.plan_repair import repair_plan
from director_brain.plan_validator import validate_plan
from director_brain.relation_inference import infer_relations
from director_brain.story_graph_builder import build_story_graph
from execution.renderer import render_edl
from observation_service.pipeline import analyze_media
from observation_service.asr import transcribe

_DEFAULT_TARGET_DURATION = 15  # 秒


def _print_edl_summary(edl, plan, validation_result, relations_count):
    """打印 EDL 摘要：镜头数、总时长、四幕分配、验证状态。"""
    edits = edl.ordered_edits
    total_us = sum(e.out_frame - e.in_frame for e in edits)
    total_s = total_us / 1_000_000

    # 四幕分配：从 rationale 中提取 act=xxx
    act_counts: dict[str, int] = {}
    for e in edits:
        act = "unknown"
        if e.rationale:
            for token in e.rationale.split(","):
                token = token.strip()
                if token.startswith("act="):
                    act = token[4:]
                    break
        act_counts[act] = act_counts.get(act, 0) + 1

    is_valid, errors = validation_result

    print("=" * 60)
    print("EDL 摘要")
    print("=" * 60)
    print(f"  镜头数:     {len(edits)}")
    print(f"  总时长:     {total_s:.2f}s ({total_us}us)")
    print(f"  四幕分配:   {act_counts}")
    print(f"  关系边数:   {relations_count}")
    print(f"  验证状态:   {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for err in errors:
            print(f"    - {err}")
    print(f"  EDL ID:     {edl.edl_id}")
    print(f"  Plan ID:    {plan.plan_id}")
    print("=" * 60)


def run_roughcut(
    input_path: str,
    output_path: str,
    target_duration: int = _DEFAULT_TARGET_DURATION,
    dry_run: bool = False,
) -> int:
    """执行端到端粗剪流程。返回 0 成功，非 0 失败。"""
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
        speech_obs = transcribe(input_path)
        print(f"      语音观测: {len(speech_obs)} 条")

        all_obs = tech_obs + speech_obs

        # ---- 2. Brief ----
        print("[3/7] 编译 Brief...")
        brief = compile_brief(
            project_id=project_id,
            video_path=input_path,
            observations=all_obs,
            target_duration_us=target_duration_us,
        )
        print(f"      Brief ID: {brief.brief_id}, 目标时长: {brief.target_duration / 1e6:.1f}s")

        # ---- 3. Story Graph ----
        print("[4/7] 构建故事图...")
        graph = build_story_graph(brief, all_obs)
        print(f"      图节点: {len(graph.nodes)}, 边: {len(graph.edges)}")

        # ---- 4. 关系推断 ----
        print("[5/7] 推断关系...")
        relations = infer_relations(all_obs, graph)
        print(f"      推断关系边: {len(relations)} 条")

        # ---- 5. 生成计划 ----
        print("[6/7] 生成导演计划...")
        reasoner = get_director_reasoner("heuristic")
        edl, plan = reasoner.generate_plan(brief, graph, all_obs)

        # ---- 6. 验证 + 修复 ----
        print("[7/7] 验证计划...")
        validation_result = validate_plan(edl, plan, all_obs)
        is_valid, errors = validation_result
        if not is_valid:
            print(f"      验证未通过 ({len(errors)} 个错误)，执行修复...")
            edl, plan = repair_plan(edl, plan, all_obs)
            validation_result = validate_plan(edl, plan, all_obs)
            is_valid, errors = validation_result
            print(f"      修复后验证: {'PASS' if is_valid else 'FAIL'}")

        # ---- 打印摘要 ----
        _print_edl_summary(edl, plan, validation_result, len(relations))

        # ---- 7. 渲染 ----
        if dry_run:
            print("[dry-run] 跳过渲染，不创建输出文件。")
            return 0

        print(f"渲染中 -> {output_path}")
        result_path = render_edl(edl, input_path, output_path)
        file_size = os.path.getsize(result_path)
        print(f"渲染完成: {result_path} ({file_size / 1024:.1f} KB)")
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
    args = parser.parse_args()
    sys.exit(run_roughcut(
        input_path=args.input,
        output_path=args.output,
        target_duration=args.target_duration,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
