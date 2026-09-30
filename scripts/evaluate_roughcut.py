"""粗剪自动评估脚本：7 维度客观评估选片质量。

对原始视频 + 粗剪视频做全链路复现获取 EDL，然后从技术质量、四幕结构、
镜头时长分布、语义多样性、覆盖率、重复检测、叙事连贯性 7 个维度评估。

用法：
    python scripts/evaluate_roughcut.py --input <原始视频> --roughcut <粗剪视频> \\
        --output <报告.json> [--target-duration 15]

注意：blur_score 为 Laplacian 方差，值越高越清晰（越低越糊）。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# 确保项目根在 sys.path 中
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from director_brain.brief_compiler import compile_brief
from director_brain.director_reasoner import get_director_reasoner
from director_brain.models.film_observation import ClaimKind
from director_brain.plan_repair import repair_plan
from director_brain.plan_validator import validate_plan
from director_brain.relation_inference import infer_relations
from director_brain.story_graph_builder import build_story_graph
from observation_service.asr import transcribe
from observation_service.pipeline import analyze_media
from observation_service.shot_discovery import discover_shots
from observation_service.vlm_observation import batch_vlm_observations

_DEFAULT_TARGET_DURATION = 15  # 秒
_TIMEBASE_US = 1_000_000


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _parse_claim(claim: str) -> dict:
    try:
        data = json.loads(claim)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _ffprobe_duration(video_path: str) -> float | None:
    """获取视频时长（秒），失败返回 None（候选⑤：统一走 media_info）。"""
    from observation_service.media_info import probe_media_meta

    meta = probe_media_meta(video_path)
    if meta.ok and meta.duration_us:
        return meta.duration_us / 1_000_000
    return None


def _ffprobe_info(video_path: str) -> dict:
    """用 ffprobe 获取视频基本信息，失败返回空 dict。"""
    info: dict = {}
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,size:stream=width,height,codec_name",
                "-of", "json", video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return info


def _text_histogram(values: list[float], bins: int = 5) -> str:
    """生成文本直方图。"""
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"all values = {lo:.2f} (n={len(values)})"
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    max_count = max(counts)
    lines = []
    for i in range(bins):
        bar_len = int(counts[i] / max_count * 20) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {lo + i * width:6.2f}-{lo + (i + 1) * width:6.2f}s │ {bar} ({counts[i]})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 全链路复现（与 roughcut.py 完全一致，确保 EDL 匹配渲染视频）
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, target_duration: int):
    """复现 roughcut.py 全链路，返回 (edl, plan, observations, relations)。"""
    project_id = Path(input_path).stem
    target_duration_us = target_duration * _TIMEBASE_US

    # 1. 确定性技术分析（与 roughcut.py 一致，不含 VLM）
    tech_obs = analyze_media(input_path)
    # 2. ASR 转写
    speech_obs = transcribe(input_path)
    all_obs = tech_obs + speech_obs

    # 3. Brief
    brief = compile_brief(
        project_id=project_id,
        video_path=input_path,
        observations=all_obs,
        target_duration_us=target_duration_us,
    )

    # 4. Story Graph
    graph = build_story_graph(brief, all_obs)

    # 5. 关系推断（确定性，不含 VLM）
    relations = infer_relations(all_obs, graph)

    # 6. 生成计划
    reasoner = get_director_reasoner("heuristic")
    edl, plan = reasoner.generate_plan(brief, graph, all_obs)

    # 7. 验证 + 修复
    validation_result = validate_plan(edl, plan, all_obs)
    is_valid, errors = validation_result
    if not is_valid:
        outcome = repair_plan(edl, plan, all_obs)
        if outcome.requires_director:
            print(f"[repair] ABSTAIN: {outcome.reason_code}: {outcome.reason}")
        else:
            edl, plan = outcome.edl, outcome.plan
            validation_result = validate_plan(edl, plan, all_obs)

    return edl, plan, all_obs, relations, graph


def run_vlm_analysis(input_path: str) -> list:
    """独立运行 VLM 语义分析（用于维度 4 和 7），失败返回空列表。"""
    try:
        shots = discover_shots(input_path)
        if not shots:
            return []
        return batch_vlm_observations(input_path, shots)
    except Exception as exc:
        print(f"[VLM] 分析失败，降级: {type(exc).__name__}: {exc}")
        return []


# ---------------------------------------------------------------------------
# 7 维度评估
# ---------------------------------------------------------------------------

def eval_technical_quality(edl, all_obs: list) -> dict:
    """维度 1：技术质量对比（选中镜头 vs 全部镜头的 blur_score）。"""
    tech_obs = [o for o in all_obs if o.observation_type == "deterministic_technical"]
    all_blurs = []
    for o in tech_obs:
        data = _parse_claim(o.claim)
        blur = data.get("blur_score")
        if isinstance(blur, (int, float)):
            all_blurs.append(float(blur))

    selected_ids = {e.source_asset_id for e in edl.ordered_edits}
    selected_blurs = []
    selected_exposure = []
    all_exposure = []
    for o in tech_obs:
        data = _parse_claim(o.claim)
        exp_ok = bool(data.get("exposure_ok", False))
        all_exposure.append(exp_ok)
        if o.media_asset_id in selected_ids:
            blur = data.get("blur_score")
            if isinstance(blur, (int, float)):
                selected_blurs.append(float(blur))
            selected_exposure.append(exp_ok)

    all_avg = statistics.mean(all_blurs) if all_blurs else 0.0
    sel_avg = statistics.mean(selected_blurs) if selected_blurs else 0.0
    diff = sel_avg - all_avg

    # blur_score 越高越清晰；"不选更糊的" → 选中平均 ≥ 全部平均
    passed = sel_avg >= all_avg if all_blurs else False

    return {
        "dimension": "technical_quality",
        "description": "选中镜头 vs 全部镜头的技术质量对比（blur_score 越高越清晰）",
        "selected_avg_blur": round(sel_avg, 2),
        "all_avg_blur": round(all_avg, 2),
        "difference": round(diff, 2),
        "selected_exposure_ok_ratio": round(
            sum(selected_exposure) / len(selected_exposure), 3
        ) if selected_exposure else 0.0,
        "all_exposure_ok_ratio": round(
            sum(all_exposure) / len(all_exposure), 3
        ) if all_exposure else 0.0,
        "selected_count": len(selected_blurs),
        "all_count": len(all_blurs),
        "threshold": "selected_avg >= all_avg (不选更糊的)",
        "passed": passed,
    }


def eval_four_act_structure(edl) -> dict:
    """维度 2：四幕结构（每幕镜头数、时长占比）。"""
    acts: dict[str, dict] = {
        "hook": {"count": 0, "duration_us": 0},
        "develop": {"count": 0, "duration_us": 0},
        "peak": {"count": 0, "duration_us": 0},
        "resolve": {"count": 0, "duration_us": 0},
    }
    total_us = 0
    for e in edl.ordered_edits:
        act = "unknown"
        if e.rationale:
            for token in e.rationale.split(","):
                token = token.strip()
                if token.startswith("act="):
                    act = token[4:]
                    break
        dur = e.out_frame - e.in_frame
        total_us += dur
        if act in acts:
            acts[act]["count"] += 1
            acts[act]["duration_us"] += dur
        else:
            acts.setdefault(act, {"count": 0, "duration_us": 0})
            acts[act]["count"] += 1
            acts[act]["duration_us"] += dur

    # 合格线：每幕至少 1 镜头；peak 时长占比 >= 20%
    all_acts_present = all(acts[a]["count"] >= 1 for a in ["hook", "develop", "peak", "resolve"])
    peak_ratio = acts["peak"]["duration_us"] / total_us if total_us > 0 else 0.0
    peak_ok = peak_ratio >= 0.20
    passed = all_acts_present and peak_ok

    act_summary = {}
    for name, info in acts.items():
        act_summary[name] = {
            "shot_count": info["count"],
            "duration_s": round(info["duration_us"] / _TIMEBASE_US, 2),
            "duration_ratio": round(info["duration_us"] / total_us, 3) if total_us > 0 else 0.0,
        }

    return {
        "dimension": "four_act_structure",
        "description": "四幕结构：hook/develop/peak/resolve 各幕镜头数与时长占比",
        "acts": act_summary,
        "total_duration_s": round(total_us / _TIMEBASE_US, 2),
        "all_acts_present": all_acts_present,
        "peak_duration_ratio": round(peak_ratio, 3),
        "threshold": "每幕>=1镜头 且 peak时长占比>=20%",
        "passed": passed,
    }


def eval_shot_duration_distribution(edl) -> dict:
    """维度 3：镜头时长分布（均值、标准差、最短、最长 + 直方图）。"""
    durations = [(e.out_frame - e.in_frame) / _TIMEBASE_US for e in edl.ordered_edits]

    if not durations:
        return {
            "dimension": "shot_duration_distribution",
            "description": "镜头时长分布统计",
            "mean": 0, "std": 0, "min": 0, "max": 0,
            "histogram": "(no shots)",
            "threshold": "min>=0.5s, max<=6s, std>0",
            "passed": False,
        }

    mean_d = statistics.mean(durations)
    std_d = statistics.stdev(durations) if len(durations) > 1 else 0.0
    min_d = min(durations)
    max_d = max(durations)

    min_ok = min_d >= 0.5
    max_ok = max_d <= 6.0
    std_ok = std_d > 0
    passed = min_ok and max_ok and std_ok

    return {
        "dimension": "shot_duration_distribution",
        "description": "镜头时长分布统计",
        "mean": round(mean_d, 2),
        "std": round(std_d, 2),
        "min": round(min_d, 2),
        "max": round(max_d, 2),
        "count": len(durations),
        "histogram": _text_histogram(durations),
        "threshold": "min>=0.5s, max<=6s, std>0 (非全部等长)",
        "passed": passed,
    }


def eval_semantic_diversity(edl, vlm_obs: list) -> dict:
    """维度 4：语义多样性（VLM shot_function / proposed_role_v2 分布）。"""
    if not vlm_obs:
        return {
            "dimension": "semantic_diversity",
            "description": "VLM 语义标签多样性",
            "status": "N/A",
            "reason": "VLM 分析不可用或未启用",
            "shot_function_distribution": {},
            "role_distribution": {},
            "threshold": "至少 2 种不同 shot_function",
            "passed": None,  # N/A 不算不达标
        }

    # 只取成功的 VLM 观测
    successful_vlm = [
        o for o in vlm_obs
        if o.observation_type == "vlm_semantic"
        and o.claim_kind == ClaimKind.MODEL_OBSERVATION
    ]

    if not successful_vlm:
        return {
            "dimension": "semantic_diversity",
            "description": "VLM 语义标签多样性",
            "status": "N/A",
            "reason": "VLM 观测全部失败（NOT_DETERMINED）",
            "shot_function_distribution": {},
            "role_distribution": {},
            "threshold": "至少 2 种不同 shot_function",
            "passed": None,
        }

    selected_ids = {e.source_asset_id for e in edl.ordered_edits}
    vlm_by_shot = {o.media_asset_id: _parse_claim(o.claim) for o in successful_vlm}

    fn_dist: dict[str, int] = {}
    role_dist: dict[str, int] = {}
    for sid in selected_ids:
        claim = vlm_by_shot.get(sid)
        if claim:
            fn = claim.get("shot_function", "UNKNOWN")
            role = claim.get("proposed_role_v2", "UNKNOWN")
            fn_dist[fn] = fn_dist.get(fn, 0) + 1
            role_dist[role] = role_dist.get(role, 0) + 1

    unique_fns = len(fn_dist)
    passed = unique_fns >= 2

    return {
        "dimension": "semantic_diversity",
        "description": "选中镜头的 VLM 语义标签多样性",
        "shot_function_distribution": fn_dist,
        "role_distribution": role_dist,
        "unique_shot_functions": unique_fns,
        "vlm_success_count": len(successful_vlm),
        "vlm_total_count": len(vlm_obs),
        "threshold": "至少 2 种不同 shot_function",
        "passed": passed,
    }


def eval_coverage(edl, input_path: str, target_duration: int) -> dict:
    """维度 5：覆盖率（选中镜头总时长 / 原始视频时长）。"""
    selected_us = sum(e.out_frame - e.in_frame for e in edl.ordered_edits)
    selected_s = selected_us / _TIMEBASE_US

    source_duration = _ffprobe_duration(input_path)
    if source_duration is None:
        source_duration = 0.0

    coverage_ratio = selected_s / source_duration if source_duration > 0 else 0.0

    # 合格线：在目标时长 ±10% 范围内
    target_low = target_duration * 0.9
    target_high = target_duration * 1.1
    in_range = target_low <= selected_s <= target_high

    return {
        "dimension": "coverage",
        "description": "选中镜头总时长 vs 原始视频时长 vs 目标时长",
        "selected_duration_s": round(selected_s, 2),
        "source_duration_s": round(source_duration, 2),
        "coverage_ratio": round(coverage_ratio, 4),
        "target_duration_s": target_duration,
        "target_range_s": [target_low, target_high],
        "threshold": f"选中时长在 {target_low:.1f}-{target_high:.1f}s 范围内",
        "passed": in_range,
    }


def eval_duplicate_detection(edl) -> dict:
    """维度 6：重复检测（同一 source_asset_id 被选多次）。"""
    id_counts: dict[str, int] = {}
    for e in edl.ordered_edits:
        id_counts[e.source_asset_id] = id_counts.get(e.source_asset_id, 0) + 1

    duplicates = {sid: cnt for sid, cnt in id_counts.items() if cnt > 1}
    passed = len(duplicates) == 0

    return {
        "dimension": "duplicate_detection",
        "description": "检测同一 source_asset_id 是否被重复选中",
        "duplicate_shots": duplicates,
        "duplicate_count": len(duplicates),
        "total_unique_shots": len(id_counts),
        "threshold": "0 重复（plan_repair rule4 应已去重）",
        "passed": passed,
    }


def eval_narrative_coherence(edl, all_obs: list, graph, vlm_obs: list) -> dict:
    """维度 7：叙事连贯性（相邻镜头关系边类型分布）。"""
    # 用含 VLM 的观测重新推断关系（VLM 语义边仅在有 VLM 观测时生效）
    full_obs = all_obs + vlm_obs if vlm_obs else all_obs
    try:
        relations = infer_relations(full_obs, graph)
    except Exception:
        relations = []

    # 统计所有关系边的类型（按 edge_id 前缀分类）
    type_counts: dict[str, int] = {}
    for edge in relations:
        eid = edge.edge_id
        if eid.startswith("semantic_continuity"):
            key = "semantic_continuity"
        elif eid.startswith("semantic_contrast"):
            key = "semantic_contrast"
        elif eid.startswith("reaction"):
            key = "reaction"
        elif eid.startswith("contrast"):
            key = "contrast"
        elif eid.startswith("montage"):
            key = "montage"
        else:
            key = "other"
        type_counts[key] = type_counts.get(key, 0) + 1

    semantic_count = type_counts.get("semantic_continuity", 0) + type_counts.get("semantic_contrast", 0)

    # 检查 VLM 是否可用
    vlm_available = any(
        o.observation_type == "vlm_semantic" and o.claim_kind == ClaimKind.MODEL_OBSERVATION
        for o in vlm_obs
    ) if vlm_obs else False

    if vlm_available:
        passed = semantic_count >= 1
        status = "evaluated"
    else:
        # 纯确定性分析时 semantic=0，标记为 N/A 不算不达标
        passed = None
        status = "N/A (VLM not available)"

    return {
        "dimension": "narrative_coherence",
        "description": "镜头间叙事关系边类型分布（含 VLM 语义边）",
        "relation_type_counts": type_counts,
        "semantic_relation_count": semantic_count,
        "total_relations": len(relations),
        "vlm_available": vlm_available,
        "status": status,
        "threshold": "至少 1 条 semantic 关系（VLM 集成后）",
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def evaluate(
    input_path: str,
    roughcut_path: str,
    output_path: str,
    target_duration: int = _DEFAULT_TARGET_DURATION,
) -> dict:
    """执行完整评估，返回报告 dict。"""
    report = {
        "input_video": input_path,
        "roughcut_video": roughcut_path,
        "target_duration_s": target_duration,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---- 粗剪视频基本信息 ----
    roughcut_info = _ffprobe_info(roughcut_path)
    report["roughcut_info"] = {
        "exists": os.path.isfile(roughcut_path),
        "size_bytes": os.path.getsize(roughcut_path) if os.path.isfile(roughcut_path) else 0,
        "ffprobe": roughcut_info,
    }

    # ---- 全链路复现获取 EDL ----
    print("[1/3] 复现全链路获取 EDL...")
    t0 = time.time()
    edl, plan, all_obs, _relations, graph = run_pipeline(input_path, target_duration)
    pipeline_time = time.time() - t0
    print(f"      EDL: {len(edl.ordered_edits)} 镜头, 耗时 {pipeline_time:.1f}s")

    # ---- VLM 分析（独立运行，用于维度 4/7）----
    print("[2/3] VLM 语义分析（用于语义多样性与叙事连贯性）...")
    t0 = time.time()
    vlm_obs = run_vlm_analysis(input_path)
    vlm_time = time.time() - t0
    vlm_success = sum(
        1 for o in vlm_obs
        if o.claim_kind == ClaimKind.MODEL_OBSERVATION
    )
    print(f"      VLM: {len(vlm_obs)} 条观测 ({vlm_success} 成功), 耗时 {vlm_time:.1f}s")

    report["pipeline"] = {
        "edl_shot_count": len(edl.ordered_edits),
        "edl_id": edl.edl_id,
        "plan_id": plan.plan_id,
        "pipeline_time_s": round(pipeline_time, 1),
        "vlm_time_s": round(vlm_time, 1),
        "vlm_obs_count": len(vlm_obs),
        "vlm_success_count": vlm_success,
    }

    # ---- 7 维度评估 ----
    print("[3/3] 执行 7 维度评估...")
    dimensions = [
        eval_technical_quality(edl, all_obs),
        eval_four_act_structure(edl),
        eval_shot_duration_distribution(edl),
        eval_semantic_diversity(edl, vlm_obs),
        eval_coverage(edl, input_path, target_duration),
        eval_duplicate_detection(edl),
        eval_narrative_coherence(edl, all_obs, graph, vlm_obs),
    ]

    report["dimensions"] = dimensions

    # ---- 汇总 ----
    passed_count = sum(1 for d in dimensions if d["passed"] is True)
    failed_count = sum(1 for d in dimensions if d["passed"] is False)
    na_count = sum(1 for d in dimensions if d["passed"] is None)
    overall_pass = passed_count >= 5

    report["summary"] = {
        "passed": passed_count,
        "failed": failed_count,
        "na": na_count,
        "total": 7,
        "overall": "PASS" if overall_pass else "FAIL",
    }

    # ---- 写入 JSON ----
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n评估报告已写入: {output_path}")

    # ---- 终端摘要 ----
    _print_summary(report)

    return report


def _print_summary(report: dict):
    """打印终端可读摘要。"""
    print("\n" + "=" * 60)
    print("粗剪自动评估摘要")
    print("=" * 60)
    print(f"  原始视频: {report['input_video']}")
    print(f"  粗剪视频: {report['roughcut_video']}")
    rc = report.get("roughcut_info", {})
    print(f"  粗剪大小: {rc.get('size_bytes', 0) / 1024:.1f} KB")
    print(f"  EDL 镜头: {report['pipeline']['edl_shot_count']}")
    print("-" * 60)

    for d in report["dimensions"]:
        status = d["passed"]
        if status is True:
            mark = "✅"
        elif status is False:
            mark = "❌"
        else:
            mark = "⚪"
        name = d["dimension"]
        # 精简指标显示
        if name == "technical_quality":
            detail = f"选中blur={d['selected_avg_blur']} vs 全部={d['all_avg_blur']} (diff={d['difference']:+.2f})"
        elif name == "four_act_structure":
            acts = d["acts"]
            detail = " | ".join(
                f"{a}:{acts[a]['shot_count']}镜/{acts[a]['duration_s']}s"
                for a in ["hook", "develop", "peak", "resolve"] if a in acts
            )
        elif name == "shot_duration_distribution":
            detail = f"mean={d['mean']}s std={d['std']} min={d['min']}s max={d['max']}s"
        elif name == "semantic_diversity":
            if d.get("status") == "N/A":
                detail = d.get("reason", "N/A")
            else:
                detail = f"shot_functions={d['shot_function_distribution']} (unique={d['unique_shot_functions']})"
        elif name == "coverage":
            detail = f"选中={d['selected_duration_s']}s / 源={d['source_duration_s']}s (目标{d['target_duration_s']}s±10%)"
        elif name == "duplicate_detection":
            detail = f"重复={d['duplicate_count']} (唯一镜头={d['total_unique_shots']})"
        elif name == "narrative_coherence":
            detail = f"关系={d['relation_type_counts']} semantic={d['semantic_relation_count']} [{d['status']}]"
        else:
            detail = str(d)
        print(f"  {mark} {name}: {detail}")
        print(f"      合格线: {d['threshold']}")

    print("-" * 60)
    s = report["summary"]
    print(f"  合计: ✅{s['passed']} ❌{s['failed']} ⚪{s['na']} / 7")
    print(f"  整体判定: {s['overall']} (合格线 >=5/7)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="粗剪自动评估：7 维度客观评估选片质量",
    )
    parser.add_argument("--input", "-i", required=True, help="原始视频路径")
    parser.add_argument("--roughcut", "-r", required=True, help="粗剪视频路径")
    parser.add_argument("--output", "-o", required=True, help="评估报告输出路径 (.json)")
    parser.add_argument(
        "--target-duration", type=int, default=_DEFAULT_TARGET_DURATION,
        help=f"目标成片时长（秒），默认 {_DEFAULT_TARGET_DURATION}",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"错误：原始视频不存在: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.roughcut):
        print(f"错误：粗剪视频不存在: {args.roughcut}", file=sys.stderr)
        sys.exit(1)

    report = evaluate(
        input_path=args.input,
        roughcut_path=args.roughcut,
        output_path=args.output,
        target_duration=args.target_duration,
    )
    sys.exit(0 if report["summary"]["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
