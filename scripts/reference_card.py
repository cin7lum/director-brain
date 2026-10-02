# -*- coding: utf-8 -*-
"""D6 · 参考片学习：参考片 → 显式风格参数卡（不做黑箱学习）。

流程（复用现有观测栈，非新自研）：
1. 参考片跑标准观测（镜头发现/五点采样；--semantic 加 VLM 情绪）；
2. 提取显式特征向量：ASL、镜长分布（P10/P50/P90）、四幕镜头占比、
   切点密度、镜长方差；
3. 特征 → ShotCard 草稿（每个参数可解释可调，落 user 卡库文件）；
4. 卡进 D1 消费链（roughcut --card ref_*），账本留痕同原生卡。

版权边界：只提取节奏/结构**参数**（纯数字），不复制内容、不存帧——
参考片文件用后即弃，卡里留参数与来源引用。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from director_brain.models.shot_card import ShotCard  # noqa: E402
from observation_service.pipeline import analyze_media  # noqa: E402
from observation_service.shot_discovery import discover_shots  # noqa: E402

_TIMEBASE_US = 1_000_000


def extract_features(src: str) -> dict:
    """参考片显式特征向量（全部为确定性测量 + 可选 VLM 情绪）。"""
    shots = discover_shots(src)
    if not shots:
        raise RuntimeError(f"参考片未检出镜头: {src}")
    durs_us = [s["source_out_us"] - s["source_in_us"] for s in shots]
    durs_s = [d / 1e6 for d in durs_us]
    total_us = shots[-1]["source_out_us"]
    total_s = total_us / 1e6

    durs_sorted = sorted(durs_us)
    n = len(durs_sorted)

    def pct(p: float) -> int:
        return durs_sorted[min(int(p * n), n - 1)]

    # 四幕镜头占比（按镜头起点比例分幕，与 story_graph 同口径）
    acts = {"hook": 0, "develop": 0, "peak": 0, "resolve": 0}
    from director_brain.acts import ACT_INTERVALS
    for s in shots:
        ratio = s["source_in_us"] / total_us
        for name, lo, hi, _label in ACT_INTERVALS:
            if lo <= ratio < hi:
                acts[name] += 1
                break
        else:
            acts["resolve"] += 1
    act_ratio = {k: round(v / n, 2) for k, v in acts.items()}

    # 情绪弧（可选：调用方先跑 --semantic 时从 VLM 观测取情绪序列）
    emotion_arc = None

    features = {
        "source": src,
        "shot_count": n,
        "duration_s": round(total_s, 2),
        "asl_s": round(statistics.mean(durs_s), 2),
        "asl_std_s": round(statistics.pstdev(durs_s), 2),
        "p10_us": pct(0.10),
        "p50_us": pct(0.50),
        "p90_us": pct(0.90),
        "cut_density_per_min": round(n / max(total_s / 60, 0.1), 1),
        "act_ratio": act_ratio,
        "emotion_arc": emotion_arc,
    }
    return features


def features_to_card(features: dict, card_name: str | None = None) -> ShotCard:
    """特征 → 显式 ShotCard 草稿（每个参数可解释、可手工再调）。"""
    p10, p50, p90 = features["p10_us"], features["p50_us"], features["p90_us"]
    # 节奏边界：P10/P90 钳制到合法域（0.3–8s）
    min_clip = int(max(300_000, min(p10, 3_000_000)))
    max_clip = int(min(8_000_000, max(p90, min_clip + 500_000)))

    asl = features["asl_s"]
    var = features["asl_std_s"]
    if asl <= 1.0 and var < 0.5:
        energy = "explosive"
    elif asl <= 2.0:
        energy = "high"
    elif asl <= 4.0:
        energy = "medium"
    else:
        energy = "calm"

    # 慢节奏参考片 → 幕间叠化更自然
    transition_policy = "dissolve_act_boundary" if p50 >= 1_500_000 else "none"

    src_name = Path(features["source"]).stem
    pitfalls = [
        f"参考片仅 {features['shot_count']} 个镜头，特征置信有限——参数可手工再调",
        f"参考片切点密度 {features['cut_density_per_min']} 切/分钟；"
        f"素材差异大时节奏可能不可复现",
    ]
    if features["emotion_arc"] is None:
        pitfalls.append("未做语义分析（无情绪弧数据）——可对参考片跑 --semantic 后重生成")

    card_id = "ref_" + str(abs(hash(src_name + str(features["asl_s"]))) % 100000).zfill(5)
    return ShotCard(
        card_id=card_id,
        version="1.0",
        name=card_name or f"参考片卡:{src_name}",
        category="参考片",
        purpose=(f"复现参考片《{src_name}》的节奏特征：ASL {asl}s、"
                 f"切点密度 {features['cut_density_per_min']}/min、"
                 f"四幕占比 {features['act_ratio']}"),
        energy=energy,
        pacing_override={"min_clip_us": min_clip, "max_clip_us": max_clip},
        emotion_arc_fit=[features["emotion_arc"]] if features["emotion_arc"] else [],
        transition_policy=transition_policy,
        min_shots=max(3, features["shot_count"] // 3),
        known_pitfalls=pitfalls,
        attribution=(f"generated from reference {src_name} "
                     f"(feature extraction only, no content copy)"),
    )


def similarity(features_ref: dict, features_cut: dict) -> dict:
    """节奏特征相似度（验收闸门）：ASL 偏差、切点密度偏差。"""
    asl_dev = abs(features_cut["asl_s"] - features_ref["asl_s"]) / max(features_ref["asl_s"], 0.1)
    dens_dev = abs(features_cut["cut_density_per_min"] - features_ref["cut_density_per_min"]) / max(features_ref["cut_density_per_min"], 0.1)
    return {
        "asl_reference_s": features_ref["asl_s"],
        "asl_cut_s": features_cut["asl_s"],
        "asl_deviation": round(asl_dev, 3),
        "density_reference": features_ref["cut_density_per_min"],
        "density_cut": features_cut["cut_density_per_min"],
        "density_deviation": round(dens_dev, 3),
        "pass": asl_dev <= 0.35 and dens_dev <= 0.35,
    }


def save_user_card(card: ShotCard) -> Path:
    """落 user 卡库（load_cards 自动装载，roughcut --card 直接可用）。"""
    user_dir = ROOT / "director_brain" / "cards" / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    out = user_dir / f"{card.card_id}.json"
    out.write_text(json.dumps([json.loads(card.model_dump_json())],
                              ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="参考片 → 显式镜头卡")
    parser.add_argument("-i", "--input", required=True, help="参考片路径")
    parser.add_argument("--name", default=None, help="卡显示名")
    args = parser.parse_args()

    print(f"[1/2] 分析参考片: {Path(args.input).name}")
    features = extract_features(args.input)
    print(f"  镜头 {features['shot_count']} ｜ ASL {features['asl_s']}s "
          f"｜ 切点密度 {features['cut_density_per_min']}/min "
          f"｜ 四幕占比 {features['act_ratio']}")

    print("[2/2] 生成显式参数卡")
    card = features_to_card(features, args.name)
    out = save_user_card(card)
    print(f"  卡: {card.name}（{card.card_id}）energy={card.energy} "
          f"节奏 {card.pacing_override['min_clip_us']/1e6:.1f}-"
          f"{card.pacing_override['max_clip_us']/1e6:.1f}s")
    print(f"  陷阱: {card.known_pitfalls[0]}")
    print(f"已写入 user 卡库: {out}")
    print(f"用法: python scripts/roughcut.py -i 素材.mp4 --card {card.card_id} "
          f"--confirm-strategy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
