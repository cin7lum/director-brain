# -*- coding: utf-8 -*-
"""阶段 P3-2 · 重要性对照 F1：成片选段 vs VLM 标注的重要性真值。

协议（QVHighlights 采纳）：VLM 打标器给源素材每镜头产 importance 1-5
（TVSum 同构）；importance ≥ 阈值（缺省 4）的镜头构成"应选真值集合"。
成片选中的镜头与真值集合算 precision / recall / F1——
"剪得好不好"的量化答案（标注员=VLM，声明 MODEL_OBSERVATION）。

用法：
    python scripts/importance_f1.py --video 素材.mp4 --edl-json 成片.edl.json \
        [--threshold 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

from observation_service.pipeline import analyze_media                # noqa: E402
from observation_service.vlm_observation import batch_vlm_observations  # noqa: E402


def importance_ground_truth(src: str, threshold: int,
                            adapter) -> tuple[set[str], dict]:
    """VLM 打标：每镜头 importance → 真值集合（importance ≥ threshold）。"""
    tech_obs = analyze_media(src)
    shots = [{"shot_id": o.media_asset_id, "source_in_us": o.start_frame,
              "source_out_us": o.end_frame, "source_media_hash": o.media_hash}
             for o in tech_obs]
    vlm_obs = batch_vlm_observations(src, shots, adapter=adapter)
    truth: set[str] = set()
    labels: dict[str, int] = {}
    for o in vlm_obs:
        try:
            claim = json.loads(o.claim) if o.claim.startswith("{") else {}
        except (json.JSONDecodeError, AttributeError):
            claim = {}
        imp = claim.get("importance")
        labels[o.media_asset_id] = imp if isinstance(imp, int) else 0
        if isinstance(imp, int) and imp >= threshold:
            truth.add(o.media_asset_id)
    return truth, labels


def f1_evaluation(selected: set[str], truth: set[str],
                  all_shots: set[str]) -> dict:
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "selected": len(selected), "truth": len(truth),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="重要性对照 F1 评估")
    parser.add_argument("--video", required=True, help="源素材视频")
    parser.add_argument("--edl-json", required=True, help="成片 EDL sidecar")
    parser.add_argument("--threshold", type=int, default=4,
                        help="importance 真值阈值（缺省 4）")
    args = parser.parse_args()

    from director_brain.config import load_settings
    s = load_settings()
    from observation_service.ollama_vlm_adapter import OllamaVLMAdapter
    adapter = OllamaVLMAdapter(model=s.vlm_model, base_url=s.ollama_base_url)

    edl = json.loads(Path(args.edl_json).read_text(encoding="utf-8"))
    selected = {e["source_asset_id"] if isinstance(e, dict) else e.source_asset_id
                for e in edl["ordered_edits"]}

    truth, labels = importance_ground_truth(args.video, args.threshold, adapter)
    all_shots = set(labels)
    result = f1_evaluation(selected, truth, all_shots)

    print("=== 重要性对照 F1（QVHighlights 协议，标注员=VLM）===")
    print(f"  真值（importance ≥ {args.threshold}）: {sorted(truth)}")
    print(f"  成片选中: {sorted(selected)}")
    print(f"  precision={result['precision']}  recall={result['recall']}  "
          f"F1={result['f1']}")
    print(f"  明细: TP={result['true_positive']} FP={result['false_positive']} "
          f"FN={result['false_negative']}")
    print(f"  全镜头标注: {json.dumps(labels, ensure_ascii=False)}")

    out = Path(args.edl_json).with_suffix(".importance_f1.json")
    out.write_text(json.dumps({
        "video": args.video, "threshold": args.threshold,
        "labels": labels, **result,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
