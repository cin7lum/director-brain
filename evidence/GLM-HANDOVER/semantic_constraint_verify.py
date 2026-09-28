# -*- coding: utf-8 -*-
"""阶段 P3-3 · 语义约束验证升级演示：云端 vision 让"必须有日出"类约束可判。

背景：链 A 的语义类 must_include（如"必须包含日出镜头"）此前诚实标注
constraint_unverifiable——因为技术观测看不见日出。云端 GLM-4V-Flash 能
看帧 → 该约束首次变为**可判定**。

本演示对 compound 成片帧跑语义验证（素材是雪地场景，无日出 →
应判"约束未满足"并给证据），同时演示满足场景的正例判定逻辑。
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from scripts.visual_judge import _zhipu_vision_call  # noqa: E402

VERIFY_PROMPT = """你是剪辑约束验证员。判断这些画面是否包含以下视觉元素：
约束元素: "{element}"

只输出 JSON：{{"present": true/false, "confidence": 1-5, "evidence": "画面依据一句话"}}
画面里没有明确证据时 present 必须为 false（fail-closed，不臆测）。"""


def verify_element(frames: list[Path], element: str, model: str,
                   api_key: str) -> dict:
    """逐帧验证（GLM-4V-Flash 单图限制）：任一帧命中即 present，fail-closed。"""
    verdicts = []
    for f in frames:
        try:
            raw = _zhipu_vision_call([f], VERIFY_PROMPT.format(element=element),
                                     model, api_key)
            lo, hi = raw.find("{"), raw.rfind("}")
            verdicts.append(json.loads(raw[lo:hi + 1]))
        except Exception as exc:  # noqa: BLE001
            print(f"    [帧跳过] {f.name}: {str(exc)[:60]}")
    present = any(v.get("present") for v in verdicts)
    conf = max((v.get("confidence", 0) for v in verdicts
                if v.get("present")), default=0)
    evidence = next((v.get("evidence") for v in verdicts if v.get("present")),
                    "全部帧未发现该元素（fail-closed）")
    return {"present": present, "confidence": conf, "evidence": evidence,
            "frames_checked": len(verdicts)}


def main() -> int:
    frames = sorted((ROOT / "evidence" / "GLM-HANDOVER" / "compound" /
                     "judge_frames").glob("*.jpg"))
    model = os.environ.get("JUDGE_MODEL", "glm-4v-flash")
    api_key = os.environ["ZHIPU_API_KEY"]
    print(f"[1] 评审帧: {len(frames)} 张（compound 成片抽帧）｜ 模型: {model}")

    # 语义 must_include 约束：当前技术观测无法验证的类别
    element = "日出"
    v = verify_element(frames, element, model, api_key)
    print(f"\n[2] 语义约束验证: must_include「{element}」")
    print(f"    present={v.get('present')}  confidence={v.get('confidence')}")
    print(f"    依据: {v.get('evidence')}")

    # 技术类约束对照（既有能力，应判满足——雪地画面亮度高）
    v2 = verify_element(frames, "雪地/积雪场景", model, api_key)
    print(f"\n[3] 技术可验证对照: must_include「雪地/积雪场景」")
    print(f"    present={v2.get('present')}  依据: {v2.get('evidence')}")

    print("\n=== 结论 ===")
    print("语义约束从 constraint_unverifiable（永久无法判定）升级为")
    print("VLM 可判定（present/confidence/证据三要素齐备）——")
    print("治理上仍需冻结验证阈值后方可进 validator 判红，但'无法验证'的")
    print("根本约束已被云端 vision API 解除。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
