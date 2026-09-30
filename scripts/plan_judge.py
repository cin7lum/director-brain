# -*- coding: utf-8 -*-
"""阶段 8c-v2 · 结构化决策评审：已准入模型按冻结五维量表给 EDL 打分。

用户打分降为可选确认后，权威评估 = L1 客观指标 + 切点自然度（权威边界参照）
+ 本模块（评审员=已准入的 doubao-seed-2-1-lite，输入是**选片元数据**而非像素
——像素级评审待 vision 模型开通后升级）。

冻结量表（与 L3 盲评同一五维，1-5 分 + 依据；评审时输入不含"哪个是机器剪的"
信息差——本模块只有一版，无成对语义）。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

JUDGE_RUBRIC = """你是专业剪辑评审。根据给出的剪辑决策元数据（镜头时长/源位置/幕结构/意图约束），
按五个维度打分（1-5 整数）并给一句依据：
1. intent: 意图达成（成片时长是否支撑目标；声明的风格约束是否体现在选片）
2. narrative: 叙事连贯（四幕结构是否完整、镜头序列是否可理解）
3. pacing: 节奏（镜头时长分布是否合理，有无过长/过短镜头）
4. shots: 镜头选择（选段在源素材上的分布是否有代表性）
5. technical: 技术质量信号（源片段是否避开已知低质区间）
只输出 JSON：{"intent": n, "narrative": n, "pacing": n, "shots": n, "technical": n,
"comment": "一句话总评"}"""


def load_env() -> None:
    env = ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def judge_plan(edl: dict, intent_text: str) -> dict:
    """独立评审调用：同一已准入模型，但走评审专用 prompt（非 DirectorDecision 管道）。

    传输统一走 llm_adapter.post_chat_json（架构体检④收编，私有 urllib 已删）。
    """
    from director_brain.llm_adapter import extract_json_object, post_chat_json

    shots = [
        {
            "source_in_s": round(e["in_frame"] / e["timebase"], 2),
            "source_out_s": round(e["out_frame"] / e["timebase"], 2),
            "dur_s": round((e["out_frame"] - e["in_frame"]) / e["timebase"], 2),
            "act": e.get("act"),
            "shot_function": e.get("shot_function"),
            "evidence_type": e.get("evidence_type"),
        }
        for e in edl["ordered_edits"]
    ]
    user_text = (
        f"用户意图: {intent_text or '（未声明，默认自动粗剪）'}\n"
        f"成片总时长: {edl['expected_duration'] / 1e6:.1f}s\n"
        f"镜头清单: {json.dumps(shots, ensure_ascii=False)}"
    )
    content = post_chat_json(
        os.environ["ARK_BASE_URL"],
        os.environ["ARK_API_KEY"],
        os.environ.get("ARK_MODEL", "doubao-seed-2-1-lite-260915"),
        JUDGE_RUBRIC,
        user_text,
        timeout=90,
    )
    return extract_json_object(content)


def main() -> int:
    load_env()
    sidecar = ROOT / "evidence" / "GLM-HANDOVER" / "compound" / "compound_cut.mp4.edl.json"
    intent_text = "快节奏剪辑，避免模糊镜头"
    edl = json.loads(sidecar.read_text(encoding="utf-8"))

    scores = judge_plan(edl, intent_text)
    print("=== 结构化决策评审（doubao-seed-2-1-lite，冻结五维量表）===")
    for k in ("intent", "narrative", "pacing", "shots", "technical"):
        print(f"  {k}: {scores.get(k)}/5")
    print(f"  总评: {scores.get('comment')}")

    out = ROOT / "evidence" / "GLM-HANDOVER" / "plan_judge_scores.json"
    out.write_text(json.dumps({
        "sidecar": str(sidecar), "intent": intent_text,
        "rubric": "frozen five-dimension (same as L3)",
        "scores": scores,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
