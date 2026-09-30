# -*- coding: utf-8 -*-
"""像素级评审（阶段 8c-v2）：本机 qwen3-vl 按冻结五维量表给成片打分。

性价比定案（专家组市调）：本机 ollama qwen3-vl（零成本/零新账号/适配器现成）
胜过方舟 vision（需开通+按量付费）与 GLM-4V-Flash（需新注册）。qwen3-vl 曾因
DirectorDecision schema 不稳被准入拒绝，但像素描述任务无该约束。

方法：EDL 每镜头中点抽 1 帧 → 本机 qwen3-vl 按冻结五维量表评分（JSON）→ 聚合。
输出：逐镜头评分 + 均值计分卡（与 L3 盲评同一五维，可对照）。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\新建豆包\AI-Director")
sys.path.insert(0, str(ROOT))

JUDGE_PROMPT = (
    "你是专业剪辑评审。这些帧来自自动剪辑成片（按镜头顺序抽取，每镜头一帧）。"
    "按五个维度打分（1-5 整数）并给一句话总评：\n"
    '1. technical: 技术质量（黑帧/冻帧/模糊/曝光问题）\n'
    '2. shots: 镜头选择（画面是否有信息量与代表性）\n'
    '3. pacing: 节奏观感（从画面序列推测）\n'
    '4. narrative: 叙事连贯（画面序列是否可理解）\n'
    '5. visual: 视觉观感（构图/色彩/美感）\n'
    '只输出 JSON：{"shots": [{"frame": 1, "technical": n, "shots": n, '
    '"pacing": n, "narrative": n, "visual": n, "note": "一句话"}], '
    '"overall": {"technical": n, "shots": n, "pacing": n, "narrative": n, '
    '"visual": n, "comment": "一句话总评"}}'
)


def extract_shot_keyframes(video: str, shot_mids_s: list[float],
                           out_dir: Path) -> list[Path]:
    """每镜头中点抽 1 帧。"""
    frames = []
    for i, t in enumerate(shot_mids_s, 1):
        fp = out_dir / f"shot_{i:02d}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", video,
             "-frames:v", "1", "-q:v", "2", str(fp)],
            check=True, capture_output=True, timeout=60)
        frames.append(fp)
    return frames


def _zhipu_vision_call(frames: list[Path], prompt: str,
                       model: str, api_key: str) -> str:
    """智谱 GLM-4V 系调用（OpenAI 兼容，免费档 GLM-4V-Flash）。

    传输统一走 llm_adapter.post_chat_json（架构体检④收编；多模态 content
    数组经 user 参数透传）。
    """
    from director_brain.llm_adapter import post_chat_json

    content: list = [{"type": "text", "text": prompt}]
    for f in frames:
        b64 = base64.b64encode(f.read_bytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    base = os.environ.get("ZHIPU_BASE_URL",
                          "https://open.bigmodel.cn/api/paas/v4")
    return post_chat_json(
        base, api_key, model, "", content, timeout=180, max_tokens=4096)


def ollama_vision_judge(frames: list[Path], model: str = "qwen3-vl:latest",
                        chunk: int = 1) -> dict:
    """分块评审（单请求体过大会 400）：每 chunk ≤2 帧，结果合并。"""
    import re
    per_shot: list[dict] = []
    for start in range(0, len(frames), chunk):
        batch = frames[start:start + chunk]
        imgs = [base64.b64encode(f.read_bytes()).decode() for f in batch]
        ids = list(range(start + 1, start + len(batch) + 1))
        prompt = JUDGE_PROMPT + (
            f"（本批帧的 frame 编号依次为 {ids}，与图片顺序一致）"
            if len(batch) > 1 else
            f"（本帧 frame 编号为 {ids[0]}）")
        payload = {
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": imgs}],
            "options": {"temperature": 0.1},
            "think": False,  # qwen3-vl 思考模式会吞掉可见输出（实测）
        }
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 400:  # 旧版 ollama 不认识 think 参数 → 去掉重试
                payload.pop("think", None)
                req = urllib.request.Request(
                    "http://localhost:11434/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=600) as r:
                    resp = json.loads(r.read())
            else:
                raise
        content = resp["message"]["content"].strip()
        # 解析统一走共享 extract_json_object（架构体检④收编；原生 ollama
        # 传输保留——非 OpenAI 兼容形状，共享传输不覆盖）
        from director_brain.llm_adapter import extract_json_object
        verdict = extract_json_object(content)
        for s in verdict.get("shots", []):
            per_shot.append(s)
    overall = verdict.get("overall", {})
    return {"shots": per_shot, "overall": overall}


def main() -> int:
    parser = argparse.ArgumentParser(description="像素级评审（本机 qwen3-vl）")
    parser.add_argument("--video", required=True)
    parser.add_argument("--edl-json", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    edl = json.loads(Path(args.edl_json).read_text(encoding="utf-8"))
    # 关键帧取成片时间线位置（8a-3 时间线数学单点复用；源时间≠成片时间）
    from director_brain.timeline import compute_output_timeline
    from director_brain.models.edl import EditorialDecisionList
    tl = compute_output_timeline(EditorialDecisionList.model_validate(edl))
    mids = [(t.out_start_us + t.duration_us / 2) / 1_000_000 for t in tl]

    out_dir = Path(args.outdir) if args.outdir else Path(args.video).parent / "judge_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = extract_shot_keyframes(args.video, mids, out_dir)
    print(f"关键帧: {len(frames)} 张（每镜头中点 1 帧）")

    provider = os.environ.get("JUDGE_PROVIDER", "local")
    if provider == "zhipu":
        # 云端免费档 GLM-4V-Flash：质量高于本地 8B，适合批量终评
        per_shot = []
        for i, f in enumerate(frames, 1):
            raw = _zhipu_vision_call(
                [f], JUDGE_PROMPT + f"（本帧 frame 编号为 {i}）",
                os.environ.get("JUDGE_MODEL", "glm-4v-flash"),
                os.environ.get("ZHIPU_API_KEY", ""))
            lo, hi = raw.find("{"), raw.rfind("}")
            verdict_one = json.loads(raw[lo:hi + 1])
            shot = verdict_one.get("shots", [{}])[0] if "shots" in verdict_one else verdict_one
            shot.setdefault("frame", i)
            per_shot.append(shot)
        verdict = {"shots": per_shot, "overall": {}}
    else:
        verdict = ollama_vision_judge(frames)

    dims = ["technical", "shots", "pacing", "narrative", "visual"]
    per_shot = verdict.get("shots", [])
    agg = {}
    for d in dims:
        vals = [s.get(d) for s in per_shot if isinstance(s.get(d), (int, float))]
        agg[d] = round(statistics.mean(vals), 2) if vals else None
    overall = verdict.get("overall", {})

    print("=== 像素级评审（本机 qwen3-vl，冻结五维量表）===")
    for s in per_shot:
        print(f"  镜头{s.get('frame')}: tech={s.get('technical')} "
              f"shots={s.get('shots')} pacing={s.get('pacing')} "
              f"narrative={s.get('narrative')} visual={s.get('visual')} — {s.get('note', '')[:40]}")
    print(f"均值: {agg}")
    print(f"总评: {overall.get('comment')}")

    out_path = Path(args.video + ".visual_judge.json")
    out_path.write_text(json.dumps({
        "video": args.video, "model": "qwen3-vl:latest",
        "per_shot": per_shot, "aggregate": agg, "overall": overall,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
