"""环境自检（T3）：验证本机环境能否运行主链。纯标准库，零第三方依赖。

用法：
    python scripts/doctor.py            # 检查并打印 [OK]/[MISSING]/[OPTIONAL]
    python scripts/doctor.py --json     # 机器可读输出

判定：
- REQUIRED 缺失 → 退出码 1（环境不完整，主链跑不起来）
- OPTIONAL 缺失 → 退出码 0（对应功能跳过，不算失败）
- 配置文件：.env 不存在不算失败（有 .env.example 模板即可），存在则列出字段名
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (模块名, pip 包名, 是否必需, 用途)
CHECKS = [
    ("pydantic", "pydantic", True, "数据契约"),
    ("numpy", "numpy", True, "确定性分析"),
    ("cv2", "opencv-python", True, "镜头切分/技术观测"),
    ("faster_whisper", "faster-whisper", False, "ASR 语音转写（可选，缺省跳过）"),
]

REQUIRED_EXTERNAL_TOOLS = [
    ("ffmpeg", "渲染成片"),
    ("ffprobe", "媒体元信息"),
]


def _module_ok(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def run_checks() -> dict:
    """执行全部检查，返回结构化结果（不打印）。"""
    python_ok = sys.version_info >= (3, 11)
    results: dict = {
        "python": {
            "ok": python_ok,
            "version": ".".join(map(str, sys.version_info[:3])),
            "required": ">=3.11",
        },
        "modules": [],
        "tools": [],
        "config": [],
    }
    all_ok = python_ok

    for module, package, required, purpose in CHECKS:
        ok = _module_ok(module)
        results["modules"].append({
            "module": module,
            "package": package,
            "required": required,
            "purpose": purpose,
            "ok": ok,
        })
        if required and not ok:
            all_ok = False

    for tool, purpose in REQUIRED_EXTERNAL_TOOLS:
        ok = shutil.which(tool) is not None
        results["tools"].append({"tool": tool, "purpose": purpose, "ok": ok})
        if not ok:
            all_ok = False

    env_path = _PROJECT_ROOT / ".env"
    example_path = _PROJECT_ROOT / ".env.example"
    results["config"] = {
        "env_exists": env_path.is_file(),
        "env_example_exists": example_path.is_file(),
        "env_fields": (
            [
                line.split("=", 1)[0].strip()
                for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not line.strip().startswith("#") and "=" in line
            ]
            if env_path.is_file()
            else []
        ),
    }

    results["all_ok"] = all_ok
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="director-brain 环境自检")
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    args = parser.parse_args()

    results = run_checks()

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0 if results["all_ok"] else 1

    py = results["python"]
    print(f"[{'OK' if py['ok'] else 'MISSING'}] python {py['version']}（要求 {py['required']}）")
    for m in results["modules"]:
        tag = "OK" if m["ok"] else ("MISSING" if m["required"] else "OPTIONAL")
        print(f"[{tag}] {m['package']}（{m['purpose']}）")
    for t in results["tools"]:
        print(f"[{'OK' if t['ok'] else 'MISSING'}] {t['tool']}（{t['purpose']}）")
    cfg = results["config"]
    if cfg["env_exists"]:
        print(f"[OK] .env 存在（字段: {', '.join(cfg['env_fields'])}；值不打印）")
    else:
        tag = "OK" if cfg["env_example_exists"] else "MISSING"
        print(f"[{tag}] .env 不存在（模板 .env.example {'在' if cfg['env_example_exists'] else '缺失'}）")

    print("=" * 50)
    if results["all_ok"]:
        print("环境自检通过：主链可运行（可选依赖缺失的功能会自动跳过）。")
        return 0
    print("环境自检未通过：存在缺失的必需项，请按上方 [MISSING] 提示安装。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
