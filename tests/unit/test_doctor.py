"""scripts/doctor.py 环境自检单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.doctor import run_checks  # noqa: E402


def test_all_ok_in_dev_environment():
    """本开发环境（numpy/cv2/ffmpeg 齐备）自检必须通过。"""
    results = run_checks()
    assert results["all_ok"] is True, results
    assert results["python"]["ok"] is True
    by_mod = {m["module"]: m for m in results["modules"]}
    # 必需项在开发环境必须就绪
    assert by_mod["pydantic"]["ok"] is True
    assert by_mod["numpy"]["ok"] is True
    assert by_mod["cv2"]["ok"] is True
    assert by_mod["faster_whisper"]["required"] is False  # 可选项标记正确
    by_tool = {t["tool"]: t for t in results["tools"]}
    assert by_tool["ffmpeg"]["ok"] is True
    assert by_tool["ffprobe"]["ok"] is True


def test_missing_required_module_flags_all_ok_false(monkeypatch):
    """必需模块缺失时 all_ok 必须为 False（这是 doctor 的判负能力）。"""
    import scripts.doctor as doctor

    monkeypatch.setattr(doctor, "_module_ok", lambda name: name != "cv2")
    results = doctor.run_checks()
    assert results["all_ok"] is False
    cv2 = next(m for m in results["modules"] if m["module"] == "cv2")
    assert cv2["ok"] is False and cv2["required"] is True


def test_missing_optional_module_keeps_all_ok_true(monkeypatch):
    """可选模块（ASR）缺失不判失败。"""
    import scripts.doctor as doctor

    monkeypatch.setattr(doctor, "_module_ok", lambda name: name != "faster_whisper")
    results = doctor.run_checks()
    assert results["all_ok"] is True
    fw = next(m for m in results["modules"] if m["module"] == "faster_whisper")
    assert fw["ok"] is False and fw["required"] is False


def test_env_fields_do_not_leak_values():
    """doctor 只报告 .env 字段名，绝不输出字段值。"""
    results = run_checks()
    text = repr(results)
    env_file = Path(_PROJECT_ROOT) / ".env"
    if not env_file.is_file():
        # CI/干净环境无 .env——无值可泄漏，断言空转通过
        return
    for line in env_file.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            value = line.split("=", 1)[1].strip()
            if value:
                assert value not in text, f".env 值泄漏到 doctor 输出: {value}"
