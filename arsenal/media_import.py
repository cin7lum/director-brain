"""Arsenal 媒体导入与代理生成。

流程：ffprobe 元数据探测 → 复制原始素材 → ffmpeg 低分辨率代理 → 写入 media_manifest.json。
SHA-256 复用 ``director_brain.utils.file_sha256``，不在此模块重复实现。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from director_brain.utils import file_sha256

#: 代理默认宽度（高度由 -2 保证偶数）
_DEFAULT_PROXY_WIDTH = 480


def _run_ffprobe(video_path: str) -> dict:
    """调用 ffprobe 读取 JSON 元数据，失败抛 RuntimeError。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _parse_fps(r_frame_rate: str) -> float:
    """将 ffprobe 的 r_frame_rate（如 "24/1"、"30000/1001"）解析为 float fps。"""
    num_str, _, den_str = r_frame_rate.partition("/")
    try:
        num = float(num_str)
        den = float(den_str) if den_str else 1.0
    except ValueError:
        return 0.0
    if den == 0:
        return 0.0
    return num / den


def probe_media(video_path: str) -> dict:
    """用 ffprobe 探测媒体元数据。

    Args:
        video_path: 视频文件路径。

    Returns:
        含以下字段的 dict：
        ``duration_us``(int) / ``width``(int) / ``height``(int) / ``fps``(float) /
        ``codec``(str) / ``audio_present``(bool) / ``file_size``(int) /
        ``sha256``(str) / ``path``(str)。

    Raises:
        RuntimeError: ffprobe 调用失败。
    """
    meta = _run_ffprobe(video_path)

    fmt = meta.get("format", {})
    v_streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    a_streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    v = v_streams[0] if v_streams else {}

    duration_s = float(fmt.get("duration", v.get("duration", 0.0)))
    duration_us = int(round(duration_s * 1_000_000))

    return {
        "duration_us": duration_us,
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "fps": _parse_fps(v.get("r_frame_rate", "0/0")),
        "codec": str(v.get("codec_name", "")),
        "audio_present": len(a_streams) > 0,
        "file_size": os.path.getsize(video_path),
        "sha256": file_sha256(video_path),
        "path": video_path,
    }


def _proxy_meta_path(proxy_path: str) -> Path:
    """代理旁车 meta 文件路径：同目录，文件名 = 代理名 + '.proxy_meta.json'。"""
    return Path(proxy_path + ".proxy_meta.json")


def generate_proxy(src_path: str, proxy_path: str, width: int = _DEFAULT_PROXY_WIDTH) -> str:
    """用 ffmpeg 生成低分辨率 H.264 代理（幂等）。

    若代理已存在且旁车 meta 中记录的 ``source_sha256`` 与源文件一致，则跳过生成。
    生成成功后写入 ``.proxy_meta.json``。

    Args:
        src_path: 源视频路径。
        proxy_path: 输出代理路径。
        width: 代理宽度，默认 480。

    Returns:
        代理路径（str）。

    Raises:
        RuntimeError: ffmpeg 调用失败。
    """
    proxy_p = Path(proxy_path)
    meta_p = _proxy_meta_path(proxy_path)

    # 幂等：代理与 meta 都存在且 source_sha256 匹配则跳过
    if proxy_p.is_file() and meta_p.is_file():
        try:
            saved = json.loads(meta_p.read_text(encoding="utf-8"))
            if saved.get("source_sha256") == file_sha256(src_path):
                return proxy_path
        except (json.JSONDecodeError, OSError):
            pass  # meta 损坏则重新生成

    # 输出目录不存在则创建
    proxy_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-vf", f"scale={width}:-2",
        "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k",
        str(proxy_p),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise RuntimeError(f"ffmpeg 代理生成失败: {tail}")

    # 写入旁车 meta
    meta = {
        "source_sha256": file_sha256(src_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "width": width,
    }
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return proxy_path


def import_media(video_path: str, project_root: Path) -> dict:
    """完整导入一个媒体文件到项目。

    1. probe_media 读取元数据；
    2. shutil.copy2 复制原始素材到 ``media/originals/``；
    3. generate_proxy 生成代理到 ``media/proxies/``；
    4. 写入/更新 ``media_manifest.json``（同 sha256 覆盖，幂等）。

    Args:
        video_path: 源媒体路径。
        project_root: ``create_project`` 返回的项目根 Path。

    Returns:
        manifest entry dict（含元数据、original_path、proxy_path、sha256、imported_at）。
    """
    project_root = Path(project_root)
    meta = probe_media(video_path)

    originals_dir = project_root / "media" / "originals"
    proxies_dir = project_root / "media" / "proxies"
    originals_dir.mkdir(parents=True, exist_ok=True)
    proxies_dir.mkdir(parents=True, exist_ok=True)

    # 复制原始素材（保留文件名与元数据）
    src_name = Path(video_path).name
    original_dst = originals_dir / src_name
    shutil.copy2(video_path, str(original_dst))

    # 代理文件名 = 原始 stem + "_proxy.mp4"
    proxy_name = Path(video_path).stem + "_proxy.mp4"
    proxy_dst = proxies_dir / proxy_name
    # 以项目内副本为源生成代理，保证项目自包含
    generate_proxy(str(original_dst), str(proxy_dst))

    entry = {
        **meta,
        "original_path": str(original_dst),
        "proxy_path": str(proxy_dst),
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }

    # 更新 media_manifest.json：同 sha256 覆盖
    manifest_path = project_root / "media_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"media": []}

    replaced = False
    for i, existing in enumerate(manifest["media"]):
        if existing.get("sha256") == entry["sha256"]:
            manifest["media"][i] = entry
            replaced = True
            break
    if not replaced:
        manifest["media"].append(entry)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return entry
