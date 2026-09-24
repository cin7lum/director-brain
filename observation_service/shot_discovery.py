"""M1.1 镜头切分：基于 ffmpeg scene detection 的确定性镜头发现。

输出稳定 ``shot_id = sha256(media_hash + in_us + out_us + version)``，对短于 0.5s
的相邻段合并、对长于 8s 的单镜头按 4s window 拆分。文件缺失或 ffmpeg/ffprobe
失败时返回空列表，不抛异常。
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess


DISCOVERY_VERSION = "v0.1"
DEFAULT_THRESHOLD = 0.25
MIN_SHOT_DURATION_US = 500_000      # 0.5s
LONG_SHOT_THRESHOLD_US = 8_000_000  # 8s：长镜头按 window 拆分
WINDOW_SIZE_US = 4_000_000          # 4s window
MIN_TAIL_US = 1_500_000             # 末段短于此则并入前一段


def _file_sha256(path: str) -> str:
    """计算文件内容的 sha256（按 1MB 块流式读取）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe_duration_us(path: str) -> int | None:
    """用 ffprobe 读取视频时长（微秒）；失败返回 None。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return None
    if r.returncode != 0:
        return None
    out = r.stdout.strip()
    if not out:
        return None
    try:
        return int(round(float(out) * 1_000_000))
    except ValueError:
        return None


def _scene_cut_times_us(path: str, threshold: float) -> list[int] | None:
    """运行 ffmpeg scene detection，从 stderr 解析 pts_time 得到切分点（微秒）。

    失败（CLI 不可用或非零退出）返回 None。
    """
    vf = f"select='gt(scene,{threshold})',showinfo"
    cmd = [
        "ffmpeg", "-i", path,
        "-vf", vf,
        "-f", "null", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return None
    if r.returncode != 0:
        return None
    times_us: list[int] = []
    for line in r.stderr.splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            times_us.append(int(round(float(m.group(1)) * 1_000_000)))
    return times_us


def _shot_id(media_hash: str, in_us: int, out_us: int, version: str) -> str:
    """稳定 shot_id：sha256(media_hash + in_us + out_us + version)，前缀 shot_。"""
    h = hashlib.sha256()
    h.update(media_hash.encode())
    h.update(str(in_us).encode())
    h.update(str(out_us).encode())
    h.update(version.encode())
    return "shot_" + h.hexdigest()[:16]


def discover_shots(video_path: str) -> list[dict]:
    """检测镜头边界并返回覆盖整个视频的镜头列表。

    规则：
    - 无切分点 → 返回单镜头覆盖整个视频。
    - 短于 ``MIN_SHOT_DURATION_US`` 的段合并到相邻段。
    - 长于 ``LONG_SHOT_THRESHOLD_US`` 的段按 ``WINDOW_SIZE_US`` 拆分。
    - 首尾相接、无间隙：首段起于 0，末段终于 duration_us。
    - 文件不存在或 ffmpeg/ffprobe 失败 → 返回空列表，不抛异常。
    """
    if not video_path or not os.path.isfile(video_path):
        return []

    try:
        media_hash = _file_sha256(video_path)
    except OSError:
        return []

    duration_us = _probe_duration_us(video_path)
    if not duration_us or duration_us <= 0:
        return []

    cuts = _scene_cut_times_us(video_path, DEFAULT_THRESHOLD)
    if cuts is None:
        return []

    # 仅保留 (0, duration) 内的切分点并去重排序
    cuts = sorted(set(c for c in cuts if 0 < c < duration_us))

    # 原始边界：[0, cuts..., duration]
    bounds = [0] + cuts + [duration_us]

    # 迭代合并过短段：若某段 < min_duration，删除产生它的切分点（保留更长邻段）
    changed = True
    while changed and len(bounds) > 2:
        changed = False
        for i in range(len(bounds) - 1):
            seg_len = bounds[i + 1] - bounds[i]
            if seg_len < MIN_SHOT_DURATION_US:
                if i == 0:
                    del bounds[1]
                elif i + 2 >= len(bounds):
                    del bounds[-2]
                else:
                    left = bounds[i] - bounds[i - 1]
                    right = bounds[i + 2] - bounds[i + 1]
                    if left <= right:
                        del bounds[i]
                    else:
                        del bounds[i + 1]
                changed = True
                break

    # 构造原始镜头区间（合并后）
    raw_intervals = [
        (bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
    ]

    # 长镜头按 window 拆分；每个子段用各自的 in/out 重新计算 shot_id
    shots: list[dict] = []
    for in_us, out_us in raw_intervals:
        duration = out_us - in_us
        if duration <= LONG_SHOT_THRESHOLD_US:
            sub_bounds = [in_us, out_us]
        else:
            window_cuts = list(range(in_us + WINDOW_SIZE_US, out_us, WINDOW_SIZE_US))
            if window_cuts and (out_us - window_cuts[-1]) < MIN_TAIL_US:
                window_cuts = window_cuts[:-1]
            sub_bounds = [in_us] + window_cuts + [out_us]

        for j in range(len(sub_bounds) - 1):
            s_in = sub_bounds[j]
            s_out = sub_bounds[j + 1]
            shots.append({
                "shot_id": _shot_id(media_hash, s_in, s_out, DISCOVERY_VERSION),
                "source_in_us": s_in,
                "source_out_us": s_out,
                "duration_us": s_out - s_in,
                "keyframes": [],
                "source_media_hash": media_hash,
            })

    return shots
