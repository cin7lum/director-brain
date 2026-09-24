"""将 EditorialDecisionList 导出为 DaVinci Resolve 可导入的 CMX 3600 EDL。

输出为纯文本 EDL：
- 头部两行：``TITLE: {edl_id}`` 与 ``FCM: NON-DROP FRAME``；
- 每个 EditItem 两行：固定宽度记录行 + ``* FROM CLIP NAME: ...`` 注释行；
- 时间码统一为 ``HH:MM:SS:FF``，基于 ``fps``（默认 24.0）。

源入/出点直接取自 EditItem 的 ``in_frame`` / ``out_frame``（微秒）；
记录入/出点从 ``00:00:00:00`` 开始按片段时长累加。
"""
from __future__ import annotations

import os

from director_brain.models.edl import EditorialDecisionList


def _us_to_tc(us: int, fps: float) -> str:
    """微秒转 ``HH:MM:SS:FF`` 时间码。

    浮点误差可能使帧数恰好等于 ``fps``，此时钳制为 ``fps - 1``，
    保证 ``FF < fps``。
    """
    total_seconds = us / 1_000_000
    hh = int(total_seconds // 3600)
    mm = int((total_seconds % 3600) // 60)
    ss = int(total_seconds % 60)
    ff = int((total_seconds - int(total_seconds)) * fps)
    ff = min(ff, int(fps) - 1)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def export_to_davinci_edl(
    edl: EditorialDecisionList,
    output_path: str,
    fps: float = 24.0,
) -> str:
    """将 EDL 导出为 CMX 3600 文本文件，返回 output_path。

    输出目录不存在时自动创建，文件以 utf-8 编码写入。
    """
    lines: list[str] = []
    lines.append(f"TITLE: {edl.edl_id}")
    lines.append("FCM: NON-DROP FRAME")

    record_us = 0  # 记录入点（微秒），从 0 累加
    for idx, edit in enumerate(edl.ordered_edits, start=1):
        src_in = _us_to_tc(edit.in_frame, fps)
        src_out = _us_to_tc(edit.out_frame, fps)
        rec_in = _us_to_tc(record_us, fps)
        duration_us = edit.out_frame - edit.in_frame
        record_us += duration_us
        rec_out = _us_to_tc(record_us, fps)

        vol = f"{edit.source_asset_id[:8]:<8s}"
        track = f"{'V':<2s}"
        trans = f"{'C':<4s}"
        lines.append(
            f"{idx:03d} {vol} {track} {trans} "
            f"{src_in} {src_out} {rec_in} {rec_out}"
        )
        lines.append(f"* FROM CLIP NAME: {edit.source_asset_id}")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return output_path
