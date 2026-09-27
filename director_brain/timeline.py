"""成片时间线计算（阶段 8a-3 地基）。

渲染器（xfade/acrossfade 链的 offset）与字幕（成片时间线映射）共用同一份
时间线数学，保证"字幕自动跟随转场"——两处各算各的必然漂移。

语义（P2-b 证据包）：
- 无转场（cut）：输出时长 = Σ镜头时长，下一镜头输出起点 = 累计时长；
- xfade 转场：重叠 D 微秒，输出时长 = Σ镜头时长 − ΣD；
  下一镜头的**内容起点** = 累计输出时长 − D（其头部 D 微秒与上一镜头尾部
  混合呈现）。
"""
from __future__ import annotations

from dataclasses import dataclass

from director_brain.models.edl import EditorialDecisionList, TransitionSpec


@dataclass
class EditTimeline:
    """单个镜头在成片时间线上的位置。"""

    edit_index: int
    source_in_us: int
    source_out_us: int
    duration_us: int
    #: 本镜头内容在成片时间线上的起始位置
    out_start_us: int
    #: 到下一镜头的转场（None 或 type=cut 表示硬切）
    transition_to_next: TransitionSpec | None


def compute_output_timeline(edl: EditorialDecisionList) -> list[EditTimeline]:
    """计算每个镜头在成片时间线上的位置（含转场重叠语义）。"""
    result: list[EditTimeline] = []
    timeline_us = 0
    edits = edl.ordered_edits
    for i, edit in enumerate(edits):
        duration = edit.out_frame - edit.in_frame
        result.append(EditTimeline(
            edit_index=i,
            source_in_us=edit.in_frame,
            source_out_us=edit.out_frame,
            duration_us=duration,
            out_start_us=timeline_us,
            transition_to_next=edit.transition,
        ))
        junction = edit.transition
        if i < len(edits) - 1 and junction is not None and junction.type == "xfade":
            timeline_us += duration - junction.duration_us
        else:
            timeline_us += duration
    return result


def total_output_duration_us(edl: EditorialDecisionList) -> int:
    """成片总时长（cut=Σd；xfade=Σd−ΣD）。"""
    timeline = compute_output_timeline(edl)
    if not timeline:
        return 0
    last = timeline[-1]
    return last.out_start_us + last.duration_us
