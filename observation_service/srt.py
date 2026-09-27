"""SRT 字幕序列化与成片时间线映射（阶段 8a 第 1 期）。

职责边界（P2-b 证据包裁定）：
- **导演层（本模块）**负责把素材时间线的语音观测映射到**成片时间线**
  （时间线数学统一取自 :mod:`director_brain.timeline`——转场重叠语义在此
  单点定义，字幕自动跟随转场，两处各算各的必然漂移）；
- **渲染器**对字幕时间戳零假设，只负责把 SRT 文件作为流/滤镜接入。

SRT 是公开固定格式（``HH:MM:SS,mmm``），本模块是薄序列化封装，非自研格式。
"""
from __future__ import annotations

from director_brain.models.edl import EditorialDecisionList
from director_brain.models.film_observation import FilmObservation
from director_brain.timeline import compute_output_timeline


def _fmt_ts(us: int) -> str:
    """微秒 → SRT 时间戳 ``HH:MM:SS,mmm``。"""
    if us < 0:
        us = 0
    hours, rem = divmod(us, 3_600_000_000)
    minutes, rem = divmod(rem, 60_000_000)
    seconds, rem2 = divmod(rem, 1_000_000)
    millis = rem2 // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def map_observations_to_output_timeline(
    speech_obs: list[FilmObservation],
    edl: EditorialDecisionList,
) -> list[dict]:
    """把素材时间线的语音观测经 EDL 映射到成片时间线。

    时间线语义（含转场重叠）统一取自 :mod:`director_brain.timeline`：
    - cut：下一镜头内容起点 = 累计时长；
    - xfade：下一镜头**内容起点** = 累计输出时长 − 重叠 D（其头部 D 微秒
      与上一镜头尾部混合呈现，字幕在该区间可能与上一条轻微叠显——真实
      crossfade 的固有语义）。

    跨镜头的语音段按镜头边界切开；落在剪辑弃用区间的语音自然丢弃。

    Returns:
        ``[{"start_us", "end_us", "text"}]``，按成片时间升序。
    """
    segments = sorted(speech_obs, key=lambda o: o.start_frame)
    timeline = compute_output_timeline(edl)
    mapped: list[dict] = []
    for tl in timeline:
        for seg in segments:
            overlap_start = max(seg.start_frame, tl.source_in_us)
            overlap_end = min(seg.end_frame, tl.source_out_us)
            if overlap_end <= overlap_start:
                continue
            text = (seg.claim or "").strip()
            if not text:
                continue
            mapped.append({
                "start_us": tl.out_start_us + (overlap_start - tl.source_in_us),
                "end_us": tl.out_start_us + (overlap_end - tl.source_in_us),
                "text": text,
            })
    mapped.sort(key=lambda m: m["start_us"])
    return mapped


def build_srt(
    speech_obs: list[FilmObservation],
    edl: EditorialDecisionList,
    *,
    min_duration_us: int = 300_000,
) -> str:
    """生成成片时间线的 SRT 文本（UTF-8）。

    Args:
        speech_obs: ASR 语音转写观测（素材时间线）。
        edl: 剪辑清单（时间线映射依据）。
        min_duration_us: 单条字幕最短显示时长（不足则补足，不与后条重叠）。
    """
    mapped = map_observations_to_output_timeline(speech_obs, edl)
    blocks: list[str] = []
    for i, seg in enumerate(mapped, 1):
        start = seg["start_us"]
        end = max(seg["end_us"], start + min_duration_us)
        if i < len(mapped):
            end = min(end, mapped[i]["start_us"])  # 不与后条重叠
        blocks.append(
            f"{i}\n{_fmt_ts(start)} --> {_fmt_ts(end)}\n{seg['text']}\n"
        )
    return "\n".join(blocks)
