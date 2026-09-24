"""M2.3 EDL Generator：从选中的镜头列表构建有序 EditorialDecisionList。

不做任何校验或重排——按 ``selected_shots`` 传入顺序原样映射为 EditItem，
校验与修复由 :mod:`director_brain.plan_validator` /
:mod:`director_brain.plan_repair` 负责。
"""
from __future__ import annotations

import time

from director_brain._utils import short_hash
from director_brain.models.edl import EditItem, EditorialDecisionList

PRODUCER = "edl_generator_v0.1"


def generate_edl(
    project_id: str,
    selected_shots: list[dict],
    timebase: int = 1_000_000,
    source_ref: str | None = None,
) -> EditorialDecisionList:
    """从选中的镜头列表构建有序 EditorialDecisionList。

    Args:
        project_id: 项目 ID。
        selected_shots: 每个 dict 含 ``source_asset_id``、``source_media_hash``、
            ``in_frame``、``out_frame``，可选 ``shot_function``、``rationale``。
            顺序即成片顺序，不重新排序。
        timebase: 时间基（微秒/秒），默认 1_000_000。
        source_ref: 源素材引用，语义应为视频路径或资产标识。未传入时回退到
            ``project_id`` 以保持向后兼容。

    Returns:
        构建好的 :class:`EditorialDecisionList`，``approval_state="draft"``。
        空列表返回空 EDL（``ordered_edits=[]``, ``expected_duration=0``），
        不抛异常。
    """
    # source_ref 语义为源素材引用，应传视频路径/资产标识；未传入时回退到
    # project_id 以保持向后兼容。
    resolved_source_ref = source_ref or project_id
    edits: list[EditItem] = []
    source_hashes: list[str] = []
    seen_hashes: set[str] = set()
    expected_duration = 0

    for shot in selected_shots:
        in_frame = int(shot["in_frame"])
        out_frame = int(shot["out_frame"])
        edits.append(EditItem(
            source_asset_id=shot["source_asset_id"],
            source_media_hash=shot["source_media_hash"],
            in_frame=in_frame,
            out_frame=out_frame,
            timebase=timebase,
            shot_function=shot.get("shot_function"),
            rationale=shot.get("rationale"),
        ))
        expected_duration += out_frame - in_frame

        h = shot["source_media_hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            source_hashes.append(h)

    fingerprint = "|".join(
        f"{e.source_asset_id}:{e.in_frame}:{e.out_frame}" for e in edits
    )
    return EditorialDecisionList(
        schema_version="1.0",
        project_id=project_id,
        created_at=int(time.time()),
        producer=PRODUCER,
        source_ref=resolved_source_ref,
        edl_id=f"edl_{project_id}_{short_hash(fingerprint)}",
        version="0.1",
        brief_version="0.1",
        context_id=f"ctx_{project_id}",
        source_asset_hashes=source_hashes,
        timebase=timebase,
        ordered_edits=edits,
        expected_duration=expected_duration,
        approval_state="draft",
    )
