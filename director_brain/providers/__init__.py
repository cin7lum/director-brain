"""director_brain.providers：导演决策的可插拔 Provider 层。

当前仅含启发式选片 Provider（:mod:`director_brain.providers.heuristic`）。
Provider 只向导演层供给"选片/修片"的确定性能力，不持有导演权。
"""
from director_brain.providers.heuristic import (
    MAX_CLIP_US,
    MIN_CLIP_US,
    HeuristicBaseline,
    compute_clip_window,
    generate_edl,
)

__all__ = [
    "MAX_CLIP_US",
    "MIN_CLIP_US",
    "HeuristicBaseline",
    "compute_clip_window",
    "generate_edl",
]
