"""节拍网格观测（D5 · 音乐节拍驱动剪辑）。

librosa（ISC，采用级准入——计划附录 A）对 BGM 提取节拍网格：
``BeatGrid{bpm, beat_times_us}``。缓存键 = 音频 sha256 + librosa 版本 +
检测参数（Analyze Once Reuse Many 纪律）。

通路：``beat_grid``（T4 协议注册，默认 EXPERIMENTAL）——网格计算与
落账本不需要转正；**切点吸附节拍**属于决策消费，须 ACTIVE
（ensure_decision_use_allowed 执法）。
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_LIBROSA_VERSION = "1.0.0"  # 采用快照；升级须复验
_TIMEBASE_US = 1_000_000


@dataclass
class BeatGrid:
    """一份 BGM 的节拍网格。"""

    bpm: float
    beat_times_us: list[int] = field(default_factory=list)
    duration_us: int = 0
    source: str = ""

    def to_dict(self) -> dict:
        return {"bpm": round(self.bpm, 1),
                "beats": len(self.beat_times_us),
                "duration_us": self.duration_us,
                "source": self.source,
                "first_beats_s": [round(t / 1e6, 2) for t in self.beat_times_us[:8]]}


def analyze_beat_grid(audio_path: str, cache_dir: str | None = None) -> BeatGrid:
    """librosa 节拍检测（带缓存）；失败抛 RuntimeError（调用方响亮降级）。"""
    ap = Path(audio_path)
    if not ap.is_file():
        raise RuntimeError(f"BGM 文件不存在: {audio_path}")
    audio_hash = hashlib.sha256(ap.read_bytes()).hexdigest()[:16]
    fingerprint = f"beatgrid_{audio_hash}_{_LIBROSA_VERSION}"

    if cache_dir:
        cache_file = Path(cache_dir) / f"{fingerprint}.json"
        if cache_file.is_file():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return BeatGrid(**data)

    import librosa
    import numpy as np

    y, sr = librosa.load(str(ap), sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    grid = BeatGrid(
        bpm=round(tempo, 1),
        beat_times_us=[int(round(float(b) * _TIMEBASE_US)) for b in beats],
        duration_us=int(round(len(y) / sr * _TIMEBASE_US)),
        source=str(ap),
    )
    if not grid.beat_times_us:
        raise RuntimeError(f"节拍检测未产出网格: {audio_path}")

    if cache_dir:
        cache_file = Path(cache_dir) / f"{fingerprint}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(grid.__dict__, ensure_ascii=False),
                              encoding="utf-8")
    logger.info("beat grid: %s bpm=%s beats=%s", ap.name, grid.bpm,
                len(grid.beat_times_us))
    return grid
