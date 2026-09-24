"""分析结果缓存层：Analyze Once, Reuse Many。

相同输入（素材哈希 + 提供者 + 模型版本 + prompt 版本 + 采样配置 + 时基 +
schema 版本）产生稳定指纹；命中缓存直接复用，不重复调用 VLM/ASR/确定性分析。
任何关键输入变化生成新指纹，不覆盖历史。

仅依赖 Python 标准库；observations 通过 pydantic 的 ``model_dump(mode="json")``
序列化为 JSON 持久化，加载时用 ``FilmObservation(**data)`` 重建。
"""
from __future__ import annotations

import json
from pathlib import Path

from director_brain._utils import short_hash
from director_brain.models.film_observation import FilmObservation


def compute_fingerprint(
    source_content_hash: str,
    provider: str,
    model_version: str,
    prompt_version: str,
    sampling_config: dict,
    timebase: int,
    schema_version: str = "1.0",
) -> str:
    """根据分析输入元组计算稳定 SHA-256 指纹。

    所有参数（包括空 ``sampling_config``）都参与计算；``sort_keys=True``
    保证 sampling_config 内部 key 顺序不影响结果。任一参数变化即产生
    不同指纹。
    """
    payload = {
        "source_content_hash": source_content_hash,
        "provider": provider,
        "model_version": model_version,
        "prompt_version": prompt_version,
        "sampling_config": sampling_config,
        "timebase": timebase,
        "schema_version": schema_version,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return short_hash(serialized)


class AnalysisCache:
    """内存 + 可选 JSON 文件持久化的观测结果缓存。

    Parameters
    ----------
    persist_path:
        若提供，put 时整体重写该 JSON 文件；实例化时若文件已存在则自动加载。
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._store: dict[str, list[FilmObservation]] = {}
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        self._hit_count = 0
        self._total_count = 0

        if self._persist_path is not None and self._persist_path.exists():
            self._load()

    def put(self, fingerprint: str, observations: list[FilmObservation]) -> None:
        """写入指纹 → observations 映射，并按需持久化。"""
        self._store[fingerprint] = list(observations)
        if self._persist_path is not None:
            self._flush()

    def get(self, fingerprint: str) -> list[FilmObservation] | None:
        """读取缓存；每次调用计入 total_count，命中计入 hit_count。"""
        self._total_count += 1
        if fingerprint in self._store:
            self._hit_count += 1
            return self._store[fingerprint]
        return None

    def invalidate(self, fingerprint: str) -> bool:
        """删除指定指纹缓存，返回是否实际删除成功。"""
        existed = self._store.pop(fingerprint, None) is not None
        if existed and self._persist_path is not None:
            self._flush()
        return existed

    def hit_rate(self) -> float:
        """命中率 hit/total；无调用时返回 0.0。"""
        if self._total_count == 0:
            return 0.0
        return self._hit_count / self._total_count

    # -- 持久化内部方法 ------------------------------------------------------

    def _flush(self) -> None:
        assert self._persist_path is not None
        payload = {
            fp: [obs.model_dump(mode="json") for obs in obs_list]
            for fp, obs_list in self._store.items()
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        assert self._persist_path is not None
        raw = self._persist_path.read_text(encoding="utf-8")
        if not raw.strip():
            return
        payload = json.loads(raw)
        for fp, obs_list in payload.items():
            self._store[fp] = [FilmObservation(**data) for data in obs_list]
