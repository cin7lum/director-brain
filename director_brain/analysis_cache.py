"""分析结果缓存层：Analyze Once, Reuse Many。

相同输入（素材哈希 + 提供者 + 模型版本 + prompt 版本 + 采样配置 + 时基 +
schema 版本）产生稳定指纹；命中缓存直接复用，不重复调用 VLM/ASR/确定性分析。
任何关键输入变化生成新指纹，不覆盖历史。

仅依赖 Python 标准库；observations 通过 pydantic 的 ``model_dump(mode="json")``
序列化为 JSON 持久化，加载时用 ``FilmObservation(**data)`` 重建。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from director_brain.utils import short_hash
from director_brain.models.film_observation import FilmObservation


class AnalysisCacheStore(Protocol):
    """Minimal cache contract shared by file and repository-backed stores."""

    def get(self, fingerprint: str) -> list[FilmObservation] | None: ...

    def put(self, fingerprint: str, observations: list[FilmObservation]) -> None: ...

    def flush(self) -> None: ...


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
        若提供，实例化时若文件已存在则自动加载。

    Notes
    -----
    ``put()`` / ``invalidate()`` 不再每次立即写盘，而是仅标记内存为 dirty。
    调用方在批量操作结束后必须显式调用 :meth:`flush` 将内存状态持久化到磁盘；
    未调用 ``flush()`` 即进程退出时，未写入的修改不会保留。
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._store: dict[str, list[FilmObservation]] = {}
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        self._hit_count = 0
        self._total_count = 0
        self._dirty: bool = False

        if self._persist_path is not None and self._persist_path.exists():
            self._load()

    def put(self, fingerprint: str, observations: list[FilmObservation]) -> None:
        """写入指纹 → observations 映射，标记为 dirty（不立即写盘）。"""
        self._store[fingerprint] = list(observations)
        if self._persist_path is not None:
            self._dirty = True

    def get(self, fingerprint: str) -> list[FilmObservation] | None:
        """读取缓存；每次调用计入 total_count，命中计入 hit_count。"""
        self._total_count += 1
        if fingerprint in self._store:
            self._hit_count += 1
            return self._store[fingerprint]
        return None

    def invalidate(self, fingerprint: str) -> bool:
        """删除指定指纹缓存，返回是否实际删除成功。标记为 dirty（不立即写盘）。"""
        existed = self._store.pop(fingerprint, None) is not None
        if existed and self._persist_path is not None:
            self._dirty = True
        return existed

    def hit_rate(self) -> float:
        """命中率 hit/total；无调用时返回 0.0。"""
        if self._total_count == 0:
            return 0.0
        return self._hit_count / self._total_count

    # -- 持久化公共接口 ------------------------------------------------------

    def flush(self) -> None:
        """将内存中的 dirty 状态持久化到磁盘；未 dirty 时不触发写入。

        调用方在批量 put / invalidate 操作结束后应显式调用此方法，
        否则修改不会写入磁盘文件。
        """
        if self._persist_path is None or not self._dirty:
            return
        self._flush()
        self._dirty = False

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


class RepositoryAnalysisCache:
    """Persist provider-level observations through the configured repository.

    Each put is committed by the repository immediately. Long local inference
    can therefore resume from completed shots after an application restart
    without writing media-derived cache files beside source footage.
    """

    def __init__(self, repository, source_content_hash: str,
                 analysis_profile: str):
        self._repository = repository
        self._source_content_hash = source_content_hash.lower()
        self._analysis_profile = analysis_profile

    def _profiled_fingerprint(self, fingerprint: str) -> str:
        """Keep historical entries isolated when a provider profile changes."""
        return hashlib.sha256(
            f"{self._analysis_profile}\0{fingerprint}".encode("utf-8")
        ).hexdigest()

    def get(self, fingerprint: str) -> list[FilmObservation] | None:
        return self._repository.get_analysis_result(
            self._profiled_fingerprint(fingerprint),
            self._source_content_hash,
            self._analysis_profile,
        )

    def put(self, fingerprint: str, observations: list[FilmObservation]) -> None:
        self._repository.save_analysis_result(
            self._profiled_fingerprint(fingerprint),
            self._source_content_hash,
            self._analysis_profile,
            observations,
        )

    def flush(self) -> None:
        """Repository-backed entries are durable at each ``put``."""
