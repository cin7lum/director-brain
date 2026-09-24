"""本地文件系统对象存储，模拟 S3 语义。

以 ``base_path`` 为桶根，``key`` 为相对对象键。写入即落盘，内容寻址用
sha256。禁止 ``..`` 等路径穿越。
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class LocalObjectStore:
    """基于本地目录的对象存储。"""

    def __init__(self, base_path: str) -> None:
        self._root = Path(base_path).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ path safety
    def _resolve(self, key: str) -> Path:
        if not key or key.startswith("/") or key.startswith("\\"):
            raise ValueError(f"非法对象键: {key!r}")
        parts = [p for p in Path(key).parts if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise ValueError(f"对象键包含路径穿越: {key!r}")
        return self._root.joinpath(*parts)

    # ------------------------------------------------------------------ S3-ish
    def put(self, key: str, data: bytes) -> str:
        """写入对象，返回内容的 sha256 十六进制摘要。"""
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def get(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def delete(self, key: str) -> bool:
        target = self._resolve(key)
        if not target.exists():
            return False
        target.unlink()
        return True

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self._resolve(prefix) if prefix else self._root
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        keys: list[str] = []
        for path in sorted(base.rglob("*")):
            if path.is_file():
                keys.append(path.relative_to(self._root).as_posix())
        return keys
