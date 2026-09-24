"""Arsenal 资产索引持久化。

基于标准库 sqlite3 的镜头 / 媒体索引，支持 upsert、按哈希查询、
分析状态检查，解决"每次运行都重新分析"的问题。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    """当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _to_int(value: Any) -> int | None:
    """bool / int → int，None 保持 None。"""
    if value is None:
        return None
    return int(bool(value))


def _to_bool(value: Any) -> bool | None:
    """int (0/1) → bool，None 保持 None。"""
    if value is None:
        return None
    return bool(value)


class AssetIndex:
    """Arsenal 资产索引：媒体记录 + 镜头记录的 SQLite 持久化。"""

    def __init__(self, db_path: str) -> None:
        """打开（或创建）索引数据库。

        Args:
            db_path: SQLite 文件路径，``":memory:"`` 使用内存库。
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        # 内存库不支持 WAL
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ------------------------------------------------------------------ schema

    def _create_tables(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS media (
            media_hash   TEXT PRIMARY KEY,
            path         TEXT,
            duration_us  INTEGER,
            width        INTEGER,
            height       INTEGER,
            fps          REAL,
            audio_present INTEGER,
            file_size    INTEGER,
            imported_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS shots (
            shot_id           TEXT PRIMARY KEY,
            media_hash        TEXT,
            start_us          INTEGER,
            end_us            INTEGER,
            duration_us       INTEGER,
            blur_score        REAL,
            exposure_ok       INTEGER,
            technical_usable  INTEGER,
            vlm_shot_function TEXT,
            vlm_role          TEXT,
            frame_description TEXT,
            analyzed_at       TEXT,
            FOREIGN KEY (media_hash) REFERENCES media(media_hash)
        );
        """
        self._conn.executescript(ddl)
        self._conn.commit()

    # ------------------------------------------------------------------ media

    def upsert_media(self, media: dict) -> None:
        """插入或更新一条媒体记录。

        Args:
            media: 含 ``media_hash`` / ``path`` / ``duration_us`` /
                ``width`` / ``height`` / ``fps`` / ``audio_present`` /
                ``file_size`` / ``imported_at`` 的 dict。
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO media "
            "(media_hash, path, duration_us, width, height, fps, "
            " audio_present, file_size, imported_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                media.get("media_hash"),
                media.get("path"),
                media.get("duration_us"),
                media.get("width"),
                media.get("height"),
                media.get("fps"),
                _to_int(media.get("audio_present")),
                media.get("file_size"),
                media.get("imported_at") or _now_iso(),
            ),
        )
        self._conn.commit()

    def list_media(self) -> list[dict]:
        """列出所有已导入的媒体记录。"""
        cur = self._conn.execute("SELECT * FROM media ORDER BY imported_at")
        return [self._media_row_to_dict(row) for row in cur.fetchall()]

    @staticmethod
    def _media_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "media_hash": row["media_hash"],
            "path": row["path"],
            "duration_us": row["duration_us"],
            "width": row["width"],
            "height": row["height"],
            "fps": row["fps"],
            "audio_present": _to_bool(row["audio_present"]),
            "file_size": row["file_size"],
            "imported_at": row["imported_at"],
        }

    # ------------------------------------------------------------------ shots

    def upsert_shot(self, shot: dict) -> None:
        """插入或更新一条镜头记录。

        Args:
            shot: 含 ``shot_id`` / ``media_hash`` / ``start_us`` /
                ``end_us`` / ``duration_us`` / ``blur_score`` /
                ``exposure_ok`` / ``technical_usable`` /
                ``vlm_shot_function`` / ``vlm_role`` /
                ``frame_description`` / ``analyzed_at`` 的 dict。
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO shots "
            "(shot_id, media_hash, start_us, end_us, duration_us, "
            " blur_score, exposure_ok, technical_usable, "
            " vlm_shot_function, vlm_role, frame_description, analyzed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                shot.get("shot_id"),
                shot.get("media_hash"),
                shot.get("start_us"),
                shot.get("end_us"),
                shot.get("duration_us"),
                shot.get("blur_score"),
                _to_int(shot.get("exposure_ok")),
                _to_int(shot.get("technical_usable")),
                shot.get("vlm_shot_function"),
                shot.get("vlm_role"),
                shot.get("frame_description"),
                shot.get("analyzed_at") or _now_iso(),
            ),
        )
        self._conn.commit()

    def get_shots(self, media_hash: str | None = None) -> list[dict]:
        """查询镜头列表。

        Args:
            media_hash: 为 None 时返回全部镜头；否则按媒体哈希过滤。

        Returns:
            镜头 dict 列表。
        """
        if media_hash is not None:
            cur = self._conn.execute(
                "SELECT * FROM shots WHERE media_hash = ? ORDER BY start_us",
                (media_hash,),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM shots ORDER BY media_hash, start_us"
            )
        return [self._shot_row_to_dict(row) for row in cur.fetchall()]

    def get_shot(self, shot_id: str) -> dict | None:
        """按 shot_id 查询单条镜头，不存在返回 None。"""
        cur = self._conn.execute(
            "SELECT * FROM shots WHERE shot_id = ?", (shot_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._shot_row_to_dict(row)

    @staticmethod
    def _shot_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "shot_id": row["shot_id"],
            "media_hash": row["media_hash"],
            "start_us": row["start_us"],
            "end_us": row["end_us"],
            "duration_us": row["duration_us"],
            "blur_score": row["blur_score"],
            "exposure_ok": _to_bool(row["exposure_ok"]),
            "technical_usable": _to_bool(row["technical_usable"]),
            "vlm_shot_function": row["vlm_shot_function"],
            "vlm_role": row["vlm_role"],
            "frame_description": row["frame_description"],
            "analyzed_at": row["analyzed_at"],
        }

    # ------------------------------------------------------------------ status

    def is_analyzed(self, media_hash: str) -> bool:
        """检查该媒体是否已有至少一条镜头索引。"""
        cur = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM shots WHERE media_hash = ?",
            (media_hash,),
        )
        row = cur.fetchone()
        return row["cnt"] > 0

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
