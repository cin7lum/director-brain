"""所有 Director Brain 顶层记录模型的抽象基类。

集中声明跨记录通用的元数据字段，并通过 ``extra="forbid"`` 禁止未定义字段。
时间以整数帧表示时由各记录自带的 ``timebase`` 字段说明帧率；本基类的
``created_at`` 为记录创建时刻的 Unix 时间戳（秒）。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseRecord(BaseModel):
    """顶层业务记录的公共字段基类。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_id: str
    created_at: int
    producer: str
    source_ref: str
