"""02 导演脑 REST API 层。

方案 §6 的 7 个端点（correlation_id 信封 + idempotency_key 幂等）。
用 FastAPI 实现，通过 ``uvicorn api:app`` 启动。

红线：接口不允许自然语言直接触发时间线写操作。
"""
