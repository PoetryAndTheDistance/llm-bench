"""WebSocket 管理器 — 实时推送压测指标"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

LOG = logging.getLogger("llm_bench.web.websocket")

# 广播循环连续异常次数上限；超过后停止循环，避免日志刷屏且无法恢复
_BROADCAST_MAX_CONSECUTIVE_ERRORS = 10


class WebSocketManager:
    """管理所有 WebSocket 连接，定期广播指标快照"""

    def __init__(self, broadcaster: Any, interval: float = 1.0):
        self.broadcaster = broadcaster
        self.interval = interval
        # 使用 dict[ws, None] 代替 set 以保留插入顺序（便于调试），同时 O(1) 增删
        self._connections: dict[Any, None] = {}
        self._task: asyncio.Task | None = None
        self._running = False
        # 已推送的历史点数量：历史全程保留，每帧只发新增点；
        # 新任务 reset 后历史变短，索引回绕为全量补发
        self._hist_sent = 0

    async def connect(self, ws: Any) -> None:
        await ws.accept()
        self._connections[ws] = None
        LOG.info("WebSocket 连接建立，当前连接数: %d", len(self._connections))

    async def disconnect(self, ws: Any) -> None:
        self._connections.pop(ws, None)
        LOG.info("WebSocket 连接断开，当前连接数: %d", len(self._connections))

    async def broadcast(self, data: dict[str, Any]) -> None:
        """向所有连接广播数据"""
        if not self._connections:
            return
        message = json.dumps(data, ensure_ascii=False, default=str)
        dead: list[Any] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.pop(ws, None)

    async def _broadcast_loop(self) -> None:
        """定时广播循环

        包含连续异常计数器：连续失败超过阈值后主动停止，
        避免异常持续刷屏日志且无法恢复。
        """
        consecutive_errors = 0
        while self._running:
            try:
                # 一次聚合同时生成快照与时序点，summary/token_stats 同源复用
                snapshot = await self.broadcaster.snapshot_with_history()
                history = await self.broadcaster.history()
                first_exchanges = await self.broadcaster.first_exchange_index()
                # 每秒最多新增 1 个点；长度变短说明任务已 reset，从头补发
                if self._hist_sent > len(history):
                    self._hist_sent = 0
                history_delta = history[self._hist_sent:]
                self._hist_sent = len(history)
                data = {
                    "type": "metrics",
                    "summary": snapshot,
                    "token_stats": snapshot,
                    "status": self.broadcaster.status,
                    "job": self.broadcaster.job_snapshot(),
                    # 首请求留证轻量索引（不含 body）；完整 body 经 HTTP 按需拉取
                    "first_exchanges": first_exchanges,
                    # 趋势历史增量（全量经 GET /api/job/history 回填）
                    "history_delta": history_delta,
                }
                await self.broadcast(data)
                consecutive_errors = 0  # 成功后重置计数器
            except Exception as e:
                consecutive_errors += 1
                LOG.error("广播异常 (%d/%d): %s",
                          consecutive_errors, _BROADCAST_MAX_CONSECUTIVE_ERRORS, e)
                if consecutive_errors >= _BROADCAST_MAX_CONSECUTIVE_ERRORS:
                    LOG.error("广播连续异常达上限，停止广播循环")
                    self._running = False
                    break
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._broadcast_loop())
            LOG.info("WebSocket 广播启动，间隔 %.1f 秒", self.interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        LOG.info("WebSocket 广播停止")

    @property
    def connection_count(self) -> int:
        return len(self._connections)
