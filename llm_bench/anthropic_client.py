"""Anthropic Messages API 客户端

与 StreamClient 共享 complete() 接口，解析 Anthropic SSE 事件格式。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from llm_bench.config import apply_default_extra_body
from llm_bench.metrics import ChunkTimelineEvent, RequestResult, iso_now, itl_stats
from llm_bench.prompts import estimate_tokens
from llm_bench.sse import (
    DONE,
    ERR_LIMIT,
    ITL_CAP,
    parse_sse_line,
)

LOG = logging.getLogger("llm_bench.anthropic_client")


class AnthropicStreamClient:
    """Anthropic Messages API 客户端"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        concurrency: int,
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
        stream: bool = True,
        capture_chunk_timeline: bool = False,
        first_exchange_sink: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.concurrency = concurrency
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 深度思考开关（与 OpenAI 客户端共用同一个 extra_body.enable_thinking 入参）
        self.thinking_enabled = bool(
            apply_default_extra_body(extra_body).get("enable_thinking", False)
        )
        self.stream = stream
        self.capture_chunk_timeline = capture_chunk_timeline
        # 首个请求留证回调（Web UI 校验请求参数/返回内容），CLI 场景为 None
        self.first_exchange_sink = first_exchange_sink
        self._first_exchange_taken = False
        # 兼容 runner.run_level() 对这些属性的访问（Anthropic 协议不使用这些选项，始终为 False）
        self.stream_options_rejected = False
        self.enable_thinking_rejected = False
        limits = httpx.Limits(
            max_connections=concurrency + 8,
            max_keepalive_connections=concurrency + 8,
            keepalive_expiry=30.0,
        )
        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            headers={
                # 鉴权双发：原生 Anthropic 认 x-api-key，走 Bearer 网关时认 Authorization
                "x-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AnthropicStreamClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _try_take_first_exchange(self) -> bool:
        """原子性地抢占首请求留证权（在单个 await 前同步调用）

        asyncio 单线程模型下，两个 await 之间的代码是原子的。
        本方法在 complete() 的首个 await 前同步调用，保证并发 worker 中
        只有第一个调用者获得留证权。
        """
        if self.first_exchange_sink is None or self._first_exchange_taken:
            return False
        self._first_exchange_taken = True
        return True

    def reset_first_exchange(self) -> None:
        """重置首请求留证标志（供连续压测场景复用客户端）"""
        self._first_exchange_taken = False

    def _body(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Anthropic 原生约束：开启 extended thinking 时 temperature 必须缺省（=1），
        # 且 thinking.budget_tokens 需满足 1024 ≤ budget < max_tokens
        if self.thinking_enabled and self.max_tokens > 1:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(1024, self.max_tokens - 1),
            }
        elif self.temperature is not None:
            body["temperature"] = self.temperature
        body["stream"] = self.stream
        return body

    async def complete(
        self,
        *,
        prompt: str,
        request_id: str,
        phase: str,
        provider: str,
        concurrency: int,
    ) -> RequestResult:
        url = f"{self.base_url}/v1/messages"
        start_perf = time.perf_counter()
        start_iso = iso_now()
        rec = RequestResult(
            request_id=request_id,
            phase=phase,
            provider=provider,
            model=self.model,
            concurrency=concurrency,
            start_time=start_iso,
            end_time=start_iso,
            start_perf=start_perf,
            end_perf=start_perf,
            streamed=self.stream,
        )
        # 首请求留证：在首个 await 前同步抢占标记，保证并发 worker 中只有第一次调用留证
        capture = self._try_take_first_exchange()
        cap = (
            {"request_body": None, "response_text": None, "response_parts": []}
            if capture
            else None
        )
        try:
            attempt = self._stream if self.stream else self._complete_json
            await asyncio.wait_for(attempt(url, prompt, rec, cap), timeout=self.timeout)
        except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException):
            rec.error_type = "timeout"
            rec.error_message = "timeout"
            rec.success = False
        except httpx.ConnectError as e:
            rec.error_type = "connect"
            rec.error_message = str(e)[:ERR_LIMIT]
            rec.success = False
        except asyncio.CancelledError:
            rec.error_type = "cancelled"
            rec.error_message = "cancelled"
            rec.success = False
        except Exception as e:
            rec.error_type = rec.error_type or "unknown"
            rec.error_message = (rec.error_message or f"{type(e).__name__}: {e}")[:ERR_LIMIT]
            rec.success = False
        rec.end_perf = time.perf_counter()
        rec.end_time = iso_now()
        if rec.input_tokens is None:
            rec.input_tokens = estimate_tokens(prompt)
            if rec.token_source is None:
                rec.token_source = "estimated"
        if rec.output_text and (rec.output_tokens or 0) == 0:
            rec.output_tokens = estimate_tokens(rec.output_text)
            rec.token_source = "estimated"
        elif rec.token_source is None:
            rec.output_tokens = estimate_tokens(rec.output_text) if rec.output_text else 0
            rec.token_source = "estimated"
        rec.total_tokens = (rec.input_tokens or 0) + (rec.output_tokens or 0)
        rec.fill_derived()
        if cap is not None and self.first_exchange_sink is not None:
            await self._emit_first_exchange(url, rec, cap)
        return rec

    async def _emit_first_exchange(
        self, url: str, rec: RequestResult, cap: dict[str, Any]
    ) -> None:
        """把本模型首个请求的完整请求/返回 body 上报给留证回调（失败不影响压测）"""
        try:
            if cap.get("response_text") is not None:
                response_text = cap["response_text"]
            else:
                # 流式：拼接原始 SSE 事件行（含 "event:"/"data:" 前缀），尽量还原线上报文
                response_text = "\n".join(cap.get("response_parts") or [])
            await self.first_exchange_sink({
                "provider": rec.provider,
                "model": rec.model,
                "phase": rec.phase,
                "request_id": rec.request_id,
                "url": url,
                "streamed": rec.streamed,
                "http_status": rec.http_status,
                "success": rec.success,
                "error_type": rec.error_type,
                "error_message": rec.error_message,
                "start_time": rec.start_time,
                "end_time": rec.end_time,
                "request_body": cap.get("request_body"),
                "response_text": response_text,
            })
        except Exception:
            LOG.debug("首请求留证上报失败", exc_info=True)

    async def _complete_json(
        self, url: str, prompt: str, rec: RequestResult, cap: dict[str, Any] | None = None
    ) -> None:
        """非流式请求"""
        rec.streamed = False
        body = self._body(prompt)
        if cap is not None:
            cap["request_body"] = body
            cap["response_text"] = None
        resp = await self._client.post(url, json=body)
        rec.http_status = resp.status_code
        if cap is not None:
            cap["response_text"] = resp.text or ""
        if resp.status_code != 200:
            raw = (resp.text or "")[:ERR_LIMIT]
            rec.error_type = "http"
            rec.error_message = raw or f"HTTP {resp.status_code}"
            rec.success = False
            LOG.warning("HTTP %d: %s", resp.status_code, raw)
            return
        try:
            obj = resp.json()
        except (json.JSONDecodeError, ValueError):
            rec.error_type = "api"
            rec.error_message = "HTTP 200 but response is not JSON"
            rec.success = False
            return
        if not isinstance(obj, dict):
            rec.error_type = "api"
            rec.error_message = "HTTP 200 but JSON is not an object"
            rec.success = False
            return
        # 解析 usage
        usage = obj.get("usage") or {}
        rec.input_tokens = usage.get("input_tokens", 0)
        rec.output_tokens = usage.get("output_tokens", 0)
        rec.cache_l1_tokens = usage.get("cache_creation_input_tokens")  # L1: 写入缓存
        rec.cache_l2_tokens = usage.get("cache_read_input_tokens")      # L2: 读取缓存
        if rec.cache_l1_tokens is not None:
            rec.cache_write_tokens = rec.cache_l1_tokens
        if rec.cache_l2_tokens is not None:
            rec.cache_read_tokens = rec.cache_l2_tokens
        # 解析 content
        content_blocks = obj.get("content") or []
        for block in content_blocks:
            if block.get("type") == "text":
                rec.output_text += block.get("text") or ""
        rec.success = True

    async def _stream(
        self, url: str, prompt: str, rec: RequestResult, cap: dict[str, Any] | None = None
    ) -> None:
        """流式请求，解析 Anthropic SSE 格式"""
        body = self._body(prompt)
        if cap is not None:
            cap["request_body"] = body
            cap["response_text"] = None
            cap["response_parts"] = []
        parse_errors = 0
        sse_frames = 0
        chunk_times: list[float] = []
        intervals: list[float] = []
        try:
            async with self._client.stream("POST", url, json=body) as resp:
                rec.http_status = resp.status_code
                if resp.status_code != 200:
                    raw_full = (await resp.aread()).decode("utf-8", errors="replace")
                    if cap is not None:
                        cap["response_text"] = raw_full
                    raw = raw_full[:ERR_LIMIT]
                    rec.error_type = "http"
                    rec.error_message = raw or f"HTTP {resp.status_code}"
                    rec.success = False
                    LOG.warning("HTTP %d: %s", resp.status_code, raw)
                    return
                async for line in resp.aiter_lines():
                    # 首请求留证：保留原始 SSE 行（含 event:/data: 前缀、注释及空行）
                    if cap is not None:
                        cap["response_parts"].append(line)
                    payload = parse_sse_line(line)
                    if payload is None:
                        continue
                    sse_frames += 1
                    if payload == DONE:
                        break
                    try:
                        evt = json.loads(payload)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue
                    if not isinstance(evt, dict):
                        parse_errors += 1
                        continue
                    etype = evt.get("type")
                    if etype == "message_start":
                        # 解析 input usage
                        msg = evt.get("message") or {}
                        u = msg.get("usage") or {}
                        rec.input_tokens = u.get("input_tokens", 0)
                        rec.cache_l1_tokens = u.get("cache_creation_input_tokens")
                        rec.cache_l2_tokens = u.get("cache_read_input_tokens")
                        if rec.cache_l1_tokens is not None:
                            rec.cache_write_tokens = rec.cache_l1_tokens
                        if rec.cache_l2_tokens is not None:
                            rec.cache_read_tokens = rec.cache_l2_tokens
                    elif etype == "content_block_delta":
                        delta = evt.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            text = delta["text"]
                            now = time.perf_counter()
                            if rec.first_token_perf is None:
                                rec.first_token_perf = now
                                rec.first_token_time = iso_now()
                            elif chunk_times:
                                intervals.append(now - chunk_times[-1])
                            chunk_times.append(now)
                            rec.output_text += text
                            if len(intervals) >= ITL_CAP:
                                intervals = intervals[:ITL_CAP]
                    elif etype == "message_delta":
                        # 解析 output usage
                        u = evt.get("usage") or {}
                        rec.output_tokens = u.get("output_tokens", rec.output_tokens or 0)
                    elif etype == "message_stop":
                        break
        except asyncio.CancelledError:
            rec.success = False
            if rec.error_type is None:
                rec.error_type = "cancelled"
                rec.error_message = "cancelled"
            raise

        if rec.error_type:
            rec.success = False
            return
        if sse_frames == 0:
            rec.error_type = "api"
            rec.error_message = "HTTP 200 but no SSE data frames"
            rec.success = False
            return
        if parse_errors and not rec.output_text:
            rec.error_type = "api"
            rec.error_message = f"sse parse failed ({parse_errors} chunks), no content"
            rec.success = False
            return
        rec.mean_itl_ms, rec.itl_p50_ms, rec.itl_p95_ms, rec.itl_count = itl_stats(intervals)
        rec.success = rec.http_status == 200
