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
    CHUNK_PREVIEW_LEN,
    CHUNK_TIMELINE_CAP,
    DONE,
    ERR_LIMIT,
    ITL_CAP,
    SSE_META_CAP,
    parse_sse_line,
)
from llm_bench.usage import usage_from_obj

LOG = logging.getLogger("llm_bench.client")

_UNKNOWN_FIELD_NEEDLES: tuple[str, ...] = (
    "enable_thinking",
    "additional properties",
    "unknown parameter",
    "unrecognized request argument",
    "unexpected keyword",
    "extra fields",
    "extra field",
)


def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if "text" in p:
                    parts.append(str(p.get("text") or ""))
                elif "content" in p:
                    parts.append(extract_text(p["content"]))
        return "".join(parts)
    return str(content)


def choice0_dict(obj: dict[str, Any]) -> dict[str, Any] | None:
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    c0 = choices[0]
    return c0 if isinstance(c0, dict) else None


def delta_content(obj: dict[str, Any]) -> str:
    choice0 = choice0_dict(obj)
    if choice0 is None:
        return ""
    if "delta" in choice0:
        delta = choice0.get("delta")
        if not isinstance(delta, dict):
            return ""
        return extract_text(delta.get("content"))
    message = choice0.get("message")
    if not isinstance(message, dict):
        return ""
    return extract_text(message.get("content"))


def finish_reason(obj: dict[str, Any]) -> str | None:
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    c0 = choices[0]
    if not isinstance(c0, dict):
        return None
    fr = c0.get("finish_reason")
    return str(fr) if fr else None


def _apply_usage(rec: RequestResult, usage: dict[str, int] | None) -> None:
    if not usage:
        return
    if usage["input_tokens"] > 0:
        rec.input_tokens = usage["input_tokens"]
    if usage["output_tokens"] > 0 or rec.output_text:
        rec.output_tokens = usage["output_tokens"]
        rec.total_tokens = usage["total_tokens"] or (
            (rec.input_tokens or 0) + usage["output_tokens"]
        )
        rec.token_source = "provider"
    if "cache_read_tokens" in usage:
        rec.cache_read_tokens = usage["cache_read_tokens"]
        rec.cache_l2_tokens = usage["cache_read_tokens"]  # F5: L2 = cache_read
    if "cache_write_tokens" in usage:
        rec.cache_write_tokens = usage["cache_write_tokens"]
        rec.cache_l1_tokens = usage["cache_write_tokens"]  # F5: L1 = cache_write


def api_error_message(obj: dict[str, Any]) -> str | None:
    err = obj.get("error")
    if err is None:
        return None
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err)
    return str(err)


def looks_like_unknown_field(message: str) -> bool:
    lower = message.lower()
    return any(needle in lower for needle in _UNKNOWN_FIELD_NEEDLES)


def create_client(
    *,
    api_style: str = "openai",
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
) -> "StreamClient | Any":
    """F3: 工厂函数 — 根据 api_style 选择客户端实现

    first_exchange_sink: 可选异步回调 sink(exchange)，传入后仅留证本客户端
    （即单个 provider/model）的首个请求完整请求/返回 body，供 Web 实时校验。
    """
    if api_style == "anthropic":
        from llm_bench.anthropic_client import AnthropicStreamClient
        return AnthropicStreamClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            concurrency=concurrency,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            stream=stream,
            capture_chunk_timeline=capture_chunk_timeline,
            first_exchange_sink=first_exchange_sink,
        )
    return StreamClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        concurrency=concurrency,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        stream=stream,
        capture_chunk_timeline=capture_chunk_timeline,
        first_exchange_sink=first_exchange_sink,
    )


class StreamClient:
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
        self.stream = stream
        self.capture_chunk_timeline = capture_chunk_timeline
        # 首个请求留证回调（Web UI 校验请求参数/返回内容），CLI 场景为 None
        self.first_exchange_sink = first_exchange_sink
        self._first_exchange_taken = False
        self.extra_body = apply_default_extra_body(extra_body)
        self.include_usage = True
        self.stream_options_rejected = False
        self.enable_thinking_rejected = False
        limits = httpx.Limits(
            max_connections=concurrency + 8,
            max_keepalive_connections=concurrency + 8,
            keepalive_expiry=30.0,
        )
        accept = "text/event-stream" if stream else "application/json"
        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            headers={
                # 鉴权双发：OpenAI 系认 Authorization: Bearer，部分网关/Anthropic 系认 X-Api-Key，
                # 多余的头通常被服务端忽略，避免用户因鉴权头不同而额外配置
                "Authorization": f"Bearer {api_key}",
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": accept,
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> StreamClient:
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
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        extra = apply_default_extra_body(self.extra_body)
        if extra:
            body.update(extra)
        # 深度思考开关双发两套方言，服务端各取所需：
        #   Qwen/百炼/部分 vLLM → enable_thinking (bool)
        #   GLM/Kimi/豆包        → thinking.type (enabled/disabled)
        # extra_body 中显式给出 thinking 时不覆盖，保留用户自定义
        if "thinking" not in body:
            body["thinking"] = {
                "type": "enabled" if body.get("enable_thinking") else "disabled"
            }
        body["stream"] = self.stream
        if not self.stream or not self.include_usage:
            body.pop("stream_options", None)
        elif "stream_options" not in body:
            body["stream_options"] = {"include_usage": True}
        if self.enable_thinking_rejected:
            # 两套方言同时剥离，一次重试到位
            body.pop("enable_thinking", None)
            body.pop("thinking", None)
        return body

    # 422：部分 FastAPI/pydantic 网关对多余字段返回 422 而非 400
    _BAD_REQUEST_STATUSES = (400, 422)

    def _redacted_headers(self) -> dict[str, str]:
        """调试日志用：脱敏鉴权头，避免密钥明文落盘"""
        return {
            k: ("***" if k.lower() in ("authorization", "x-api-key") else v)
            for k, v in self._client.headers.items()
        }

    def _maybe_disable_stream_options(self, status: int, message: str) -> None:
        if status in self._BAD_REQUEST_STATUSES and "stream_options" in message.lower():
            self.include_usage = False
            self.stream_options_rejected = True

    def _maybe_disable_enable_thinking(
        self, status: int, message: str, *, sent_thinking_params: bool
    ) -> None:
        # 任一思考参数字段被服务端拒绝（未知字段），下次重试两套方言一起剥离
        if (
            status in self._BAD_REQUEST_STATUSES
            and sent_thinking_params
            and looks_like_unknown_field(message)
        ):
            self.enable_thinking_rejected = True

    def _timeline_state(self) -> dict[str, Any]:
        return {
            "seq": 0,
            "last_perf": None,
            "cum_chars": 0,
            "sse_meta_count": 0,
            "cap_noted": False,
        }

    def _append_chunk_event(
        self,
        rec: RequestResult,
        state: dict[str, Any],
        *,
        event_type: str,
        chars: int = 0,
        text: str = "",
        sse_frame: int = 0,
    ) -> None:
        if not self.capture_chunk_timeline:
            return
        if state.get("cap_noted"):
            return
        if len(rec.chunk_timeline) >= CHUNK_TIMELINE_CAP:
            state["cap_noted"] = True
            now = time.perf_counter()
            offset_ms = (now - rec.start_perf) * 1000.0
            delta_ms = None
            if state["last_perf"] is not None:
                delta_ms = (now - state["last_perf"]) * 1000.0
            rec.chunk_timeline.append(
                ChunkTimelineEvent(
                    seq=state["seq"],
                    event_type="timeline_cap",
                    offset_ms=offset_ms,
                    delta_ms=delta_ms,
                    chars=0,
                    cumulative_chars=state["cum_chars"],
                    sse_frame=sse_frame,
                    text_preview=f"cap={CHUNK_TIMELINE_CAP}",
                )
            )
            state["seq"] += 1
            return
        now = time.perf_counter()
        offset_ms = (now - rec.start_perf) * 1000.0
        delta_ms = None
        if state["last_perf"] is not None:
            delta_ms = (now - state["last_perf"]) * 1000.0
        state["last_perf"] = now
        state["cum_chars"] += chars
        preview = text[:CHUNK_PREVIEW_LEN].replace("\n", "\\n").replace("\r", "\\r")
        rec.chunk_timeline.append(
            ChunkTimelineEvent(
                seq=state["seq"],
                event_type=event_type,
                offset_ms=offset_ms,
                delta_ms=delta_ms,
                chars=chars,
                cumulative_chars=state["cum_chars"],
                sse_frame=sse_frame,
                text_preview=preview,
            )
        )
        state["seq"] += 1

    async def complete(
        self,
        *,
        prompt: str,
        request_id: str,
        phase: str,
        provider: str,
        concurrency: int,
    ) -> RequestResult:
        url = f"{self.base_url}/v1/chat/completions"
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
            so_before = self.stream_options_rejected
            et_before = self.enable_thinking_rejected
            attempt = self._stream if self.stream else self._complete_json
            await asyncio.wait_for(attempt(url, prompt, rec, cap), timeout=self.timeout)
            retry_so = self.stream and self.stream_options_rejected and not so_before
            retry_et = self.enable_thinking_rejected and not et_before
            if rec.http_status in self._BAD_REQUEST_STATUSES and (retry_so or retry_et):
                self._reset_attempt(rec)
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
        # 供应商已上报 total_tokens 时保留其口径；估算场景才用 输入+输出 兜底
        if rec.token_source != "provider" or rec.total_tokens is None:
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
                # 流式：拼接原始 SSE 帧行（含 "data:" 前缀），尽量还原线上报文
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

    def _reset_attempt(self, rec: RequestResult) -> None:
        rec.error_type = None
        rec.error_message = None
        rec.http_status = None
        rec.success = False
        rec.output_text = ""
        rec.first_token_perf = None
        rec.first_token_time = None
        rec.input_tokens = None
        rec.output_tokens = None
        rec.total_tokens = None
        rec.token_source = None
        rec.cache_read_tokens = None
        rec.cache_write_tokens = None
        rec.mean_itl_ms = None
        rec.itl_p50_ms = None
        rec.itl_p95_ms = None
        rec.itl_count = 0
        rec.chunk_timeline = []
        rec.start_perf = time.perf_counter()
        rec.start_time = iso_now()

    async def _complete_json(
        self, url: str, prompt: str, rec: RequestResult, cap: dict[str, Any] | None = None
    ) -> None:
        rec.streamed = False
        body = self._body(prompt)
        if cap is not None:
            # 重试会重新进入本方法：以最后一次实际发送的 body / 原始返回为准
            cap["request_body"] = body
            cap["response_text"] = None
        timeline = self._timeline_state()
        resp = await self._client.post(url, json=body)
        rec.http_status = resp.status_code
        if cap is not None:
            cap["response_text"] = resp.text or ""
        if resp.status_code == 200:
            self._append_chunk_event(rec, timeline, event_type="http_200")
        raw = (resp.text or "")[:ERR_LIMIT]
        if resp.status_code != 200:
            rec.error_type = "http"
            rec.error_message = raw or f"HTTP {resp.status_code}"
            rec.success = False
            # 打印请求参数和 API Key 用于调试
            LOG.warning("HTTP %d request debug: url=%s api_key=%s headers=%s body_keys=%s response_headers=%s",
                        resp.status_code, url,
                        (self.api_key[:6] + "***") if self.api_key else "(empty)",
                        self._redacted_headers(), list(body.keys()), dict(resp.headers))
            self._maybe_disable_enable_thinking(
                resp.status_code,
                rec.error_message,
                sent_enable_thinking="enable_thinking" in body,
            )
            return
        try:
            obj = json.loads(resp.text) if resp.text else None
        except json.JSONDecodeError:
            rec.error_type = "api"
            rec.error_message = "HTTP 200 but response is not JSON"
            rec.success = False
            return
        if not isinstance(obj, dict):
            rec.error_type = "api"
            rec.error_message = "HTTP 200 but JSON is not an object"
            rec.success = False
            return
        err = api_error_message(obj)
        if err:
            rec.error_type = "api"
            rec.error_message = err[:ERR_LIMIT]
            rec.success = False
            return
        usage = usage_from_obj(obj)
        _apply_usage(rec, usage)
        rec.output_text = delta_content(obj)
        if rec.output_text:
            self._append_chunk_event(
                rec,
                timeline,
                event_type="json_body",
                chars=len(rec.output_text),
                text=rec.output_text,
            )
        if rec.output_text or usage is not None:
            self._append_chunk_event(rec, timeline, event_type="stream_end")
            rec.success = True
            return
        rec.error_type = "api"
        rec.error_message = "HTTP 200 but no content or usage"
        rec.success = False

    async def _stream(
        self, url: str, prompt: str, rec: RequestResult, cap: dict[str, Any] | None = None
    ) -> None:
        body = self._body(prompt)
        if cap is not None:
            cap["request_body"] = body
            cap["response_text"] = None
            cap["response_parts"] = []
        parse_errors = 0
        sse_frames = 0
        saw_done = False
        saw_finish = False
        chunk_times: list[float] = []
        intervals: list[float] = []
        timeline = self._timeline_state()
        try:
            async with self._client.stream("POST", url, json=body) as resp:
                rec.http_status = resp.status_code
                if resp.status_code == 200:
                    self._append_chunk_event(rec, timeline, event_type="http_200")
                if resp.status_code != 200:
                    raw_full = (await resp.aread()).decode("utf-8", errors="replace")
                    if cap is not None:
                        cap["response_text"] = raw_full
                    raw = raw_full[:ERR_LIMIT]
                    rec.error_type = "http"
                    rec.error_message = raw or f"HTTP {resp.status_code}"
                    rec.success = False
                    # 打印请求参数和 API Key 用于调试
                    LOG.warning("HTTP %d request debug: url=%s api_key=%s headers=%s body_keys=%s response_headers=%s",
                                resp.status_code, url,
                                (self.api_key[:6] + "***") if self.api_key else "(empty)",
                                self._redacted_headers(), list(body.keys()), dict(resp.headers))
                    self._maybe_disable_stream_options(resp.status_code, rec.error_message)
                    self._maybe_disable_enable_thinking(
                        resp.status_code,
                        rec.error_message,
                        sent_thinking_params=("enable_thinking" in body or "thinking" in body),
                    )
                    return
                async for line in resp.aiter_lines():
                    # 首请求留证：保留原始 SSE 帧行（含 data: 前缀/注释/空行）
                    if cap is not None:
                        cap["response_parts"].append(line)
                    payload = parse_sse_line(line)
                    if payload is None:
                        continue
                    sse_frames += 1
                    if payload == DONE:
                        saw_done = True
                        self._append_chunk_event(
                            rec, timeline, event_type="done", sse_frame=sse_frames
                        )
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue
                    if not isinstance(obj, dict):
                        parse_errors += 1
                        continue
                    err = api_error_message(obj)
                    if err:
                        rec.error_type = "api"
                        rec.error_message = err[:ERR_LIMIT]
                        rec.success = False
                        return
                    usage = usage_from_obj(obj)
                    if usage and (usage["output_tokens"] > 0 or finish_reason(obj)):
                        _apply_usage(rec, usage)
                    elif usage and usage["input_tokens"] > 0:
                        _apply_usage(rec, usage)
                    if finish_reason(obj):
                        saw_finish = True
                        self._append_chunk_event(
                            rec, timeline, event_type="finish", sse_frame=sse_frames
                        )
                    text = delta_content(obj)
                    c0 = choice0_dict(obj)
                    is_delta_frame = c0 is not None and "delta" in c0
                    if c0 is not None and not is_delta_frame and rec.output_text:
                        # 非流式风格的整体 message 帧，避免把完整正文重复累计
                        text = ""
                    # 思考链（reasoning_content）也是模型实时吐出的 token：
                    # 只思考模型可能在 max_tokens 内正文为空，但 TTFT 仍应可测
                    reasoning = ""
                    if is_delta_frame:
                        delta = c0.get("delta")
                        if isinstance(delta, dict):
                            reasoning = extract_text(delta.get("reasoning_content"))
                    # 计时按"模型生成的任意文本"（正文优先，其次思考链）
                    generated = text or reasoning
                    if not generated:
                        if (
                            self.capture_chunk_timeline
                            and timeline["sse_meta_count"] < SSE_META_CAP
                        ):
                            self._append_chunk_event(
                                rec,
                                timeline,
                                event_type="sse_meta",
                                sse_frame=sse_frames,
                            )
                            timeline["sse_meta_count"] += 1
                        continue
                    now = time.perf_counter()
                    if rec.first_token_perf is None:
                        rec.first_token_perf = now
                        rec.first_token_time = iso_now()
                    elif chunk_times:
                        intervals.append(now - chunk_times[-1])
                    chunk_times.append(now)
                    self._append_chunk_event(
                        rec,
                        timeline,
                        event_type="content" if text else "reasoning",
                        chars=len(generated),
                        text=generated,
                        sse_frame=sse_frames,
                    )
                    # output_text 只保留可见正文，思考链不计入
                    rec.output_text += text
                    if len(intervals) >= ITL_CAP:
                        intervals = intervals[:ITL_CAP]
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
        if not saw_done and not saw_finish:
            rec.error_type = "truncated"
            rec.error_message = "stream ended without [DONE] or finish_reason"
            rec.success = False
            return
        if parse_errors and not rec.output_text:
            rec.error_type = "api"
            rec.error_message = f"sse parse failed ({parse_errors} chunks), no content"
            rec.success = False
            return
        rec.mean_itl_ms, rec.itl_p50_ms, rec.itl_p95_ms, rec.itl_count = itl_stats(intervals)
        self._append_chunk_event(rec, timeline, event_type="stream_end")
        rec.success = rec.http_status == 200
