from __future__ import annotations

import asyncio
import hashlib
import logging
import platform
import re
import signal
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from llm_bench import __version__
from llm_bench.client import StreamClient
from llm_bench.config import RunConfig
from llm_bench.metrics import RequestResult, iso_now, timeseries_rows
from llm_bench.prompts import build_prompt, estimate_tokens, prompt_profile_hints
from llm_bench.report import (
    build_run_artifacts,
    format_minute_row,
    write_json,
)

LOG = logging.getLogger("llm_bench")


def _uncancel_current_task() -> None:
    task = asyncio.current_task()
    if task is not None:
        while task.cancelling():
            task.uncancel()


def sanitize_fs(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return s or "model"


def sanitize_fs_readable(name: str) -> str:
    """任务组/批次目录内的可读命名：保留中文等非 ASCII 字符，仅替换文件系统非法字符"""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    s = re.sub(r"\s+", "_", s).strip(" ._")
    return s or "unknown"


async def public_ip() -> str:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("https://api.ipify.org")
            if r.status_code == 200:
                return r.text.strip()[:64] or "unknown"
    except Exception:
        pass
    return "unknown"


def _logger_id_for_path(path: Path) -> str:
    raw = str(path.resolve())
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _close_logger_handlers(log: logging.Logger) -> None:
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()


def setup_file_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    ident = _logger_id_for_path(path)
    log = logging.getLogger(f"llm_bench.run.{ident}")
    log.setLevel(logging.INFO)
    log.propagate = False
    _close_logger_handlers(log)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


class _Budget:
    def __init__(self, n: int | None) -> None:
        self.n = n
        self.i = 0
        self.lock = asyncio.Lock()

    async def take(self) -> int | None:
        async with self.lock:
            if self.n is not None and self.i >= self.n:
                return None
            self.i += 1
            return self.i

    async def untake(self) -> None:
        async with self.lock:
            if self.i > 0:
                self.i -= 1


async def _one(
    client: StreamClient,
    *,
    prompt: str,
    request_id: str,
    phase: str,
    provider: str,
    concurrency: int,
    retry: int,
    log: logging.Logger,
) -> RequestResult:
    last: RequestResult | None = None
    attempts = 1 + max(0, retry)
    for a in range(attempts):
        rec = await client.complete(
            prompt=prompt,
            request_id=request_id if a == 0 else f"{request_id}-r{a}",
            phase=phase,
            provider=provider,
            concurrency=concurrency,
        )
        last = rec
        if rec.success or rec.error_type == "cancelled":
            return rec
        if a + 1 < attempts:
            log.warning("retry %s error=%s status=%s message=%s", request_id, rec.error_type, rec.http_status, rec.error_message)
    assert last is not None
    return last


async def _run_workers(
    *,
    client: StreamClient,
    prompt: str,
    phase: str,
    provider: str,
    concurrency: int,
    n_workers: int,
    budget: _Budget,
    stop: asyncio.Event,
    retry: int,
    records: list[RequestResult],
    lock: asyncio.Lock,
    printed_minutes: set[str],
    log: logging.Logger,
    wait_time: float = 0.0,  # F9
    broadcaster: Any = None,  # F1: Web UI
) -> None:
    # 3.2: 记录上次检查分钟打印时的分钟值，避免每次请求都拷贝列表
    last_minute_check = ""

    async def worker() -> None:
        nonlocal last_minute_check
        while not stop.is_set():
            idx = await budget.take()
            if idx is None:
                return
            if stop.is_set():
                await budget.untake()
                return
            rec = await _one(
                client,
                prompt=prompt,
                request_id=f"{phase}-{idx:06d}",
                phase=phase,
                provider=provider,
                concurrency=concurrency,
                retry=retry,
                log=log,
            )
            async with lock:
                records.append(rec)
            # 3.2: 分钟级统计打印：仅在分钟变化时拷贝列表，避免每次请求都全量拷贝
            now_minute = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
            if now_minute != last_minute_check:
                last_minute_check = now_minute
                async with lock:
                    snap = list(records)
                _maybe_print_closed_minutes(snap, printed_minutes, log)
            # 通知 Web UI：所有阶段都记录（指标只统计 run，预热阶段用于展示任务进度）
            if broadcaster:
                await broadcaster.record(rec)
            # F9: worker 请求间隔
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
    await asyncio.gather(*workers, return_exceptions=True)


def _maybe_print_closed_minutes(
    records: list[RequestResult], printed: set[str], log: logging.Logger
) -> None:
    now_key = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    rows = {r["minute"]: r for r in timeseries_rows(records)}
    for minute, row in rows.items():
        if minute < now_key and minute not in printed:
            printed.add(minute)
            line = format_minute_row(row)
            print(line)
            log.info(line)


async def run_level(
    cfg: RunConfig,
    *,
    concurrency: int,
    duration: float | None,
    requests: int | None,
    out_dir: Path,
    machine: dict[str, str],
    install_signals: bool = True,
    stop_external: asyncio.Event | None = None,
    broadcaster: Any = None,  # F1: Web UI
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_file_logger(out_dir / "benchmark.log")
    # 自定义提示词
    if cfg.prompt == "custom" and cfg.custom_prompt:
        prompt = cfg.custom_prompt
    else:
        prompt = build_prompt(cfg.prompt, cfg.max_tokens)
    records: list[RequestResult] = []
    lock = asyncio.Lock()
    stop = asyncio.Event()
    printed: set[str] = set()
    interrupted = False
    watcher: asyncio.Task[None] | None = None

    def on_sigint() -> None:
        nonlocal interrupted
        if interrupted:
            log.warning("second SIGINT: cancelling in-flight")
            stop.set()
            for t in asyncio.all_tasks():
                if t is not asyncio.current_task():
                    t.cancel()
            return
        interrupted = True
        log.warning("SIGINT: stopping new requests")
        print("SIGINT: stopping new requests, waiting in-flight...", file=sys.stderr)
        stop.set()

    loop = asyncio.get_running_loop()
    if install_signals:
        try:
            loop.add_signal_handler(signal.SIGINT, on_sigint)
        except NotImplementedError:
            signal.signal(signal.SIGINT, lambda *_: on_sigint())

    if stop_external is not None:

        async def _watch_external() -> None:
            nonlocal interrupted
            await stop_external.wait()
            interrupted = True
            stop.set()

        watcher = asyncio.create_task(_watch_external())

    started_at = iso_now()
    # F3: 使用工厂函数根据 api_style 选择客户端
    from llm_bench.client import create_client
    client = create_client(
        api_style=cfg.api_style,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        timeout=cfg.timeout,
        concurrency=concurrency,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        extra_body=cfg.extra_body,
        stream=cfg.stream,
        capture_chunk_timeline=cfg.chunk_timeline,
        # Web UI：留证本模型首个请求的完整请求/返回 body（CLI 无 broadcaster 时为 None）
        first_exchange_sink=(
            broadcaster.record_first_exchange if broadcaster is not None else None
        ),
    )
    # 3.11: 注册 reset 回调，确保 broadcaster.reset() 时客户端状态也被重置
    if broadcaster is not None and hasattr(client, 'reset_first_exchange'):
        broadcaster.register_on_reset(client.reset_first_exchange)
    try:
        if cfg.cache_mode == "warmup" and cfg.cache_warmup > 0:
            log.info("cache_warmup n=%s conc=1", cfg.cache_warmup)
            await _run_workers(
                client=client,
                prompt=prompt,
                phase="cache_warmup",
                provider=cfg.provider_name,
                concurrency=concurrency,
                n_workers=1,
                budget=_Budget(cfg.cache_warmup),
                stop=stop,
                retry=0,
                records=records,
                lock=lock,
                printed_minutes=printed,
                log=log,
            )
        if cfg.warmup > 0:
            log.info("warmup n=%s conc=%s", cfg.warmup, min(2, concurrency))
            await _run_workers(
                client=client,
                prompt=prompt,
                phase="warmup",
                provider=cfg.provider_name,
                concurrency=concurrency,
                n_workers=min(2, concurrency),
                budget=_Budget(cfg.warmup),
                stop=stop,
                retry=0,
                records=records,
                lock=lock,
                printed_minutes=printed,
                log=log,
            )
        if not stop.is_set():
            stopper: asyncio.Task[None] | None = None
            if duration is not None:

                async def _stop_later() -> None:
                    await asyncio.sleep(duration)
                    stop.set()

                stopper = asyncio.create_task(_stop_later())
            log.info(
                "run conc=%s duration=%s requests=%s",
                concurrency,
                duration,
                requests,
            )
            t0 = time.perf_counter()
            await _run_workers(
                client=client,
                prompt=prompt,
                phase="run",
                provider=cfg.provider_name,
                concurrency=concurrency,
                n_workers=concurrency,
                budget=_Budget(requests),
                stop=stop,
                retry=cfg.retry,
                records=records,
                lock=lock,
                printed_minutes=printed,
                log=log,
                wait_time=cfg.wait_time,  # F9
                broadcaster=broadcaster,  # F1
            )
            if stopper is not None:
                stopper.cancel()
            wall_measured = time.perf_counter() - t0
            log.info("level finished wall=%.3fs records=%s", wall_measured, len(records))
        else:
            interrupted = True
    except asyncio.CancelledError:
        interrupted = True
        log.warning("run cancelled, writing partial results")
        _uncancel_current_task()
    finally:
        if install_signals:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, AttributeError):
                pass
        if watcher is not None:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
        await client.aclose()

    ended_at = iso_now()
    header = {
        "provider": cfg.provider_name,
        "model": cfg.model,
        "concurrency": concurrency,
        "prompt": cfg.prompt,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "started_at": started_at,
        "ended_at": ended_at,
        "interrupted": interrupted,
        "llm_bench_version": __version__,
        "stream_options_include_usage": (not client.stream_options_rejected),
        "enable_thinking_disabled_during_run": client.enable_thinking_rejected,
        "stream": cfg.stream,
        "cache_mode": cfg.cache_mode,
        "chunk_timeline": cfg.chunk_timeline,
    }
    profile = prompt_profile_hints(cfg.prompt)
    config_json = {
        **cfg.redacted_dict(),
        **machine,
        "concurrency": concurrency,
        "duration": duration,
        "requests": requests,
        "prompt_size": cfg.prompt,
        "prompt_chars": len(prompt),
        "estimated_prompt_tokens": estimate_tokens(prompt),
        "prompt_profile_hints": profile,
        "cache_mode": cfg.cache_mode,
        "cache_warmup": cfg.cache_warmup,
        "chunk_timeline": cfg.chunk_timeline,
        "stream_options_include_usage": not client.stream_options_rejected,
        "stream_options_disabled_during_run": client.stream_options_rejected,
        "enable_thinking_disabled_during_run": client.enable_thinking_rejected,
        "started_at": started_at,
        "ended_at": ended_at,
        "hostname": machine.get("hostname"),
        "python": machine.get("python"),
        "platform": machine.get("platform"),
        "public_ip": machine.get("public_ip"),
        "llm_bench_version": __version__,
        "interrupted": interrupted,
    }
    try:
        payload = build_run_artifacts(
            out_dir,
            records=records,
            header=header,
            config_json=config_json,
            configured_duration=None if interrupted else duration,
            log=log,
        )
        payload["output_dir"] = str(out_dir)
        return payload
    finally:
        _close_logger_handlers(log)


def level_duration_requests(
    cfg: RunConfig, n_levels: int
) -> tuple[float | None, int | None]:
    if cfg.requests is not None:
        return None, cfg.requests
    if n_levels == 1:
        return cfg.duration, None
    if cfg.step_duration_explicit:
        return cfg.step_duration, None
    return cfg.duration, None


async def run_benchmark(
    cfg: RunConfig,
    *,
    install_signals: bool = True,
    stop_external: asyncio.Event | None = None,
    broadcaster: Any = None,  # F1: Web UI
    run_name: str | None = None,  # 任务组/批次目录内的可读子目录名（供应商_模型）
) -> Path:
    if run_name is not None:
        # 任务组（job_时间戳）/批次（suite_时间戳）目录已承担分组与时间戳职责，
        # 模型子目录直接使用可读命名并保留中文；同名目标并发时由 mkdir 原子去重
        readable = sanitize_fs_readable(run_name)
        base = cfg.output_dir / readable
        dup = 2
        while True:
            try:
                base.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                base = cfg.output_dir / f"{readable}-{dup}"
                dup += 1
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # CLI 直跑在平铺的 results/ 下：中文等非 ASCII 供应商名经 sanitize 后会
        # 全部丢失，追加名称短哈希，避免不同供应商选同名模型时结果目录互相覆盖
        p_name = sanitize_fs(cfg.provider_name)
        if not p_name or any(ord(ch) > 127 for ch in cfg.provider_name):
            p_name = (p_name or "provider") + "-" + hashlib.sha1(
                cfg.provider_name.encode("utf-8")).hexdigest()[:6]
        base = cfg.output_dir / f"{p_name}_{sanitize_fs(cfg.model)}_{stamp}"
        # 同秒同供应商/模型并发的兜底
        if base.exists():
            base = base.with_name(base.name + "-" + hashlib.sha1(
                f"{cfg.base_url}".encode("utf-8")).hexdigest()[:6])
        base.mkdir(parents=True, exist_ok=True)
    try:
        machine = {
            "hostname": socket.gethostname(),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "public_ip": await public_ip(),
        }
        levels = cfg.concurrency
        dur, reqs = level_duration_requests(cfg, len(levels))
        if len(levels) == 1:
            await run_level(
                cfg,
                concurrency=levels[0],
                duration=dur,
                requests=reqs,
                out_dir=base,
                machine=machine,
                install_signals=install_signals,
                stop_external=stop_external,
                broadcaster=broadcaster,  # F1
            )
            print(f"results: {base}")
            return base

        steps: list[dict[str, Any]] = []
        for c in levels:
            sub = base / f"c{c}"
            print(f"\n=== concurrency {c} -> {sub} ===")
            payload = await run_level(
                cfg,
                concurrency=c,
                duration=dur,
                requests=reqs,
                out_dir=sub,
                machine=machine,
                install_signals=install_signals,
                stop_external=stop_external,
                broadcaster=broadcaster,  # F1
            )
            steps.append(
                {
                    "concurrency": c,
                    "dir": str(sub),
                    "success_rate": payload.get("availability", {}).get("success_rate"),
                    "ttft_p50": payload.get("ttft", {}).get("p50"),
                    "decode_tps_p50": payload.get("decode_tps", {}).get("p50"),
                    "rpm": payload.get("throughput", {}).get("rpm"),
                    "aggregate_output_tps": payload.get("throughput", {}).get("aggregate_output_tps"),
                }
            )
            if payload.get("interrupted"):
                break
        write_json(base / "steps.json", {"steps": steps, "config": cfg.redacted_dict()})
        print(f"results: {base}")
        return base
    except asyncio.CancelledError:
        _uncancel_current_task()
        print(f"results: {base}")
        return base
