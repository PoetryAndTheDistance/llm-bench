from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def percentile(sorted_values: list[float], p: float) -> float | None:
    """Linear interpolation; p in 0-100. Empty -> None."""
    if not sorted_values:
        return None
    if p < 0 or p > 100:
        raise ValueError(f"percentile p must be in 0-100, got {p}")
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (n - 1)
    i = int(math.floor(rank))
    f = rank - i
    if i + 1 >= n:
        return float(sorted_values[-1])
    a = float(sorted_values[i])
    b = float(sorted_values[i + 1])
    return a * (1.0 - f) + b * f


def mean(values: Iterable[float]) -> float | None:
    xs = list(values)
    if not xs:
        return None
    return sum(xs) / len(xs)


def stddev(values: Iterable[float]) -> float | None:
    xs = list(values)
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def cv(values: Iterable[float]) -> float | None:
    xs = list(values)
    m = mean(xs)
    s = stddev(xs)
    if m is None or s is None or m == 0:
        return None
    return s / m


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class ChunkTimelineEvent:
    seq: int
    event_type: str
    offset_ms: float
    delta_ms: float | None
    chars: int
    cumulative_chars: int
    sse_frame: int
    text_preview: str = ""


@dataclass
class RequestResult:
    request_id: str
    phase: str
    provider: str
    model: str
    concurrency: int
    start_time: str
    end_time: str
    start_perf: float
    end_perf: float
    first_token_time: str | None = None
    first_token_perf: float | None = None
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cache_l1_tokens: int | None = None  # F5: cache_creation / cache_write (写入缓存)
    cache_l2_tokens: int | None = None  # F5: cache_read (读取缓存)
    token_source: str | None = None
    output_text: str = ""
    ttft: float | None = None
    e2e: float | None = None
    decode_duration: float | None = None
    decode_tps: float | None = None
    effective_tps: float | None = None
    prefill_tps: float | None = None  # F8: input_tokens / ttft
    tpot_ms: float | None = None
    mean_itl_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_count: int = 0
    success: bool = False
    streamed: bool = True
    chunk_timeline: list[ChunkTimelineEvent] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def fill_derived(self) -> None:
        self.e2e = self.end_perf - self.start_perf if self.end_perf >= self.start_perf else None
        ot = self.output_tokens or 0
        if self.e2e is not None and self.e2e > 0 and ot > 0:
            self.effective_tps = ot / self.e2e
        else:
            self.effective_tps = None
        if not self.streamed:
            self.ttft = self.e2e
            self.decode_duration = None
            self.decode_tps = None
            self.tpot_ms = None
            # F8: Prefill TPS for non-stream mode
            if self.ttft is not None and self.ttft > 0 and (self.input_tokens or 0) > 0:
                self.prefill_tps = self.input_tokens / self.ttft
            else:
                self.prefill_tps = None
            return
        if self.first_token_perf is not None:
            self.ttft = self.first_token_perf - self.start_perf
            self.decode_duration = self.end_perf - self.first_token_perf
        else:
            self.ttft = None
            self.decode_duration = None
        if (
            self.first_token_perf is not None
            and self.decode_duration is not None
            and self.decode_duration > 0
            and ot > 0
        ):
            self.decode_tps = ot / self.decode_duration
        else:
            self.decode_tps = None
        if self.decode_duration is not None and self.decode_duration > 0 and ot >= 2:
            self.tpot_ms = self.decode_duration / (ot - 1) * 1000.0
        else:
            self.tpot_ms = None
        # F8: Prefill TPS = input_tokens / ttft
        if self.ttft is not None and self.ttft > 0 and (self.input_tokens or 0) > 0:
            self.prefill_tps = self.input_tokens / self.ttft
        else:
            self.prefill_tps = None


def itl_stats(intervals_s: list[float]) -> tuple[float | None, float | None, float | None, int]:
    if not intervals_s:
        return None, None, None, 0
    ms = [x * 1000.0 for x in intervals_s]
    ms_sorted = sorted(ms)
    return mean(ms), percentile(ms_sorted, 50), percentile(ms_sorted, 95), len(ms)


_HTTP_KEYS = (200, 400, 401, 403, 429, 500, 502, 503, 504)


def _pct_fields(values: list[float], ps: tuple[int, ...], with_minmax: bool) -> dict[str, float | None]:
    s = sorted(values)
    out: dict[str, float | None] = {"avg": mean(s)}
    if with_minmax:
        out["min"] = s[0] if s else None
        out["max"] = s[-1] if s else None
    for p in ps:
        out[f"p{p}"] = percentile(s, p)
    return out


def summarize(
    records: list[RequestResult],
    *,
    configured_duration: float | None,
) -> dict[str, Any]:
    run = [r for r in records if r.phase == "run"]
    ok = [r for r in run if r.success]
    total = len(run)
    success_n = len(ok)
    failed_n = total - success_n
    timeouts = sum(1 for r in run if r.error_type == "timeout")
    connect_errors = sum(1 for r in run if r.error_type == "connect")
    status_counts: dict[str, int] = {str(k): 0 for k in _HTTP_KEYS}
    status_counts["other"] = 0
    http_5xx = 0
    for r in run:
        st = r.http_status
        if st is None:
            continue
        if st in _HTTP_KEYS:
            status_counts[str(st)] += 1
        else:
            status_counts["other"] += 1
        if st is not None and 500 <= st <= 599:
            http_5xx += 1

    if run:
        wall = max(r.end_perf for r in run) - min(r.start_perf for r in run)
    else:
        wall = 0.0
    if configured_duration is not None:
        wall = max(configured_duration, wall)
    if wall <= 0:
        wall = 1e-9
    minutes = wall / 60.0

    in_sum = sum(r.input_tokens or 0 for r in ok)
    out_sum = sum(r.output_tokens or 0 for r in ok)
    # 优先使用供应商 usage.total_tokens 的口径
    total_sum = sum(r.total_tokens or 0 for r in ok)

    ttft_vals = [r.ttft for r in ok if r.ttft is not None]
    decode_vals = [r.decode_tps for r in ok if r.decode_tps is not None]
    eff_vals = [r.effective_tps for r in ok if r.effective_tps is not None]
    tpot_vals = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    e2e_vals = [r.e2e for r in ok if r.e2e is not None]
    prefill_vals = [r.prefill_tps for r in ok if r.prefill_tps is not None]  # F8

    availability = {
        "total": total,
        "success": success_n,
        "failed": failed_n,
        "success_rate": (success_n / total) if total else 0.0,
        "timeout_rate": (timeouts / total) if total else 0.0,
        "connect_errors": connect_errors,
        "http_5xx": http_5xx,
        **{f"http_{k}": status_counts[str(k)] for k in _HTTP_KEYS},
        "http_other": status_counts["other"],
    }

    cache_reported = [
        r
        for r in ok
        if r.cache_read_tokens is not None and (r.input_tokens or 0) > 0
    ]
    cache_hit_rates = [
        r.cache_read_tokens / (r.input_tokens or 1)
        for r in cache_reported
        if r.cache_read_tokens is not None
    ]
    cache_read_vals = [float(r.cache_read_tokens) for r in cache_reported]
    cache_write_vals = [
        float(r.cache_write_tokens)
        for r in ok
        if r.cache_write_tokens is not None
    ]
    # 请求维度命中率：成功请求中至少命中 1 个缓存 token（L2）的请求数
    cache_hit_requests = sum(1 for r in ok if (r.cache_read_tokens or 0) > 0)
    cache = {
        "reported_requests": len(cache_reported),
        "reported_rate": (len(cache_reported) / success_n) if success_n else 0.0,
        "hit_requests": cache_hit_requests,
        "hit_request_rate": (cache_hit_requests / success_n) if success_n else 0.0,
        "cache_read_sum": int(sum(cache_read_vals)) if cache_read_vals else None,
        "cache_read_avg": mean(cache_read_vals),
        "cache_write_sum": int(sum(cache_write_vals)) if cache_write_vals else None,
        "cache_write_avg": mean(cache_write_vals),
        "cache_l1_sum": sum(r.cache_l1_tokens or 0 for r in ok),  # F5
        "cache_l2_sum": sum(r.cache_l2_tokens or 0 for r in ok),  # F5
        "hit_rate_avg": mean(cache_hit_rates),
        "hit_rate_p50": percentile(sorted(cache_hit_rates), 50) if cache_hit_rates else None,
        "hit_rate_p95": percentile(sorted(cache_hit_rates), 95) if cache_hit_rates else None,
    }

    return {
        "availability": availability,
        "tokens": {
            "input_sum": in_sum,
            "input_avg": (in_sum / success_n) if success_n else None,
            "output_sum": out_sum,
            "output_avg": (out_sum / success_n) if success_n else None,
            "total_sum": total_sum,
            "total_avg": (total_sum / success_n) if success_n else None,
        },
        "throughput": {
            "wall_seconds": wall,
            "rpm": success_n / minutes,
            "input_tpm": in_sum / minutes,
            "output_tpm": out_sum / minutes,
            "total_tpm": total_sum / minutes,
            "aggregate_output_tps": out_sum / wall,
        },
        "ttft": _pct_fields(ttft_vals, (1, 10, 50, 90, 95, 99), True),
        "decode_tps": _pct_fields(decode_vals, (1, 10, 50, 90, 95, 99), True),
        "effective_tps": _pct_fields(eff_vals, (1, 50, 95, 99), False),
        "prefill_tps": _pct_fields(prefill_vals, (1, 10, 50, 90, 95, 99), True),  # F8
        "tpot_ms": _pct_fields(tpot_vals, (50, 95, 99), False),
        "e2e": _pct_fields(e2e_vals, (50, 90, 95, 99), False),
        "stability": {
            "ttft_std": stddev(ttft_vals),
            "ttft_cv": cv(ttft_vals),
            "decode_tps_std": stddev(decode_vals),
            "decode_tps_cv": cv(decode_vals),
            "e2e_std": stddev(e2e_vals),
            "e2e_cv": cv(e2e_vals),
        },
        "cache": cache,
    }


def minute_bucket(iso_ts: str) -> str:
    # "2026-03-02T20:31:05.123+08:00" -> "2026-03-02 20:31"
    try:
        dt = datetime.fromisoformat(iso_ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_ts[:16].replace("T", " ")


def _minute_elapsed(minute: str, *, is_last: bool, now: datetime) -> float:
    if not is_last:
        return 60.0
    try:
        start = datetime.strptime(minute, "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
    except ValueError:
        return 60.0
    elapsed = (now - start).total_seconds()
    if elapsed <= 0:
        return 1e-9
    return 60.0 if elapsed >= 60.0 else elapsed


def timeseries_rows(
    records: list[RequestResult],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    run = [r for r in records if r.phase == "run"]
    buckets: dict[str, list[RequestResult]] = {}
    for r in run:
        buckets.setdefault(minute_bucket(r.start_time), []).append(r)
    clock = now or datetime.now().astimezone()
    keys = sorted(buckets)
    last_key = keys[-1] if keys else None
    rows: list[dict[str, Any]] = []
    for minute in keys:
        rs = buckets[minute]
        ok = [r for r in rs if r.success]
        n = len(rs)
        success_n = len(ok)
        failed_n = n - success_n
        in_sum = sum(r.input_tokens or 0 for r in ok)
        out_sum = sum(r.output_tokens or 0 for r in ok)
        ttft_vals = sorted(r.ttft for r in ok if r.ttft is not None)
        dec_vals = sorted(r.decode_tps for r in ok if r.decode_tps is not None)
        eff_vals = sorted(r.effective_tps for r in ok if r.effective_tps is not None)
        e2e_vals = sorted(r.e2e for r in ok if r.e2e is not None)
        elapsed = _minute_elapsed(minute, is_last=(minute == last_key), now=clock)
        rows.append(
            {
                "minute": minute,
                "n": n,
                "requests": n,
                "success": success_n,
                "failed": failed_n,
                "rpm": success_n * 60.0 / elapsed,
                "input_tokens": in_sum,
                "output_tokens": out_sum,
                "avg_input_tokens": (in_sum / success_n) if success_n else None,
                "avg_output_tokens": (out_sum / success_n) if success_n else None,
                "ttft_avg": mean(ttft_vals),
                "ttft_p50": percentile(ttft_vals, 50),
                "ttft_p95": percentile(ttft_vals, 95),
                "ttft_p99": percentile(ttft_vals, 99),
                "decode_tps_p1": percentile(dec_vals, 1),
                "decode_tps_p50": percentile(dec_vals, 50),
                "decode_tps_p95": percentile(dec_vals, 95),
                "effective_tps_p1": percentile(eff_vals, 1),
                "effective_tps_p50": percentile(eff_vals, 50),
                "effective_tps_p95": percentile(eff_vals, 95),
                "e2e_p50": percentile(e2e_vals, 50),
                "e2e_p95": percentile(e2e_vals, 95),
                "e2e_p99": percentile(e2e_vals, 99),
                "http_429": sum(1 for r in rs if r.http_status == 429),
                "http_5xx": sum(
                    1 for r in rs if r.http_status is not None and 500 <= r.http_status <= 599
                ),
                "timeout_count": sum(1 for r in rs if r.error_type == "timeout"),
                "error_rate": (failed_n / n) if n else 0.0,
                "output_tps": out_sum / elapsed,
            }
        )
    return rows
