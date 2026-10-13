from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from llm_bench.config import FALLBACK_WEIGHTS, load_score_weights
from llm_bench.metrics import RequestResult, clamp, summarize, timeseries_rows

CSV_COLUMNS = [
    "request_id",
    "phase",
    "provider",
    "model",
    "concurrency",
    "start_time",
    "first_token_time",
    "end_time",
    "http_status",
    "error_type",
    "error_message",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_l1_tokens",   # F5
    "cache_l2_tokens",   # F5
    "token_source",
    "ttft",
    "e2e",
    "decode_duration",
    "decode_tps",
    "effective_tps",
    "prefill_tps",       # F8
    "tpot_ms",
    "mean_itl_ms",
]

CHUNK_COLUMNS = [
    "request_id",
    "phase",
    "seq",
    "event_type",
    "offset_ms",
    "delta_ms",
    "chars",
    "cumulative_chars",
    "sse_frame",
    "text_preview",
]

TS_COLUMNS = [
    "minute",
    "n",
    "requests",
    "success",
    "failed",
    "rpm",
    "input_tokens",
    "output_tokens",
    "avg_input_tokens",
    "avg_output_tokens",
    "ttft_avg",
    "ttft_p50",
    "ttft_p95",
    "ttft_p99",
    "decode_tps_p1",
    "decode_tps_p50",
    "decode_tps_p95",
    "effective_tps_p1",
    "effective_tps_p50",
    "effective_tps_p95",
    "e2e_p50",
    "e2e_p95",
    "e2e_p99",
    "http_429",
    "http_5xx",
    "timeout_count",
    "error_rate",
    "output_tps",
]


def _n(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%"


def format_summary_banner(summary: dict[str, Any], *, header: dict[str, Any]) -> str:
    a = summary["availability"]
    tok = summary["tokens"]
    th = summary["throughput"]
    ttft = summary["ttft"]
    dec = summary["decode_tps"]
    eff = summary["effective_tps"]
    prefill = summary.get("prefill_tps") or {}  # F8
    tpot = summary["tpot_ms"]
    e2e = summary["e2e"]
    st = summary["stability"]
    cache = summary.get("cache") or {}
    line = "=" * 72
    sep = "-" * 72
    conc = header.get("concurrency")
    rows = [
        line,
        " LLM API Benchmark",
        f" provider={header.get('provider')}  model={header.get('model')}  concurrency={conc}",
        f" prompt={header.get('prompt')}  max_tokens={header.get('max_tokens')}  temp={header.get('temperature')}",
        f" wall={_n(th.get('wall_seconds'), 2)}s  started={header.get('started_at')}  ended={header.get('ended_at')}",
        line,
        " Availability",
        f"   total            {a['total']}",
        f"   success          {a['success']}",
        f"   failed           {a['failed']}",
        f"   success_rate     {_pct(a['success_rate'])}",
        f"   timeout_rate     {_pct(a['timeout_rate'])}",
        (
            "   http            "
            f"200={a['http_200']} 400={a['http_400']} 401={a['http_401']} "
            f"403={a['http_403']} 429={a['http_429']} 500={a['http_500']} "
            f"502={a['http_502']} 503={a['http_503']} 504={a['http_504']} "
            f"other={a['http_other']}"
        ),
        f"   5xx              {a['http_5xx']}",
        f"   connect_errors   {a['connect_errors']}",
        sep,
        " Tokens (success)",
        f"   input            sum={tok['input_sum']}  avg={_n(tok['input_avg'], 1)}",
        f"   output           sum={tok['output_sum']}  avg={_n(tok['output_avg'], 1)}",
        sep,
        " Cache (success, provider-reported)",
        (
            f"   reported         {cache.get('reported_requests', 0)}"
            f"  rate={_pct(cache.get('reported_rate'))}"
        ),
        (
            f"   hit_rate(token)  avg={_pct(cache.get('hit_rate_avg'))}"
            f"  p50={_pct(cache.get('hit_rate_p50'))}"
            f"  p95={_pct(cache.get('hit_rate_p95'))}"
        ),
        (
            f"   hit_req(request) {cache.get('hit_requests', 0)}/{a['success']}"
            f"  rate={_pct(cache.get('hit_request_rate'))}"
        ),
        (
            f"   cache_read       sum={cache.get('cache_read_sum') or 'n/a'}"
            f"  avg={_n(cache.get('cache_read_avg'), 1)}"
        ),
        (
            f"   cache_write      sum={cache.get('cache_write_sum') or 'n/a'}"
            f"  avg={_n(cache.get('cache_write_avg'), 1)}"
        ),
        sep,
        " Throughput",
        f"   RPM              {_n(th['rpm'], 2)}",
        f"   Input TPM        {_n(th['input_tpm'], 2)}",
        f"   Output TPM       {_n(th['output_tpm'], 2)}",
        f"   Total TPM        {_n(th['total_tpm'], 2)}",
        f"   Aggregate Out TPS {_n(th['aggregate_output_tps'], 2)}",
        sep,
        " TTFT (s)",
        (
            f"   avg={_n(ttft['avg'])}  min={_n(ttft['min'])}  max={_n(ttft['max'])}  "
            f"p1={_n(ttft['p1'])}  p10={_n(ttft['p10'])}  p50={_n(ttft['p50'])}  "
            f"p90={_n(ttft['p90'])}  p95={_n(ttft['p95'])}  p99={_n(ttft['p99'])}"
        ),
        sep,
        " Decode TPS (tokens/s, excludes TTFT)",
        (
            f"   avg={_n(dec['avg'], 2)}  min={_n(dec['min'], 2)}  max={_n(dec['max'], 2)}  "
            f"p1={_n(dec['p1'], 2)}  p10={_n(dec['p10'], 2)}  p50={_n(dec['p50'], 2)}  "
            f"p90={_n(dec['p90'], 2)}  p95={_n(dec['p95'], 2)}  p99={_n(dec['p99'], 2)}"
        ),
        sep,
        " Prefill TPS (tokens/s, input_tokens / TTFT)",  # F8
        (
            f"   avg={_n(prefill.get('avg'), 2)}  min={_n(prefill.get('min'), 2)}  max={_n(prefill.get('max'), 2)}  "
            f"p1={_n(prefill.get('p1'), 2)}  p10={_n(prefill.get('p10'), 2)}  p50={_n(prefill.get('p50'), 2)}  "
            f"p90={_n(prefill.get('p90'), 2)}  p95={_n(prefill.get('p95'), 2)}  p99={_n(prefill.get('p99'), 2)}"
        ),
        sep,
        " Effective TPS (tokens/s, includes TTFT)",
        (
            f"   avg={_n(eff['avg'], 2)}  p1={_n(eff['p1'], 2)}  p50={_n(eff['p50'], 2)}  "
            f"p95={_n(eff['p95'], 2)}  p99={_n(eff['p99'], 2)}"
        ),
        sep,
        " TPOT (ms)",
        (
            f"   avg={_n(tpot['avg'], 2)}  p50={_n(tpot['p50'], 2)}  "
            f"p95={_n(tpot['p95'], 2)}  p99={_n(tpot['p99'], 2)}"
        ),
        sep,
        " E2E (s)",
        (
            f"   avg={_n(e2e['avg'])}  p50={_n(e2e['p50'])}  p90={_n(e2e['p90'])}  "
            f"p95={_n(e2e['p95'])}  p99={_n(e2e['p99'])}"
        ),
        sep,
        " Stability",
        f"   TTFT        std={_n(st['ttft_std'])}  cv={_n(st['ttft_cv'])}",
        f"   Decode TPS  std={_n(st['decode_tps_std'], 2)}  cv={_n(st['decode_tps_cv'])}",
        f"   E2E         std={_n(st['e2e_std'])}  cv={_n(st['e2e_cv'])}",
        line,
    ]
    return "\n".join(rows)


def format_minute_row(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[{row['minute']}]",
            (
                f"  n={row['n']} success={row['success']} failed={row['failed']} "
                f"rpm={_n(row['rpm'], 2)} error_rate={_pct(row.get('error_rate'))}"
            ),
            (
                f"  tokens in={row.get('input_tokens')} out={row.get('output_tokens')} "
                f"avg_in={_n(row.get('avg_input_tokens'), 1)} "
                f"avg_out={_n(row.get('avg_output_tokens'), 1)}"
            ),
            (
                f"  ttft avg={_n(row.get('ttft_avg'))} p50={_n(row.get('ttft_p50'))} "
                f"p95={_n(row.get('ttft_p95'))} p99={_n(row.get('ttft_p99'))}"
            ),
            (
                f"  decode_tps p1={_n(row.get('decode_tps_p1'), 2)} "
                f"p50={_n(row.get('decode_tps_p50'), 2)} p95={_n(row.get('decode_tps_p95'), 2)}"
            ),
            (
                f"  effective_tps p1={_n(row.get('effective_tps_p1'), 2)} "
                f"p50={_n(row.get('effective_tps_p50'), 2)} p95={_n(row.get('effective_tps_p95'), 2)}"
            ),
            (
                f"  e2e p50={_n(row.get('e2e_p50'))} p95={_n(row.get('e2e_p95'))} "
                f"p99={_n(row.get('e2e_p99'))}"
            ),
            (
                f"  429={row.get('http_429')} 5xx={row.get('http_5xx')} "
                f"timeout={row.get('timeout_count')} out_tps={_n(row.get('output_tps'), 2)}"
            ),
        ]
    )


def _csv_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_requests_csv(path: Path, records: list[RequestResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in records:
            w.writerow({k: _csv_val(getattr(r, k, None)) for k in CSV_COLUMNS})


def write_chunks_csv(path: Path, records: list[RequestResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CHUNK_COLUMNS)
        w.writeheader()
        for rec in records:
            for ev in rec.chunk_timeline:
                w.writerow(
                    {
                        "request_id": rec.request_id,
                        "phase": rec.phase,
                        "seq": ev.seq,
                        "event_type": ev.event_type,
                        "offset_ms": _csv_val(ev.offset_ms),
                        "delta_ms": _csv_val(ev.delta_ms),
                        "chars": ev.chars,
                        "cumulative_chars": ev.cumulative_chars,
                        "sse_frame": ev.sse_frame,
                        "text_preview": ev.text_preview,
                    }
                )


def write_timeseries_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TS_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: _csv_val(row.get(k)) for k in TS_COLUMNS})


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ms(v: float | None) -> str:
    """秒 -> 毫秒字符串；None 显示 n/a（流式但无正文时 TTFT 可能缺失）"""
    return "n/a" if v is None else f"{v * 1000:.1f} ms"


def _num(v: Any, spec: str = ".2f", na: str = "n/a") -> str:
    return na if v is None else format(float(v), spec)


def _int_num(v: Any, na: str = "n/a") -> str:
    return na if v is None else f"{int(v):,}"


_DT_FRAC_RE = re.compile(r"^(.+T\d{2}:\d{2}:\d{2})\.(\d+)(.*)$")


def _fmt_dt(v: Any) -> str:
    """ISO 时间 -> 'YYYY-MM-DD HH:MM:SS'；无法解析时原样返回"""
    if not v:
        return "n/a"
    s = str(v)
    try:
        return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Python 3.10 的 fromisoformat 只接受 3/6 位小数秒，补齐到 6 位后再试
        m = _DT_FRAC_RE.match(s)
        if m:
            head, frac, tail = m.groups()
            try:
                return datetime.fromisoformat(
                    f"{head}.{(frac + '000000')[:6]}{tail}"
                ).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        return s


def _cfg_val(v: Any) -> str:
    """配置值在报告中的展示：布尔 -> 是/否，None -> n/a"""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "是" if v else "否"
    return str(v)


def _lat_ms_row(name: str, stats: dict[str, Any], keys: list[str]) -> list[str]:
    """延迟分位表行：summary 中的秒值统一转毫秒展示"""
    return [name] + [_ms((stats or {}).get(k)) for k in keys]


def _clean_md_cell(v: Any, limit: int = 120) -> str:
    """错误信息等放入 Markdown 表格前：转义竖线、压平换行、截断"""
    s = str(v or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()
    return (s[:limit] + "…") if len(s) > limit else s


def _collect_error_breakdown(records: list[RequestResult]) -> list[dict[str, Any]]:
    """聚合 run 阶段失败请求：按错误类型计数，每类保留首条错误信息作为样例"""
    counts: dict[str, int] = {}
    samples: dict[str, str] = {}
    for r in records:
        if r.phase != "run" or r.success:
            continue
        etype = (r.error_type or "").strip()
        if not etype:
            etype = f"http_{r.http_status}" if r.http_status else "unknown"
        counts[etype] = counts.get(etype, 0) + 1
        if etype not in samples and r.error_message:
            samples[etype] = str(r.error_message)
    return [
        {"type": etype, "count": count, "sample": samples.get(etype, "")}
        for etype, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


# HTTP 状态码分布展示顺序
_HTTP_BUCKETS = [
    (200, "http_200"), (400, "http_400"), (401, "http_401"), (403, "http_403"),
    (429, "http_429"), (500, "http_500"), (502, "http_502"),
    (503, "http_503"), (504, "http_504"),
]

# 逐分钟时序表列（key 对应 timeseries_rows 输出）
_TS_COLUMNS = [
    ("分钟", "minute", "str"),
    ("n", "n", "int"),
    ("成功", "success", "int"),
    ("失败", "failed", "int"),
    ("RPM", "rpm", "rpm"),
    ("错误率", "error_rate", "pct"),
    ("TTFT p50", "ttft_p50", "ms"),
    ("Decode p50", "decode_tps_p50", "tps"),
    ("E2E p50", "e2e_p50", "ms"),
    ("429", "http_429", "int"),
    ("5xx", "http_5xx", "int"),
    ("超时", "timeout_count", "int"),
]


def format_single_run_markdown(
    summary: dict[str, Any],
    header: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    error_breakdown: list[dict[str, Any]] | None = None,
    timeseries: list[dict[str, Any]] | None = None,
) -> str:
    """生成单次压测的 Markdown 报告"""
    cfg = config or {}
    interrupted = bool(header.get("interrupted"))
    lines: list[str] = [
        "# llm-bench 压测报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    if interrupted:
        lines.extend([
            "> ⚠️ **本次压测被手动停止（中断）**，以下指标仅基于停止前已完成的请求，"
            "不代表完整压测结果。",
            "",
        ])

    # ── 基本信息 ──
    th = summary.get("throughput", {})
    info: list[tuple[str, Any]] = [
        ("供应商", header.get("provider")),
        ("模型", header.get("model")),
        ("并发数", header.get("concurrency")),
        ("Prompt", header.get("prompt")),
        ("max_tokens", header.get("max_tokens")),
        ("temperature", header.get("temperature")),
        ("流式", "是" if header.get("stream") else "否"),
    ]
    if cfg.get("api_style") is not None:
        info.append(("请求协议", cfg.get("api_style")))
    if cfg.get("requests") is not None:
        info.append(("压测模式", f"按请求数（计划 {cfg['requests']} 次）"))
    elif cfg.get("duration") is not None:
        info.append(("压测模式", f"按时长（计划 {float(cfg['duration']):g}s）"))
    for label, key in (
        ("超时(秒)", "timeout"),
        ("重试次数", "retry"),
        ("请求间隔(秒)", "wait_time"),
        ("缓存模式", "cache_mode"),
        ("缓存预热次数", "cache_warmup"),
        ("预热次数", "warmup"),
        ("chunk 时间线", "chunk_timeline"),
    ):
        if cfg.get(key) is not None:
            info.append((label, cfg.get(key)))
    info.extend([
        ("开始时间", _fmt_dt(header.get("started_at"))),
        ("结束时间", _fmt_dt(header.get("ended_at"))),
        ("实际墙钟时长", f"{_num(th.get('wall_seconds'), '.1f')} s"),
    ])
    if cfg.get("hostname"):
        info.append(("运行机器", cfg.get("hostname")))

    lines.extend(["## 基本信息", ""])
    lines.extend(f"- {label}：{_cfg_val(val)}" for label, val in info)
    lines.append("")

    # 仅当 prompt 档位为 custom 时才显示自定义提示词
    if cfg.get("custom_prompt") and cfg.get("prompt") == "custom":
        chars = cfg.get("prompt_chars")
        est_tokens = cfg.get("estimated_prompt_tokens")
        meta = []
        if chars is not None:
            meta.append(f"{chars} 字符")
        if est_tokens is not None:
            meta.append(f"约 {est_tokens} tokens")
        meta_text = f"（{'，'.join(meta)}）" if meta else ""
        lines.append(f"**自定义提示词{meta_text}：**")
        lines.append("")
        for pline in str(cfg["custom_prompt"]).splitlines() or [""]:
            lines.append(f"> {pline}")
        lines.append("")

    # ── 可用性 ──
    av = summary.get("availability", {})
    lines.extend([
        "## 可用性",
        "",
        f"- 总请求数：{av.get('total', 0)}",
        f"- 成功数：{av.get('success', 0)}",
        f"- 失败数：{av.get('failed', 0)}",
        f"- 成功率：{_num(av.get('success_rate', 0), '.2%')}",
        f"- 超时率：{_num(av.get('timeout_rate', 0), '.2%')}",
        f"- 连接错误：{av.get('connect_errors', 0)}",
        f"- HTTP 5xx：{av.get('http_5xx', 0)}",
        "- HTTP 状态分布："
        + "，".join(f"{code}={av.get(key, 0)}" for code, key in _HTTP_BUCKETS)
        + f"，other={av.get('http_other', 0)}",
    ])
    if error_breakdown:
        lines.extend(["", "### 错误分类", ""])
        lines.extend(
            _md_table(
                ["错误类型", "次数", "样例信息（截断）"],
                [
                    [str(e["type"]), str(e["count"]), _clean_md_cell(e.get("sample")) or "—"]
                    for e in error_breakdown
                ],
            )
        )
    lines.append("")

    # ── 延迟指标（全分位，毫秒）──
    ttft = summary.get("ttft", {})
    e2e = summary.get("e2e", {})
    tpot = summary.get("tpot_ms", {})
    lines.extend([
        "## 延迟指标",
        "",
        "### TTFT（首 Token 延迟，ms）",
        "",
    ])
    lines.extend(
        _md_table(
            ["指标", "avg", "min", "max", "p1", "p10", "p50", "p90", "p95", "p99"],
            [_lat_ms_row("TTFT", ttft, ["avg", "min", "max", "p1", "p10", "p50", "p90", "p95", "p99"])],
        )
    )
    lines.extend(["", "### E2E（端到端延迟，ms）", ""])
    lines.extend(
        _md_table(
            ["指标", "avg", "p50", "p90", "p95", "p99"],
            [_lat_ms_row("E2E", e2e, ["avg", "p50", "p90", "p95", "p99"])],
        )
    )
    lines.extend(["", "### TPOT（每输出 Token 耗时，ms）", ""])
    lines.extend(
        _md_table(
            ["指标", "avg", "p50", "p95", "p99"],
            [["TPOT"] + [_num(tpot.get(k), ".1f") for k in ("avg", "p50", "p95", "p99")]],
        )
    )
    lines.append("")

    # ── 吞吐指标 ──
    lines.extend([
        "## 吞吐指标",
        "",
        f"- RPM（请求/分钟）：{_num(th.get('rpm'), '.2f')}",
        f"- Input TPM（输入 token/分钟）：{_num(th.get('input_tpm'), '.0f')}",
        f"- Output TPM（输出 token/分钟）：{_num(th.get('output_tpm'), '.0f')}",
        f"- Total TPM（总 token/分钟）：{_num(th.get('total_tpm'), '.0f')}",
        f"- Aggregate Output TPS（聚合吞吐）：{_num(th.get('aggregate_output_tps'), '.2f')}",
        "",
        "### TPS 详情（tokens/s）",
        "",
    ])
    tps_keys = ["avg", "min", "max", "p1", "p10", "p50", "p90", "p95", "p99"]
    tps_rows = []
    for label, block in (
        ("Decode TPS（首 token 之后，不含 TTFT）", "decode_tps"),
        ("Prefill TPS（input_tokens / TTFT）", "prefill_tps"),
        ("Effective TPS（端到端，含 TTFT）", "effective_tps"),
    ):
        st = summary.get(block, {})
        tps_rows.append([label] + [_num(st.get(k), ".2f") for k in tps_keys])
    lines.extend(
        _md_table(["指标", "avg", "min", "max", "p1", "p10", "p50", "p90", "p95", "p99"], tps_rows)
    )
    lines.append("")

    # ── Token 消耗与缓存 ──
    tok = summary.get("tokens", {})
    cache = summary.get("cache", {})
    success_n = av.get("success", 0)
    write_sum = cache.get("cache_write_sum")
    read_sum = cache.get("cache_read_sum")
    lines.extend([
        "## Token 消耗",
        "",
        f"- 输入 Token：总计 {_int_num(tok.get('input_sum', 0))}，"
        f"平均每请求 {_num(tok.get('input_avg'), '.1f')}",
        f"- 输出 Token：总计 {_int_num(tok.get('output_sum', 0))}，"
        f"平均每请求 {_num(tok.get('output_avg'), '.1f')}",
        f"- Token 总计（供应商 usage.total_tokens）：总计 {_int_num(tok.get('total_sum'))}，"
        f"平均每请求 {_num(tok.get('total_avg'), '.1f')}",
        "",
        "### 缓存（仅统计成功请求，供应商上报口径）",
        "",
        f"- 上报缓存字段的请求：{_int_num(cache.get('reported_requests'), '0')}/{success_n}"
        f"（上报率 {_pct(cache.get('reported_rate'))}）",
        f"- 请求维度命中率（cache_read>0 的请求 / 成功请求）："
        f"{_int_num(cache.get('hit_requests'), '0')}/{success_n}"
        f"（{_pct(cache.get('hit_request_rate'))}）",
        f"- Token 维度命中率（每请求 cache_read / input 的统计）："
        f"avg={_pct(cache.get('hit_rate_avg'))}，"
        f"p50={_pct(cache.get('hit_rate_p50'))}，p95={_pct(cache.get('hit_rate_p95'))}",
        f"- 缓存写入 token（cache_write / L1 创建）：总计 {_int_num(write_sum)}"
        + ("" if write_sum is not None else "（供应商未上报）")
        + f"，平均每请求 {_num(cache.get('cache_write_avg'), '.1f')}",
        f"- 缓存读取 token（cache_read / L2 命中）：总计 {_int_num(read_sum)}"
        + ("" if read_sum is not None else "（供应商未上报）")
        + f"，平均每请求 {_num(cache.get('cache_read_avg'), '.1f')}",
        f"- L1 缓存 token 累计：{_int_num(cache.get('cache_l1_sum'), '0')}",
        f"- L2 缓存 token 累计：{_int_num(cache.get('cache_l2_sum'), '0')}",
        "",
    ])

    # ── 稳定性（std + CV）──
    stb = summary.get("stability", {})
    lines.extend([
        "## 稳定性",
        "",
        f"- TTFT：std={_num(stb.get('ttft_std'), '.3f')} s，CV={_num(stb.get('ttft_cv'), '.3f')}",
        f"- Decode TPS：std={_num(stb.get('decode_tps_std'), '.2f')}，"
        f"CV={_num(stb.get('decode_tps_cv'), '.3f')}",
        f"- E2E：std={_num(stb.get('e2e_std'), '.3f')} s，CV={_num(stb.get('e2e_cv'), '.3f')}",
        "",
    ])

    # ── 逐分钟时序 ──
    if timeseries:
        lines.extend(["## 逐分钟时序", ""])
        ts_rows: list[list[str]] = []
        for row in timeseries:
            cells: list[str] = []
            for _label, key, kind in _TS_COLUMNS:
                v = row.get(key)
                if kind == "int":
                    cells.append(str(v if v is not None else 0))
                elif kind == "pct":
                    cells.append(_pct(v))
                elif kind == "ms":
                    cells.append(_ms(v))
                elif kind == "tps":
                    cells.append(_num(v, ".2f"))
                elif kind == "rpm":
                    cells.append(_num(v, ".2f"))
                else:
                    cells.append(str(v if v is not None else ""))
            ts_rows.append(cells)
        lines.extend(_md_table([c[0] for c in _TS_COLUMNS], ts_rows))
        lines.append("")

    lines.extend(["---", "", "*由 llm-bench 自动生成*"])
    return "\n".join(lines)


def build_run_artifacts(
    out_dir: Path,
    *,
    records: list[RequestResult],
    header: dict[str, Any],
    config_json: dict[str, Any],
    configured_duration: float | None,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records, configured_duration=configured_duration)
    payload = {**header, **summary}
    write_json(out_dir / "summary.json", payload)
    write_json(out_dir / "config.json", config_json)
    write_requests_csv(out_dir / "requests.csv", records)
    if any(r.chunk_timeline for r in records):
        write_chunks_csv(out_dir / "chunks.csv", records)
    ts = timeseries_rows(records)
    write_timeseries_csv(out_dir / "timeseries.csv", ts)

    # 生成单次压测 Markdown 报告（含错误分类与逐分钟时序）
    error_breakdown = _collect_error_breakdown(records)
    report_md = format_single_run_markdown(
        summary,
        header,
        config=config_json,
        error_breakdown=error_breakdown,
        timeseries=ts,
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    banner = format_summary_banner(summary, header=header)
    print(banner)
    if log:
        log.info("\n%s", banner)
    title = "-" * 72 + "\n Per-minute\n" + "-" * 72
    print(title)
    if log:
        log.info(title)
    for row in ts:
        line = format_minute_row(row)
        print(line)
        if log:
            log.info(line)
    return payload


def _get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def minmax_norm(values: list[float | None], *, invert: bool = False) -> list[float]:
    xs: list[float | None] = []
    for v in values:
        if v is None:
            xs.append(None)
        elif invert:
            xs.append(None if v == 0 else 1.0 / float(v))
        else:
            xs.append(float(v))
    present = [x for x in xs if x is not None]
    if not present:
        return [0.0] * len(values)
    lo, hi = min(present), max(present)
    out: list[float] = []
    for x in xs:
        if x is None:
            out.append(0.0)
        elif hi == lo:
            out.append(1.0)
        else:
            out.append((x - lo) / (hi - lo))
    return out


def tail_stability_raw(summary: dict[str, Any]) -> float | None:
    cv = _get(summary, "stability", "ttft_cv")
    a = 1.0 - clamp(float(cv), 0.0, 1.0) if cv is not None else None
    p50 = _get(summary, "ttft", "p50")
    p99 = _get(summary, "ttft", "p99")
    b: float | None = None
    if p50 and p99 is not None and float(p50) > 0:
        ratio = float(p99) / float(p50)
        b = 1.0 / ratio if ratio > 0 else None
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2.0


def _error_reliability(summary: dict[str, Any]) -> float:
    total = float(_get(summary, "availability", "total") or 0.0)
    timeout_rate = float(_get(summary, "availability", "timeout_rate") or 0.0)
    http_5xx = float(_get(summary, "availability", "http_5xx") or 0.0)
    rate_5xx = (http_5xx / total) if total else 0.0
    return clamp(1.0 - timeout_rate - rate_5xx, 0.0, 1.0)


def compute_scores(
    summaries: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    n = len(summaries)
    rows: list[dict[str, Any] | None] = [None] * n
    groups: dict[Any, list[int]] = {}
    for i, s in enumerate(summaries):
        groups.setdefault(s.get("concurrency"), []).append(i)
    for idxs in groups.values():
        sub = [summaries[i] for i in idxs]
        scored = _compute_scores_flat(sub, weights)
        for i, sc in zip(idxs, scored, strict=True):
            rows[i] = sc
    return [r for r in rows if r is not None]


def _compute_scores_flat(
    summaries: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    w = dict(FALLBACK_WEIGHTS if weights is None else weights)
    n = len(summaries)
    success = [float(_get(s, "availability", "success_rate") or 0.0) for s in summaries]
    error_comp = [_error_reliability(s) for s in summaries]
    ttft_p50 = [_get(s, "ttft", "p50") for s in summaries]
    tps_p50 = [_get(s, "decode_tps", "p50") for s in summaries]
    agg = [_get(s, "throughput", "aggregate_output_tps") for s in summaries]
    tail = [tail_stability_raw(s) for s in summaries]
    ttft_n = minmax_norm(ttft_p50, invert=True)
    tps_n = minmax_norm(tps_p50, invert=False)
    thr_n = minmax_norm(agg, invert=False)
    tail_n = minmax_norm(tail, invert=False)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        parts = {
            "availability": success[i],
            "ttft": ttft_n[i],
            "decode_tps": tps_n[i],
            "throughput": thr_n[i],
            "tail_stability": tail_n[i],
            "error_rate": error_comp[i],
        }
        score = 100.0 * sum(w[k] * parts[k] for k in w)
        rows.append({"score": score, "components": parts})
    return rows


COMPARE_FIELDS = [
    ("provider", "provider", None),
    ("model", "model", None),
    ("concurrency", "concurrency", None),
    ("success_rate", ("availability", "success_rate"), "high"),
    ("ttft_p50", ("ttft", "p50"), "low"),
    ("cache_hit_p50", ("cache", "hit_rate_p50"), "high"),
    ("cache_req_hit", ("cache", "hit_request_rate"), "high"),
    ("ttft_p95", ("ttft", "p95"), "low"),
    ("ttft_p99", ("ttft", "p99"), "low"),
    ("ttft_cv", ("stability", "ttft_cv"), "low"),
    ("decode_tps_p1", ("decode_tps", "p1"), "high"),
    ("decode_tps_p50", ("decode_tps", "p50"), "high"),
    ("prefill_tps_p50", ("prefill_tps", "p50"), "high"),  # F8
    ("eff_tps_p50", ("effective_tps", "p50"), "high"),
    ("rpm", ("throughput", "rpm"), "high"),
    ("output_tpm", ("throughput", "output_tpm"), "high"),
    ("agg_out_tps", ("throughput", "aggregate_output_tps"), "high"),
    ("e2e_p50", ("e2e", "p50"), "low"),
    ("score", "score", "high"),
]


def _cell(summary: dict[str, Any], key: str | tuple[str, ...]) -> Any:
    if key == "score":
        return summary.get("score")
    if isinstance(key, tuple):
        return _get(summary, *key)
    return summary.get(key)


def _best_indices(vals: list[Any], direction: str) -> set[int]:
    nums: list[tuple[int, float]] = []
    for i, v in enumerate(vals):
        if isinstance(v, (int, float)):
            nums.append((i, float(v)))
    if not nums:
        return set()
    if direction == "low":
        best = min(x[1] for x in nums)
    else:
        best = max(x[1] for x in nums)
    return {i for i, v in nums if v == best}


def format_compare_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [f[0] for f in COMPARE_FIELDS]
    raw: dict[str, list[Any]] = {h: [_cell(r, k) for r in rows] for h, k, _ in COMPARE_FIELDS}
    best: dict[str, set[int]] = {}
    for name, _k, direction in COMPARE_FIELDS:
        if direction:
            best[name] = _best_indices(raw[name], direction)

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for i, _row in enumerate(rows):
        cells: list[str] = []
        for name, _k, _d in COMPARE_FIELDS:
            text = fmt(raw[name][i])
            if name in best and i in best[name] and text:
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Bold = best in column (lowest TTFT/CV/E2E, highest TPS/RPM/TPM/success/score).")
    return "\n".join(lines) + "\n"


def _as_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _uniq(vals: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for v in vals:
        k = repr(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _label(s: dict[str, Any]) -> str:
    return f"{s.get('provider')}/{s.get('model')}"


def _fmt_number(
    v: Any,
    suffix: str = "",
    digits: int = 2,
    scientific_threshold: float | None = None,
) -> str:
    """通用数字格式化函数

    Args:
        v: 要格式化的值
        suffix: 后缀（如 's' 表示秒）
        digits: 小数位数
        scientific_threshold: 科学计数法阈值，超过此值使用科学计数法
    """
    x = _as_float(v)
    if x is None:
        return "n/a"
    if scientific_threshold is not None and abs(x) >= scientific_threshold:
        return f"{x:.{digits}e}{suffix}"
    return f"{x:.{digits}f}{suffix}"


def _fmt_sec(v: Any) -> str:
    return _fmt_number(v, suffix="s", digits=3)


def _fmt_tps(v: Any) -> str:
    return _fmt_number(v, digits=2, scientific_threshold=10000)


def _fmt_score(v: Any) -> str:
    return _fmt_number(v, digits=1)


def _fmt_qty(v: Any, digits: int = 1) -> str:
    return _fmt_number(v, digits=digits)


def _fmt_bool(v: Any) -> str:
    if v is None:
        return "未知"
    return "开" if v else "关"


def _fmt_duration(v: Any) -> str:
    x = _as_float(v)
    if x is None:
        return "n/a"
    if x == int(x):
        return f"{int(x)}s"
    return f"{x:.1f}s"


def _thinking_of(s: dict[str, Any]) -> bool | None:
    extra = s.get("extra_body")
    if isinstance(extra, dict) and "enable_thinking" in extra:
        return bool(extra["enable_thinking"])
    if "enable_thinking" in s:
        return bool(s["enable_thinking"])
    return None


def _duration_of(s: dict[str, Any]) -> float | None:
    return _as_float(s.get("duration"))


def _zero_decode_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    w = dict(FALLBACK_WEIGHTS if weights is None else weights)
    w["decode_tps"] = 0.0
    total = sum(w.values())
    if total <= 0:
        return w
    return {k: v / total for k, v in w.items()}


_CONFIG_MERGE_KEYS = (
    "prompt",
    "prompt_size",
    "prompt_chars",
    "estimated_prompt_tokens",
    "cache_mode",
    "cache_warmup",
    "warmup",
    "chunk_timeline",
    "stream",
    "temperature",
    "timeout",
    "max_tokens",
)


def _fill_from_config(data: dict[str, Any], summary_file: Path) -> None:
    cfg_path = summary_file.parent / "config.json"
    if not cfg_path.is_file():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(cfg, dict):
        return
    if data.get("duration") is None and cfg.get("duration") is not None:
        data["duration"] = cfg["duration"]
    if data.get("extra_body") is None and cfg.get("extra_body") is not None:
        data["extra_body"] = cfg["extra_body"]
    for key in _CONFIG_MERGE_KEYS:
        if data.get(key) is None and cfg.get(key) is not None:
            data[key] = cfg[key]


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _burst_decode(s: dict[str, Any]) -> bool:
    dec = _as_float(_get(s, "decode_tps", "p50"))
    eff = _as_float(_get(s, "effective_tps", "p50"))
    ttft = _as_float(_get(s, "ttft", "p50"))
    e2e = _as_float(_get(s, "e2e", "p50"))
    if dec is None or eff is None or ttft is None or e2e is None:
        return False
    if eff <= 0 or e2e <= 0:
        return False
    huge = dec >= 20.0 * eff and dec >= 1000.0
    close = abs(ttft - e2e) / e2e <= 0.15
    return huge and close


def _winners(rows: list[dict[str, Any]], values: list[Any], direction: str) -> list[dict[str, Any]]:
    return [rows[i] for i in sorted(_best_indices(values, direction))]


def _names(rows: list[dict[str, Any]]) -> str:
    return "、".join(_label(s) for s in rows)


def _ensure_scores(
    summaries: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> None:
    if all(s.get("score") is not None for s in summaries):
        return
    scores = compute_scores(summaries, weights)
    for s, sc in zip(summaries, scores, strict=True):
        s["score"] = sc["score"]
        s["score_components"] = sc["components"]


def _apply_analysis_scores(
    summaries: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> bool:
    if any(_burst_decode(s) for s in summaries):
        scores = compute_scores(summaries, _zero_decode_weights(weights))
        for s, sc in zip(summaries, scores, strict=True):
            s["score"] = sc["score"]
            s["score_components"] = sc["components"]
        return True
    _ensure_scores(summaries, weights)
    return False


def _cond_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["无运行结果。"]

    # 按供应商分组显示模型
    provider_models: dict[str, list[str]] = {}
    for s in rows:
        provider = str(s.get("provider") or "未知")
        model = str(s.get("model") or "未知")
        if provider not in provider_models:
            provider_models[provider] = []
        if model not in provider_models[provider]:
            provider_models[provider].append(model)

    lines = ["- 供应商与模型："]
    for provider, models in provider_models.items():
        lines.append(f"  - {provider}：")
        for model in models:
            lines.append(f"    - {model}")

    fields: list[tuple[str, list[Any], Any]] = [
        ("prompt", [s.get("prompt") or s.get("prompt_size") for s in rows], lambda s: s.get("prompt") or s.get("prompt_size")),
        ("est_input_tokens", [s.get("estimated_prompt_tokens") for s in rows], lambda s: s.get("estimated_prompt_tokens")),
        ("cache_mode", [s.get("cache_mode") for s in rows], lambda s: s.get("cache_mode")),
        ("cache_warmup", [s.get("cache_warmup") for s in rows], lambda s: s.get("cache_warmup")),
        ("warmup", [s.get("warmup") for s in rows], lambda s: s.get("warmup")),
        ("concurrency", [s.get("concurrency") for s in rows], lambda s: s.get("concurrency")),
        ("duration", [_duration_of(s) for s in rows], _duration_of),
        ("max_tokens", [s.get("max_tokens") for s in rows], lambda s: s.get("max_tokens")),
        ("stream", [s.get("stream") for s in rows], lambda s: s.get("stream")),
        ("chunk_timeline", [s.get("chunk_timeline") for s in rows], lambda s: s.get("chunk_timeline")),
        ("thinking", [_thinking_of(s) for s in rows], _thinking_of),
    ]
    mixed: list[str] = []
    same: list[str] = []
    for name, vals, getter in fields:
        uniq = _uniq(vals)
        if len(uniq) <= 1:
            same.append(f"{name}={_fmt_cond(name, uniq[0] if uniq else None)}")
        else:
            bits = [f"{_label(s)}={_fmt_cond(name, getter(s))}" for s in rows]
            mixed.append(f"- {name}：{'；'.join(bits)}")
    if same:
        lines.append("- 共同条件：")
        for item in same:
            lines.append(f"  - {item}")
    if mixed:
        lines.append("- 不一致项：")
        lines.extend(mixed)
    return lines


def _join_vals(vals: list[Any]) -> str:
    return "、".join("未知" if v is None else str(v) for v in vals)


def _fmt_cond(name: str, v: Any) -> str:
    if name == "duration":
        return _fmt_duration(v)
    if name in ("stream", "thinking", "chunk_timeline"):
        return _fmt_bool(v)
    if name == "est_input_tokens":
        x = _as_float(v)
        return "n/a" if x is None else f"{int(x):,}"
    if v is None:
        return "未知"
    return str(v)


def _conclusion_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["没有可汇总的运行。"]
    ranked = sorted(
        rows,
        key=lambda s: (
            _as_float(s.get("score")) is None,
            -(_as_float(s.get("score")) or 0.0),
        ),
    )
    top = ranked[0]
    lines = [
        (
            f"本批 {len(rows)} 次运行，综合分最高为 {_label(top)}"
            f"（{_fmt_score(top.get('score'))}）。"
        )
    ]
    if len(ranked) >= 2:
        second = ranked[1]
        lines.append(
            f"其次为 {_label(second)}（{_fmt_score(second.get('score'))}）。"
        )
    if any(_burst_decode(s) for s in rows):
        lines.append("本批存在突发 Decode TPS，分析综合分已去掉该项后重算。")
    failed = [
        s
        for s in rows
        if (_as_float(_get(s, "availability", "success_rate")) is not None)
        and float(_get(s, "availability", "success_rate")) < 1.0
    ]
    for s in failed:
        sr = _get(s, "availability", "success_rate")
        suc = _get(s, "availability", "success")
        tot = _get(s, "availability", "total")
        extra = ""
        if suc is not None and tot is not None:
            extra = f"，{suc}/{tot}"
        lines.append(f"{_label(s)} 成功率为 {_pct(sr)}{extra}。")
        if len(lines) >= 8:
            return lines[:8]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for s in rows:
        by_model.setdefault(str(s.get("model")), []).append(s)
    for model, group in by_model.items():
        providers = _uniq([s.get("provider") for s in group])
        if len(providers) < 2:
            continue
        ttft_w = _winners(group, [_get(s, "ttft", "p50") for s in group], "low")
        e2e_w = _winners(group, [_get(s, "e2e", "p50") for s in group], "low")
        rpm_w = _winners(group, [_get(s, "throughput", "rpm") for s in group], "high")
        bits: list[str] = []
        if ttft_w:
            bits.append(
                f"TTFT p50 最低 {_names(ttft_w)}"
                f"（{_fmt_sec(_get(ttft_w[0], 'ttft', 'p50'))}）"
            )
        if e2e_w:
            bits.append(
                f"E2E p50 最低 {_names(e2e_w)}"
                f"（{_fmt_sec(_get(e2e_w[0], 'e2e', 'p50'))}）"
            )
        if rpm_w:
            bits.append(
                f"RPM 最高 {_names(rpm_w)}"
                f"（{_fmt_qty(_get(rpm_w[0], 'throughput', 'rpm'))}）"
            )
        if bits:
            lines.append(f"同模型 {model}：{'；'.join(bits)}。")
        if len(lines) >= 8:
            return lines[:8]
    return lines[:8]


def _rank_table(rows: list[dict[str, Any]]) -> list[str]:
    ranked = sorted(
        rows,
        key=lambda s: (
            _as_float(s.get("score")) is None,
            -(_as_float(s.get("score")) or 0.0),
        ),
    )
    headers = [
        "provider",
        "model",
        "score",
        "success_rate",
        "ttft_p50",
        "e2e_p50",
        "decode_p50",
        "eff_tps_p50",
        "cache_hit_p50",
        "cache_req_hit",
        "rpm",
        "output_tpm",
    ]
    specs: list[tuple[str, Any, str | None]] = [
        ("provider", lambda s: s.get("provider"), None),
        ("model", lambda s: s.get("model"), None),
        ("score", lambda s: s.get("score"), "high"),
        ("success_rate", lambda s: _get(s, "availability", "success_rate"), "high"),
        ("ttft_p50", lambda s: _get(s, "ttft", "p50"), "low"),
        ("e2e_p50", lambda s: _get(s, "e2e", "p50"), "low"),
        ("decode_p50", lambda s: _get(s, "decode_tps", "p50"), "high"),
        ("eff_tps_p50", lambda s: _get(s, "effective_tps", "p50"), "high"),
        ("cache_hit_p50", lambda s: _get(s, "cache", "hit_rate_p50"), "high"),
        ("cache_req_hit", lambda s: _get(s, "cache", "hit_request_rate"), "high"),
        ("rpm", lambda s: _get(s, "throughput", "rpm"), "high"),
        ("output_tpm", lambda s: _get(s, "throughput", "output_tpm"), "high"),
    ]
    raw = {name: [fn(s) for s in ranked] for name, fn, _d in specs}
    best: dict[str, set[int]] = {}
    for name, _fn, direction in specs:
        if direction:
            best[name] = _best_indices(raw[name], direction)

    def cell(name: str, i: int) -> str:
        v = raw[name][i]
        if name in ("provider", "model"):
            text = "" if v is None else str(v)
        elif name == "score":
            text = _fmt_score(v)
        elif name == "success_rate":
            text = _pct(v)
        elif name in ("ttft_p50", "e2e_p50"):
            text = _fmt_sec(v)
        elif name in ("decode_p50", "eff_tps_p50"):
            text = _fmt_tps(v)
        elif name in ("cache_hit_p50", "cache_req_hit"):
            text = _pct(v)
        else:
            text = _fmt_qty(v)
        if name in best and i in best[name] and text and text != "n/a":
            text = f"**{text}**"
        return text

    body = [[cell(h, i) for h in headers] for i in range(len(ranked))]
    return _md_table(headers, body)


def _dimension_section(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    sr_vals = [_get(s, "availability", "success_rate") for s in rows]
    sr_w = _winners(rows, sr_vals, "high")
    ttft_w = _winners(rows, [_get(s, "ttft", "p50") for s in rows], "low")
    e2e_w = _winners(rows, [_get(s, "e2e", "p50") for s in rows], "low")
    rpm_w = _winners(rows, [_get(s, "throughput", "rpm") for s in rows], "high")
    tpm_w = _winners(rows, [_get(s, "throughput", "output_tpm") for s in rows], "high")
    agg_w = _winners(
        rows, [_get(s, "throughput", "aggregate_output_tps") for s in rows], "high"
    )
    cv_w = _winners(rows, [_get(s, "stability", "ttft_cv") for s in rows], "low")
    uniq_sr = _uniq(sr_vals)
    if len(uniq_sr) <= 1:
        lines.append(f"- 可用性：全部成功率 {_pct(uniq_sr[0] if uniq_sr else None)}")
    elif sr_w:
        lines.append(
            f"- 可用性：{_names(sr_w)}"
            f"（成功率 {_pct(_get(sr_w[0], 'availability', 'success_rate'))}）"
        )
    if ttft_w or e2e_w:
        bits = []
        if ttft_w:
            bits.append(
                f"TTFT p50 {_names(ttft_w)} {_fmt_sec(_get(ttft_w[0], 'ttft', 'p50'))}"
            )
        if e2e_w:
            bits.append(
                f"E2E p50 {_names(e2e_w)} {_fmt_sec(_get(e2e_w[0], 'e2e', 'p50'))}"
            )
        lines.append("- 延迟：" + "；".join(bits))
    if rpm_w or tpm_w or agg_w:
        bits = []
        if rpm_w:
            bits.append(f"RPM {_names(rpm_w)} {_fmt_qty(_get(rpm_w[0], 'throughput', 'rpm'))}")
        if tpm_w:
            bits.append(
                f"output TPM {_names(tpm_w)} {_fmt_qty(_get(tpm_w[0], 'throughput', 'output_tpm'))}"
            )
        if agg_w:
            bits.append(
                f"aggregate TPS {_names(agg_w)} "
                f"{_fmt_tps(_get(agg_w[0], 'throughput', 'aggregate_output_tps'))}"
            )
        lines.append("- 吞吐：" + "；".join(bits))
    if cv_w:
        lines.append(
            f"- 稳定性：TTFT CV 最低 {_names(cv_w)}"
            f"（{_fmt_qty(_get(cv_w[0], 'stability', 'ttft_cv'), 3)}）"
        )
    if not lines:
        lines.append("- 本批缺少可比较的分维指标。")
    return lines


def _same_model_section(rows: list[dict[str, Any]]) -> list[str]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for s in rows:
        by_model.setdefault(str(s.get("model")), []).append(s)
    comparable = [
        (model, group)
        for model, group in by_model.items()
        if len(_uniq([s.get("provider") for s in group])) >= 2
    ]
    if not comparable:
        return ["本批没有同一模型覆盖两家及以上供应商。"]
    lines: list[str] = []
    for model, group in comparable:
        providers = _join_vals(_uniq([s.get("provider") for s in group]))
        lines.append(f"### {model}")
        lines.append("")
        lines.append(f"供应商：{providers}。")
        lines.append("")
        headers = [
            "provider",
            "score",
            "success_rate",
            "ttft_p50",
            "e2e_p50",
            "eff_tps_p50",
            "rpm",
        ]
        body: list[list[str]] = []
        for s in group:
            body.append(
                [
                    str(s.get("provider") or ""),
                    _fmt_score(s.get("score")),
                    _pct(_get(s, "availability", "success_rate")),
                    _fmt_sec(_get(s, "ttft", "p50")),
                    _fmt_sec(_get(s, "e2e", "p50")),
                    _fmt_tps(_get(s, "effective_tps", "p50")),
                    _fmt_qty(_get(s, "throughput", "rpm")),
                ]
            )
        lines.extend(_md_table(headers, body))
        ttft_w = _winners(group, [_get(s, "ttft", "p50") for s in group], "low")
        e2e_w = _winners(group, [_get(s, "e2e", "p50") for s in group], "low")
        rpm_w = _winners(group, [_get(s, "throughput", "rpm") for s in group], "high")
        bits: list[str] = []
        if ttft_w:
            bits.append(f"TTFT 更低的是 {_names(ttft_w)}")
        if e2e_w:
            bits.append(f"E2E 更低的是 {_names(e2e_w)}")
        if rpm_w:
            bits.append(f"RPM 更高的是 {_names(rpm_w)}")
        if bits:
            lines.append("")
            lines.append("；".join(bits) + "。")
        lines.append("")
    return lines


def _cache_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = [
        "- 缓存命中率仅统计正式 `run` 阶段；`cache_warmup` / `warmup` 不计入 summary。",
        "- `cache_mode=warmup` 时，顺序为：cache_warmup → warmup → run。",
        "- 命中率分两种维度：req_hit=请求维度（cache_read>0 的请求占比），"
        "hit=Token 维度（每请求 cache_read/input 的统计）。",
    ]
    for s in rows:
        cache = s.get("cache") or {}
        reported = cache.get("reported_requests", 0)
        total = _get(s, "availability", "success") or 0
        hit_requests = cache.get("hit_requests", 0)
        mode = s.get("cache_mode") or "record"
        cw = s.get("cache_warmup")
        extra = f"，cache_warmup={cw}" if mode == "warmup" and cw is not None else ""
        lines.append(
            f"- {_label(s)}（{mode}{extra}）："
            f"reported {reported}/{total}，"
            f"req_hit={_pct(cache.get('hit_request_rate'))}（{hit_requests}/{total}），"
            f"hit avg={_pct(cache.get('hit_rate_avg'))}，"
            f"p50={_pct(cache.get('hit_rate_p50'))}，"
            f"p95={_pct(cache.get('hit_rate_p95'))}，"
            f"cache_read avg={_n(cache.get('cache_read_avg'), 0)}"
        )
    return lines


def _result_dir_of(summary: dict[str, Any]) -> Path | None:
    raw = summary.get("_path")
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_file():
        return p.parent
    if p.is_dir() and (p / "summary.json").is_file():
        return p
    if p.is_dir():
        return p
    return None


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _successful_run_rows(requests_path: Path) -> list[dict[str, str]]:
    return [
        r
        for r in _load_csv_rows(requests_path)
        if r.get("phase") == "run"
        and r.get("http_status") == "200"
        and not (r.get("error_type") or "").strip()
    ]


def _run_chunk_timing(
    chunks: list[dict[str, str]],
    *,
    timeline_collected: bool,
) -> dict[str, Any]:
    if not timeline_collected:
        return {"timeline_collected": False}
    timeline_truncated = any(c.get("event_type") == "timeline_cap" for c in chunks)
    content = [c for c in chunks if c.get("event_type") == "content"]
    stream_end = next((c for c in chunks if c.get("event_type") == "stream_end"), None)
    # sse_meta: non-content SSE data frames (role, usage, empty delta, etc.), not comment pings.
    sse_meta = [c for c in chunks if c.get("event_type") == "sse_meta"]
    first_content_ms: float | None = None
    post_first_ms: float | None = None
    sub_delta_avg_ms: float | None = None
    sub_delta_max_ms: float | None = None
    if content:
        first_content_ms = float(content[0]["offset_ms"])
        if stream_end:
            post_first_ms = float(stream_end["offset_ms"]) - first_content_ms
        if len(content) > 1:
            offsets = [float(c["offset_ms"]) for c in content]
            deltas = [offsets[i] - offsets[i - 1] for i in range(1, len(offsets))]
            sub_delta_avg_ms = sum(deltas) / len(deltas)
            sub_delta_max_ms = max(deltas)
    pre_first_non_content = 0
    if first_content_ms is not None:
        pre_first_non_content = sum(
            1 for c in sse_meta if float(c["offset_ms"]) < first_content_ms
        )
    return {
        "timeline_collected": True,
        "timeline_truncated": timeline_truncated,
        "content_chunks": len(content),
        "first_content_s": first_content_ms / 1000.0 if first_content_ms is not None else None,
        "post_first_ms": post_first_ms,
        "sub_delta_avg_ms": sub_delta_avg_ms,
        "sub_delta_max_ms": sub_delta_max_ms,
        "pre_first_non_content_frames": pre_first_non_content,
    }


def _streaming_pattern_kind(timing: dict[str, Any]) -> str:
    if not timing.get("timeline_collected", True):
        return "未采集 chunk 时间线"
    n = int(timing.get("content_chunks") or 0)
    post = timing.get("post_first_ms")
    if n == 0:
        return "无 content chunk"
    if n == 1 and post is not None and post < 50:
        return "整块输出（首响后一次性到齐）"
    if n > 1:
        return "增量流式输出"
    return f"{n} 块 content"


def _streaming_pattern_detail(timing: dict[str, Any]) -> str:
    n = int(timing.get("content_chunks") or 0)
    post = timing.get("post_first_ms")
    if n == 0:
        return "无 content chunk"
    if n == 1 and post is not None and post < 50:
        return "整块输出（首响后一次性到齐）"
    if n > 1:
        avg = timing.get("sub_delta_avg_ms")
        avg_s = f"，块间隔 avg {avg:.0f}ms" if avg is not None else ""
        post_s = f"{post:.0f}ms" if post is not None else "n/a"
        return f"增量输出（{n} 块，首响后 {post_s}{avg_s}）"
    return f"{n} 块 content"


def _fmt_ms(v: Any) -> str:
    x = _as_float(v)
    if x is None:
        return "n/a"
    if x < 1000:
        return f"{x:.1f}ms"
    return f"{x / 1000:.2f}s"


def _ttft_deep_section(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["无运行。"]
    lines: list[str] = [
        "基于 `summary.json` 与 `requests.csv`（及可选 `chunks.csv`）的 **run 阶段**逐请求分析。",
        "",
        "### TTFT 分位数（summary）",
        "",
    ]
    pct_headers = ["provider", "model", "p50", "p95", "p99", "avg", "CV"]
    pct_body: list[list[str]] = []
    for s in rows:
        ttft = s.get("ttft") if isinstance(s.get("ttft"), dict) else {}
        st = s.get("stability") if isinstance(s.get("stability"), dict) else {}
        pct_body.append(
            [
                str(s.get("provider") or ""),
                str(s.get("model") or ""),
                _fmt_sec(ttft.get("p50")),
                _fmt_sec(ttft.get("p95")),
                _fmt_sec(ttft.get("p99")),
                _fmt_sec(ttft.get("avg")),
                _fmt_qty(st.get("ttft_cv"), 3),
            ]
        )
    lines.extend(_md_table(pct_headers, pct_body))

    by_model: dict[str, list[dict[str, Any]]] = {}
    for s in rows:
        by_model.setdefault(str(s.get("model")), []).append(s)
    cross_bits: list[str] = []
    for model, group in by_model.items():
        valid = [g for g in group if _as_float(_get(g, "ttft", "p50")) is not None]
        if len(_uniq([g.get("provider") for g in valid])) < 2:
            continue
        ttft_w = _winners(valid, [_get(g, "ttft", "p50") for g in valid], "low")
        if not ttft_w:
            continue
        best = ttft_w[0]
        worst = max(valid, key=lambda g: _as_float(_get(g, "ttft", "p50")) or 0.0)
        best_p = _as_float(_get(best, "ttft", "p50"))
        worst_p = _as_float(_get(worst, "ttft", "p50"))
        if best_p is not None and worst_p is not None:
            if best_p > 0 and worst_p >= best_p:
                ratio = worst_p / best_p
                gap = f"差距约 **{ratio:.1f}×**"
            else:
                gap = "倍率无法计算（存在零或缺失分母）"
            cross_bits.append(
                f"- **{model}**：TTFT p50 最低 {_label(best)}（{_fmt_sec(best_p)}），"
                f"最高 {_label(worst)}（{_fmt_sec(worst_p)}），{gap}"
            )
    if cross_bits:
        lines.extend(["", "### 跨供应商 TTFT 对比", "", *cross_bits])

    return lines


def _streaming_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for s in rows:
        dec = _as_float(_get(s, "decode_tps", "p50"))
        eff = _as_float(_get(s, "effective_tps", "p50"))
        ttft = _as_float(_get(s, "ttft", "p50"))
        e2e = _as_float(_get(s, "e2e", "p50"))
        bits = [
            f"TTFT p50={_fmt_sec(ttft)}",
            f"E2E p50={_fmt_sec(e2e)}",
            f"Decode p50={_fmt_tps(dec)}",
            f"Effective p50={_fmt_tps(eff)}",
        ]
        tag = ""
        if _burst_decode(s):
            tag = "，**疑似伪流式/突发到齐**"
        lines.append(f"- {_label(s)}：{'；'.join(bits)}{tag}")
    if not lines:
        return ["无运行。"]
    return lines


def _compare_table_section(rows: list[dict[str, Any]]) -> list[str]:
    headers = [f[0] for f in COMPARE_FIELDS]
    raw: dict[str, list[Any]] = {h: [_cell(r, k) for r in rows] for h, k, _ in COMPARE_FIELDS}
    best: dict[str, set[int]] = {}
    for h, _k, direction in COMPARE_FIELDS:
        if direction:
            best[h] = _best_indices(raw[h], direction)

    def cell(name: str, i: int) -> str:
        v = raw[name][i]
        if name in ("provider", "model"):
            text = "" if v is None else str(v)
        elif name == "success_rate":
            text = _pct(v)
        elif name in ("ttft_p50", "ttft_p95", "ttft_p99", "e2e_p50"):
            text = _fmt_sec(v)
        elif name in ("cache_hit_p50", "cache_req_hit"):
            text = _pct(v)
        elif name in ("decode_tps_p1", "decode_tps_p50", "eff_tps_p50", "agg_out_tps"):
            text = _fmt_tps(v)
        elif name == "score":
            text = _fmt_score(v)
        elif name in ("rpm", "output_tpm", "ttft_cv"):
            text = _fmt_qty(v)
        else:
            text = "" if v is None else str(v)
        if name in best and i in best[name] and text and text != "n/a":
            text = f"**{text}**"
        return text

    body = [[cell(h, i) for h in headers] for i in range(len(rows))]
    return _md_table(headers, body)


def _path_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for s in rows:
        path = s.get("_path")
        if path:
            lines.append(f"- {_label(s)}：`{path}`")
    return lines or ["无路径信息。"]


def _detail_section(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "provider",
        "model",
        "prompt",
        "conc",
        "success_rate",
        "ttft_p50",
        "e2e_p50",
        "decode_p50",
        "eff_p50",
        "cache_hit_p50",
        "cache_req_hit",
        "rpm",
    ]
    walls = [_get(s, "throughput", "wall_seconds") for s in rows]
    show_wall = any(w is not None for w in walls)
    if show_wall:
        headers.append("wall")
    body: list[list[str]] = []
    for s, wall in zip(rows, walls, strict=True):
        conc = s.get("concurrency")
        row = [
            str(s.get("provider") or ""),
            str(s.get("model") or ""),
            str(s.get("prompt") or s.get("prompt_size") or ""),
            "" if conc is None else str(conc),
            _pct(_get(s, "availability", "success_rate")),
            _fmt_sec(_get(s, "ttft", "p50")),
            _fmt_sec(_get(s, "e2e", "p50")),
            _fmt_tps(_get(s, "decode_tps", "p50")),
            _fmt_tps(_get(s, "effective_tps", "p50")),
            _pct(_get(s, "cache", "hit_rate_p50")),
            _pct(_get(s, "cache", "hit_request_rate")),
            _fmt_qty(_get(s, "throughput", "rpm")),
        ]
        if show_wall:
            row.append(_fmt_duration(wall))
        body.append(row)
    return _md_table(headers, body)


def format_analysis_markdown(
    summaries: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> str:
    rows = [dict(s) for s in summaries]
    if rows:
        _apply_analysis_scores(rows, weights)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        "# llm-bench 对比分析",
        "",
        f"生成时间：{now}  运行次数：{len(rows)}",
        "",
        "## 测试条件",
        "",
    ]
    lines.extend(_cond_lines(rows))
    lines.extend(["", "## 缓存与预热", ""])
    lines.extend(_cache_lines(rows) if rows else ["无运行。"])
    lines.extend(["", "## 结论摘要", ""])
    lines.extend(_conclusion_lines(rows))
    lines.extend(["", "## 综合排名", ""])
    if rows:
        lines.extend(_rank_table(rows))
    else:
        lines.append("无运行。")
    lines.extend(["", "## 分维度", ""])
    if rows:
        lines.extend(_dimension_section(rows))
    else:
        lines.append("无运行。")
    lines.extend(["", "## 流式与解码特征", ""])
    lines.extend(_streaming_lines(rows) if rows else ["无运行。"])
    lines.extend(["", "## TTFT 深度分析", ""])
    lines.extend(_ttft_deep_section(rows) if rows else ["无运行。"])
    lines.extend(["", "## 全量指标对比", ""])
    if rows:
        lines.extend(_compare_table_section(rows))
    else:
        lines.append("无运行。")
    lines.extend(["", "## 同模型跨供应商", ""])
    lines.extend(_same_model_section(rows))
    lines.extend(["", "## 各次运行明细", ""])
    if rows:
        lines.extend(_detail_section(rows))
    else:
        lines.append("无运行。")
    lines.extend(["", "## 结果目录", ""])
    lines.extend(_path_lines(rows) if rows else ["无运行。"])
    lines.extend(
        [
            "",
            "## 读数说明",
            "",
            "- **TTFT**：排队与预填到首 token，不是解码速度。",
            "- **TTFT 深度分析**：run 阶段逐请求的 TTFT、首响后传输耗时与 content 块间隔；需 `chunks.csv` 时请在交互中开启 chunk 时间线。",
            "- **sse_meta**：非正文 SSE 数据帧（role、usage、空 delta 等），不是注释心跳 `: ping`（后者当前未写入时间线）。",
            "- **Decode TPS**：首 token 之后的生成速度，不含 TTFT。",
            "- **Effective TPS**：含 TTFT 的端到端 token/s，更接近用户体感。",
            "- **Aggregate / RPM / TPM**：窗口总吞吐，不是单流速度。",
            "- 综合分是本批内 min-max 加权，不是绝对质量分。",
        ]
    )
    bursts = [s for s in rows if _burst_decode(s)]
    if bursts:
        names = "、".join(_label(s) for s in bursts)
        lines.append(
            f"- **突发到达**：{names} 的 Decode TPS 远高于 Effective TPS，"
            "且 TTFT 接近 E2E，token 在等待后短时到齐。"
            "Decode TPS 不能代表用户体感速度，排名以 TTFT / E2E / Effective TPS / RPM / 成功率为准。"
        )
    return "\n".join(lines) + "\n"


def _step_summary_sort_key(path: Path) -> tuple[int, int | str]:
    name = path.parent.name
    if len(name) > 1 and name[0] == "c" and name[1:].isdigit():
        return (0, int(name[1:]))
    return (1, name)


def resolve_summary_files(p: Path) -> list[Path]:
    p = Path(p)
    if p.is_file():
        return [p]
    direct = p / "summary.json"
    if direct.is_file():
        return [direct]
    if (p / "steps.json").is_file():
        found = sorted(p.glob("c*/summary.json"), key=_step_summary_sort_key)
        if found:
            return found
    raise FileNotFoundError(f"no summary.json at {p}")


def compare_paths(
    paths: Iterable[Path],
    *,
    output: Path,
    weights_path: Path | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    for p in paths:
        for summary_file in resolve_summary_files(p):
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            _fill_from_config(data, summary_file)
            data["_path"] = str(summary_file.parent if summary_file.name == "summary.json" else summary_file)
            summaries.append(data)
    weights = load_score_weights(weights_path)
    scores = compute_scores(summaries, weights)
    for s, sc in zip(summaries, scores, strict=True):
        s["score"] = sc["score"]
        s["score_components"] = sc["components"]
    md = format_compare_markdown(summaries)
    analysis = format_analysis_markdown(summaries, weights=weights)
    output.mkdir(parents=True, exist_ok=True)
    (output / "compare.md").write_text(md, encoding="utf-8")
    (output / "analysis.md").write_text(analysis, encoding="utf-8")
    csv_path = output / "compare.csv"
    fieldnames = [f[0] for f in COMPARE_FIELDS] + ["path"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            row = {name: _cell(s, key) for name, key, _ in COMPARE_FIELDS}
            row["path"] = s.get("_path")
            w.writerow(row)
    print(md)
    print("Weights:", weights)
    for s in summaries:
        print(
            f"  {s.get('provider')} {s.get('model')} score={s['score']:.2f} "
            f"components={ {k: round(v, 4) for k, v in s['score_components'].items()} }"
        )
    return md, summaries
