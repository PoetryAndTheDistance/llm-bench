"""压测指标广播器 — 收集请求结果，向 Web UI 提供实时快照"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from llm_bench.metrics import RequestResult, percentile

# 实时指标使用的分位数
# 选择理由：P1/P99 关注尾部延迟，P25/P75 反映分布宽度，P50 是中位数基准
_PCTS = (1, 25, 50, 75, 95, 99)

# 阶段先后顺序（多个 target 并发时，取所有 target 中最靠后的阶段）
_PHASE_ORDER = {"cache_warmup": 1, "warmup": 2, "run": 3}
# 任务进行中的阶段：仅这些阶段允许趋势历史继续追加时序点
_ACTIVE_STAGES = frozenset({"starting", "cache_warmup", "warmup", "run"})
# 任务终态：进入后补最后一个结束点并冻结历史，时间轴不再延伸
_TERMINAL_STAGES = frozenset({"finished", "stopped", "error"})
_PHASE_LABEL = {
    "starting": "准备中",
    "cache_warmup": "缓存预热中",
    "warmup": "预热请求中",
    "run": "正式压测中",
    "finished": "压测结束",
    "stopped": "已停止",
    "error": "执行异常",
    "idle": "空闲",
}


class MetricsBroadcaster:
    """将压测指标暴露给 Web UI

    在 runner.py 中使用，每次请求完成后调用 record()。
    Web UI 通过 WebSocket 定时获取 snapshot()。
    """

    def __init__(self):
        self._records: list[RequestResult] = []
        self._lock = asyncio.Lock()
        self._history: list[dict] = []
        self._running = False
        self._config: dict[str, Any] = {}
        self._stop_event = asyncio.Event()
        # 任务级状态（阶段 / 进度）
        self._job: dict[str, Any] = {"stage": "idle"}
        # 本次任务产出的报告：[{"id": 结果目录名, "provider": ..., "model": ...}]
        self._results: list[dict[str, Any]] = []
        # 每个 provider/model 的首个请求留证（完整请求/返回 body，仅用于 Web 实时校验）：
        # {f"{provider}/{model}": exchange}。体量大（prompt 可能很长），
        # 不进每秒 WebSocket 帧，只在前端展开时经 HTTP 按需取。
        self._first_exchanges: dict[str, dict[str, Any]] = {}
        # 3.11: reset 时的客户端状态重置回调列表（由外部注册）
        self._on_reset_callbacks: list[Callable[[], None]] = []
        # 3.3: 增量快照缓存 — 记录上次快照时的 record 数量，仅处理新增记录
        self._last_snapshot_count: int = 0
        # 3.3: 分组索引缓存 — 避免每次全量重建分组
        self._cached_groups: dict[str, list[RequestResult]] = {}
        # 趋势历史已冻结（任务进入终态并补完结束点后置 True，reset 前不再追加）
        self._history_closed = False

    # 首请求留证中阶段码对应的中文标签（与指标页阶段措辞保持一致）
    _EXCHANGE_PHASE_LABEL = {
        "cache_warmup": "缓存预热",
        "warmup": "预热",
        "run": "正式压测",
    }

    def init_job(
        self,
        *,
        mode: str,
        total_requests: int | None,
        duration: float | None,
        target_count: int,
        targets: list[dict[str, str]] | None = None,
        initial_stage: str = "starting",
        job_id: str | None = None,
    ) -> None:
        """任务开始时初始化任务状态"""
        # 新任务解除上一轮终态的历史冻结（正常流程 start_job 会先 reset，这里双保险）
        self._history_closed = False
        self._job = {
            "stage": initial_stage,
            "job_id": job_id,  # 结果任务目录名（results/<job_id>）
            "targets": targets or [],  # [{"provider","model"}]，用于预置分组行
            "mode": mode,  # requests / duration
            "total_requests": total_requests,  # 所有 target 合计的总请求数
            "duration": duration,
            "target_count": target_count,
            "started_perf": time.perf_counter(),
            "run_started_perf": None,  # 第一条正式压测请求完成的时刻
        }

    def finish_job(self, stage: str = "finished") -> None:
        job = dict(self._job)
        # 冻结结束时刻（perf 计时用于 elapsed，墙钟用于趋势图最后一个点的时间戳），
        # 避免任务结束后耗时继续增长、趋势时间轴继续向右延伸
        end_perf = time.perf_counter()
        job["ended_perf"] = end_perf
        job["ended_time"] = time.time()
        # 冻结结束时的进度，避免停止后 elapsed 继续增长导致进度条归零/回退
        if job.get("mode") == "duration":
            if stage == "finished":
                job["progress_override"] = 1.0
            elif job.get("run_started_perf") and job.get("duration"):
                job["progress_override"] = min(
                    (end_perf - job["run_started_perf"]) / float(job["duration"]), 1.0
                )
        self._job = {**job, "stage": stage}

    def set_results(self, results: list[dict[str, Any]]) -> None:
        """任务结束时登记产出的报告（供 Web UI 展示查看/下载/飞书操作）"""
        self._results = list(results)

    async def record_first_exchange(self, exchange: dict[str, Any]) -> None:
        """留证某个 provider/model 的首个请求（完整请求/返回 body）

        由客户端在本模型第一次 complete() 结束后回调；同一 key 只保留最早一次。
        exchange 至少含 provider/model/phase，body 字段体量可能很大。
        """
        key = f"{exchange.get('provider')}/{exchange.get('model')}"
        async with self._lock:
            if key in self._first_exchanges:
                return
            self._first_exchanges[key] = exchange

    async def first_exchange_index(self) -> dict[str, dict[str, Any]]:
        """首请求留证的轻量索引（剔除 body），随每秒 WebSocket 帧下发"""
        async with self._lock:
            return {
                key: {
                    "provider": ex.get("provider"),
                    "model": ex.get("model"),
                    "phase": ex.get("phase"),
                    "phase_label": self._EXCHANGE_PHASE_LABEL.get(ex.get("phase"), ex.get("phase")),
                    "http_status": ex.get("http_status"),
                    "success": ex.get("success"),
                    "streamed": ex.get("streamed"),
                    "error_type": ex.get("error_type"),
                    "error_message": ex.get("error_message"),
                    "start_time": ex.get("start_time"),
                    "end_time": ex.get("end_time"),
                    "has_request": ex.get("request_body") is not None,
                    "has_response": bool(ex.get("response_text")),
                }
                for key, ex in self._first_exchanges.items()
            }

    async def get_first_exchange(self, provider: str, model: str) -> dict[str, Any] | None:
        """按 provider/model 取完整首请求留证（含 body），供 HTTP 按需拉取"""
        key = f"{provider}/{model}"
        async with self._lock:
            ex = self._first_exchanges.get(key)
            return dict(ex) if ex is not None else None

    async def all_first_exchanges(self) -> dict[str, dict[str, Any]]:
        """全部首请求留证（含完整请求/返回 body），供任务收口时冻结进页面快照"""
        async with self._lock:
            return {key: dict(ex) for key, ex in self._first_exchanges.items()}

    def job_snapshot(self) -> dict[str, Any]:
        """任务级状态：当前阶段 + 进度（0~1）"""
        job = dict(self._job)
        stage = job.get("stage", "idle")
        run_records = [r for r in self._records if r.phase == "run"]
        done = len(run_records)
        success = sum(1 for r in run_records if r.success)
        job.update(
            stage=stage,
            stage_label=_PHASE_LABEL.get(stage, stage),
            done=done,
            success=success,
            failed=done - success,
        )
        progress: float | None = None
        now = time.perf_counter()
        # 终态耗时以冻结的结束时刻计算，避免结束后 elapsed_s 随墙钟继续增长
        if stage in _TERMINAL_STAGES and job.get("ended_perf"):
            now = job["ended_perf"]
        if job.get("progress_override") is not None:
            progress = job["progress_override"]
        elif stage == "finished" and job.get("mode") == "requests":
            progress = 1.0
        elif job.get("mode") == "requests":
            total = job.get("total_requests") or 0
            progress = min(done / total, 1.0) if total else None
        elif job.get("mode") == "duration":
            run_start = job.get("run_started_perf")
            duration = job.get("duration")
            if stage == "run" and run_start and duration:
                progress = min((now - run_start) / float(duration), 1.0)
            else:
                progress = 0.0
        job["progress"] = progress
        if job.get("run_started_perf"):
            job["elapsed_s"] = round(now - job["run_started_perf"], 1)
        else:
            job["elapsed_s"] = round(now - job["started_perf"], 1) if job.get("started_perf") else 0
        job["results"] = list(self._results)
        return job

    async def record(self, rec: RequestResult) -> None:
        """记录一条请求结果（含预热/缓存预热记录，用于阶段判断与进度）"""
        async with self._lock:
            self._records.append(rec)
            job = self._job
            order = _PHASE_ORDER.get(rec.phase, 0)
            cur = _PHASE_ORDER.get(job.get("stage"), 0)
            if order > cur:
                job["stage"] = rec.phase
            if rec.phase == "run" and job.get("run_started_perf") is None:
                job["run_started_perf"] = time.perf_counter()

    # 趋势历史点只保留图表用到的指标（逐秒一个点、任务全程保留），
    # 以控制 WebSocket 增量帧与离线 report.html 的体积
    @staticmethod
    def _slim_group(g: dict[str, Any]) -> dict[str, Any]:
        return {
            "ttft": {"p50": (g.get("ttft") or {}).get("p50")},
            "e2e": {"p50": (g.get("e2e") or {}).get("p50")},
            "decode_tps": {"p50": (g.get("decode_tps") or {}).get("p50")},
            "prefill_tps": {"p50": (g.get("prefill_tps") or {}).get("p50")},
            "effective_tps": {"p50": (g.get("effective_tps") or {}).get("p50")},
            "rpm": g.get("rpm"),
            "input_tpm": g.get("input_tpm"),
            "output_tpm": g.get("output_tpm"),
        }

    async def snapshot_with_history(self) -> dict[str, dict[str, Any]]:
        """广播循环按固定间隔调用：计算一次分组快照，同时追加时序点

        一次聚合同时服务快照推送与趋势图，避免重复计算。
        历史全程保留（活跃期每秒 1 点，reset 随任务清空），不做点数截断。
        任务进入终态时只补最后一个结束点（时间戳取冻结的结束时刻），
        之后历史冻结，广播循环继续调用也不会让趋势图时间轴向右延伸。
        """
        async with self._lock:
            grouped = self._grouped_locked()
            self._append_history_locked(grouped)
            return grouped

    def _append_history_locked(self, grouped: dict[str, dict[str, Any]]) -> None:
        """按任务阶段决定是否追加趋势点（调用时已持锁）"""
        # 终态：补唯一一个结束点后冻结；离线快照收口与广播循环都会走到这里
        if self._history_closed:
            return
        stage = self._job.get("stage", "idle")
        if stage in _TERMINAL_STAGES:
            # 预置的空分组行不进趋势历史：至少有一条 run 记录才记点
            if any(r.phase == "run" for r in self._records):
                slim = {k: self._slim_group(g) for k, g in grouped.items()}
                self._history.append({"t": self._job.get("ended_time") or time.time(), "data": slim})
            self._history_closed = True
            return
        # 活跃阶段：逐秒追加；idle（reset 后无任务）不追加
        if stage in _ACTIVE_STAGES and any(r.phase == "run" for r in self._records):
            slim = {k: self._slim_group(g) for k, g in grouped.items()}
            self._history.append({"t": time.time(), "data": slim})

    def _grouped_locked(self) -> dict[str, dict[str, Any]]:
        """按 provider/model 分组计算完整指标（调用时已持锁）

        参考 Locust MetricsStore.summary() 的设计，
        每个分组独立计算延迟分位数、TPS、RPM、Token、缓存等。
        RPM/TPM 基于请求自身的时间跨度（首个请求开始 ~ 最后一个请求结束），
        避免压测刚开始时因墙钟时间过短导致数值虚高。

        3.3 性能优化：使用增量分组缓存，仅处理上次快照后新增的记录。
        """
        # 增量处理：仅处理 _last_snapshot_count 之后的新记录
        new_records = self._records[self._last_snapshot_count:]
        has_new = len(new_records) > 0

        if has_new or not self._cached_groups:
            # 首次或有新记录时，更新分组缓存
            if not self._cached_groups:
                # 首次构建：全量处理
                self._cached_groups = {}
                # 预置目标分组行
                for t in (self._job.get("targets") or []):
                    self._cached_groups.setdefault(f"{t['provider']}/{t['model']}", [])
                for r in self._records:
                    if r.phase == "run":
                        key = f"{r.provider}/{r.model}"
                        self._cached_groups.setdefault(key, []).append(r)
            else:
                # 增量追加：只处理新记录
                for r in new_records:
                    if r.phase == "run":
                        key = f"{r.provider}/{r.model}"
                        self._cached_groups.setdefault(key, []).append(r)
            self._last_snapshot_count = len(self._records)

        groups = self._cached_groups

        # 每个目标当前所处阶段（含预热/缓存预热记录）
        # 注意：phase_seen 仍需全量扫描（预热记录不在 groups 中）
        phase_seen: dict[str, str] = {}
        for r in self._records:
            key = f"{r.provider}/{r.model}"
            cur = phase_seen.get(key)
            if cur is None or _PHASE_ORDER.get(r.phase, 0) > _PHASE_ORDER.get(cur, 0):
                phase_seen[key] = r.phase

        # 全局时间跨度（所有 run 记录）
        run_records = [r for r in self._records if r.phase == "run"]
        if run_records:
            wall = max(r.end_perf for r in run_records) - min(r.start_perf for r in run_records)
        else:
            wall = 0.0
        elapsed_min = max(wall / 60.0, 1e-9)

        target_names = {
            f"{t['provider']}/{t['model']}": (t["provider"], t["model"])
            for t in (self._job.get("targets") or [])
        }

        out: dict[str, dict[str, Any]] = {}
        for key, recs in groups.items():
            ok = [r for r in recs if r.success]
            total = len(recs)
            success_n = len(ok)
            provider_name = recs[0].provider if recs else target_names.get(key, (key, ""))[0]
            model_name = recs[0].model if recs else target_names.get(key, ("", key))[1]
            stage = phase_seen.get(key, "starting")

            def pct(values: list[float]) -> dict[str, float | None]:
                s = sorted(values)
                result = {"avg": sum(s) / len(s) if s else None}
                for p in _PCTS:
                    result[f"p{p}"] = percentile(s, p)
                return result

            ttft_vals = [r.ttft for r in ok if r.ttft is not None]
            e2e_vals = [r.e2e for r in ok if r.e2e is not None]
            decode_vals = [r.decode_tps for r in ok if r.decode_tps is not None]
            prefill_vals = [r.prefill_tps for r in ok if r.prefill_tps is not None]
            eff_vals = [r.effective_tps for r in ok if r.effective_tps is not None]

            in_sum = sum(r.input_tokens or 0 for r in ok)
            out_sum = sum(r.output_tokens or 0 for r in ok)
            total_sum = sum(r.total_tokens or 0 for r in ok)
            l1_sum = sum(r.cache_l1_tokens or 0 for r in ok)
            l2_sum = sum(r.cache_l2_tokens or 0 for r in ok)
            # 请求维度缓存命中：L2 至少读到 1 个缓存 token 的成功请求数
            cache_hit_requests = sum(
                1 for r in ok if (r.cache_read_tokens or 0) > 0
            )

            out[key] = {
                "provider": provider_name,
                "model": model_name,
                "stage": stage,
                "stage_label": _PHASE_LABEL.get(stage, stage),
                "total": total,
                "success": success_n,
                "failed": total - success_n,
                "success_rate": success_n / total if total else 0,
                "ttft": pct(ttft_vals),
                "e2e": pct(e2e_vals),
                "decode_tps": pct(decode_vals),
                "prefill_tps": pct(prefill_vals),
                "effective_tps": pct(eff_vals),
                "rpm": total / elapsed_min,
                "input_tpm": in_sum / elapsed_min,
                "output_tpm": out_sum / elapsed_min,
                "input_tokens": in_sum,
                "output_tokens": out_sum,
                "cache_l1_tokens": l1_sum,
                "cache_l2_tokens": l2_sum,
                # 请求维度命中率：命中请求数 / 成功请求数
                "cache_hit_requests": cache_hit_requests,
                "cache_hit_request_rate": (
                    cache_hit_requests / success_n if success_n else 0
                ),
                # 供应商 usage 中是否真的上报了缓存字段（没有则前端显示"未上报"）
                "cache_l1_reported": any(r.cache_l1_tokens is not None for r in ok),
                "cache_l2_reported": any(r.cache_l2_tokens is not None for r in ok),
                "total_tokens": total_sum,
            }
        return out

    async def snapshot(self) -> dict[str, Any]:
        """返回按 provider/model 分组的完整实时指标"""
        async with self._lock:
            return self._grouped_locked()

    async def token_stats(self) -> dict[str, dict[str, Any]]:
        """返回按 provider/model 分组的指标（与 snapshot 同源）"""
        async with self._lock:
            return self._grouped_locked()

    async def history(self) -> list[dict]:
        """返回时序快照历史"""
        async with self._lock:
            return list(self._history)

    @property
    def status(self) -> str:
        return "running" if self._running else "idle"

    def set_running(self, running: bool) -> None:
        self._running = running

    @property
    def stop_event(self) -> asyncio.Event:
        return self._stop_event

    def request_stop(self) -> None:
        self._stop_event.set()

    def reset(self) -> None:
        """重置所有数据

        同时调用已注册的 on_reset 回调，通知客户端重置状态（如首请求留证标志）。
        """
        self._records.clear()
        self._history.clear()
        self._history_closed = False
        self._stop_event.clear()
        self._running = False
        self._job = {"stage": "idle"}
        self._results = []
        self._first_exchanges = {}
        # 3.3: 重置增量缓存
        self._last_snapshot_count = 0
        self._cached_groups = {}
        # 3.11: 通知客户端重置状态（回调在 reset 之后的 run_level 中重新注册，
        # 通知完即可清空，避免跨任务累积已关闭客户端的回调）
        for callback in self._on_reset_callbacks:
            try:
                callback()
            except Exception:
                pass
        self._on_reset_callbacks = []

    def register_on_reset(self, callback: Callable[[], None]) -> None:
        """注册 reset 时的回调（用于通知客户端重置状态）"""
        self._on_reset_callbacks.append(callback)

    def update_config(self, config: dict[str, Any]) -> None:
        """更新运行时配置"""
        self._config.update(config)

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)
