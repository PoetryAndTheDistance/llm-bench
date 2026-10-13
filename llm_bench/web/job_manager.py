"""压测任务管理器

- 全局同时只允许一个任务运行
- 一个任务可选择多个供应商/模型，它们之间并发执行
- 任务内所有请求收集到同一个 MetricsBroadcaster
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_bench.config import DEFAULT_WEIGHTS_PATH, merge_run_config, resolve_api_key
from llm_bench.runner import run_benchmark
from llm_bench.web.broadcaster import MetricsBroadcaster

LOG = logging.getLogger("llm_bench.web.job_manager")


def _write_job_manifest(job_dir: Path, manifest: dict[str, Any]) -> None:
    """落盘/更新任务清单 job.json（任务元数据与产物索引的唯一真实源）"""
    (job_dir / "job.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


class JobManager:
    """管理单个压测任务（任务内多 provider/model 并发）"""

    def __init__(self, results_dir: Path, providers_manager):
        self.results_dir = results_dir
        self.providers = providers_manager
        self.broadcaster = MetricsBroadcaster()
        self._task: asyncio.Task | None = None
        self._job_info: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def get_status(self) -> dict[str, Any]:
        """返回当前任务状态"""
        if not self.is_running and self._job_info is None:
            return {"status": "idle", "job": None}
        if self.is_running:
            return {"status": "running", "job": self._job_info}
        # 任务已结束，保留最后一次任务信息
        info = dict(self._job_info) if self._job_info else {}
        info["status"] = "completed"
        return {"status": "completed", "job": info}

    async def start_job(self, job_config: dict[str, Any]) -> dict[str, Any]:
        """启动压测任务

        job_config:
            targets: [{"provider": "xxx", "model": "yyy"}, ...]
            prompt, custom_prompt, max_tokens, concurrency, duration, requests,
            stream, enable_thinking, cache_mode, cache_warmup, wait_time, warmup,
            retry, timeout, chunk_timeline, temperature ...
        """
        async with self._lock:
            if self.is_running:
                return {"error": "已有压测任务正在运行，请等待完成或停止后再创建新任务"}

            targets = job_config.get("targets", [])
            if not targets:
                return {"error": "请至少选择一个供应商/模型"}

            # 校验 duration / requests 互斥
            duration = job_config.get("duration")
            requests = job_config.get("requests")
            if duration and requests:
                return {"error": "压测时长和请求数不能同时设置，请二选一"}
            if not duration and not requests:
                return {"error": "请设置压测时长或请求数"}

            # 重置 broadcaster
            self.broadcaster.reset()

            # 创建任务级结果目录：results/job_<stamp>/，每个模型产出独立子目录
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_dir = self.results_dir / f"job_{stamp}"
            if job_dir.exists():  # 同秒兜底（理论上全局单任务锁已避免并发）
                job_dir = job_dir.with_name(
                    job_dir.name
                    + "-"
                    + hashlib.sha1(str(targets).encode("utf-8")).hexdigest()[:6]
                )
            job_dir.mkdir(parents=True, exist_ok=False)
            started_at = datetime.now().isoformat()

            # 任务信息
            self._job_info = {
                "job_id": job_dir.name,
                "job_dir": str(job_dir),
                "started_at": started_at,
                "targets": targets,
                "target_count": len(targets),
                "config": {
                    k: v for k, v in job_config.items() if k != "targets"
                },
            }

            # 初始清单（异常中断时详情页仍可识别该任务及状态）
            _write_job_manifest(job_dir, {
                "job_id": job_dir.name,
                "kind": "job",
                "stage": "starting",
                "started_at": started_at,
                "ended_at": None,
                "config": job_config,
                "targets": [
                    {"provider": t.get("provider"), "model": t.get("model"),
                     "dir": None, "status": "pending", "error": None}
                    for t in targets
                ],
                "artifacts": {"page": False, "analysis": False, "compare_md": False},
            })

            # 启动后台任务
            self._task = asyncio.create_task(self._run_job(job_config, targets, job_dir))
            return {"status": "started", "job": self._job_info}

    async def stop_job(self) -> dict[str, Any]:
        """停止当前任务"""
        if not self.is_running:
            return {"error": "没有正在运行的任务"}
        self.broadcaster.request_stop()
        return {"status": "stopping"}

    async def _run_job(
        self,
        job_config: dict[str, Any],
        targets: list[dict],
        job_dir: Path,
    ) -> None:
        """实际执行任务：多个 provider/model 并发，产物归档到任务目录 job_dir"""
        stage = "error"
        # 任务清单中的目标状态（与 job.json 对应）
        target_entries: list[dict[str, Any]] = [
            {"provider": t.get("provider"), "model": t.get("model"),
             "dir": None, "status": "pending", "error": None}
            for t in targets
        ]

        def _mark_entry(provider_name: str, model: str, status: str, error: str) -> None:
            """标记所有匹配的 pending 目标项

            3.6: 修复原实现只标记第一个匹配项的问题。
            如果同一个 provider/model 在 targets 中出现多次，全部标记。
            """
            for e in target_entries:
                if e["provider"] == provider_name and e["model"] == model and e["status"] == "pending":
                    e["status"] = status
                    e["error"] = error

        try:
            duration = job_config.get("duration")
            requests_total = job_config.get("requests")
            mode = "requests" if requests_total else "duration"

            # 第一遍：校验并收集有效 target，请求总数按 target 均分
            valid_targets: list[tuple[str, str, dict]] = []
            for target in targets:
                provider_name = target.get("provider")
                model = target.get("model")
                provider = self.providers.get_provider(provider_name)
                if provider is None:
                    LOG.error("供应商不存在: %s", provider_name)
                    print(f"[压测] 供应商不存在，跳过: {provider_name}", flush=True)
                    _mark_entry(provider_name, model, "failed", "供应商不存在")
                    continue
                valid_targets.append((provider_name, model, provider))

            if not valid_targets:
                raise RuntimeError("没有有效的压测目标（供应商均不存在）")

            n_targets = len(valid_targets)
            # 请求次数是【所有模型合计】：均分到各 target，余数分给前面的 target
            per_target_requests: dict[str, int] = {}
            if requests_total:
                base, rem = divmod(int(requests_total), n_targets)
                for i, (provider_name, model, _provider) in enumerate(valid_targets):
                    per_target_requests[f"{provider_name}/{model}"] = base + (1 if i < rem else 0)

            print(
                f"\n{'='*70}\n[压测开始] 目标 {n_targets} 个，模式: "
                + (f"按请求数，总请求 {requests_total} 次"
                   if mode == "requests"
                   else f"按时长，{duration} 秒")
                + f"（缓存预热 {job_config.get('cache_warmup', 0)} 次，"
                  f"预热 {job_config.get('warmup', 10)} 次）\n{'='*70}",
                flush=True,
            )
            LOG.info("压测开始: %d 个目标, mode=%s, total_requests=%s, duration=%s",
                     n_targets, mode, requests_total, duration)

            # 初始阶段与 runner 的执行顺序一致：缓存预热 → 预热 → 正式压测
            if job_config.get("cache_mode") == "warmup" and job_config.get("cache_warmup", 0) > 0:
                initial_stage = "cache_warmup"
            elif job_config.get("warmup", 0) > 0:
                initial_stage = "warmup"
            else:
                initial_stage = "run"

            self.broadcaster.init_job(
                mode=mode,
                total_requests=int(requests_total) if requests_total else None,
                duration=float(duration) if duration else None,
                target_count=n_targets,
                targets=[{"provider": pn, "model": mdl} for pn, mdl, _p in valid_targets],
                initial_stage=initial_stage,
                job_id=job_dir.name,
            )

            # 为每个 target 构建 RunConfig
            # cfg_targets 与 configs 同序（API Key 解析失败的 target 会被跳过，
            # 不能直接用 valid_targets 与 gather 结果配对）
            configs = []
            cfg_targets: list[tuple[str, str]] = []
            for provider_name, model, provider in valid_targets:
                # 解析 API Key
                try:
                    api_key, api_key_env = resolve_api_key(
                        cli_key=None,
                        cli_env=None,
                        yaml_env=provider.get("api_key_env"),
                        yaml_key=provider.get("api_key"),
                    )
                except SystemExit as e:
                    LOG.error("API key 错误 (%s): %s", provider_name, e)
                    print(f"[压测] API key 错误，跳过 {provider_name}/{model}: {e}", flush=True)
                    _mark_entry(provider_name, model, "failed", f"API key 错误: {e}")
                    continue

                concurrency = job_config.get("concurrency", [1])
                if isinstance(concurrency, (int, float)):
                    concurrency = [int(concurrency)]
                elif isinstance(concurrency, str):
                    concurrency = [int(x) for x in concurrency.split(",") if x.strip()]

                cfg = merge_run_config(
                    yaml_provider=provider,
                    provider_name=provider_name,
                    base_url=provider.get("base_url"),
                    api_key=api_key,
                    api_key_env=api_key_env,
                    model=model,
                    timeout=job_config.get("timeout", 120),
                    prompt=job_config.get("prompt", "medium"),
                    max_tokens=job_config.get("max_tokens", 500),
                    temperature=job_config.get("temperature", 0.0),
                    concurrency=concurrency,
                    duration=duration,
                    requests=per_target_requests.get(f"{provider_name}/{model}"),
                    step_duration=300.0,
                    step_duration_explicit=False,
                    warmup=job_config.get("warmup", 10),
                    retry=job_config.get("retry", 1),
                    output_dir=job_dir,
                    # 页面「深度思考」开关 → 请求体 enable_thinking；
                    # 缺省 False 与 apply_default_extra_body 的默认值一致
                    extra_body={"enable_thinking": bool(job_config.get("enable_thinking", False))},
                    stream=job_config.get("stream", True),
                    cache_mode=job_config.get("cache_mode", "record"),
                    cache_warmup=job_config.get("cache_warmup", 3),
                    chunk_timeline=job_config.get("chunk_timeline", False),
                    custom_prompt=job_config.get("custom_prompt"),
                    wait_time=job_config.get("wait_time", 0.0),
                    api_style=job_config.get("api_style", "openai"),
                )
                configs.append(cfg)
                cfg_targets.append((provider_name, model))

            if not configs:
                raise RuntimeError("没有有效的压测配置（API Key 解析失败）")

            # 所有 target 并发执行，共享 broadcaster
            self.broadcaster.set_running(True)
            LOG.info("启动压测任务，共 %d 个 target", len(configs))

            tasks = [
                asyncio.create_task(
                    run_benchmark(
                        cfg,
                        stop_external=self.broadcaster.stop_event,
                        broadcaster=self.broadcaster,
                        run_name=f"{provider_name}_{model}",
                    )
                )
                for cfg, (provider_name, model) in zip(configs, cfg_targets)
            ]
            monitor = asyncio.create_task(self._progress_monitor())
            results = await asyncio.gather(*tasks, return_exceptions=True)
            monitor.cancel()

            # 把每个 target 的产出目录（任务目录下的模型子目录）与状态登记到任务清单
            ok_dirs: list[Path] = []
            for (provider_name, model), r in zip(cfg_targets, results):
                if isinstance(r, Exception):
                    LOG.error("target 执行失败 (%s/%s): %s", provider_name, model, r)
                    print(f"[压测] target 执行失败 ({provider_name}/{model}): {r}", flush=True)
                    _mark_entry(provider_name, model, "failed", str(r))
                elif r:
                    _mark_entry(provider_name, model, "ok", None)
                    for e in target_entries:
                        if e["provider"] == provider_name and e["model"] == model:
                            e["dir"] = Path(r).name
                    ok_dirs.append(Path(r))

            stopped = self.broadcaster.stop_event.is_set()
            stage = "stopped" if stopped else "finished"
            job = self.broadcaster.job_snapshot()

            print(
                f"\n{'='*70}\n[压测结束] 状态: {'已停止' if stopped else '正常完成'}，"
                f"正式请求 {job['done']} 次（成功 {job['success']} / 失败 {job['failed']}），"
                f"{len(ok_dirs)}/{len(configs)} 个目标产出报告\n{'='*70}",
                flush=True,
            )
            LOG.info("压测任务完成，%d 个 target 成功", len(ok_dirs))

        except Exception as e:
            stage = "error"
            LOG.error("任务执行异常: %s", e, exc_info=True)
            print(f"[压测异常] {e}", flush=True)
            if self._job_info:
                self._job_info["error"] = str(e)
        finally:
            self.broadcaster.set_running(False)
            self.broadcaster.finish_job(stage)
            # 任务收口（完成/停止/异常）：清单 + 页面对比快照 + 本次任务内 MD 对比
            await self._finalize_job(stage, job_dir, job_config, target_entries)

    async def _finalize_job(
        self,
        stage: str,
        job_dir: Path,
        job_config: dict[str, Any],
        target_entries: list[dict[str, Any]],
    ) -> None:
        """收口产物：job.json、任务级 report.html；多模型时追加 analysis.md/compare.md"""
        ended_at = datetime.now().isoformat()
        if self._job_info:
            self._job_info["ended_at"] = ended_at

        model_dirs = [job_dir / e["dir"] for e in target_entries if e.get("dir")]
        multi_model = len(model_dirs) >= 2

        # 本次任务内多模型对比（不跨任务）；与 suite 批次复用同一对比逻辑
        has_analysis = False
        if multi_model:
            try:
                from llm_bench.report import compare_paths

                weights = DEFAULT_WEIGHTS_PATH if DEFAULT_WEIGHTS_PATH.is_file() else None
                compare_paths(model_dirs, output=job_dir, weights_path=weights)
                has_analysis = (job_dir / "analysis.md").is_file()
                print(f"[压测报告] 本次任务对比分析已生成：{job_dir / 'analysis.md'}", flush=True)
            except Exception as e:
                LOG.warning("生成本次任务对比报告失败: %s", e, exc_info=True)
                print(f"[压测报告] 生成对比分析失败: {e}", flush=True)

        # 任务级页面快照（含全部模型分组表与趋势图，单文件可离线打开）
        has_page = False
        try:
            from llm_bench.web.snapshot import build_job_snapshot, write_job_report

            payload = await build_job_snapshot(
                stage=stage,
                job_config=job_config,
                job_info=self._job_info,
                broadcaster=self.broadcaster,
                result_dirs=[str(p) for p in model_dirs],
                job_id=job_dir.name,
                targets=target_entries,
                multi_model=multi_model,
            )
            write_job_report(job_dir, payload)
            has_page = True
            print(f"[压测报告] 页面快照已生成：{job_dir / 'report.html'}", flush=True)
        except Exception as e:
            LOG.warning("生成 HTML 页面快照失败: %s", e, exc_info=True)

        # 剩余未标记的目标统一记为未完成（异常收口路径）
        for e in target_entries:
            if e["status"] == "pending":
                e["status"] = "failed"
                e["error"] = "任务未正常结束"

        if self._job_info:
            self._job_info["ended_at"] = ended_at
            self._job_info["targets"] = target_entries

        try:
            _write_job_manifest(job_dir, {
                "job_id": job_dir.name,
                "kind": "job",
                "stage": stage,
                "started_at": (self._job_info or {}).get("started_at"),
                "ended_at": ended_at,
                "config": job_config,
                "targets": target_entries,
                "multi_model": multi_model,
                "artifacts": {
                    "page": has_page,
                    "analysis": has_analysis,
                    "compare_md": (job_dir / "compare.md").is_file(),
                },
                "error": (self._job_info or {}).get("error"),
            })
        except OSError as e:
            LOG.warning("写入任务清单失败: %s", e)

    async def _progress_monitor(self, interval: float = 5.0) -> None:
        """控制台周期打印压测进度（阶段 / 进度 / 成功失败）"""
        last_done = -1
        while True:
            await asyncio.sleep(interval)
            try:
                job = self.broadcaster.job_snapshot()
            except Exception:
                continue
            done = job.get("done", 0)
            stage_label = job.get("stage_label", "")
            if job.get("mode") == "requests":
                total = job.get("total_requests") or 0
                pct = f"{(job.get('progress') or 0) * 100:.0f}%"
                line = (f"[压测进度] 阶段: {stage_label}  {done}/{total}（{pct}）"
                        f"  成功 {job.get('success', 0)} 失败 {job.get('failed', 0)}")
            else:
                line = (f"[压测进度] 阶段: {stage_label}  已用时 {job.get('elapsed_s', 0)}s"
                        f"（{(job.get('progress') or 0) * 100:.0f}%）"
                        f"  正式请求 {done} 次  成功 {job.get('success', 0)} 失败 {job.get('failed', 0)}")
            # 避免预热阶段刷屏
            if done != last_done or job.get("stage") == "run":
                print(line, flush=True)
            last_done = done
