from __future__ import annotations

import asyncio
import json
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_bench.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEIGHTS_PATH,
    PACKAGE_ROOT,
    apply_default_extra_body,
    load_yaml,
    merge_run_config,
    parse_concurrency,
    resolve_api_key,
)
from llm_bench.prompts import PROMPT_SIZES
from llm_bench.report import compare_paths
from llm_bench.runner import run_benchmark


class SuiteConfigError(Exception):
    pass


class _SuiteAbort(Exception):
    """Non-Cancelled abort so TaskGroup cancels sibling provider groups."""


_OVERRIDE_KEYS = (
    "prompt",
    "max_tokens",
    "temperature",
    "concurrency",
    "duration",
    "requests",
    "warmup",
    "retry",
    "timeout",
    "extra_body",
    "step_duration",
    "stream",
)

_LOAD_FP_KEYS = ("prompt", "max_tokens", "temperature", "duration", "requests")


def _require_bool(raw: Any, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    raise SuiteConfigError(f"{name} must be a boolean")


def _as_float(raw: Any, name: str) -> float:
    if isinstance(raw, bool) or raw is None:
        raise SuiteConfigError(f"{name} must be a number")
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(raw)
    except (TypeError, ValueError) as e:
        raise SuiteConfigError(f"{name} must be a number") from e


def _as_int(raw: Any, name: str) -> int:
    if isinstance(raw, bool) or raw is None:
        raise SuiteConfigError(f"{name} must be an integer")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    raise SuiteConfigError(f"{name} must be an integer")


def _dedupe_preserve(vals: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@dataclass
class PlannedJob:
    provider_name: str
    model: str
    base_url: str
    api_key_env: str | None
    yaml_api_key: str | None
    prompt: str
    max_tokens: int
    temperature: float
    concurrency: list[int]
    duration: float | None
    requests: int | None
    warmup: int
    retry: int
    timeout: float
    extra_body: dict[str, Any]
    step_duration: float
    step_duration_explicit: bool
    stream: bool = True
    weight: int = 1  # F4: 权重（控制并发分配比）

    def public_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model,
            "concurrency": self.concurrency,
            "duration": self.duration,
            "requests": self.requests,
            "stream": self.stream,
        }

    def describe(self) -> str:
        conc = ",".join(str(c) for c in self.concurrency)
        if self.duration is not None:
            load = f"duration={self.duration:g}"
        else:
            load = f"requests={self.requests}"
        return (
            f"provider={self.provider_name} model={self.model} "
            f"concurrency={conc} {load} stream={self.stream}"
        )


@dataclass
class SuitePlan:
    jobs: list[PlannedJob]
    output_dir: Path
    compare: bool
    continue_on_error: bool
    weights: Path | None
    parallel_providers: bool


def _resolve_path(raw: str | Path | None, default: Path) -> Path:
    if raw is None:
        return default
    p = Path(raw)
    if not p.is_absolute():
        p = PACKAGE_ROOT / p
    return p


def _parse_concurrency(raw: Any) -> list[int]:
    if isinstance(raw, bool) or raw is None:
        raise ValueError("concurrency must not be empty")
    if isinstance(raw, int):
        if raw < 1:
            raise ValueError("concurrency values must be >= 1")
        vals = [raw]
    elif isinstance(raw, float) and raw.is_integer():
        return _parse_concurrency(int(raw))
    elif isinstance(raw, list):
        if not raw:
            raise ValueError("concurrency must not be empty")
        vals = parse_concurrency(",".join(str(x) for x in raw))
    else:
        vals = parse_concurrency(str(raw))
    vals = _dedupe_preserve(vals)
    if not vals:
        raise ValueError("concurrency must not be empty")
    return vals


def _pick(layers: list[dict[str, Any]], key: str, default: Any = None) -> Any:
    for layer in reversed(layers):
        if key in layer and layer[key] is not None:
            return layer[key]
    return default


def _duration_requests(layers: list[dict[str, Any]]) -> tuple[float | None, int | None]:
    chosen: dict[str, Any] | None = None
    for layer in layers:
        has_d = layer.get("duration") is not None
        has_r = layer.get("requests") is not None
        if has_d or has_r:
            chosen = layer
    if chosen is None:
        raise SuiteConfigError("exactly one of duration or requests is required")
    has_d = chosen.get("duration") is not None
    has_r = chosen.get("requests") is not None
    if has_d == has_r:
        raise SuiteConfigError("exactly one of duration or requests is required")
    if has_d:
        duration = _as_float(chosen["duration"], "duration")
        if duration <= 0:
            raise SuiteConfigError("duration must be > 0")
        return duration, None
    requests = _as_int(chosen["requests"], "requests")
    if requests <= 0:
        raise SuiteConfigError("requests must be > 0")
    return None, requests


def _model_entries(provider: dict[str, Any]) -> list[dict[str, Any]]:
    if "models" in provider and provider["models"] is not None:
        raw = provider["models"]
        if not isinstance(raw, list) or not raw:
            raise SuiteConfigError(
                f"provider {provider.get('name')!r} models must be a non-empty list"
            )
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                out.append({"name": item})
            elif isinstance(item, dict):
                name = item.get("name")
                if not name:
                    raise SuiteConfigError(
                        f"provider {provider.get('name')!r} model dict requires name"
                    )
                extra = {k: item[k] for k in _OVERRIDE_KEYS if k in item}
                extra["name"] = str(name)
                out.append(extra)
            else:
                raise SuiteConfigError(
                    f"provider {provider.get('name')!r} model entry must be string or mapping"
                )
        return out
    model = provider.get("model")
    if not model:
        raise SuiteConfigError(
            f"provider {provider.get('name')!r} needs models or model"
        )
    return [{"name": str(model)}]


def _build_job(
    provider: dict[str, Any], model_entry: dict[str, Any], defaults: dict[str, Any]
) -> PlannedJob:
    name = provider.get("name")
    url = provider.get("base_url")
    if not name:
        raise SuiteConfigError("provider name required")
    if not url:
        raise SuiteConfigError(f"provider {name!r} missing base_url")
    layers = [defaults, provider, model_entry]
    try:
        conc = _parse_concurrency(_pick(layers, "concurrency", 20))
    except ValueError as e:
        raise SuiteConfigError(f"provider {name!r}: {e}") from e
    duration, requests = _duration_requests(layers)
    prompt = str(_pick(layers, "prompt", "medium"))
    if prompt not in PROMPT_SIZES:
        raise SuiteConfigError(
            f"provider {name!r} prompt must be one of {', '.join(PROMPT_SIZES)}"
        )
    extra = _pick(layers, "extra_body", {})
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise SuiteConfigError(f"provider {name!r} extra_body must be a mapping")
    sd_raw = _pick(layers, "step_duration")
    step_explicit = sd_raw is not None
    if step_explicit:
        step_duration = _as_float(sd_raw, "step_duration")
        if step_duration <= 0:
            raise SuiteConfigError("step_duration must be > 0")
    else:
        step_duration = 300.0
    stream = _require_bool(_pick(layers, "stream", True), "stream")
    weight = _as_int(_pick(layers, "weight", 1), "weight")  # F4
    return PlannedJob(
        provider_name=str(name),
        model=str(model_entry["name"]),
        base_url=str(url).rstrip("/"),
        api_key_env=str(provider["api_key_env"]) if provider.get("api_key_env") else None,
        yaml_api_key=str(provider["api_key"]) if provider.get("api_key") else None,
        prompt=prompt,
        max_tokens=_as_int(_pick(layers, "max_tokens", 500), "max_tokens"),
        temperature=_as_float(_pick(layers, "temperature", 0.0), "temperature"),
        concurrency=conc,
        duration=duration,
        requests=requests,
        warmup=_as_int(_pick(layers, "warmup", 10), "warmup"),
        retry=_as_int(_pick(layers, "retry", 1), "retry"),
        timeout=_as_float(_pick(layers, "timeout", 120), "timeout"),
        extra_body=apply_default_extra_body(dict(extra)),
        step_duration=step_duration,
        step_duration_explicit=step_explicit,
        stream=stream,
        weight=weight,  # F4
    )


def _apply_plan_overrides(
    jobs: list[PlannedJob],
    *,
    stream: bool | None,
    enable_thinking: bool | None,
    duration: float | None,
    requests: int | None,
    concurrency: list[int] | None,
    max_tokens: int | None,
) -> None:
    for job in jobs:
        if stream is not None:
            job.stream = stream
        if enable_thinking is not None:
            extra = apply_default_extra_body(dict(job.extra_body))
            extra["enable_thinking"] = enable_thinking
            job.extra_body = extra
        if duration is not None:
            job.duration = duration
            job.requests = None
        if requests is not None:
            job.requests = requests
            job.duration = None
        if concurrency is not None:
            job.concurrency = list(concurrency)
        if max_tokens is not None:
            job.max_tokens = max_tokens


def load_suite_plan(
    path: Path,
    *,
    output_dir: Path | None = None,
    only: list[str] | None = None,
    skip_compare: bool = False,
    only_models: list[str] | None = None,
    stream: bool | None = None,
    enable_thinking: bool | None = None,
    duration: float | None = None,
    requests: int | None = None,
    concurrency: list[int] | None = None,
    max_tokens: int | None = None,
) -> SuitePlan:
    path = Path(path)
    if not path.is_file():
        raise SuiteConfigError(f"config not found: {path}")
    try:
        data = load_yaml(path)
    except Exception as e:
        raise SuiteConfigError(f"invalid yaml: {e}") from e
    if not isinstance(data, dict):
        raise SuiteConfigError(f"{path} must be a mapping")
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SuiteConfigError("defaults must be a mapping")
    providers = data.get("providers")
    if not isinstance(providers, list) or not providers:
        raise SuiteConfigError(f"{path} has no providers list")
    only_set = set(only) if only else None
    model_set = set(only_models) if only_models else None
    jobs: list[PlannedJob] = []
    for prov in providers:
        if not isinstance(prov, dict):
            raise SuiteConfigError("each provider must be a mapping")
        if "enabled" in prov:
            if not _require_bool(prov["enabled"], "enabled"):
                continue
        pname = str(prov.get("name", ""))
        if only_set is not None and pname not in only_set:
            continue
        for entry in _model_entries(prov):
            if model_set is not None and str(entry["name"]) not in model_set:
                continue
            jobs.append(_build_job(prov, entry, defaults))
    if not jobs:
        raise SuiteConfigError("no providers")
    _apply_plan_overrides(
        jobs,
        stream=stream,
        enable_thinking=enable_thinking,
        duration=duration,
        requests=requests,
        concurrency=concurrency,
        max_tokens=max_tokens,
    )
    yaml_out = data.get("output_dir")
    resolved = _resolve_path(output_dir if output_dir is not None else yaml_out, DEFAULT_OUTPUT_DIR)
    weights_raw = data.get("weights")
    weights = _resolve_path(weights_raw, DEFAULT_WEIGHTS_PATH) if weights_raw else (
        DEFAULT_WEIGHTS_PATH if DEFAULT_WEIGHTS_PATH.is_file() else None
    )
    compare_raw = data["compare"] if "compare" in data else True
    continue_raw = data["continue_on_error"] if "continue_on_error" in data else True
    parallel_raw = data["parallel_providers"] if "parallel_providers" in data else True
    compare = _require_bool(compare_raw, "compare") and not skip_compare
    continue_on_error = _require_bool(continue_raw, "continue_on_error")
    parallel_providers = _require_bool(parallel_raw, "parallel_providers")
    return SuitePlan(
        jobs=jobs,
        output_dir=resolved,
        compare=compare,
        continue_on_error=continue_on_error,
        weights=weights,
        parallel_providers=parallel_providers,
    )


def _job_record(job: PlannedJob, status: str, error: str | None, result: Path | None) -> dict[str, Any]:
    rec = job.public_dict()
    rec["status"] = status
    rec["error"] = error
    rec["result"] = str(result) if result else None
    return rec


def _load_fingerprint(job: PlannedJob) -> tuple[Any, ...]:
    return tuple(getattr(job, k) for k in _LOAD_FP_KEYS)


def _read_json_obj(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _result_summaries(result: Path) -> list[dict[str, Any]]:
    parent = result / "summary.json"
    if parent.is_file():
        data = _read_json_obj(parent)
        return [data] if data else []
    found: list[dict[str, Any]] = []
    for path in sorted(result.glob("c*/summary.json")):
        data = _read_json_obj(path)
        if data is not None:
            found.append(data)
    return found


def _availability_success(summary: dict[str, Any]) -> int:
    av = summary.get("availability")
    if not isinstance(av, dict):
        return 0
    raw = av.get("success")
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _classify_result(result: Path) -> tuple[str, str | None]:
    summaries = _result_summaries(result)
    if any(s.get("interrupted") is True for s in summaries):
        return "interrupted", "interrupted"
    if not summaries or all(_availability_success(s) == 0 for s in summaries):
        return "fail", "all requests failed"
    return "ok", None


def _write_jobs(batch: Path, records: list[dict[str, Any]]) -> None:
    payload = {"batch_dir": str(batch), "jobs": records}
    (batch / "jobs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _jobs_by_provider(jobs: list[PlannedJob]) -> list[list[tuple[int, PlannedJob]]]:
    grouped: dict[str, list[tuple[int, PlannedJob]]] = {}
    order: list[str] = []
    for i, job in enumerate(jobs):
        name = job.provider_name
        if name not in grouped:
            order.append(name)
            grouped[name] = []
        grouped[name].append((i, job))
    return [grouped[name] for name in order]


def _to_run_config(job: PlannedJob, batch_dir: Path, api_key: str, api_key_env: str | None):
    return merge_run_config(
        yaml_provider={"name": job.provider_name, "base_url": job.base_url},
        provider_name=job.provider_name,
        base_url=job.base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        model=job.model,
        timeout=job.timeout,
        prompt=job.prompt,
        max_tokens=job.max_tokens,
        temperature=job.temperature,
        concurrency=job.concurrency,
        duration=job.duration,
        requests=job.requests,
        step_duration=job.step_duration,
        step_duration_explicit=job.step_duration_explicit,
        warmup=job.warmup,
        retry=job.retry,
        output_dir=batch_dir,
        extra_body=job.extra_body,
        stream=job.stream,
    )


def _run_suite_sequential(plan: SuitePlan, batch: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    n = len(plan.jobs)
    abort_rest = False
    skip_providers: set[str] = set()
    skip_reason: dict[str, str] = {}

    for i, job in enumerate(plan.jobs, start=1):
        if abort_rest:
            rec = _job_record(job, "skip", "aborted", None)
            records.append(rec)
            print(f"skip {i}/{n} {job.describe()} error=aborted", file=sys.stderr)
            continue
        if job.provider_name in skip_providers:
            err = skip_reason.get(job.provider_name, "skipped")
            rec = _job_record(job, "skip", err, None)
            records.append(rec)
            print(f"skip {i}/{n} {job.describe()} error={err}", file=sys.stderr)
            _write_jobs(batch, records)
            continue
        try:
            key, key_env = resolve_api_key(
                cli_key=None,
                cli_env=None,
                yaml_env=job.api_key_env,
                yaml_key=job.yaml_api_key,
            )
        except SystemExit as e:
            err = str(e)
            rec = _job_record(job, "skip", err, None)
            records.append(rec)
            print(f"skip {i}/{n} {job.describe()} error={err}", file=sys.stderr)
            skip_providers.add(job.provider_name)
            skip_reason[job.provider_name] = err
            if not plan.continue_on_error:
                abort_rest = True
            _write_jobs(batch, records)
            continue

        print(f"start {i}/{n} {job.describe()}")
        result: Path | None = None
        try:
            cfg = _to_run_config(job, batch, key, key_env)
            result = asyncio.run(run_benchmark(
                cfg, run_name=f"{job.provider_name}_{job.model}"
            ))
        except (KeyboardInterrupt, asyncio.CancelledError):
            if result is not None:
                status, err = _classify_result(result)
                rec = _job_record(job, status, err, result)
            else:
                rec = _job_record(job, "interrupted", "interrupted", None)
            records.append(rec)
            print(f"interrupted {i}/{n} {job.describe()}", file=sys.stderr)
            abort_rest = True
            _write_jobs(batch, records)
            continue
        except Exception as e:
            err = str(e)
            rec = _job_record(job, "fail", err, None)
            records.append(rec)
            print(f"fail {i}/{n} {job.describe()} error={err}", file=sys.stderr)
            if not plan.continue_on_error:
                abort_rest = True
            _write_jobs(batch, records)
            continue
        status, err = _classify_result(result)
        rec = _job_record(job, status, err, result)
        records.append(rec)
        if status == "ok":
            print(f"ok {i}/{n} {job.describe()} result={result}")
        elif status == "interrupted":
            print(f"interrupted {i}/{n} {job.describe()} result={result}", file=sys.stderr)
            abort_rest = True
        else:
            print(f"fail {i}/{n} {job.describe()} error={err}", file=sys.stderr)
            if not plan.continue_on_error:
                abort_rest = True
            else:
                skip_providers.add(job.provider_name)
                skip_reason[job.provider_name] = err or "all requests failed"
        _write_jobs(batch, records)

    _write_jobs(batch, records)
    return records


def _snapshot_records(
    plan: SuitePlan, records: list[dict[str, Any] | None]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for job, rec in zip(plan.jobs, records, strict=True):
        out.append(rec if rec is not None else _job_record(job, "pending", None, None))
    return out


async def _run_suite_parallel(
    plan: SuitePlan, batch: Path, records: list[dict[str, Any] | None]
) -> None:
    """F4: 所有 job 并行执行，不再按 provider 分组"""
    abort = asyncio.Event()
    lock = asyncio.Lock()
    n = len(plan.jobs)

    async def commit(i: int, rec: dict[str, Any]) -> None:
        async with lock:
            records[i] = rec
            _write_jobs(batch, _snapshot_records(plan, records))

    async def run_one(i: int, job: PlannedJob) -> None:
        """F4: 执行单个 job"""
        if abort.is_set():
            await commit(i, _job_record(job, "skip", "aborted", None))
            print(f"skip {i + 1}/{n} {job.describe()} error=aborted", file=sys.stderr)
            return
        try:
            key, key_env = resolve_api_key(
                cli_key=None,
                cli_env=None,
                yaml_env=job.api_key_env,
                yaml_key=job.yaml_api_key,
            )
        except SystemExit as e:
            err = str(e)
            await commit(i, _job_record(job, "skip", err, None))
            print(f"skip {i + 1}/{n} {job.describe()} error={err}", file=sys.stderr)
            if not plan.continue_on_error:
                abort.set()
            return

        print(f"start {i + 1}/{n} {job.describe()}")
        result: Path | None = None
        try:
            cfg = _to_run_config(job, batch, key, key_env)
            result = await run_benchmark(
                cfg,
                install_signals=False,
                stop_external=abort,
                run_name=f"{job.provider_name}_{job.model}",
            )
        except KeyboardInterrupt:
            if result is not None:
                status, err = _classify_result(result)
                await commit(i, _job_record(job, status, err, result))
            else:
                await commit(i, _job_record(job, "interrupted", "interrupted", None))
            print(f"interrupted {i + 1}/{n} {job.describe()}", file=sys.stderr)
            abort.set()
            raise
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                while task.cancelling():
                    task.uncancel()
            if result is not None:
                status, err = _classify_result(result)
                await commit(i, _job_record(job, status, err, result))
            else:
                await commit(i, _job_record(job, "interrupted", "interrupted", None))
            print(f"interrupted {i + 1}/{n} {job.describe()}", file=sys.stderr)
            abort.set()
            raise _SuiteAbort()
        except Exception as e:
            err = str(e)
            await commit(i, _job_record(job, "fail", err, None))
            print(f"fail {i + 1}/{n} {job.describe()} error={err}", file=sys.stderr)
            if not plan.continue_on_error:
                abort.set()
            return
        status, err = _classify_result(result)
        await commit(i, _job_record(job, status, err, result))
        if status == "ok":
            print(f"ok {i + 1}/{n} {job.describe()} result={result}")
        elif status == "interrupted":
            print(
                f"interrupted {i + 1}/{n} {job.describe()} result={result}",
                file=sys.stderr,
            )
            abort.set()
        else:
            print(f"fail {i + 1}/{n} {job.describe()} error={err}", file=sys.stderr)
            if not plan.continue_on_error:
                abort.set()

    loop = asyncio.get_running_loop()

    def on_sigint() -> None:
        if not abort.is_set():
            abort.set()
            print(
                "SIGINT: stopping new requests, waiting in-flight...",
                file=sys.stderr,
            )
            return
        print("second SIGINT: cancelling in-flight", file=sys.stderr)
        for t in asyncio.all_tasks():
            if t is not asyncio.current_task():
                t.cancel()

    try:
        try:
            loop.add_signal_handler(signal.SIGINT, on_sigint)
        except NotImplementedError:
            signal.signal(signal.SIGINT, lambda *_: on_sigint())
        # F4: 所有 job 直接并行，不再按 provider 分组
        # 使用 gather 兼容 Python 3.10
        tasks = [asyncio.create_task(run_one(i, job)) for i, job in enumerate(plan.jobs)]
        await asyncio.gather(*tasks, return_exceptions=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        abort.set()
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, AttributeError):
            pass
        for i, job in enumerate(plan.jobs):
            if records[i] is None:
                records[i] = _job_record(job, "skip", "aborted", None)
        _write_jobs(batch, [r for r in records if r is not None])


def run_suite(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    only: list[str] | None = None,
    skip_compare: bool = False,
    dry_run: bool = False,
    only_models: list[str] | None = None,
    stream: bool | None = None,
    enable_thinking: bool | None = None,
    duration: float | None = None,
    requests: int | None = None,
    concurrency: list[int] | None = None,
    max_tokens: int | None = None,
) -> int:
    plan = load_suite_plan(
        config_path,
        output_dir=output_dir,
        only=only,
        skip_compare=skip_compare,
        only_models=only_models,
        stream=stream,
        enable_thinking=enable_thinking,
        duration=duration,
        requests=requests,
        concurrency=concurrency,
        max_tokens=max_tokens,
    )
    if dry_run:
        for job in plan.jobs:
            print(job.describe())
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch = plan.output_dir / f"suite_{stamp}"
    batch.mkdir(parents=True, exist_ok=False)

    if plan.parallel_providers:
        slots: list[dict[str, Any] | None] = [None] * len(plan.jobs)
        asyncio.run(_run_suite_parallel(plan, batch, slots))
        records = [r for r in slots if r is not None]
    else:
        records = _run_suite_sequential(plan, batch)

    _write_jobs(batch, records)
    print(f"results: {batch}")

    ok_paths: list[Path] = []
    fingerprints: set[tuple[Any, ...]] = set()
    for rec, job in zip(records, plan.jobs, strict=True):
        if rec["status"] == "ok" and rec["result"]:
            ok_paths.append(Path(rec["result"]))
            fingerprints.add(_load_fingerprint(job))
    if plan.compare and ok_paths:
        if len(fingerprints) > 1:
            print(
                "warning: skip compare: ok jobs differ in prompt/max_tokens/temperature/duration/requests",
                file=sys.stderr,
            )
        else:
            try:
                compare_paths(ok_paths, output=batch, weights_path=plan.weights)
            except Exception as e:
                print(f"error: compare failed: {e}", file=sys.stderr)
                return 1

    if all(r["status"] == "ok" for r in records):
        return 0
    return 1
