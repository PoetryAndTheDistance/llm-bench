from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FALLBACK_WEIGHTS: dict[str, float] = {
    "availability": 0.25,
    "ttft": 0.20,
    "decode_tps": 0.20,
    "throughput": 0.15,
    "tail_stability": 0.15,
    "error_rate": 0.05,
}

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = PACKAGE_ROOT / "configs" / "score_weights.yaml"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results"


@dataclass
class RunConfig:
    provider_name: str
    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0
    api_key_env: str | None = None
    prompt: str = "medium"
    max_tokens: int = 500
    temperature: float = 0.0
    concurrency: list[int] = field(default_factory=lambda: [20])
    duration: float | None = None
    requests: int | None = None
    step_duration: float = 300.0
    warmup: int = 10
    retry: int = 1
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    extra_body: dict[str, Any] = field(default_factory=dict)
    step_duration_explicit: bool = False
    stream: bool = True
    cache_mode: str = "record"
    cache_warmup: int = 3
    chunk_timeline: bool = False
    custom_prompt: str | None = None    # 自定义提示词（当 prompt="custom" 时使用）
    wait_time: float = 0.0            # F9: worker 请求间隔（秒）
    api_style: str = "openai"         # F3: openai | anthropic

    def redacted_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "base_url": self.base_url,
            "api_key": "[redacted]",
            "api_key_env": self.api_key_env,
            "model": self.model,
            "timeout": self.timeout,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "concurrency": self.concurrency,
            "duration": self.duration,
            "requests": self.requests,
            "step_duration": self.step_duration,
            "warmup": self.warmup,
            "retry": self.retry,
            "extra_body": self.extra_body,
            "stream": self.stream,
            "cache_mode": self.cache_mode,
            "cache_warmup": self.cache_warmup,
            "chunk_timeline": self.chunk_timeline,
            "custom_prompt": self.custom_prompt,  # 自定义提示词
            "wait_time": self.wait_time,  # F9
            "api_style": self.api_style,  # F3
        }


def parse_dotenv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def find_dotenv(start: Path | None = None, package_root: Path | None = None) -> Path | None:
    root = Path(package_root) if package_root is not None else PACKAGE_ROOT
    canonical = root / "configs" / ".env"
    if canonical.is_file():
        return canonical.resolve()
    cur = (start or Path.cwd()).expanduser().resolve()
    if not cur.is_dir():
        cur = cur.parent
    while True:
        candidate = cur / ".env"
        if candidate.is_file():
            return candidate
        if (cur / ".git").exists():
            return None
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent


def apply_dotenv(
    start: Path | None = None,
    environ: dict | None = None,
    package_root: Path | None = None,
) -> Path | None:
    env = os.environ if environ is None else environ
    path = find_dotenv(start, package_root=package_root)
    if path is None:
        return None
    parsed = parse_dotenv(path.read_text(encoding="utf-8"))
    for key, value in parsed.items():
        if not env.get(key, ""):
            env[key] = value
    return path


def parse_concurrency(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("concurrency must not be empty")
    vals = [int(p) for p in parts]
    if any(v < 1 for v in vals):
        raise ValueError("concurrency values must be >= 1")
    return vals


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_providers(path: Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    providers = data.get("providers") if isinstance(data, dict) else None
    if not isinstance(providers, list):
        raise ValueError(f"{path} has no providers list")
    return providers


def pick_provider(providers: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for p in providers:
        if str(p.get("name")) == name:
            return p
    names = [str(p.get("name")) for p in providers]
    raise ValueError(f"provider {name!r} not in {names}")


def resolve_api_key(
    *,
    cli_key: str | None,
    cli_env: str | None,
    yaml_env: str | None,
    yaml_key: str | None,
) -> tuple[str, str | None]:
    if cli_key:
        return cli_key, cli_env
    if cli_env:
        val = os.environ.get(cli_env, "")
        if not val:
            raise SystemExit(f"environment variable {cli_env} is empty or unset")
        return val, cli_env
    if yaml_env:
        val = os.environ.get(yaml_env, "")
        if val:
            return val, yaml_env
        if not yaml_key:
            raise SystemExit(f"environment variable {yaml_env} is empty or unset")
    if yaml_key:
        return str(yaml_key), None
    raise SystemExit(
        "no API key: pass --api-key, --api-key-env, YAML api_key_env, or YAML api_key"
    )


def load_score_weights(path: Path | None = None) -> dict[str, float]:
    p = path or DEFAULT_WEIGHTS_PATH
    if not p.is_file():
        return dict(FALLBACK_WEIGHTS)
    data = load_yaml(p)
    if not isinstance(data, dict):
        return dict(FALLBACK_WEIGHTS)
    out = dict(FALLBACK_WEIGHTS)
    for k in FALLBACK_WEIGHTS:
        if k in data:
            out[k] = float(data[k])
    return out


def parse_extra_body(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--extra-body must be a JSON object")
    return data


def apply_default_extra_body(extra_body: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(extra_body or {})
    if "enable_thinking" not in out:
        out["enable_thinking"] = False
    return out


def merge_run_config(
    *,
    yaml_provider: dict[str, Any] | None,
    provider_name: str | None,
    base_url: str | None,
    api_key: str | None,
    api_key_env: str | None,
    model: str | None,
    timeout: float | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
    concurrency: list[int],
    duration: float | None,
    requests: int | None,
    step_duration: float,
    step_duration_explicit: bool,
    warmup: int,
    retry: int,
    output_dir: Path,
    extra_body: dict[str, Any],
    stream: bool = True,
    cache_mode: str = "record",
    cache_warmup: int = 3,
    chunk_timeline: bool = False,
    custom_prompt: str | None = None,  # 自定义提示词
    wait_time: float = 0.0,  # F9
    api_style: str = "openai",  # F3
) -> RunConfig:
    y = yaml_provider or {}
    name = provider_name or y.get("name")
    url = base_url or y.get("base_url")
    mdl = model or y.get("model")
    to = timeout if timeout is not None else y.get("timeout", 120)
    if not name:
        raise SystemExit("provider name required (--provider-name or YAML name)")
    if not url:
        raise SystemExit("base_url required (--base-url or YAML)")
    if not mdl:
        raise SystemExit("model required (--model or YAML)")
    key, key_env = resolve_api_key(
        cli_key=api_key,
        cli_env=api_key_env,
        yaml_env=y.get("api_key_env"),
        yaml_key=y.get("api_key"),
    )
    # F2: 已简化，用 max_tokens=1 替代 ttft_only
    return RunConfig(
        provider_name=str(name),
        base_url=str(url).rstrip("/"),
        api_key=key,
        api_key_env=key_env,
        model=str(mdl),
        timeout=float(to),
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        concurrency=concurrency,
        duration=duration,
        requests=requests,
        step_duration=step_duration,
        warmup=warmup,
        retry=retry,
        output_dir=output_dir,
        extra_body=extra_body,
        step_duration_explicit=step_duration_explicit,
        stream=stream,
        cache_mode=cache_mode,
        cache_warmup=cache_warmup,
        chunk_timeline=chunk_timeline,
        custom_prompt=custom_prompt or y.get("custom_prompt"),  # 自定义提示词
        wait_time=wait_time if wait_time > 0 else y.get("wait_time", 0.0),  # F9
        api_style=api_style if api_style != "openai" else y.get("api_style", "openai"),  # F3
    )
