"""提示词构建与 token 估算

- estimate_tokens: 使用 tiktoken 精确估算；失败时回退到 len/4
- build_prompt: 按尺寸规格生成标准提示词（含 filler 填充）
- _pad_filler_large: 大 token 数（≥50k）填充，带最大迭代保护
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPT_SIZES = ("short", "medium", "long", "very_long", "990k", "1m", "custom")
TARGET_INPUT_TOKENS = {
    "short": 100,
    "medium": 1000,
    "long": 8000,
    "very_long": 32000,
    "990k": 990_000,
    "1m": 1_000_000,
    "custom": 0,  # 自定义提示词不使用 token 估算
}

# 近百万输入档位（990k / 1m）共用的压测参数建议：单并发、长超时
_NEAR_1M_HINTS: dict[str, Any] = {
    "suggested_concurrency": 1,
    "suggested_max_tokens": 32,
    "suggested_timeout": 900.0,
    "suggested_duration": 120,
    "suggested_warmup": 2,
    "suggested_cache_warmup": 3,
}

PROFILE_HINTS: dict[str, dict[str, Any]] = {
    "990k": _NEAR_1M_HINTS,
    "1m": _NEAR_1M_HINTS,
}

FILLER_UNIT = (
    "Benchmark filler text for input-length control. "
    "Measure TTFT, decode throughput, and tail latency independently. "
    "This paragraph is deterministic and contains no task-relevant instructions. "
)

_LARGE_FILLER_THRESHOLD = 50_000

# _pad_filler_large 的最大迭代次数，防止 estimate_tokens 返回异常值时无限循环
_PAD_LARGE_MAX_ITERATIONS = 100


@lru_cache(maxsize=1)
def _get_encoding() -> Any:
    """获取 tiktoken 编码器（惰性单例，lru_cache 保证线程安全）"""
    import tiktoken  # type: ignore[import-not-found]
    return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数

    优先使用 tiktoken 精确编码；任何异常（未安装、网络错误等）时
    回退到 len(text) // 4 的粗略估算。
    """
    try:
        enc = _get_encoding()
        return max(1, len(enc.encode(text)))
    except Exception:
        return max(1, len(text) // 4)


def prompt_profile_hints(size: str) -> dict[str, Any] | None:
    return PROFILE_HINTS.get(size)


def _pad_filler_large(target_tokens: int) -> str:
    """为大 token 数（≥50k）生成填充文本

    使用倍增策略快速逼近目标，再用缩减策略微调。
    包含最大迭代次数保护，防止 estimate_tokens 返回异常值时无限循环。
    """
    unit_tok = max(1, estimate_tokens(FILLER_UNIT))
    reps = max(1, (target_tokens // unit_tok) + 2)
    text = FILLER_UNIT * reps
    iterations = 0
    while estimate_tokens(text) < target_tokens:
        iterations += 1
        if iterations > _PAD_LARGE_MAX_ITERATIONS:
            break
        need = target_tokens - estimate_tokens(text)
        reps = max(1, need // unit_tok + 1)
        text += FILLER_UNIT * reps
    slack = max(32, target_tokens // 100)
    iterations = 0
    while estimate_tokens(text) > target_tokens + slack and len(text) > len(FILLER_UNIT):
        iterations += 1
        if iterations > _PAD_LARGE_MAX_ITERATIONS:
            break
        text = text[: max(len(FILLER_UNIT), int(len(text) * 0.995))]
    return text


def _pad_filler(target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    if target_tokens >= _LARGE_FILLER_THRESHOLD:
        return _pad_filler_large(target_tokens)
    unit_tok = max(1, estimate_tokens(FILLER_UNIT))
    reps = max(1, (target_tokens // unit_tok) + 4)
    text = FILLER_UNIT * reps
    lo, hi = 0, len(text)
    best = text
    for _ in range(24):
        mid = (lo + hi) // 2
        chunk = text[: max(mid, 1)]
        n = estimate_tokens(chunk)
        if n < target_tokens:
            lo = mid + 1
            best = chunk
        else:
            hi = mid
            best = chunk
            if n == target_tokens:
                break
    n = estimate_tokens(best)
    if n < target_tokens:
        extra = FILLER_UNIT * (max(1, (target_tokens - n) // unit_tok + 2))
        best = best + extra
        while estimate_tokens(best) > target_tokens + max(8, target_tokens // 50) and len(best) > 1:
            best = best[: int(len(best) * 0.98)]
    return best


def _instruction(target_output_tokens: int) -> str:
    n = max(1, int(target_output_tokens))
    return (
        f"Output ONLY the word ping repeated exactly {n} times, "
        "separated by spaces. No other text."
    )


@lru_cache(maxsize=16)
def build_prompt(size: str, target_output_tokens: int) -> str:
    if size not in TARGET_INPUT_TOKENS:
        raise ValueError(f"unknown prompt size {size!r}; expected one of {PROMPT_SIZES}")
    if size == "custom":
        raise ValueError("custom prompt should be provided directly, not via build_prompt()")
    instr = _instruction(target_output_tokens)
    target = TARGET_INPUT_TOKENS[size]
    instr_tok = estimate_tokens(instr + "\n\n")
    filler = _pad_filler(max(1, target - instr_tok))
    return f"{filler}\n\n{instr}"
