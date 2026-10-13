from __future__ import annotations

from typing import Any

CACHE_MODES = ("record", "warmup")


def _int_field(u: dict[str, Any], *keys: str) -> int | None:
    for k in keys:
        if k in u and u[k] is not None:
            try:
                return int(u[k])
            except (TypeError, ValueError):
                return None
    return None


def _nested_int(u: dict[str, Any], parent: str, *keys: str) -> int | None:
    block = u.get(parent)
    if not isinstance(block, dict):
        return None
    return _int_field(block, *keys)


def usage_from_obj(obj: dict[str, Any]) -> dict[str, int] | None:
    u = obj.get("usage")
    if not isinstance(u, dict):
        return None
    if not any(
        k in u
        for k in (
            "prompt_tokens",
            "input_tokens",
            "completion_tokens",
            "output_tokens",
            "total_tokens",
        )
    ):
        return None
    try:
        inp = _int_field(u, "prompt_tokens", "input_tokens") or 0
        out = _int_field(u, "completion_tokens", "output_tokens")
        if out is None:
            out = 0
        tot = _int_field(u, "total_tokens")
        if tot is None:
            tot = inp + out
    except (TypeError, ValueError):
        return None

    cache_read = _int_field(
        u,
        "cache_read_tokens",
        "cached_tokens",
        "prompt_cache_hit_tokens",
        "input_tokens_cache_read",
    )
    if cache_read is None:
        cache_read = _nested_int(u, "prompt_tokens_details", "cached_tokens", "cache_read_tokens")
    if cache_read is None:
        cache_read = _nested_int(u, "input_tokens_details", "cached_tokens", "cache_read_tokens")

    cache_write = _int_field(
        u,
        "cache_write_tokens",
        "cache_creation_input_tokens",
        "input_tokens_cache_write",
    )
    if cache_write is None:
        cache_write = _nested_int(
            u,
            "prompt_tokens_details",
            "cache_creation_tokens",
            "cache_write_tokens",
        )

    out_dict: dict[str, int] = {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": tot,
    }
    if cache_read is not None:
        out_dict["cache_read_tokens"] = cache_read
    if cache_write is not None:
        out_dict["cache_write_tokens"] = cache_write
    return out_dict
