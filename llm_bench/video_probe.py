from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from llm_bench.config import load_providers, pick_provider, resolve_api_key
from llm_bench.runner import sanitize_fs

DEFAULT_FPS = 2.0
MIN_FPS = 0.1
MAX_FPS = 10.0
PROBE_TEXT = "请简要描述这段视频的内容。"


def clamp_fps(fps: float) -> float:
    if fps < MIN_FPS or fps > MAX_FPS:
        raise ValueError(f"fps must be in [{MIN_FPS}, {MAX_FPS}], got {fps}")
    return fps


def parse_frame_urls(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    return parts


def _video_message_formats(
    video_url: str,
    *,
    fps: float,
    frame_urls: list[str] | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    text = {"type": "text", "text": PROBE_TEXT}
    formats: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "video_url",
            [
                text,
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                    "fps": fps,
                },
            ],
        ),
    ]
    frames = frame_urls or []
    if len(frames) >= 4:
        formats.append(
            (
                "video_list",
                [
                    text,
                    {
                        "type": "video",
                        "video": frames,
                        "fps": fps,
                    },
                ],
            )
        )
    return formats


async def _head_size_mb(client: httpx.AsyncClient, video_url: str) -> float | None:
    try:
        resp = await client.head(video_url, follow_redirects=True)
        length = resp.headers.get("content-length")
        if length and length.isdigit():
            return int(length) / (1024 * 1024)
    except Exception:
        return None
    return None


async def probe_video_input(
    *,
    base_url: str,
    api_key: str,
    model: str,
    video_url: str,
    timeout: float = 300.0,
    max_tokens: int = 32,
    fps: float = DEFAULT_FPS,
    frame_urls: list[str] | None = None,
) -> dict[str, Any]:
    fps = clamp_fps(fps)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "model": model,
        "video_url": video_url,
        "fps": fps,
        "frame_urls": frame_urls or [],
        "supported": False,
        "attempts": [],
    }
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=min(30.0, timeout)),
        headers={
            # 与主压测客户端一致：两种鉴权头双发
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "Content-Type": "application/json",
        },
    ) as client:
        size_mb = await _head_size_mb(client, video_url)
        result["video_size_mb"] = size_mb
        url = f"{base_url.rstrip('/')}/chat/completions"
        for idx, (format_name, content) in enumerate(
            _video_message_formats(video_url, fps=fps, frame_urls=frame_urls)
        ):
            body = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
            attempt: dict[str, Any] = {
                "format_index": idx,
                "format_name": format_name,
                "fps": fps,
                "http_status": None,
            }
            t0 = time.perf_counter()
            try:
                resp = await client.post(url, json=body)
                attempt["http_status"] = resp.status_code
                attempt["latency_s"] = time.perf_counter() - t0
                raw = (resp.text or "")[:2000]
                attempt["body_preview"] = raw
                if resp.status_code == 200:
                    result["supported"] = True
                    result["working_format"] = format_name
                    result["working_format_index"] = idx
                    try:
                        obj = resp.json()
                        attempt["response_id"] = obj.get("id")
                    except json.JSONDecodeError:
                        pass
                    result["attempts"].append(attempt)
                    break
                attempt["error"] = raw
            except Exception as e:
                attempt["latency_s"] = time.perf_counter() - t0
                attempt["error"] = f"{type(e).__name__}: {e}"
            result["attempts"].append(attempt)
    skipped = len(frame_urls or []) < 4
    result["video_list_skipped"] = skipped
    if skipped:
        result["video_list_skip_reason"] = (
            "video_list 需要至少 4 个帧图片 URL；仅提供视频文件 URL 时只测 video_url。"
        )
    result["elapsed_s"] = time.perf_counter() - started
    return result


async def run_video_probe_job(
    *,
    config_path: Path,
    provider_name: str,
    model: str,
    video_url: str,
    output_dir: Path,
    timeout: float = 300.0,
    max_tokens: int = 32,
    fps: float = DEFAULT_FPS,
    frame_urls: list[str] | None = None,
) -> Path:
    providers = load_providers(config_path)
    y = pick_provider(providers, provider_name)
    key, _env = resolve_api_key(
        cli_key=None,
        cli_env=None,
        yaml_env=y.get("api_key_env"),
        yaml_key=y.get("api_key"),
    )
    base_url = str(y.get("base_url", "")).rstrip("/")
    result = await probe_video_input(
        base_url=base_url,
        api_key=key,
        model=model,
        video_url=video_url,
        timeout=timeout,
        max_tokens=max_tokens,
        fps=fps,
        frame_urls=frame_urls,
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"{sanitize_fs(provider_name)}_{sanitize_fs(model)}_video_probe_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": provider_name,
        "probe_type": "video_input",
        "bailian_formats": ["video_url+fps", "video_list+fps"],
        **result,
    }
    (out / "probe_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "probe_analysis.md").write_text(
        format_video_probes_markdown([payload]),
        encoding="utf-8",
    )
    return out


def _extract_error_message(attempt: dict[str, Any]) -> str:
    raw = attempt.get("error") or attempt.get("body_preview") or ""
    if not raw:
        return "n/a"
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
    except json.JSONDecodeError:
        pass
    return str(raw)[:240]


def _probe_label(p: dict[str, Any]) -> str:
    return f"{p.get('provider')}/{p.get('model')}"


def format_video_probes_markdown(probes: list[dict[str, Any]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        "# llm-bench 视频输入能力探测",
        "",
        f"生成时间：{now}  探测次数：{len(probes)}",
        "",
        "## 测试条件",
        "",
    ]
    if probes:
        p0 = probes[0]
        lines.append(f"- 视频 URL：`{p0.get('video_url', 'n/a')}`")
        size = p0.get("video_size_mb")
        if size is not None:
            lines.append(f"- HEAD 体积：{size:.2f} MB")
        lines.append(f"- fps：{p0.get('fps', DEFAULT_FPS)}（百炼 OpenAI 兼容默认 2）")
        frames = p0.get("frame_urls") or []
        if frames:
            lines.append(f"- 帧列表 URL：{len(frames)} 个（用于 `video_list` 格式）")
        providers = sorted({str(p.get("provider")) for p in probes if p.get("provider")})
        models = sorted({str(p.get("model")) for p in probes if p.get("model")})
        lines.append(f"- 供应商：{'、'.join(providers) or 'n/a'}")
        lines.append(f"- 模型：{'、'.join(models) or 'n/a'}")
    else:
        lines.append("- 无探测记录。")
    lines.extend(["", "## 结论摘要", ""])
    if not probes:
        lines.append("无探测记录。")
    else:
        for p in probes:
            verdict = "支持" if p.get("supported") else "**不支持**"
            size = p.get("video_size_mb")
            size_s = f"{size:.2f} MB" if size is not None else "未知"
            fmt = p.get("working_format") or ""
            extra = f"，格式 `{fmt}`" if fmt else ""
            lines.append(f"- {_probe_label(p)}：{verdict}（视频 {size_s}{extra}）")
            if not p.get("supported"):
                attempts = p.get("attempts") or []
                if attempts:
                    lines.append(f"  - 首条错误：{_extract_error_message(attempts[0])}")
            if p.get("video_list_skipped"):
                lines.append(f"  - {p.get('video_list_skip_reason', '')}")
    lines.extend(["", "## 探测结果对比", ""])
    if probes:
        lines.extend(
            [
                "| provider | model | size_mb | fps | supported | latency_s | 说明 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for p in probes:
            attempts = p.get("attempts") or []
            latency = attempts[0].get("latency_s") if attempts else None
            if p.get("supported"):
                note = f"format={p.get('working_format')}"
            elif attempts:
                note = _extract_error_message(attempts[0])[:120]
            else:
                note = "n/a"
            size = p.get("video_size_mb")
            lines.append(
                "| {provider} | {model} | {size} | {fps} | {supported} | {latency} | {note} |".format(
                    provider=p.get("provider") or "",
                    model=p.get("model") or "",
                    size=f"{size:.2f}" if size is not None else "n/a",
                    fps=p.get("fps", DEFAULT_FPS),
                    supported="是" if p.get("supported") else "否",
                    latency=f"{latency:.3f}" if isinstance(latency, (int, float)) else "n/a",
                    note=note.replace("|", "/"),
                )
            )
    else:
        lines.append("无探测记录。")
    lines.extend(["", "## 各次尝试明细", ""])
    for p in probes:
        lines.append(f"### {_probe_label(p)}")
        lines.append("")
        for att in p.get("attempts") or []:
            fmt = att.get("format_name") or att.get("format_index")
            status = att.get("http_status")
            lat = att.get("latency_s")
            lat_s = f"{lat:.3f}s" if isinstance(lat, (int, float)) else "n/a"
            fps_v = att.get("fps", p.get("fps"))
            if status == 200:
                lines.append(f"- `{fmt}`（fps={fps_v}）：HTTP {status}，{lat_s}")
            else:
                lines.append(
                    f"- `{fmt}`（fps={fps_v}）：HTTP {status}，{lat_s} — {_extract_error_message(att)}"
                )
        lines.append("")
    lines.extend(
        [
            "## 读数说明",
            "",
            "- 视频文件 URL：百炼 OpenAI 兼容格式为 `type=video_url` + `fps`（0.1–10，默认 2）。",
            "- 视频帧列表：百炼格式为 `type=video` + `video: [url,...]`（至少 4 张图）+ `fps`。",
            "- 需使用支持视觉理解的模型（如 qwen3.7-plus、qwen3-vl-plus）；纯文本模型会返回 not supported。",
            "- HEAD 体积来自 `Content-Length`；非流式视频理解建议 timeout ≥ 300s。",
            "",
        ]
    )
    return "\n".join(lines)


def write_video_probe_analysis(output: Path, probe_json_paths: list[Path]) -> Path:
    probes: list[dict[str, Any]] = []
    for path in probe_json_paths:
        probes.append(json.loads(path.read_text(encoding="utf-8")))
    md = format_video_probes_markdown(probes)
    out = output / "probe_analysis.md"
    out.write_text(md, encoding="utf-8")
    return out
