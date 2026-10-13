"""任务级 HTML 页面快照

任务收口（正常完成 / 手动停止 / 异常）时冻结最终分组指标与全量趋势历史，
注入自包含模板，在任务根目录生成唯一一份 report.html，离线双击即可还原
当时的压测页面（多模型时即天然的本次任务多模型对比视图）。

注意：job_config 只含压测参数与 targets，不含 API Key，无需脱敏；
首请求留证只含请求/返回 body 与 URL，同样不含请求头与 API Key。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

LOG = logging.getLogger("llm_bench.web.snapshot")

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATE_PATH = STATIC_DIR / "job-report-template.html"
# 与 dashboard.html 共用的前端组件，生成快照时内联进 HTML，保证 report.html
# 单文件离线可打开，且趋势图/首请求校验与实时压测页同一套实现
INLINE_ASSETS = ("trend-chart.js", "first-exchange.js")
_PLACEHOLDER = "__SNAPSHOT_PAYLOAD__"
_ASSETS_PLACEHOLDER = "__SNAPSHOT_ASSETS__"
SCHEMA_VERSION = 1

_template_cache: str | None = None
_assets_cache: str | None = None


def _load_template() -> str:
    global _template_cache
    if _template_cache is None:
        _template_cache = TEMPLATE_PATH.read_text(encoding="utf-8")
    return _template_cache


def _load_inline_assets() -> str:
    """读取公共 JS 并包成内联 <script> 块（缓存）"""
    global _assets_cache
    if _assets_cache is None:
        blocks = []
        for name in INLINE_ASSETS:
            code = (STATIC_DIR / name).read_text(encoding="utf-8")
            # 防御：组件源码中若出现字面量 </script>，避免提前闭合快照脚本块
            code = code.replace("</script>", "<\\/script>")
            blocks.append(f"<script>\n{code}\n</script>")
        _assets_cache = "\n".join(blocks)
    return _assets_cache


def render_job_report(payload: dict[str, Any]) -> str:
    """把快照数据注入模板，返回完整 HTML 文本"""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    # 防止 custom_prompt 等字段中的 </script> 提前闭合脚本块；
    # < 在 JSON 字符串内转成合法的 unicode 转义序列，前端解析后语义不变
    data = data.replace("<", "\\u003c")
    return (
        _load_template()
        .replace(_ASSETS_PLACEHOLDER, _load_inline_assets())
        .replace(_PLACEHOLDER, data)
    )


async def build_job_snapshot(
    *,
    stage: str,
    job_config: dict[str, Any],
    job_info: dict[str, Any] | None,
    broadcaster: Any,
    result_dirs: list[str],
    job_id: str | None = None,
    targets: list[dict[str, Any]] | None = None,
    multi_model: bool = False,
) -> dict[str, Any]:
    """收口时刻冻结 broadcaster 终态：分组指标 + 全量趋势点 + 任务状态"""
    # 追加最后一个趋势点，保证曲线终点与表格终值一致
    grouped = await broadcaster.snapshot_with_history()
    history = await broadcaster.history()
    # 首请求完整留证（请求/返回 body），离线快照中也能展开核对
    first_exchanges = await broadcaster.all_first_exchanges()
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "multi_model": multi_model,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": stage,
        "job": broadcaster.job_snapshot(),
        "job_info": {
            "started_at": (job_info or {}).get("started_at"),
            "ended_at": (job_info or {}).get("ended_at"),
            "target_count": (job_info or {}).get("target_count"),
            "error": (job_info or {}).get("error"),
        },
        "config": job_config or {},
        "targets": targets or [],
        "groups": grouped,
        "history": history,
        "first_exchanges": first_exchanges,
        "result_dirs": list(result_dirs),
    }


def write_job_report(job_dir: Path, payload: dict[str, Any]) -> Path:
    """把任务级快照写入任务根目录的 report.html（每次任务一份）"""
    target = job_dir / "report.html"
    target.write_text(render_job_report(payload), encoding="utf-8")
    return target
