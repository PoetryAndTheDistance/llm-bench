"""FastAPI 应用 — 左侧菜单布局，供应商管理 + 任务管理 + 历史记录"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)

from llm_bench.web.broadcaster import MetricsBroadcaster
from llm_bench.web.config_store import load_yaml, new_map, save_yaml, to_plain, update_yaml
from llm_bench.web.job_manager import JobManager
from llm_bench.web.providers_api import ProvidersManager
from llm_bench.web.websocket import WebSocketManager

LOG = logging.getLogger("llm_bench.web.server")

STATIC_DIR = Path(__file__).parent / "static"

# 配置/结果目录：打包后在 exe 同级目录，开发时在项目根目录
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(".")

CONFIGS_DIR = BASE_DIR / "configs"
RESULTS_DIR = BASE_DIR / "results"

# 每个模型子目录下可下载的产物文件
TARGET_FILES = (
    "report.md", "requests.csv", "timeseries.csv", "chunks.csv",
    "summary.json", "config.json", "benchmark.log",
)


def _read_json_obj(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _classify_dir(d: Path) -> str | None:
    """识别结果目录类型：job（任务目录）/ suite（批量批次）/ legacy（旧版平铺单模型）"""
    if not d.is_dir() or d.name.startswith("compare_"):
        return None
    if (d / "job.json").is_file():
        return "job"
    if (d / "jobs.json").is_file():
        return "suite"
    if (d / "summary.json").is_file():
        return "legacy"
    return None


def _safe_result_file(job_id: str, rel_path: str) -> Path | None:
    """解析任务目录内的相对文件路径，拒绝目录遍历；不存在返回 None

    3.10: 修复路径遍历防护的前缀误匹配问题。
    使用路径分隔符确保完整目录匹配，避免 job_123 匹配到 job_1234。
    """
    job_dir = (RESULTS_DIR / job_id)
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    try:
        base_resolved = job_dir.resolve()
        target = (job_dir.joinpath(*parts)).resolve()
        # 3.10: 使用路径分隔符确保完整匹配
        # 目标路径必须等于基础目录，或在基础目录的子目录下
        target_str = str(target)
        base_str = str(base_resolved)
        if not (target_str == base_str or target_str.startswith(base_str + os.sep)):
            return None
        if not target.is_file():
            return None
        return target
    except OSError:
        return None


def _build_target_entry(
    job_dir: Path,
    provider: str | None,
    model: str | None,
    *,
    subdir: str | None,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    tdir = job_dir / subdir if subdir else job_dir
    summary = _read_json_obj(tdir / "summary.json")
    cfg = _read_json_obj(tdir / "config.json")
    if (provider is None or model is None) and summary:
        provider = provider or summary.get("provider")
        model = model or summary.get("model")
    files = {name: (tdir / name).is_file() for name in TARGET_FILES}
    return {
        "provider": provider or "unknown",
        "model": model or "unknown",
        "dir": subdir or "",
        "status": status,
        "error": error,
        "config": cfg,
        "summary": summary,
        "files": files,
    }


def _root_artifacts(job_dir: Path) -> dict[str, bool]:
    return {
        "page": (job_dir / "report.html").is_file(),
        "analysis": (job_dir / "analysis.md").is_file(),
        "compare_md": (job_dir / "compare.md").is_file(),
        "compare_csv": (job_dir / "compare.csv").is_file(),
    }


def _suite_started_at(name: str) -> str:
    # suite_YYYYmmdd_HHMMSS_ffffff
    try:
        parts = name.split("_")
        return datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S").isoformat()
    except (IndexError, ValueError):
        return ""


def _build_job_detail(job_dir: Path) -> dict[str, Any] | None:
    """组装任务详情页所需的完整数据"""
    kind = _classify_dir(job_dir)
    if kind is None:
        return None
    artifacts = _root_artifacts(job_dir)

    if kind == "job":
        manifest = _read_json_obj(job_dir / "job.json") or {}
        targets = []
        for t in manifest.get("targets", []):
            targets.append(_build_target_entry(
                job_dir, t.get("provider"), t.get("model"),
                subdir=t.get("dir"), status=t.get("status", "pending"),
                error=t.get("error"),
            ))
        return {
            "id": job_dir.name, "kind": "job",
            "stage": manifest.get("stage", "unknown"),
            "started_at": manifest.get("started_at", ""),
            "ended_at": manifest.get("ended_at", ""),
            "multi_model": bool(manifest.get("multi_model")),
            "config": manifest.get("config", {}),
            "error": manifest.get("error"),
            "targets": targets,
            "artifacts": artifacts,
        }

    if kind == "legacy":
        summary = _read_json_obj(job_dir / "summary.json") or {}
        interrupted = bool(summary.get("interrupted"))
        targets = [_build_target_entry(
            job_dir, summary.get("provider"), summary.get("model"),
            subdir=None,
            status="ok" if summary.get("availability", {}).get("total") else "failed",
        )]
        return {
            "id": job_dir.name, "kind": "legacy",
            "stage": "stopped" if interrupted else "finished",
            "started_at": summary.get("started_at", ""),
            "ended_at": summary.get("ended_at", ""),
            "multi_model": False,
            "config": {},
            "error": None,
            "targets": targets,
            "artifacts": artifacts,
        }

    # suite 批次
    jobs_doc = _read_json_obj(job_dir / "jobs.json") or {}
    targets = []
    for rec in jobs_doc.get("jobs", []):
        result = rec.get("result")
        subdir = Path(result).name if result and Path(result).parent == job_dir else None
        targets.append(_build_target_entry(
            job_dir, rec.get("provider"), rec.get("model"),
            subdir=subdir,
            status="ok" if rec.get("status") == "ok" else "failed",
            error=rec.get("error"),
        ))
    return {
        "id": job_dir.name, "kind": "suite",
        "stage": "finished",
        "started_at": _suite_started_at(job_dir.name),
        "ended_at": "",
        "multi_model": len([t for t in targets if t["dir"]]) >= 2,
        "config": {},
        "error": None,
        "targets": targets,
        "artifacts": artifacts,
    }


def _list_jobs() -> list[dict[str, Any]]:
    """历史记录列表：一次压测任务一行"""
    rows: list[dict[str, Any]] = []
    if not RESULTS_DIR.exists():
        return rows
    for d in RESULTS_DIR.iterdir():
        detail = _build_job_detail(d) if _classify_dir(d) else None
        if detail is None:
            continue
        total = success = failed = 0
        names: list[str] = []
        ok_targets = 0
        for t in detail["targets"]:
            names.append(f"{t['provider']}/{t['model']}")
            if t["status"] == "ok":
                ok_targets += 1
            av = (t.get("summary") or {}).get("availability") or {}
            total += int(av.get("total") or 0)
            success += int(av.get("success") or 0)
            failed += int(av.get("failed") or 0)
        rows.append({
            "id": detail["id"],
            "kind": detail["kind"],
            "stage": detail["stage"],
            "_started": detail.get("started_at") or "",
            "started_at": detail["started_at"],
            "ended_at": detail["ended_at"],
            "target_count": len(detail["targets"]),
            "ok_count": ok_targets,
            "target_names": names,
            "total": total,
            "success": success,
            "failed": failed,
            "multi_model": detail["multi_model"],
            "has_page": detail["artifacts"]["page"],
            "has_analysis": detail["artifacts"]["analysis"],
        })
    # 按开始时间倒序（同时间回退目录名），避免目录前缀不同导致顺序失真
    rows.sort(key=lambda r: (r["_started"], r["id"]), reverse=True)
    for r in rows:
        r.pop("_started", None)
    return rows


def create_app() -> Any:
    """创建 FastAPI 应用"""
    app = FastAPI(title="llm-bench Web UI", version="2.0")

    providers_mgr = ProvidersManager(CONFIGS_DIR)
    job_mgr = JobManager(RESULTS_DIR, providers_mgr)
    broadcaster = job_mgr.broadcaster
    ws_manager = WebSocketManager(broadcaster)

    # ─────────────────────────────────────────────
    # 页面
    # ─────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        html_path = STATIC_DIR / "dashboard.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<h1>dashboard.html not found</h1>"

    @app.get("/viewer", response_class=HTMLResponse)
    async def report_viewer():
        """Markdown 报告渲染预览页（自包含，离线可用），通过 ?src= 指定取数地址"""
        html_path = STATIC_DIR / "report-viewer.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<h1>report-viewer.html not found</h1>"

    @app.get("/job/{job_id}", response_class=HTMLResponse)
    async def job_detail_page(job_id: str):
        """单次压测任务详情页（任务级报告 + 各模型结果）"""
        html_path = STATIC_DIR / "job-detail.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<h1>job-detail.html not found</h1>"

    # dashboard 与离线快照共用的前端公共组件（白名单，避免任意文件读取）
    STATIC_ASSETS = {"trend-chart.js", "first-exchange.js"}

    @app.get("/static/{name}")
    async def static_asset(name: str):
        if name not in STATIC_ASSETS:
            return JSONResponse({"error": "not found"}, status_code=404)
        asset = STATIC_DIR / name
        if not asset.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(asset, media_type="application/javascript; charset=utf-8")

    # ─────────────────────────────────────────────
    # 供应商配置 API
    # ─────────────────────────────────────────────

    @app.get("/api/providers")
    async def list_providers():
        return JSONResponse({"providers": providers_mgr.list_providers()})

    @app.post("/api/providers")
    async def add_provider(provider: dict[str, Any]):
        try:
            return JSONResponse(providers_mgr.add_provider(provider))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.put("/api/providers/{name}")
    async def update_provider(name: str, updates: dict[str, Any]):
        try:
            return JSONResponse(providers_mgr.update_provider(name, updates))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.delete("/api/providers/{name}")
    async def delete_provider(name: str):
        try:
            return JSONResponse(providers_mgr.delete_provider(name))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/providers/{name}/models")
    async def add_model(name: str, body: dict[str, Any]):
        try:
            model_name = body.get("model", "").strip()
            if not model_name:
                return JSONResponse({"error": "模型名不能为空"}, status_code=400)
            return JSONResponse(providers_mgr.add_model(name, model_name))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.delete("/api/providers/{name}/models/{model}")
    async def delete_model(name: str, model: str):
        try:
            return JSONResponse(providers_mgr.delete_model(name, model))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/providers/{name}/fetch-models")
    async def fetch_models(name: str, body: dict[str, Any] | None = None):
        try:
            api_style = (body or {}).get("api_style", "openai")
            api_key_override = (body or {}).get("api_key", "")
            base_url_override = (body or {}).get("base_url", "")
            models = await providers_mgr.fetch_models(
                name, api_style, api_key_override, base_url_override
            )
            payload: dict[str, Any] = {"models": models}
            if not models:
                # 百应类网关在 key 与路径前缀不匹配时会返回 200 + 空列表，给出可操作的提示
                payload["warning"] = (
                    "接口请求成功但模型列表为空。若该供应商有多套环境，"
                    "请检查 base_url 前缀（如 /dev）与 API Key 是否属于同一环境"
                )
            return JSONResponse(payload)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": f"拉取模型失败: {e}"}, status_code=500)

    @app.post("/api/providers/{name}/test-connection")
    async def test_connection(name: str, body: dict[str, Any] | None = None):
        data_body = body or {}
        api_style = data_body.get("api_style", "openai")
        api_key_override = data_body.get("api_key", "")
        model = data_body.get("model", "")
        base_url_override = data_body.get("base_url", "")
        result = await providers_mgr.ping(
            name, api_style, api_key_override, model, base_url_override
        )
        return JSONResponse(result)

    # ─────────────────────────────────────────────
    # 飞书配置 API
    # ─────────────────────────────────────────────

    @app.get("/api/config/feishu")
    async def get_feishu():
        data = load_yaml(CONFIGS_DIR / "feishu.yaml", default=new_map())
        return JSONResponse(to_plain(data.get("feishu", {}) if data else {}))

    @app.post("/api/config/feishu")
    async def save_feishu(config: dict[str, Any]):
        # round-trip 更新：只改字段值，保留文件中已有注释
        def mutate(data):
            section = data.setdefault("feishu", new_map())
            for key in ("enabled", "app_id", "app_secret", "folder_token", "title_template"):
                if key in config:
                    section[key] = config[key]

        update_yaml(CONFIGS_DIR / "feishu.yaml", mutate)
        return JSONResponse({"status": "ok"})

    @app.post("/api/config/feishu/test")
    async def test_feishu(config: dict[str, Any]):
        """测试飞书连接：用当前表单配置创建一个 demo 文档"""
        from llm_bench.feishu import test_feishu_connection

        app_id = (config.get("app_id") or "").strip()
        app_secret = (config.get("app_secret") or "").strip()
        folder_token = (config.get("folder_token") or "").strip()

        if not app_id or not app_secret:
            return JSONResponse({"ok": False, "error": "请填写 App ID 和 App Secret"},
                                status_code=400)

        # 表单中的 secret 若为掩码，则回退读已保存的配置
        if "***" in app_secret:
            data = load_yaml(CONFIGS_DIR / "feishu.yaml", default=new_map())
            saved = (data or {}).get("feishu", {})
            app_secret = saved.get("app_secret", "")

        result = test_feishu_connection(app_id, app_secret, folder_token)
        return JSONResponse(result)

    # ─────────────────────────────────────────────
    # 压测任务 API
    # ─────────────────────────────────────────────

    @app.get("/api/config/preferences")
    async def get_preferences():
        data = load_yaml(CONFIGS_DIR / "preferences.yaml", default=new_map())
        return JSONResponse(to_plain(data.get("last_used", {}) if data else {}))

    @app.get("/api/job")
    async def get_job():
        return JSONResponse(job_mgr.get_status())

    @app.post("/api/job/start")
    async def start_job(job_config: dict[str, Any]):
        result = await job_mgr.start_job(job_config)
        if "error" in result:
            return JSONResponse(result, status_code=400)
        # 保存用户偏好（下次打开压测页自动回显）
        try:
            from llm_bench.web.config_store import new_seq

            targets_seq = new_seq()
            for t in job_config.get("targets", []):
                item = new_map()
                item["provider"] = t.get("provider")
                item["model"] = t.get("model")
                targets_seq.append(item)

            prefs = new_map()
            prefs["targets"] = targets_seq
            prefs["api_style"] = job_config.get("api_style", "openai")
            prefs["prompt"] = job_config.get("prompt")
            prefs["custom_prompt"] = job_config.get("custom_prompt")
            prefs["max_tokens"] = job_config.get("max_tokens")
            prefs["concurrency"] = job_config.get("concurrency")
            prefs["budget_mode"] = "duration" if job_config.get("duration") else "requests"
            prefs["duration"] = job_config.get("duration")
            prefs["requests"] = job_config.get("requests")
            prefs["stream"] = job_config.get("stream")
            prefs["enable_thinking"] = job_config.get("enable_thinking")
            prefs["cache_mode"] = job_config.get("cache_mode")
            prefs["cache_warmup"] = job_config.get("cache_warmup")
            prefs["warmup"] = job_config.get("warmup")
            prefs["retry"] = job_config.get("retry")
            prefs["timeout"] = job_config.get("timeout")
            prefs["wait_time"] = job_config.get("wait_time")

            root = new_map()
            root["last_used"] = prefs
            root.yaml_set_start_comment(
                "用户偏好（自动生成，记录上次压测参数）\n"
                "每次启动压测时保存，打开压测页自动回显；可手动删除恢复默认"
            )
            save_yaml(CONFIGS_DIR / "preferences.yaml", root)
        except Exception as e:
            LOG.warning("保存偏好失败: %s", e)
        return JSONResponse(result)

    @app.post("/api/job/stop")
    async def stop_job():
        result = await job_mgr.stop_job()
        if "error" in result:
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)

    @app.get("/api/job/first-exchange")
    async def get_first_exchange(provider: str, model: str):
        """按需取某个 provider/model 的首个请求完整请求/返回 body（前端展开时拉取）"""
        exchange = await broadcaster.get_first_exchange(provider, model)
        if exchange is None:
            return JSONResponse({"error": "该模型暂无首请求留证"}, status_code=404)
        return JSONResponse(exchange)

    @app.get("/api/job/history")
    async def get_job_history():
        """趋势历史全量回填：页面打开/重连时一次性取任务全程的采样点"""
        return JSONResponse({"history": await broadcaster.history()})

    @app.get("/api/metrics")
    async def get_metrics():
        return JSONResponse(await broadcaster.snapshot())

    @app.get("/api/status")
    async def get_status():
        return JSONResponse({
            "status": broadcaster.status,
            "job": job_mgr.get_status().get("job"),
            "connections": ws_manager.connection_count,
        })

    # ─────────────────────────────────────────────
    # 历史结果 API
    # ─────────────────────────────────────────────

    @app.get("/api/results/history")
    async def results_history():
        """历史记录：一次压测任务一行（含旧版平铺结果与 suite 批次）"""
        try:
            return JSONResponse({"results": _list_jobs()})
        except Exception as e:
            LOG.error("读取历史记录失败: %s", e, exc_info=True)
            return JSONResponse({"results": [], "error": str(e)}, status_code=500)

    @app.get("/api/results/{job_id}/detail")
    async def job_detail(job_id: str):
        job_dir = RESULTS_DIR / job_id
        if _classify_dir(job_dir) is None:
            return JSONResponse({"error": "任务不存在"}, status_code=404)
        detail = _build_job_detail(job_dir)
        if detail is None:
            return JSONResponse({"error": "任务不存在"}, status_code=404)
        return JSONResponse(detail)

    @app.get("/api/results/{job_id}/download/{file_path:path}")
    async def download_result(job_id: str, file_path: str, inline: bool = False):
        # 先校验任务目录合法，再做嵌套路径遍历防护
        if _classify_dir(RESULTS_DIR / job_id) is None:
            return JSONResponse({"error": "任务不存在"}, status_code=404)
        resolved = _safe_result_file(job_id, file_path)
        if resolved is None:
            return JSONResponse({"error": "file not found"}, status_code=404)
        media_types = {
            ".json": "application/json",
            ".csv": "text/csv",
            ".md": "text/markdown",
            ".html": "text/html; charset=utf-8",
            ".log": "text/plain",
        }
        media_type = media_types.get(resolved.suffix, "application/octet-stream")
        download_name = resolved.name
        if inline:
            # 文本类统一 text/plain 便于浏览器内预览；html 保持 text/html 直接渲染
            if resolved.suffix in {".md", ".log", ".csv", ".json", ".txt"}:
                media_type = "text/plain; charset=utf-8"
            return FileResponse(
                resolved, media_type=media_type, filename=download_name,
                content_disposition_type="inline",
            )
        return FileResponse(resolved, media_type=media_type, filename=download_name)

    @app.get("/api/results/{job_id}/zip")
    async def download_job_zip(job_id: str):
        job_dir = RESULTS_DIR / job_id
        if _classify_dir(job_dir) is None:
            return JSONResponse({"error": "任务不存在"}, status_code=404)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(p for p in job_dir.rglob("*") if p.is_file()):
                zf.write(f, f.relative_to(job_dir))
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
        )

    def _feishu_title(
        detail: dict[str, Any],
        md_path: Path,
        target: dict[str, Any] | None = None,
    ) -> str:
        """按飞书标题模板生成文档标题；对比报告使用全部目标名"""
        from llm_bench.feishu import load_feishu_config

        template = load_feishu_config().get(
            "title_template", "llm-bench 压测报告 - {provider}/{model} - {date}"
        )
        if target is not None and md_path.name == "report.md":
            provider = target.get("provider", "unknown")
            model = target.get("model", "unknown")
            return template.format(
                provider=provider, model=model, providers=provider, models=model,
                count=1, date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
        providers = "、".join(sorted({t["provider"] for t in detail["targets"]}))
        models = "、".join(sorted({t["model"] for t in detail["targets"]}))
        return template.format(
            provider="多模型", model="多模型", providers=providers, models=models,
            count=len(detail["targets"]), date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    @app.post("/api/results/{job_id}/feishu")
    async def export_job_feishu(job_id: str, body: dict[str, Any] | None = None):
        """导出任务内单个 Markdown：body.path 为相对任务目录的路径（如 analysis.md）"""
        from llm_bench.feishu import export_to_feishu, is_feishu_configured

        if not is_feishu_configured():
            return JSONResponse({"error": "飞书未配置，请先在配置页面设置"}, status_code=400)
        job_dir = RESULTS_DIR / job_id
        detail = _build_job_detail(job_dir)
        if detail is None:
            return JSONResponse({"error": "任务不存在"}, status_code=404)

        rel = (body or {}).get("path") or ("analysis.md" if detail["artifacts"]["analysis"] else "")
        md_path = _safe_result_file(job_id, rel)
        if md_path is None or md_path.suffix != ".md":
            return JSONResponse({"error": "Markdown 文件不存在"}, status_code=404)

        target = next(
            (t for t in detail["targets"] if t["dir"] and str(md_path).endswith(
                str(Path(t["dir"]) / "report.md"))),
            None,
        )
        title = _feishu_title(detail, md_path, target)
        result = export_to_feishu(md_path, title)
        if result.get("ok"):
            resp = {"status": "ok", "url": result["url"]}
            if result.get("warning"):
                resp["warning"] = result["warning"]
            return JSONResponse(resp)
        return JSONResponse(
            {"error": result.get("error") or "导出失败", "url": result.get("url")},
            status_code=500,
        )

    @app.post("/api/results/{job_id}/feishu-all")
    async def export_job_feishu_all(job_id: str):
        """一键导出：任务级报告（对比分析、对比表、交互式报告）+ 每个模型的 report.md"""
        from llm_bench.feishu import export_to_feishu, is_feishu_configured

        if not is_feishu_configured():
            return JSONResponse({"error": "飞书未配置，请先在配置页面设置"}, status_code=400)
        job_dir = RESULTS_DIR / job_id
        detail = _build_job_detail(job_dir)
        if detail is None:
            return JSONResponse({"error": "任务不存在"}, status_code=404)

        md_items: list[tuple[str, Path, dict[str, Any] | None]] = []
        # 任务级报告：analysis.md、compare.md（HTML 不支持飞书导出）
        if detail["artifacts"]["analysis"]:
            md_items.append(("analysis.md", job_dir / "analysis.md", None))
        if detail["artifacts"].get("compare_md"):
            md_items.append(("compare.md", job_dir / "compare.md", None))
        # 各模型的 report.md
        for t in detail["targets"]:
            if t.get("dir") and (job_dir / t["dir"] / "report.md").is_file():
                md_items.append((f"{t['dir']}/report.md", job_dir / t["dir"] / "report.md", t))

        results = []
        for rel, path, target in md_items:
            title = _feishu_title(detail, path, target)
            r = export_to_feishu(path, title)
            results.append({"path": rel, "ok": bool(r.get("ok")),
                            "url": r.get("url"), "error": r.get("error"),
                            "warning": r.get("warning")})
        return JSONResponse({"results": results})

    # ─────────────────────────────────────────────
    # WebSocket
    # ─────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        ws_manager.start()
        await ws_manager.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            # 客户端正常断开
            pass
        except Exception as e:
            # 3.5: 记录非预期的异常，但不过度捕获导致掩盖问题
            LOG.debug("WebSocket 连接异常: %s: %s", type(e).__name__, e)
        finally:
            await ws_manager.disconnect(ws)

    return app


def run_web_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """启动 Web 服务器

    3.7: 配置 30 秒优雅关闭超时。信号处理（首次 Ctrl+C 优雅关闭、
    二次强制退出）由 uvicorn 原生提供，无需自行注册处理器。
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("需要安装: pip install fastapi uvicorn websockets")

    app = create_app()

    print(f"\n{'='*50}")
    print("  llm-bench Web UI")
    print(f"  访问: http://{host}:{port}")
    print(f"  按 Ctrl+C 停止服务")
    print(f"{'='*50}\n")

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        ws="auto",
        # 优雅关闭超时（秒）：等待在途请求/连接处理完成
        timeout_graceful_shutdown=30,
    )
    server = uvicorn.Server(config)
    server.run()
