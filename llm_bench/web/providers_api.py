"""供应商配置管理 — providers.yaml 的 CRUD、模型拉取、连接测试"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from llm_bench.client import looks_like_unknown_field
from llm_bench.web.config_store import load_yaml, new_map, new_seq, save_yaml

LOG = logging.getLogger("llm_bench.web.providers")

# 上游错误响应体在页面上展示的最大长度
RAW_BODY_LIMIT = 800


def _format_upstream_error(resp: httpx.Response, *, action: str) -> str:
    """把上游原始响应（状态码 + 响应体）拼成可直接展示的错误信息"""
    body = resp.text.strip()
    if len(body) > RAW_BODY_LIMIT:
        body = body[:RAW_BODY_LIMIT] + "…（已截断）"
    return f"{action}失败：HTTP {resp.status_code} {resp.request.method} {resp.request.url}\n上游原始返回:\n{body or '（空响应体）'}"


class ProvidersManager:
    """管理 configs/providers.yaml 的增删改查（round-trip，保留注释）"""

    def __init__(self, configs_dir: Path):
        self.config_path = configs_dir / "providers.yaml"

    def _load_raw(self) -> dict[str, Any]:
        data = load_yaml(self.config_path, default=new_map())
        if data is None:
            data = new_map()
        if "providers" not in data or data["providers"] is None:
            data["providers"] = new_seq()
        return data

    def _save_raw(self, data: dict[str, Any]) -> None:
        save_yaml(self.config_path, data)

    def list_providers(self) -> list[dict[str, Any]]:
        """返回所有供应商（包含 api_key 掩码）"""
        data = self._load_raw()
        result = []
        for p in data.get("providers", []):
            item = dict(p)
            # 掩码处理 api_key，不向前端返回完整密钥
            if item.get("api_key"):
                key = str(item["api_key"])
                item["api_key"] = key[:8] + "***" if len(key) > 8 else "***"
                item["has_api_key"] = True
            else:
                item["has_api_key"] = bool(item.get("api_key_env"))
            models = item.get("models", [])
            item["models"] = [m if isinstance(m, str) else m.get("name", "") for m in models]
            # 清理已废弃字段
            item.pop("api_style", None)
            item.pop("api_path", None)
            result.append(item)
        return result

    def get_provider(self, name: str) -> dict[str, Any] | None:
        data = self._load_raw()
        for p in data.get("providers", []):
            if p.get("name") == name:
                return p
        return None

    def add_provider(self, provider: dict[str, Any]) -> dict[str, Any]:
        """新增供应商"""
        data = self._load_raw()
        providers = data.setdefault("providers", [])

        name = provider.get("name", "").strip()
        if not name:
            raise ValueError("供应商名称不能为空")

        # 检查重名
        if any(p.get("name") == name for p in providers):
            raise ValueError(f"供应商 {name} 已存在")

        new_provider = new_map()
        new_provider["name"] = name
        new_provider["base_url"] = provider.get("base_url", "").rstrip("/")
        new_provider["enabled"] = provider.get("enabled", True)
        new_provider["models"] = new_seq(provider.get("models", []))
        # api_key 仅在非掩码且非空时保存
        api_key = provider.get("api_key", "")
        if api_key and "***" not in str(api_key):
            new_provider["api_key"] = api_key

        providers.append(new_provider)
        self._save_raw(data)
        LOG.info("新增供应商: %s", name)
        return {"status": "ok", "name": name}

    def update_provider(self, name: str, updates: dict[str, Any]) -> dict[str, Any]:
        """更新供应商"""
        data = self._load_raw()
        providers = data.setdefault("providers", [])

        target = None
        for p in providers:
            if p.get("name") == name:
                target = p
                break

        if target is None:
            raise ValueError(f"供应商 {name} 不存在")

        # 可更新字段
        for field in ("base_url", "api_key_env", "enabled"):
            if field in updates:
                target[field] = updates[field]
                if field == "base_url" and isinstance(target[field], str):
                    target[field] = target[field].rstrip("/")

        # api_key 仅在非掩码且非空时更新
        api_key = updates.get("api_key", "")
        if api_key and "***" not in str(api_key):
            target["api_key"] = api_key

        # models 全量同步（弹窗编辑时）
        if "models" in updates and isinstance(updates["models"], list):
            target["models"] = [m for m in updates["models"] if m]

        # 重命名
        new_name = updates.get("new_name", "").strip()
        if new_name and new_name != name:
            if any(p.get("name") == new_name for p in providers):
                raise ValueError(f"供应商 {new_name} 已存在")
            target["name"] = new_name

        self._save_raw(data)
        LOG.info("更新供应商: %s", name)
        return {"status": "ok"}

    def delete_provider(self, name: str) -> dict[str, Any]:
        """删除供应商"""
        data = self._load_raw()
        providers = data.setdefault("providers", [])
        before = len(providers)
        data["providers"] = [p for p in providers if p.get("name") != name]
        if len(data["providers"]) == before:
            raise ValueError(f"供应商 {name} 不存在")
        self._save_raw(data)
        LOG.info("删除供应商: %s", name)
        return {"status": "ok"}

    def add_model(self, provider_name: str, model_name: str) -> dict[str, Any]:
        """为供应商添加模型"""
        provider = self.get_provider(provider_name)
        if provider is None:
            raise ValueError(f"供应商 {provider_name} 不存在")

        models = provider.setdefault("models", [])
        norm_models = [m if isinstance(m, str) else m.get("name", "") for m in models]
        if model_name in norm_models:
            raise ValueError(f"模型 {model_name} 已存在")

        models.append(model_name)
        data = self._load_raw()
        for p in data["providers"]:
            if p.get("name") == provider_name:
                p["models"] = models
                break
        self._save_raw(data)
        return {"status": "ok"}

    def delete_model(self, provider_name: str, model_name: str) -> dict[str, Any]:
        """删除供应商的模型"""
        provider = self.get_provider(provider_name)
        if provider is None:
            raise ValueError(f"供应商 {provider_name} 不存在")

        models = provider.get("models", [])
        new_models = []
        for m in models:
            mname = m if isinstance(m, str) else m.get("name", "")
            if mname != model_name:
                new_models.append(m)

        if len(new_models) == len(models):
            raise ValueError(f"模型 {model_name} 不存在")

        data = self._load_raw()
        for p in data["providers"]:
            if p.get("name") == provider_name:
                p["models"] = new_models
                break
        self._save_raw(data)
        return {"status": "ok"}

    def _resolve_provider(
        self,
        provider_name: str,
        api_key_override: str = "",
        base_url_override: str = "",
    ) -> dict[str, Any]:
        """解析本次请求使用的供应商配置

        - 已保存的供应商：以配置文件为基础，用弹窗表单中的 base_url / api_key 覆盖
          （这样未保存的修改也能立即测试）
        - 尚未保存的新供应商：供应商不存在时，直接用表单数据构造临时配置，
          不读写 providers.yaml
        """
        provider = self.get_provider(provider_name)
        if provider is not None:
            return provider
        if not base_url_override:
            raise ValueError(
                f"供应商 {provider_name} 尚未保存，且弹窗中未填写 Base URL，无法发起请求"
            )
        LOG.info("供应商 %s 尚未保存，使用表单临时参数测试连接", provider_name)
        return {"name": provider_name, "base_url": base_url_override,
                "api_key": api_key_override, "models": []}

    async def fetch_models(
        self,
        provider_name: str,
        api_style: str = "openai",
        api_key_override: str = "",
        base_url_override: str = "",
    ) -> list[str]:
        """调用供应商接口拉取模型列表

        - openai 协议：GET {base_url}/v1/models
        - anthropic 协议：Anthropic 无公开模型列表接口，不支持自动拉取
        - 供应商未保存时，使用表单传入的 base_url / api_key 临时请求
        """
        provider = self._resolve_provider(provider_name, api_key_override, base_url_override)

        if api_style == "anthropic":
            raise ValueError("Anthropic 协议不支持自动拉取模型列表，请手动填写")

        base_url = (base_url_override or provider.get("base_url", "")).rstrip("/")
        api_key = api_key_override if api_key_override and "***" not in api_key_override else provider.get("api_key", "")

        headers = {}
        if api_key:
            # 与压测客户端一致：两种鉴权头同时下发
            headers["Authorization"] = f"Bearer {api_key}"
            headers["x-api-key"] = api_key

        models_url = f"{base_url}/v1/models"

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(models_url, headers=headers)
            if resp.status_code >= 400:
                raise ValueError(_format_upstream_error(resp, action="拉取模型列表"))
            try:
                data = resp.json()
            except ValueError:
                raise ValueError(
                    f"拉取模型列表失败：响应不是合法 JSON（HTTP {resp.status_code}）\n"
                    f"上游原始返回:\n{resp.text[:RAW_BODY_LIMIT] or '（空响应体）'}"
                )

        # OpenAI 格式: {"data": [{"id": "model-name"}, ...]}
        if isinstance(data, dict) and "data" in data:
            return [m.get("id", "") for m in data["data"] if m.get("id")]
        if isinstance(data, list):
            return [m.get("id", "") if isinstance(m, dict) else str(m) for m in data]
        raise ValueError(
            "无法解析模型列表响应，上游原始返回:\n"
            + json.dumps(data, ensure_ascii=False)[:RAW_BODY_LIMIT]
        )

    @staticmethod
    def _first_model(provider: dict[str, Any]) -> str:
        """取供应商配置中的第一个模型名（models 元素可能是字符串或 {name: ...}）"""
        for m in provider.get("models") or []:
            name = m if isinstance(m, str) else (m.get("name") if isinstance(m, dict) else "")
            if name:
                return name
        return ""

    async def ping(
        self,
        provider_name: str,
        api_style: str = "openai",
        api_key_override: str = "",
        model: str = "",
        base_url_override: str = "",
    ) -> dict[str, Any]:
        """测试供应商连接 —— 与「拉取模型列表」完全独立

        用页面上下拉框选中的模型，给模型发一句「你好」：
        - 成功：返回模型的实际回复内容
        - 失败：返回上游 HTTP 状态码与原始响应体
        - 供应商未保存时，使用表单传入的 base_url / api_key 临时请求
        """
        try:
            provider = self._resolve_provider(provider_name, api_key_override, base_url_override)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        base_url = (base_url_override or provider.get("base_url", "")).rstrip("/")
        api_key = api_key_override if api_key_override and "***" not in api_key_override else provider.get("api_key", "")
        model = (model or "").strip() or self._first_model(provider)
        if not model:
            return {"status": "error",
                    "message": "测试连接需要指定模型：请先拉取或手动添加模型，并在下拉框中选中一个模型"}

        probe_prompt = "你好"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                if api_style == "anthropic":
                    return await self._ping_anthropic(
                        client, base_url, api_key, model, probe_prompt
                    )
                return await self._ping_openai(
                    client, base_url, api_key, model, probe_prompt
                )
        except httpx.HTTPError as e:
            return {"status": "error", "message": f"请求异常：{e}"}
        except Exception as e:
            return {"status": "error", "message": f"请求异常：{e}"}

    async def _ping_openai(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        model: str,
        probe_prompt: str,
    ) -> dict[str, Any]:
        """openai 协议：非流式 POST /v1/chat/completions，发「你好」并取回模型回复"""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            # 与 llm_bench/client.py 压测客户端保持一致，避免"测试通过但压测失败"
            # 两种鉴权头、两套深度思考方言同时下发
            headers["Authorization"] = f"Bearer {api_key}"
            headers["x-api-key"] = api_key
        url_openai = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": probe_prompt}],
            # 推理模型需要先消耗思考 token，64 可能拿不到正文，给足余量
            "max_tokens": 1024,
            "stream": False,
            # 与压测客户端默认行为一致（深度思考关闭，两套方言双发）
            "enable_thinking": False,
            "thinking": {"type": "disabled"},
        }
        resp = await client.post(url_openai, headers=headers, json=payload)
        # 严格网关拒绝未知思考字段时，剥离两套方言后重试一次（与压测客户端同策略）
        if resp.status_code in (400, 422) and looks_like_unknown_field(resp.text or ""):
            payload.pop("enable_thinking", None)
            payload.pop("thinking", None)
            resp = await client.post(url_openai, headers=headers, json=payload)
        if resp.status_code >= 400:
            return {"status": "error", "message": _format_upstream_error(resp, action="对话测试")}
        try:
            data = resp.json()
        except ValueError:
            return {"status": "error",
                    "message": f"对话测试失败：响应不是合法 JSON（HTTP {resp.status_code}）\n"
                               f"上游原始返回:\n{resp.text[:RAW_BODY_LIMIT] or '（空响应体）'}"}
        # 部分网关 HTTP 200 也可能返回错误体
        if isinstance(data, dict) and data.get("error"):
            return {"status": "error",
                    "message": "对话测试失败：HTTP 200 但响应中包含错误\n上游原始返回:\n"
                               + json.dumps(data["error"], ensure_ascii=False)[:RAW_BODY_LIMIT]}
        reply = ""
        try:
            reply = (data["choices"][0].get("message") or {}).get("content") or ""
        except (KeyError, IndexError, AttributeError, TypeError):
            reply = ""
        reply = str(reply).strip()
        if not reply:
            # HTTP 200 但没有正文（可能是思考 token 用尽、网关异常体等），不能算成功
            return {"status": "error",
                    "message": "对话测试失败：HTTP 200 但未解析到模型回复\n上游原始返回:\n"
                               + json.dumps(data, ensure_ascii=False)[:RAW_BODY_LIMIT]}
        return {
            "status": "ok",
            "model": model,
            "reply": reply,
            "message": f"连接成功，模型 {model} 对「{probe_prompt}」的回复：\n{reply}",
        }

    async def _ping_anthropic(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        model: str,
        probe_prompt: str,
    ) -> dict[str, Any]:
        """anthropic 协议：POST /v1/messages，发「你好」并取回模型回复"""
        resp = await client.post(
            f"{base_url}/v1/messages",
            headers={
                # 与 llm_bench/anthropic_client.py 一致：两种鉴权头双发
                "x-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": probe_prompt}],
            },
        )
        if resp.status_code >= 400:
            return {"status": "error", "message": _format_upstream_error(resp, action="对话测试")}
        try:
            data = resp.json()
        except ValueError:
            return {"status": "error",
                    "message": f"对话测试失败：响应不是合法 JSON（HTTP {resp.status_code}）\n"
                               f"上游原始返回:\n{resp.text[:RAW_BODY_LIMIT] or '（空响应体）'}"}
        if isinstance(data, dict) and data.get("type") == "error":
            return {"status": "error",
                    "message": "对话测试失败，上游原始返回:\n"
                               + json.dumps(data.get("error", data), ensure_ascii=False)[:RAW_BODY_LIMIT]}
        reply = ""
        for block in (data.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "text":
                reply += block.get("text", "")
        reply = reply.strip()
        if not reply:
            return {"status": "error",
                    "message": "对话测试失败：HTTP 200 但未解析到模型回复\n上游原始返回:\n"
                               + json.dumps(data, ensure_ascii=False)[:RAW_BODY_LIMIT]}
        return {
            "status": "ok",
            "model": model,
            "reply": reply,
            "message": f"连接成功（Anthropic 协议），模型 {model} 对「{probe_prompt}」的回复：\n{reply}",
        }

    async def test_connection(self, provider_name: str, api_key_override: str = "") -> dict[str, Any]:
        """兼容旧接口（默认 openai 协议）"""
        return await self.ping(provider_name, "openai", api_key_override)
