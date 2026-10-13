"""Web UI 模块 — 左侧菜单布局，配置/压测/历史三大页面"""
from llm_bench.web.broadcaster import MetricsBroadcaster
from llm_bench.web.server import create_app

__all__ = ["create_app", "MetricsBroadcaster"]
