"""SSE 解析公共工具与常量

提取 client.py 与 anthropic_client.py 的共用 SSE 解析逻辑和常量，
消除代码重复，统一维护。
"""
from __future__ import annotations

# ── 常量定义 ──────────────────────────────────
DONE = "[DONE]"

# 错误消息截断上限（字符数）
# 选择理由：2000 字符足以包含有意义的错误信息，同时避免超长响应撑爆内存/日志
ERR_LIMIT = 2000

# 单请求内最多保留的 ITL（Inter-Token Latency）采样点数
# 选择理由：256 点足够计算统计分位数，同时防止长对话无限膨胀
ITL_CAP = 256

# chunk_timeline 最多保留的事件数
# 选择理由：4096 事件覆盖大多数请求的完整流式时间线
CHUNK_TIMELINE_CAP = 4096

# chunk preview 文本截取长度
# 选择理由：120 字符足够预览内容，同时保持 WebSocket 帧体积可控
CHUNK_PREVIEW_LEN = 120

# SSE 元数据帧（非 content 帧）最大记录数
# 选择理由：64 帧用于记录 SSE 控制事件（如心跳），超出则忽略以避免干扰
SSE_META_CAP = 64


def parse_sse_line(line: str) -> str | None:
    """解析单行 SSE 数据

    - 空行 / 注释行（:开头）→ None
    - "data: xxx" → "xxx"（去除前缀及前导空白）
    - 其他行 → None
    """
    s = line.strip()
    if not s or s.startswith(":"):
        return None
    if s.startswith("data:"):
        return s[5:].lstrip()
    return None
