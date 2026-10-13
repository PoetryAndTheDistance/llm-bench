"""飞书报告导出（基于飞书 Open API）"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

LOG = logging.getLogger("llm_bench.feishu")

# 飞书 Open API 端点
FEISHU_TOKEN_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
FEISHU_DOC_CREATE_URL = "https://open.feishu.cn/open-apis/docx/v1/documents"
FEISHU_DOC_BLOCK_URL = "https://open.feishu.cn/open-apis/docx/v1/documents/{document_id}/blocks/{block_id}/children"


def load_feishu_config(configs_dir: Path | str | None = None) -> dict[str, Any]:
    """加载飞书配置"""
    if configs_dir is not None:
        config_path = Path(configs_dir) / "feishu.yaml"
    else:
        config_path = Path("configs/feishu.yaml")
    if not config_path.exists():
        return {}

    import yaml
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("feishu", {})


def is_feishu_configured(configs_dir: Path | str | None = None) -> bool:
    """检查飞书是否已配置"""
    config = load_feishu_config(configs_dir)
    return bool(config.get("enabled") and config.get("app_id") and config.get("app_secret"))


def get_tenant_access_token(app_id: str, app_secret: str) -> dict[str, Any] | None:
    """获取 tenant_access_token

    返回飞书接口的完整响应（成功时含 tenant_access_token）；
    网络异常或响应无法解析时返回 None，由调用方拼装原始错误展示给页面。
    """
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                FEISHU_TOKEN_URL,
                json={"app_id": app_id, "app_secret": app_secret},
            )
            try:
                data = resp.json()
            except ValueError:
                LOG.error("获取 token 失败：非 JSON 响应，HTTP %s: %s",
                          resp.status_code, resp.text[:300])
                return {
                    "_http_status": resp.status_code,
                    "_raw": resp.text[:500],
                }
            if data.get("code") != 0:
                LOG.error(f"获取 token 失败: {data}")
            return data
    except Exception as e:
        LOG.error(f"获取 token 异常: {e}")
        return {"_exception": str(e)}


def _parse_inline_formatting(text: str) -> list[dict]:
    """解析行内格式（加粗、斜体、链接）"""
    elements = []
    i = 0
    current_text = ""

    while i < len(text):
        # 加粗 **text**
        if text[i:i+2] == "**":
            if current_text:
                elements.append({"text_run": {"content": current_text}})
                current_text = ""
            end = text.find("**", i + 2)
            if end > i + 2:
                elements.append({
                    "text_run": {
                        "content": text[i+2:end],
                        "text_element_style": {"bold": True}
                    }
                })
                i = end + 2
            else:
                current_text += text[i]
                i += 1
        # 斜体 *text*
        elif text[i] == "*" and (i == 0 or text[i-1] != "*"):
            if current_text:
                elements.append({"text_run": {"content": current_text}})
                current_text = ""
            end = text.find("*", i + 1)
            if end > i + 1 and (end + 1 >= len(text) or text[end+1] != "*"):
                elements.append({
                    "text_run": {
                        "content": text[i+1:end],
                        "text_element_style": {"italic": True}
                    }
                })
                i = end + 1
            else:
                current_text += text[i]
                i += 1
        # 链接 [text](url)
        elif text[i] == "[":
            if current_text:
                elements.append({"text_run": {"content": current_text}})
                current_text = ""
            end_bracket = text.find("]", i)
            if end_bracket != -1 and end_bracket + 1 < len(text) and text[end_bracket+1] == "(":
                end_paren = text.find(")", end_bracket + 2)
                if end_paren != -1:
                    link_text = text[i+1:end_bracket]
                    url = text[end_bracket+2:end_paren]
                    elements.append({
                        "text_run": {
                            "content": link_text,
                            "text_element_style": {"link": {"url": url}}
                        }
                    })
                    i = end_paren + 1
                else:
                    current_text += text[i]
                    i += 1
            else:
                current_text += text[i]
                i += 1
        else:
            current_text += text[i]
            i += 1

    if current_text:
        elements.append({"text_run": {"content": current_text}})

    return elements if elements else [{"text_run": {"content": text}}]


# 飞书 docx BlockType：3~11 对应 heading1~heading9
_HEADING_TYPES = {n: (n + 2, f"heading{n}") for n in range(1, 10)}
# 飞书单批最多创建 50 个子块；单个 text_run 内容按保守长度切分，避免超长导致整批失败
_MAX_RUN_CHARS = 1800
_ORDERED_RE = re.compile(r"^\d+\.\s+(.*)$")


def _chunked_text_runs(content: str) -> list[dict]:
    """把可能超长的纯文本切成多个 text_run（飞书对单个文本元素有长度限制）"""
    if not content:
        return [{"text_run": {"content": ""}}]
    return [
        {"text_run": {"content": content[i:i + _MAX_RUN_CHARS]}}
        for i in range(0, len(content), _MAX_RUN_CHARS)
    ]


def markdown_to_feishu_blocks(markdown: str) -> list[dict]:
    """将 Markdown 转换为飞书文档 Block 格式

    支持：标题（h1~h9）、段落、代码块、表格（两步骤创建）、
    无序/有序列表（支持嵌套）、引用、加粗、斜体、链接

    注意：表格和嵌套列表需要特殊处理，通过标记在后续步骤中创建
    """
    blocks = []
    lines = markdown.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # 标题：# ~ #########
        heading = re.match(r"^(#{1,9})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            block_type, key = _HEADING_TYPES[level]
            blocks.append({
                "block_type": block_type,
                key: {"elements": _parse_inline_formatting(heading.group(2).strip())},
            })
            i += 1
            continue

        # 代码块
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过结束围栏（文件结束时 i 可能越界，while 会结束）
            if code_lines:
                blocks.append({
                    "block_type": 14,  # code
                    "code": {
                        "elements": _chunked_text_runs("\n".join(code_lines)),
                        "style": {"language": 1, "wrap": True},  # 1=纯文本
                    },
                })
            continue

        # 表格：检测 Markdown 表格语法（| ... | ... |）
        # 飞书 docx 建表结构复杂，逐行转为等宽对齐的文本块
        if line.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", lines[i + 1].strip()):
            # 跳过分隔行，逐行转为文本块
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                row = lines[i].strip()
                cells = [c.strip() for c in row.strip("|").split("|")]
                is_separator = cells and all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells)
                if cells and not is_separator:
                    blocks.append({
                        "block_type": 2,
                        "text": {"elements": _chunked_text_runs("| " + " | ".join(cells) + " |")},
                    })
                i += 1
            continue

        # 无序列表 - 支持嵌套（通过缩进判断层级）
        # 飞书 API 的列表嵌套需要多步骤创建，这里使用视觉缩进方式
        list_match = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if list_match:
            list_items = []
            while i < len(lines):
                m = re.match(r"^(\s*)[-*+]\s+(.*)$", lines[i])
                if not m:
                    break
                indent = len(m.group(1))
                content = m.group(2).strip()
                # 根据缩进计算层级（每 2 个空格为一级）
                level = min(indent // 2, 7)
                list_items.append({"level": level, "content": content})
                i += 1

            # 创建带视觉缩进的 bullet blocks
            for item in list_items:
                # 添加空格缩进 + 不同层级的符号
                indent_str = " " * (item["level"] * 2)
                bullet_chars = ["•", "◦", "▪", "▫", "◇", "◆", "○", "●"]
                bullet = bullet_chars[min(item["level"], len(bullet_chars) - 1)]
                display_text = f"{indent_str}{bullet} {item['content']}"
                blocks.append({
                    "block_type": 2,  # text（使用文本块而不是 bullet 块，以便控制缩进）
                    "text": {
                        "elements": _parse_inline_formatting(display_text),
                    },
                })
            continue

        # 有序列表 1. item（支持两位数以上序号；使用视觉缩进）
        if _ORDERED_RE.match(line):
            m = _ORDERED_RE.match(lines[i])
            ordered_items = []
            item_num = 1
            while m:
                # 检测缩进
                indent_match = re.match(r"^(\s*)", lines[i])
                indent = len(indent_match.group(1)) if indent_match else 0
                level = min(indent // 2, 7)
                ordered_items.append({"level": level, "content": m.group(1).strip(), "num": item_num})
                item_num += 1
                i += 1
                m = _ORDERED_RE.match(lines[i]) if i < len(lines) else None

            for item in ordered_items:
                indent_str = " " * (item["level"] * 2)
                display_text = f"{indent_str}{item['num']}. {item['content']}"
                blocks.append({
                    "block_type": 2,  # text
                    "text": {
                        "elements": _parse_inline_formatting(display_text),
                    },
                })
            continue

        # 引用 > item（飞书 BlockType 15=quote）
        if line.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].startswith(">"):
                quote_lines.append(lines[i].lstrip("> ").strip())
                i += 1
            blocks.append({
                "block_type": 15,
                "quote": {"elements": _chunked_text_runs("\n".join(x for x in quote_lines if x))},
            })
            continue

        # 普通段落
        if line.strip():
            blocks.append({
                "block_type": 2,
                "text": {"elements": _parse_inline_formatting(line.strip())},
            })
        i += 1

    return blocks


def create_feishu_document(
    title: str,
    markdown_content: str,
    app_id: str | None = None,
    app_secret: str | None = None,
    folder_token: str | None = None,
) -> dict[str, Any]:
    """创建飞书文档

    凭证优先使用显式传入的参数，否则从配置文件读取。

    Returns:
        {"ok": bool, "url": str|None, "error": str|None}
    """
    # 显式参数优先，否则读配置
    if not (app_id and app_secret):
        config = load_feishu_config()
        app_id = app_id or config.get("app_id")
        app_secret = app_secret or config.get("app_secret")
        folder_token = folder_token if folder_token is not None else config.get("folder_token")

    if not (app_id and app_secret):
        return {"ok": False, "url": None, "error": "飞书未配置 App ID / App Secret"}

    # 1. 获取 tenant_access_token
    token_data = get_tenant_access_token(app_id, app_secret)
    if not token_data or token_data.get("code") != 0:
        if token_data and "_exception" in token_data:
            raw = f"请求异常：{token_data['_exception']}"
        elif token_data and "_raw" in token_data:
            raw = f"HTTP {token_data.get('_http_status')}，原始返回:\n{token_data['_raw']}"
        else:
            raw = "飞书原始返回:\n" + json.dumps(token_data, ensure_ascii=False)[:500]
        return {"ok": False, "url": None,
                "error": f"获取 tenant_access_token 失败，{raw}"}
    token = token_data.get("tenant_access_token")

    # 2. 创建文档
    folder_warning = None
    try:
        with httpx.Client(timeout=30) as client:
            def _create_doc(target_folder: str | None) -> dict:
                body = {"title": title}
                if target_folder:
                    body["folder_token"] = target_folder
                return client.post(
                    FEISHU_DOC_CREATE_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                ).json()

            data = _create_doc(folder_token)

            # 指定文件夹无权限（1770040）时，回退到默认位置重试
            if data.get("code") == 1770040 and folder_token:
                folder_warning = (
                    "指定的文件夹无写入权限（未将该文件夹授权给应用），"
                    "已改为在默认位置创建文档。如需写入指定文件夹，请在飞书云空间把该文件夹共享给应用。"
                )
                LOG.warning("文件夹无权限，回退默认位置: %s", data.get("msg"))
                data = _create_doc(None)

            if data.get("code") != 0:
                msg = data.get("msg", "创建文档失败")
                code = data.get("code")
                LOG.error("创建文档失败: %s", data)
                hint = ""
                if code == 1770040:
                    hint = "（应用无权在该位置创建文档，请把目标文件夹共享给应用，或清空文件夹 Token）"
                return {"ok": False, "url": None,
                        "error": f"创建文档失败（code={code}）: {msg}{hint}"}

            document_id = data["data"]["document"]["document_id"]
            # 从响应中取 URL（不同租户域名不同），取不到再用通用域名
            document_url = (
                data.get("data", {}).get("document", {}).get("url")
                or f"https://feishu.cn/docx/{document_id}"
            )

            # 3. 添加内容（处理嵌套列表的视觉缩进）
            blocks = markdown_to_feishu_blocks(markdown_content)

            # 批量写入（每批最多 50 块）
            batch_size = 50
            batch_errors: list[str] = []
            for i in range(0, len(blocks), batch_size):
                batch_no = i // batch_size + 1
                resp = client.post(
                    FEISHU_DOC_BLOCK_URL.format(document_id=document_id, block_id=document_id),
                    headers={"Authorization": f"Bearer {token}"},
                    json={"children": blocks[i:i + batch_size]},
                )
                try:
                    resp_data = resp.json()
                except ValueError:
                    batch_errors.append(
                        f"第 {batch_no} 批 HTTP {resp.status_code}：{resp.text[:200]}"
                    )
                    LOG.error("飞书写入第 %s 批返回非 JSON（HTTP %s）: %s",
                              batch_no, resp.status_code, resp.text[:300])
                    continue
                if resp_data.get("code") != 0:
                    batch_errors.append(
                        f"第 {batch_no} 批 code={resp_data.get('code')} {resp_data.get('msg', '')}"
                    )
                    LOG.error("添加内容批次 %s 失败: %s", batch_no, resp_data)

            if batch_errors:
                # 文档已创建但内容不完整（甚至全空）：必须按失败返回，不能给出空白文档链接
                return {
                    "ok": False,
                    "url": document_url,
                    "error": "文档已创建但内容写入失败：" + "；".join(batch_errors[:3]),
                }

            LOG.info("飞书文档创建成功: %s", document_url)
            result = {"ok": True, "url": document_url, "error": None}
            if folder_warning:
                result["warning"] = folder_warning
            return result

    except Exception as e:
        LOG.error("创建文档异常: %s", e)
        return {"ok": False, "url": None, "error": str(e)}


# 测试连接用的 demo 文档内容
DEMO_MARKDOWN = """# llm-bench 飞书连接测试

这是一份由 **llm-bench** 自动创建的测试文档。

## 测试项

- 鉴权（tenant_access_token）
- 文档创建
- 内容写入（标题、段落、列表）

## 结论

如果你能看到这份文档，说明飞书集成配置正确。

> 可安全删除此文档。
"""


def test_feishu_connection(
    app_id: str,
    app_secret: str,
    folder_token: str = "",
) -> dict[str, Any]:
    """测试飞书连接：创建一个 demo 文档

    Returns:
        {"ok": bool, "url": str|None, "error": str|None}
    """
    title = f"llm-bench 飞书连接测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return create_feishu_document(
        title=title,
        markdown_content=DEMO_MARKDOWN,
        app_id=app_id,
        app_secret=app_secret,
        folder_token=folder_token or None,
    )


def export_to_feishu(
    analysis_path: Path,
    title: str | None = None,
) -> dict[str, Any]:
    """将分析报告导出为飞书文档

    Returns:
        {"ok": bool, "url": str|None, "error": str|None, "warning": str|None}
    """
    if not analysis_path.is_file():
        LOG.error("分析报告不存在: %s", analysis_path)
        return {"ok": False, "url": None, "error": f"报告文件不存在: {analysis_path.name}"}

    if title is None:
        title = f"llm-bench 压测报告 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    content = analysis_path.read_text(encoding="utf-8")
    return create_feishu_document(title, content)
