# llm-bench

> LLM API 性能压测工具 — 通过 Web UI 配置和执行压测

## 项目简介

llm-bench 是一个专业的 LLM API 性能压测工具，通过 **Web UI** 进行所有配置和操作。

### 核心功能

| 功能 | 说明 |
|------|------|
| **Web UI 实时看板** | Tab 分组：延迟指标、吞吐指标、Token 消耗、历史记录、配置面板 |
| **Token 消耗监控** | 按供应商/模型分组，实时显示输入/输出/缓存创建/缓存命中 |
| **一键启动压测** | 在配置面板选择供应商/模型，点击"开始压测" |
| **Anthropic 协议** | 同时支持 OpenAI 和 Anthropic 两种 API 风格 |
| **Prefill TPS** | 衡量模型处理输入的速度 |
| **L1/L2 缓存分层** | 区分缓存创建和缓存命中 |
| **自定义提示词** | 选择 custom 档位后可输入任意提示词 |
| **Markdown 报告** | 每次压测自动生成 report.md 分析报告 |
| **历史记录浏览** | 查看所有历史压测结果，支持下载和导出 |
| **飞书集成** | 基于飞书 Open API，自动上传报告到飞书文档 |

## 快速开始

### 1. 安装依赖

```bash
cd llm-bench
pip install -r requirements.txt
```

### 2. 启动 Web UI

```bash
python -m llm_bench web --port 8080
```

### 3. 配置与使用

1. 浏览器访问 `http://127.0.0.1:8080`
2. 切换到 **配置** Tab
3. 在 Web UI 中添加供应商、填写 API Key、选择模型
4. 配置压测参数
5. 点击 **▶️ 开始压测**
6. 切换到其他 Tab 查看实时指标

> 💡 所有配置均可通过 Web UI 完成，无需手动编辑配置文件。

## Web UI 界面

### Tab 1: ⏱️ 延迟指标
- TTFT / E2E 分位数表格
- TTFT 趋势图

### Tab 2: 📊 吞吐指标
- RPM / TPM / TPS 指标
- Decode TPS / Prefill TPS 趋势图

### Tab 3: 🎯 Token 消耗
- 按供应商/模型分组的实时 Token 统计
- 输入/输出/缓存创建/缓存命中/总 Token
- 合计行汇总

### Tab 4: ⚙️ 配置
- 供应商/模型选择
- Prompt 档位（含 custom 自定义）
- 压测参数配置
- 开始/停止按钮

## 项目结构

```
llm-bench/
├── llm_bench/
│   ├── __main__.py         # 程序入口
│   ├── cli.py              # CLI 入口（web/compare/suite 命令）
│   ├── config.py           # 配置管理
│   ├── config_store.py     # 配置存储
│   ├── client.py           # OpenAI 客户端
│   ├── anthropic_client.py # Anthropic 客户端
│   ├── runner.py           # 压测引擎
│   ├── metrics.py          # 指标采集
│   ├── prompts.py          # Prompt 生成
│   ├── report.py           # 报告生成
│   ├── suite.py            # 批量模式
│   ├── job_manager.py      # 任务管理
│   ├── feishu.py           # 飞书导出
│   ├── broadcaster.py      # 消息广播
│   ├── sse.py              # Server-Sent Events
│   ├── snapshot.py         # 快照管理
│   ├── usage.py            # 使用统计
│   ├── providers_api.py    # 供应商 API
│   ├── video_probe.py      # 视频探测
│   └── web/                # Web UI
│       ├── server.py       # FastAPI 服务
│       ├── websocket.py    # WebSocket 管理
│       └── static/         # 静态资源
│           ├── dashboard.html
│           ├── job-detail.html
│           ├── report-viewer.html
│           ├── job-report-template.html
│           ├── trend-chart.js
│           └── first-exchange.js
├── configs/
│   ├── providers.yaml          # 供应商/模型配置（API Key 在此配置）
│   ├── providers_template.yaml # 供应商配置模板
│   ├── benchmark.yaml          # 压测参数默认值
│   ├── preferences.yaml        # 用户偏好（自动生成）
│   ├── preferences_template.yaml # 用户偏好模板
│   ├── feishu.yaml             # 飞书集成配置
│   ├── feishu_template.yaml    # 飞书配置模板
│   └── score_weights.yaml      # 评分权重配置
├── build/                      # 打包脚本
│   ├── build.py
│   └── README.md
└── requirements.txt
```

## 配置文件

配置文件分为 5 个独立文件，职责清晰：

| 文件 | 用途 | 说明 |
|------|------|------|
| `configs/providers.yaml` | 供应商/模型配置 | API 密钥也配置在此文件 |
| `configs/benchmark.yaml` | 压测参数默认值 | 压测时的默认参数（prompt、并发数等） |
| `configs/preferences.yaml` | 用户偏好 | **自动生成**，记录上次使用的参数，重启后自动回显 |
| `configs/feishu.yaml` | 飞书集成配置 | 飞书报告导出相关设置 |
| `configs/score_weights.yaml` | 评分权重配置 | 综合评分的各维度权重 |

> 💡 **配置模板**：每个可配置文件都有对应的 `*_template.yaml` 模板文件。如果配置出错，可以用模板重置。

### configs/providers.yaml

```yaml
providers:
  - name: provider_name
    base_url: https://api.example.com
    api_style: openai  # 或 anthropic
    api_key: "your-api-key-here"  # 直接填写密钥
    enabled: true
    models:
      - model_name_1
      - model_name_2
```

### configs/benchmark.yaml

```yaml
defaults:
  prompt: medium
  max_tokens: 500
  concurrency: 1
  duration: 60
  stream: true
  cache_mode: record  # record=只记录 / warmup=先预热再测；输入需大于 1K tokens 才会触发模型缓存机制
  # ... 更多参数
```

### configs/preferences.yaml

此文件由程序自动维护，记录用户上次选择的参数。下次启动 Web UI 时自动回显。

## 压测结果

每次压测在 `results/` 目录下生成一个 `job_{时间戳}` 文件夹，包含：

```
job_20261013_152620/
├── analysis.md         # 对比分析报告（批量测试时生成）
├── compare.csv         # 对比数据（CSV 格式）
├── compare.md          # 对比报告（Markdown 格式）
├── job.json            # 任务信息
├── report.html         # HTML 格式报告
└── {供应商}_{模型}/    # 每个供应商/模型的测试结果
    ├── benchmark.log   # 运行日志
    ├── config.json     # 本次压测的完整配置快照
    ├── report.md       # Markdown 格式的分析报告
    ├── requests.csv    # 每个请求的详细数据
    ├── summary.json    # 汇总指标（JSON 格式）
    └── timeseries.csv  # 时间序列数据
```

### 历史记录与对比报告

通过 Web UI 的"历史记录" Tab 可以：

1. **浏览所有历史结果**
   - 查看所有压测结果列表
   - 显示关键指标：时间、供应商/模型、成功率、TTFT p50、RPM

2. **查看单次报告**
   - 点击"查看"按钮打开 Markdown 报告
   - 点击"下载"按钮下载报告文件
   - 点击"飞书"按钮上传到飞书文档

3. **生成对比报告**
   - 勾选多个压测结果（至少 2 个）
   - 点击"生成对比报告"按钮
   - 系统自动生成 `analysis.md` 和 `compare.md` 对比分析报告
   - 包含：综合评分排名、各维度对比、同模型跨供应商对比等

对比报告包含以下分析：
- **综合评分**：基于可用性、延迟、吞吐、稳定性等维度的加权评分
- **延迟对比**：TTFT、E2E 的分位数对比
- **吞吐对比**：RPM、TPM、TPS 的对比
- **稳定性分析**：CV 变异系数对比
- **缓存效果**：缓存命中率对比
- **同模型跨供应商**：同一模型在不同供应商上的表现对比

## 指标说明

| 指标 | 说明 |
|------|------|
| **TTFT** | Time To First Token — 首 token 延迟 |
| **E2E** | End-to-End — 端到端延迟 |
| **Decode TPS** | 解码吞吐（不含 TTFT） |
| **Prefill TPS** | 预填充吞吐 = input_tokens / TTFT |
| **Effective TPS** | 有效吞吐 = output_tokens / E2E |
| **RPM** | Requests Per Minute |
| **TPM** | Tokens Per Minute |
| **Cache L1** | 缓存创建 token 数 |
| **Cache L2** | 缓存命中 token 数 |

## 文档索引

| 文档 | 说明 |
|------|------|
| [评分规则说明](./评分规则说明.md) | 综合评分的计算逻辑、评分维度、权重配置、计算示例 |

## 飞书集成

基于飞书 Open API，无需安装额外工具。

### 配置步骤

1. **创建飞书应用**
   - 访问 https://open.feishu.cn/app
   - 创建企业自建应用
   - 在"凭证与基础信息"中获取 App ID 和 App Secret

2. **添加权限**
   - 在"权限管理"中添加：
     - `docx:document`（读写文档）
     - `drive:drive`（访问云空间）

3. **发布应用**
   - 发布应用并等待审批通过

4. **在 Web UI 中配置**
   - 切换到"配置" Tab
   - 在"飞书集成配置"区域填写：
     - App ID
     - App Secret
     - 文件夹 Token（可选，留空则保存到根目录）
   - 点击"保存飞书配置"

### 使用方法

1. 在"历史记录" Tab 中找到要导出的压测结果
2. 点击"飞书"按钮
3. 如果已配置飞书，会自动创建文档并返回链接
4. 如果未配置，会提示先在配置页面设置

### 可选：指定保存文件夹

如需将文档保存到特定文件夹：
1. 在飞书云空间中创建或找到目标文件夹
2. 从文件夹 URL 中提取 folder_token
   - 例如：`https://bytedance.feishu.cn/drive/folder/ABC123`
   - folder_token 为 `ABC123`
3. 在 Web UI 配置中填入该 token
