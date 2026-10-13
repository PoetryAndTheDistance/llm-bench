# llm-bench 打包说明

## 快速打包

```bash
# 在项目根目录执行
python build/build.py
```

打包完成后，发布包位于 `build/dist/llm-bench/` 目录。

## 发布包结构

```
build/dist/llm-bench/
├── llm-bench.exe              # 主程序（单文件）
├── configs/                   # 配置文件（用户可修改）
│   ├── providers.yaml         # 供应商/模型/API Key（从模板初始化）
│   ├── providers_template.yaml # 供应商配置模板（用于重置）
│   ├── preferences.yaml       # 用户偏好（从模板初始化，自动维护）
│   ├── preferences_template.yaml # 用户偏好模板（用于重置）
│   ├── feishu.yaml            # 飞书集成配置（从模板初始化）
│   ├── feishu_template.yaml   # 飞书配置模板（用于重置）
│   ├── benchmark.yaml         # 压测参数默认值
│   └── score_weights.yaml     # 评分权重配置
├── results/                   # 压测结果输出目录
└── README.md                  # 使用说明
```

## 设计理念

### 为什么配置文件外部化？

如果将配置文件打包进 exe，每次重启程序都会恢复默认配置，用户修改的供应商、API Key、偏好设置都会丢失。

通过配置文件外部化：
- 用户可以修改 `providers.yaml` 添加自己的供应商和 API Key
- 用户偏好自动保存到 `preferences.yaml`，重启后自动回显
- 压测结果保存在 `results/` 目录，持久化存储

### 路径解析逻辑

程序启动时自动检测运行环境：
- **开发环境**：配置文件在项目根目录的 `configs/`
- **打包环境**：配置文件在 exe 同级目录的 `configs/`

```python
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后
    CONFIGS_DIR = Path(sys.executable).parent / "configs"
else:
    # 开发环境
    CONFIGS_DIR = Path("configs")
```

## 打包依赖

```bash
pip install pyinstaller
```

其他依赖（fastapi, uvicorn, httpx, pyyaml）已在 requirements.txt 中。

## 注意事项

1. **PyInstaller 缓存**：首次打包较慢，后续会快一些
2. **文件大小**：单文件 exe 约 20-30 MB（包含所有依赖）
3. **杀毒软件**：PyInstaller 打包的 exe 可能被误报，需要添加白名单
4. **跨平台**：Windows 上打包生成 .exe，Linux/Mac 上生成对应可执行文件

## 自定义打包

如需自定义打包参数，编辑 `build/build.py` 中的 `build_exe()` 函数。

常用参数：
- `--onefile`：单文件模式（当前使用）
- `--windowed`：无控制台窗口（适合 GUI 应用）
- `--icon=icon.ico`：自定义图标
- `--add-data`：包含额外数据文件
