#!/usr/bin/env python3
"""
llm-bench 打包脚本

功能：
1. 使用 PyInstaller 打包为单个 exe 文件
2. 自动创建发布目录，包含 exe + configs + results
3. 确保配置文件外部化，用户修改可持久化

用法：
    python build/build.py

发布目录结构：
    dist/llm-bench/
    ├── llm-bench.exe          # 主程序
    ├── configs/               # 配置文件（用户可修改）
    │   ├── providers.yaml
    │   ├── benchmark.yaml
    │   ├── preferences.yaml
    │   └── feishu.yaml
    └── results/               # 压测结果输出目录
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = BUILD_DIR / "dist"
RELEASE_DIR = DIST_DIR / "llm-bench"

# PyInstaller 临时文件目录
PYINSTALLER_WORK_DIR = BUILD_DIR / "pyinstaller_work"

# 需要复制的配置文件（模板文件 → 目标文件）
CONFIG_FILES = [
    ("providers_template.yaml", "providers.yaml"),
    ("preferences_template.yaml", "preferences.yaml"),
    ("feishu_template.yaml", "feishu.yaml"),
    ("benchmark.yaml", "benchmark.yaml"),
    ("score_weights.yaml", "score_weights.yaml"),
]

# 需要额外复制的模板文件（保留 _template 后缀，供用户重置使用）
TEMPLATE_BACKUP_FILES = [
    "providers_template.yaml",
    "preferences_template.yaml",
    "feishu_template.yaml",
]


def remove_readonly(func, path, excinfo):
    """错误处理回调：处理只读文件和锁定文件"""
    try:
        # 尝试移除只读属性并重试
        os.chmod(path, 0o666)
        func(path)
    except Exception:
        # 如果仍然失败，跳过该文件
        print(f"  ⚠ 无法删除（文件被锁定）: {path}")
        print(f"     请关闭占用该文件的程序后重试")


def clean_directory_with_retry(path, max_retries=3, delay=1):
    """带重试机制的目录清理函数"""
    if not path.exists():
        return True

    for attempt in range(max_retries):
        try:
            shutil.rmtree(path, onerror=remove_readonly)
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ 目录被锁定，{delay}秒后重试 ({attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"  ⚠ 无法删除目录（多次重试失败）: {path}")
                print(f"     错误: {e}")
                print(f"     请关闭占用该目录的程序后重试")
                return False
        except Exception as e:
            print(f"  ⚠ 删除目录时出错: {e}")
            return False

    return False


def check_dependencies():
    """检查打包依赖"""
    print("🔍 检查打包依赖...")

    try:
        import PyInstaller
        print(f"  ✓ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("  ✗ PyInstaller 未安装")
        print("  请运行: pip install pyinstaller")
        sys.exit(1)

    # 检查其他依赖
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "httpx": "httpx",
        "pyyaml": "yaml",
    }
    for display_name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} 未安装")
            sys.exit(1)


def clean_build():
    """清理旧的构建文件（只删除 PyInstaller 生成的文件）"""
    print("\n🧹 清理旧的构建文件...")

    # 只清理 PyInstaller 的临时目录和输出目录
    dirs_to_remove = [
        PYINSTALLER_WORK_DIR,  # PyInstaller 临时工作目录
        DIST_DIR,              # 最终输出目录
    ]

    for d in dirs_to_remove:
        if d.exists():
            if clean_directory_with_retry(d):
                print(f"  ✓ 已删除: {d}")
            else:
                print(f"  ⚠ 跳过: {d}")

    # 清理 PyInstaller .spec 文件（在项目根目录）
    spec_file = PROJECT_ROOT / "llm-bench.spec"
    if spec_file.exists():
        try:
            spec_file.unlink()
            print(f"  ✓ 已删除: {spec_file}")
        except Exception as e:
            print(f"  ⚠ 无法删除 {spec_file}: {e}")


def clean_after_build():
    """构建完成后清理临时文件"""
    print("\n🧹 清理构建临时文件...")

    # 删除 PyInstaller 临时工作目录
    if PYINSTALLER_WORK_DIR.exists():
        if clean_directory_with_retry(PYINSTALLER_WORK_DIR):
            print(f"  ✓ 已删除: {PYINSTALLER_WORK_DIR}")
        else:
            print(f"  ⚠ 跳过: {PYINSTALLER_WORK_DIR}")

    # 删除 PyInstaller .spec 文件
    spec_file = PROJECT_ROOT / "llm-bench.spec"
    if spec_file.exists():
        try:
            spec_file.unlink()
            print(f"  ✓ 已删除: {spec_file}")
        except Exception as e:
            print(f"  ⚠ 无法删除 {spec_file}: {e}")


def build_exe():
    """使用 PyInstaller 构建 exe"""
    print("\n🔨 构建 exe 文件...")

    # PyInstaller 参数
    cmd = [
        "pyinstaller",
        "--onefile",              # 单个 exe 文件
        "--name", "llm-bench",   # exe 名称
        "--clean",                # 清理临时文件
        "--noconfirm",            # 不确认覆盖
        "--workpath", str(PYINSTALLER_WORK_DIR),  # 临时工作目录
        "--distpath", str(DIST_DIR),              # 输出目录
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "fastapi",
        "--hidden-import", "websockets",
        "--add-data", f"{PROJECT_ROOT / 'llm_bench' / 'web' / 'static'}{os.pathsep}llm_bench/web/static",
        str(PROJECT_ROOT / "llm_bench" / "__main__.py"),
    ]

    print(f"  执行: {' '.join(cmd[:5])}...")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=False)
        print("  ✓ exe 构建成功")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 构建失败: {e}")
        sys.exit(1)

    # 检查 exe 是否生成
    exe_path = DIST_DIR / "llm-bench.exe"
    if not exe_path.exists():
        print(f"  ✗ exe 文件未找到: {exe_path}")
        sys.exit(1)

    print(f"  ✓ 文件大小: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")


def create_release():
    """创建发布目录"""
    print("\n📦 创建发布目录...")

    # 创建发布目录
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  创建目录: {RELEASE_DIR}")

    # 移动 exe 到发布目录
    exe_src = DIST_DIR / "llm-bench.exe"
    exe_dst = RELEASE_DIR / "llm-bench.exe"
    shutil.move(str(exe_src), str(exe_dst))
    print(f"  ✓ 移动 exe: {exe_dst.name}")

    # 复制配置文件
    configs_src = PROJECT_ROOT / "configs"
    configs_dst = RELEASE_DIR / "configs"

    if configs_dst.exists():
        shutil.rmtree(configs_dst)

    configs_dst.mkdir()

    for src_name, dst_name in CONFIG_FILES:
        src = configs_src / src_name
        dst = configs_dst / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ 复制配置: {src_name} → {dst_name}")
        else:
            print(f"  ⚠ 配置文件不存在: {src}")

    # 复制模板备份文件（供用户重置配置使用）
    for template_name in TEMPLATE_BACKUP_FILES:
        src = configs_src / template_name
        dst = configs_dst / template_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ 复制模板: {template_name}")
        else:
            print(f"  ⚠ 模板文件不存在: {src}")

    # 创建 results 目录
    results_dir = RELEASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"  ✓ 创建结果目录: results/")

    # 创建 README（面向最终用户）
    create_release_readme()

    print(f"\n✅ 发布包创建完成: {RELEASE_DIR}")


def create_release_readme():
    """创建发布目录的 README（面向最终用户）"""
    readme_content = """# llm-bench

> LLM API 性能压测工具

## 目录结构

```
llm-bench/
├── llm-bench.exe       # 主程序
├── configs/            # 配置文件目录
│   ├── providers.yaml          # 供应商和模型配置
│   ├── providers_template.yaml # 供应商配置模板（用于重置）
│   ├── preferences.yaml        # 用户偏好（自动保存）
│   ├── preferences_template.yaml # 用户偏好模板（用于重置）
│   ├── feishu.yaml             # 飞书集成配置
│   ├── feishu_template.yaml    # 飞书配置模板（用于重置）
│   ├── benchmark.yaml          # 压测参数默认值
│   └── score_weights.yaml      # 评分权重配置
└── results/            # 压测结果输出目录
```

## 快速开始

### 1. 启动程序

**方式一：双击运行**
```
双击 llm-bench.exe
```

**方式二：命令行启动**
```bash
.\\llm-bench.exe web --port 8080
```

### 2. 使用 Web UI

1. 浏览器访问 http://127.0.0.1:8080
2. 所有配置（供应商、模型、压测参数等）均可在 Web UI 的 **配置** Tab 中完成
3. 点击 **▶️ 开始压测** 即可开始
4. 切换到其他 Tab 查看实时指标和历史记录

## 配置文件说明

所有配置均可通过 Web UI 进行可视化操作，无需手动编辑配置文件。

| 文件 | 用途 |
|------|------|
| `providers.yaml` | 供应商和模型配置（API Key 在此配置） |
| `preferences.yaml` | 用户偏好（自动生成，记录上次使用的参数） |
| `feishu.yaml` | 飞书集成配置（用于上传报告到飞书文档） |
| `benchmark.yaml` | 压测参数默认值 |
| `score_weights.yaml` | 综合评分的各维度权重 |

### 模板文件

`*_template.yaml` 文件是配置模板备份。如果修改配置后出现问题，可以：
1. 删除对应的配置文件（如 `providers.yaml`）
2. 复制模板文件并重命名（如 `providers_template.yaml` → `providers.yaml`）

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

## 飞书集成

支持将压测报告自动上传到飞书文档，可在 Web UI 中配置飞书凭证。

## 问题反馈

如有问题，请联系开发团队。
"""

    readme_path = RELEASE_DIR / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"  ✓ 创建 README.md")


def main():
    """主函数"""
    print("=" * 60)
    print("llm-bench 打包工具")
    print("=" * 60)

    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)

    # 执行打包流程
    check_dependencies()
    clean_build()
    build_exe()
    create_release()
    clean_after_build()

    print("\n" + "=" * 60)
    print("✅ 打包完成！")
    print(f"📁 发布目录: {RELEASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
