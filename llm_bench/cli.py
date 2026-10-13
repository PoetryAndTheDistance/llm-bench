"""CLI entry point for llm-bench

主要使用方式：通过 Web UI 进行所有操作。
CLI 仅保留 web（启动服务）、compare（对比分析）、suite（批量任务）命令。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_bench import __version__
from llm_bench.config import DEFAULT_OUTPUT_DIR, DEFAULT_WEIGHTS_PATH


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-bench",
        description="LLM API 性能压测工具 — 通过 Web UI 配置和执行压测",
    )
    p.add_argument("--version", action="version", version=f"llm-bench {__version__}")
    sub = p.add_subparsers(dest="command", required=False)

    # ── web 子命令（主要入口）──
    web = sub.add_parser("web", help="启动 Web UI（主要使用方式）")
    web.add_argument("--host", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
    web.add_argument("--port", type=int, default=8080, help="监听端口 (默认: 8080)")

    # ── compare 子命令 ──
    cmp = sub.add_parser("compare", help="对比多次压测结果")
    cmp.add_argument("paths", nargs="+", type=Path, help="结果目录（包含 summary.json）")
    cmp.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="输出目录 (默认: 当前目录)",
    )
    cmp.add_argument("--weights", type=Path, default=None, help="评分权重文件")

    # ── suite 子命令 ──
    suite = sub.add_parser("suite", help="批量执行多个供应商/模型的压测")
    suite.add_argument("--config", type=Path, required=True, help="YAML 配置文件")
    suite.add_argument("--output-dir", type=Path, default=None)
    suite.add_argument("--only", action="append", default=None, help="只运行指定供应商")
    suite.add_argument("--model", action="append", default=None, help="只运行指定模型")
    suite.add_argument("--duration", type=float, default=None, help="压测时长（秒）")
    suite.add_argument("--requests", type=int, default=None, help="请求次数")
    suite.add_argument("--concurrency", default=None, help="并发数")
    suite.add_argument("--max-tokens", type=int, default=None)
    suite.add_argument("--skip-compare", action="store_true")
    suite.add_argument("--dry-run", action="store_true")

    return p


def _cmd_web(args: argparse.Namespace) -> int:
    """启动 Web UI"""
    from llm_bench.web.server import run_web_server
    try:
        run_web_server(
            host=args.host,
            port=args.port,
        )
    except ImportError as e:
        print(f"错误: {e}", file=sys.stderr)
        print("请安装 Web UI 依赖: pip install fastapi uvicorn websockets")
        return 2
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """对比多次压测结果"""
    from llm_bench.report import compare_paths
    weights = args.weights or (DEFAULT_WEIGHTS_PATH if DEFAULT_WEIGHTS_PATH.is_file() else None)
    try:
        compare_paths(args.paths, output=args.output, weights_path=weights)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    """批量执行压测"""
    from llm_bench.suite import SuiteConfigError, run_suite
    from llm_bench.config import parse_concurrency

    try:
        conc = parse_concurrency(args.concurrency) if args.concurrency is not None else None
    except (ValueError, TypeError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2

    if (args.duration is not None) == (args.requests is not None):
        if args.duration is not None or args.requests is not None:
            print("错误: --duration 和 --requests 二选一", file=sys.stderr)
            return 2

    try:
        return run_suite(
            args.config,
            output_dir=args.output_dir,
            only=args.only,
            skip_compare=args.skip_compare,
            dry_run=args.dry_run,
            only_models=args.model,
            duration=args.duration,
            requests=args.requests,
            concurrency=conc,
            max_tokens=args.max_tokens,
        )
    except SuiteConfigError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"未知错误: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    from llm_bench.config import apply_dotenv
    apply_dotenv()

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 默认启动 Web UI（双击 exe 时没有参数）
    if args.command is None:
        args.command = "web"
        args.host = "127.0.0.1"
        args.port = 8080

    if args.command == "web":
        return _cmd_web(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "suite":
        return _cmd_suite(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
