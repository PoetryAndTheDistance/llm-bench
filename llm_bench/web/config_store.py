"""注释保留型 YAML 配置读写

基于 ruamel.yaml 的 round-trip 模式：加载 → 修改 → 保存时
完整保留注释、键顺序、引号风格和空行。

适用于会被 Web UI 反复编辑、且人工维护注释的配置文件：
- providers.yaml（供应商/模型）
- feishu.yaml（飞书配置）
- preferences.yaml（用户偏好）

3.4: 简化的文件锁机制（跨平台兼容）
- 使用临时文件 + 重命名的原子写入
- 轻量级文件锁（适用于低并发场景）
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 4096  # 避免长字符串被折行
    return y


class _FileLock:
    """轻量级文件锁（跨平台兼容）

    使用 lock 文件实现简单的互斥锁。
    适用于低并发场景（如 Web UI 的配置编辑）。

    注意：这不是一个完全可靠的锁机制，但在单进程、低并发场景下足够用。
    高并发场景建议使用数据库或专业的锁服务。
    """

    def __init__(self, path: Path, timeout: float = 5.0):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.timeout = timeout
        self._fd = -1

    def acquire(self) -> None:
        """获取锁"""
        start = time.monotonic()
        while True:
            try:
                # 原子创建锁文件（O_CREAT | O_EXCL）
                self._fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644
                )
                # 写入 PID 便于调试
                os.write(self._fd, str(os.getpid()).encode())
                return
            except FileExistsError:
                # 锁已被持有，检查是否过期（防止死锁）
                try:
                    mtime = os.path.getmtime(str(self.lock_path))
                    if time.time() - mtime > 60:  # 60秒过期（更宽松）
                        os.unlink(str(self.lock_path))
                        continue
                except OSError:
                    # 锁文件可能被其他进程删除，重试
                    pass

                if time.monotonic() - start > self.timeout:
                    raise TimeoutError(f"获取文件锁超时: {self.lock_path}")
                time.sleep(0.1)
            except OSError as e:
                raise RuntimeError(f"获取文件锁失败: {e}")

    def release(self) -> None:
        """释放锁"""
        # 关闭文件描述符
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = -1

        # 删除锁文件
        try:
            os.unlink(str(self.lock_path))
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def load_yaml(path: Path | str, default: Any = None) -> Any:
    """加载 YAML，文件不存在时返回 default"""
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        data = _yaml().load(f)
    return data if data is not None else default


def save_yaml(path: Path | str, data: Any) -> None:
    """保存 YAML（保留已有注释）

    使用文件锁 + 原子写入防止并发写入导致数据损坏。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with _FileLock(p):
        # 写入临时文件
        dir_path = p.parent
        fd, tmp_path_str = tempfile.mkstemp(
            dir=dir_path,
            prefix=f".{p.name}.",
            suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                _yaml().dump(data, tmp_f)

            # os.replace 在 Windows/Unix 上都是原子操作（直接覆盖目标），
            # 避免「先删除再重命名」中途失败导致原文件丢失
            os.replace(tmp_path_str, str(p))
        except Exception:
            # 清理临时文件
            try:
                os.unlink(tmp_path_str)
            except OSError:
                pass
            raise


def update_yaml(path: Path | str, mutator) -> Any:
    """加载 YAML → 用 mutator(data) 原地修改 → 保存

    mutator 接收顶层数据（通常是 CommentedMap）并原地修改。
    文件不存在时以空 dict 初始化。

    Returns:
        修改后的数据
    """
    p = Path(path)

    with _FileLock(p):
        # 在锁内完成读-改-写
        data = load_yaml(p, default=CommentedMap())
        if data is None:
            data = CommentedMap()
        mutator(data)

        # 直接写入（已持有锁）
        p.parent.mkdir(parents=True, exist_ok=True)
        dir_path = p.parent
        fd, tmp_path_str = tempfile.mkstemp(
            dir=dir_path,
            prefix=f".{p.name}.",
            suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                _yaml().dump(data, tmp_f)

            # os.replace 在 Windows/Unix 上都是原子操作（直接覆盖目标）
            os.replace(tmp_path_str, str(p))
        except Exception:
            try:
                os.unlink(tmp_path_str)
            except OSError:
                pass
            raise

    return data


def to_plain(obj: Any) -> Any:
    """将 ruamel 的 CommentedMap/CommentedSeq 递归转为普通 dict/list

    用于向前端 / JSON 返回数据（FastAPI 通常也能序列化，显式转换更稳妥）。
    """
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def new_map() -> CommentedMap:
    return CommentedMap()


def new_seq(items: list | None = None) -> CommentedSeq:
    seq = CommentedSeq()
    for item in (items or []):
        seq.append(item)
    return seq
