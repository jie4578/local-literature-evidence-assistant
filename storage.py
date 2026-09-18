"""安全的本地 JSON 原子写入工具。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(value: Any, path: str | Path) -> Path:
    """在目标文件同目录完成写入、flush/fsync 后原子替换。

    os.replace 在 Windows 上同样提供同一文件系统内的替换语义；
    失败时保留旧文件，并清理临时文件。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                # 某些文件系统不支持 fsync；仍保留 flush 后的原子替换。
                pass
        os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass
