import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    """在工作区创建可写临时目录，避免受限 Windows 临时目录权限影响。"""
    path = Path.cwd() / ".test_tmp" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
