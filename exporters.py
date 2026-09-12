"""本地结构化结果导出。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from docx import Document


def export_json(value: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def export_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    return target


def export_excel(rows: list[dict[str, Any]], path: str | Path) -> Path:
    from openpyxl import Workbook
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    keys = list(rows[0].keys()) if rows else []
    sheet.append(keys)
    for row in rows:
        sheet.append([json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in keys])
    workbook.save(target)
    return target


def export_word(rows: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_heading("本地结构化文献对比结果", level=1)
    document.add_paragraph("本文件仅包含本地规则提取的结构化结果和原文证据，不是 AI 生成的完整文献综述。")
    for row in rows:
        document.add_heading(str(row.get("title_candidate") or row.get("file_name") or "未命名论文"), level=2)
        for key, value in row.items():
            document.add_paragraph(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value}")
    document.save(target)
    return target
