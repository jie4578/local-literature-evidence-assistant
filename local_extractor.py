"""完全离线的分页 PDF 提取、章节识别、元数据和事实证据整理。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from paper_pipeline import PipelineConfig, build_chunks, extract_pdf_pages
from scientific_facts import extract_scientific_facts
from section_parser import parse_sections


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _metadata(pages: list[Any]) -> dict[str, Any]:
    first_lines = [line.strip() for page in pages[:2] for line in page.text.splitlines() if line.strip()]
    title = first_lines[0] if first_lines else None
    authors = None
    if len(first_lines) > 1 and not re.search(r"(?:purpose|objective|abstract|摘要|研究目的)", first_lines[1], re.IGNORECASE):
        authors = first_lines[1]
    joined = "\n".join(page.text for page in pages[:2])
    doi = DOI_RE.search(joined)
    years = YEAR_RE.findall(joined)
    return {"title_candidate": title, "author_candidate": authors,
            "year_candidate": int(years[0]) if years else None,
            "doi": doi.group(0).rstrip(".,)") if doi else None}


def extract_local_paper(file_path: str | Path, config: PipelineConfig | None = None) -> dict[str, Any]:
    path = Path(file_path)
    result: dict[str, Any] = {"file_name": path.name, "pages": [], "chunks": [], "sections": {}, "facts": [], "warnings": [], "errors": []}
    try:
        pages = extract_pdf_pages(path)
    except ValueError as exc:
        result["errors"] = [str(exc)]
        return result
    result["pages"] = [page.to_dict() for page in pages]
    result["chunks"] = [chunk.to_dict() for chunk in build_chunks(pages, config)]
    result.update(_metadata(pages))
    result["sections"] = parse_sections(pages)
    result["facts"] = extract_scientific_facts(pages)
    empty_pages = [page.page_number for page in pages if page.is_empty]
    if empty_pages:
        result["warnings"].append(f"第 {','.join(map(str, empty_pages))} 页无可提取文字，可能是扫描页")
    if not result["sections"]:
        result["warnings"].append("未识别到明确章节标题")
    return result
