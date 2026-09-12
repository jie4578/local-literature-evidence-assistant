"""完全离线的分页 PDF 提取、章节识别、元数据和事实证据整理。"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz

from paper_pipeline import PipelineConfig, build_chunks, extract_pdf_pages
from scientific_facts import extract_scientific_facts
from section_parser import parse_sections


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


GENERIC_MARKERS = {"research article", "open access", "original article", "article", "abstract", "plos one"}


def _valid_title(value: str | None) -> bool:
    if not value:
        return False
    clean = re.sub(r"\s+", " ", value).strip(" -")
    return bool(clean and clean.casefold() not in GENERIC_MARKERS and not DOI_RE.search(clean) and "open access" not in clean.casefold())


def _clean_page_text(raw: str, repeated_lines: set[str]) -> tuple[str, list[str]]:
    flags = []
    if "\u00ad" in raw:
        flags.append("soft_hyphen")
    if "�" in raw:
        flags.append("replacement_character")
    text = unicodedata.normalize("NFKC", raw.replace("\u00ad", ""))
    safe_chars = []
    for char in text:
        if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf"}:
            safe_chars.append(char)
        else:
            flags.append("raw_control_character_removed_for_display")
    text = "".join(safe_chars)
    lines = text.splitlines()
    kept = []
    for number, line in enumerate(lines):
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized in repeated_lines and (number < 3 or number >= len(lines) - 3):
            flags.append("repeated_header_footer")
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = re.sub(r"(?<=[A-Za-z])-[ \t]*(?:\r?\n)+[ \t]*(?=[a-z])", "", text)
    return text, sorted(set(flags))


def _metadata(path: Path, pages: list[Any]) -> dict[str, Any]:
    with fitz.open(path) as doc:
        metadata = doc.metadata or {}
        first_page = doc[0]
        blocks = first_page.get_text("dict").get("blocks", [])
        first_lines = [line.strip() for page in pages[:2] for line in page.text.splitlines() if line.strip()]
        title = metadata.get("title") if _valid_title(metadata.get("title")) else None
        title_source, title_confidence = ("pdf_metadata", 0.98) if title else (None, 0.0)
        if not title:
            candidates = []
            for block in blocks:
                spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
                large = [span.get("text", "").strip() for span in spans if float(span.get("size", 0)) >= 13]
                candidate = re.sub(r"\s+", " ", " ".join(x for x in large if x)).strip()
                if _valid_title(candidate):
                    candidates.append(candidate)
            if candidates:
                title, title_source, title_confidence = candidates[0], "first_page_layout", 0.8
            elif first_lines and _valid_title(first_lines[0]):
                title, title_source, title_confidence = first_lines[0], "first_page_text", 0.55
        author_value = metadata.get("author") if metadata.get("author") else None
        if author_value:
            author_value = author_value.split(",")[0].strip()
        if not author_value or author_value.casefold() in GENERIC_MARKERS:
            author_value = None
            for line in first_lines[1:8]:
                if ("," in line or re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", line)) and not re.search(r"open access|doi:|@|university|pharmaceutical|abstract", line, re.IGNORECASE):
                    author_value = line.split(",")[0].strip(" *†")
                    break
        author_source = "pdf_metadata" if metadata.get("author") else ("first_page_layout" if author_value else None)
        author_confidence = 0.98 if author_source == "pdf_metadata" else (0.65 if author_value else 0.0)
        joined = "\n".join(page.text for page in pages[:3])
        published = re.search(r"(?:published|publication date)\s*[:：]?[^\n]{0,80}?\b((?:19|20)\d{2})\b", joined, re.IGNORECASE)
        citation = re.search(r"(?:citation|\b(?:BMC Medicine|PLOS ONE)\b)[^\n]{0,100}?\b((?:19|20)\d{2})\b", joined, re.IGNORECASE)
        metadata_year = re.search(r"\b((?:19|20)\d{2})\b", metadata.get("creationDate", ""))
        year_match = published or citation or metadata_year
        year_source = "published_date" if published else ("citation" if citation else ("pdf_metadata" if metadata_year else None))
        return {"title_candidate": title, "title_source": title_source, "title_confidence": title_confidence,
                "author_candidate": author_value, "author_source": author_source, "author_confidence": author_confidence,
                "year_candidate": int(year_match.group(1)) if year_match else None, "year_source": year_source,
                "year_confidence": 0.98 if published else (0.9 if citation else (0.75 if metadata_year else 0.0)),
                "doi": (DOI_RE.search(metadata.get("subject", "")) or DOI_RE.search(joined)).group(0).rstrip(".,)") if (DOI_RE.search(metadata.get("subject", "")) or DOI_RE.search(joined)) else None}


def extract_local_paper(file_path: str | Path, config: PipelineConfig | None = None) -> dict[str, Any]:
    path = Path(file_path)
    result: dict[str, Any] = {"file_name": path.name, "pages": [], "chunks": [], "sections": {}, "facts": [], "warnings": [], "errors": []}
    try:
        pages = extract_pdf_pages(path)
    except ValueError as exc:
        result["errors"] = [str(exc)]
        return result
    raw_lines = []
    for page in pages:
        raw_lines.extend(re.sub(r"\s+", " ", line).strip() for line in page.text.splitlines() if line.strip())
    counts = {line: raw_lines.count(line) for line in set(raw_lines)}
    repeated = {line for line, count in counts.items() if count >= 3}
    page_dicts = []
    for page in pages:
        display, flags = _clean_page_text(page.text, repeated)
        item = page.to_dict()
        item.update({"raw_text": page.text, "display_text": display, "text_quality_flags": flags})
        page_dicts.append(item)
    result["pages"] = page_dicts
    result["text_quality_flags"] = sorted({flag for page in page_dicts for flag in page.get("text_quality_flags", [])})
    result["chunks"] = [chunk.to_dict() for chunk in build_chunks(pages, config)]
    result.update(_metadata(path, pages))
    result["sections"] = parse_sections(page_dicts)
    result["facts"] = extract_scientific_facts(page_dicts)
    empty_pages = [page.page_number for page in pages if page.is_empty]
    if empty_pages:
        result["warnings"].append(f"第 {','.join(map(str, empty_pages))} 页无可提取文字，可能是扫描页")
    if "replacement_character" in result["text_quality_flags"]:
        result["warnings"].append("文本包含不可恢复的替换字符，相关证据已保留质量标记，未猜测恢复")
    if "repeated_header_footer" in result["text_quality_flags"]:
        result["warnings"].append("已从展示文本和规则事实候选中排除重复页眉页脚，原始文本仍保留")
    if not result["sections"]:
        result["warnings"].append("未识别到明确章节标题")
    return result
