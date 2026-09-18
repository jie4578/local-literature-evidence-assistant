"""离线证据段落构建。

该模块只把已有 PDF 页面和保守章节识别结果整理成可检索的原文段落，
不做语义推断、摘要、翻译或事实真值判断。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from section_parser import _heading_kind


DEFAULT_PASSAGE_MAX_CHARS = 1000

_SECTION_LABELS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "methods": "Methods",
    "results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "references": "References",
}


@dataclass(frozen=True)
class EvidencePassage:
    passage_id: str
    source_file: str
    document_id: str | None
    section: str
    text: str
    pdf_page_start: int
    pdf_page_end: int
    ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _page_value(page: Any, field: str, default: Any = "") -> Any:
    if isinstance(page, Mapping):
        return page.get(field, default)
    return getattr(page, field, default)


def _display_text(page: Any) -> str:
    display = _page_value(page, "display_text", None)
    if display is None:
        display = _page_value(page, "text", "")
    return str(display or "")


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _split_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?。！？；;])\s+", clean)
    return [part.strip() for part in parts if part.strip()]


def _split_long_text(text: str, max_chars: int) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) <= max_chars:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if current and len(candidate) > max_chars:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        remaining = sentence
        while len(remaining) > max_chars:
            cut = remaining.rfind(" ", 0, max_chars + 1)
            cut = cut if cut >= max_chars // 2 else max_chars
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            current = remaining
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece]


def _page_units(pages: Iterable[Any]) -> list[dict[str, Any]]:
    """按页和保守标题重建段落单元，保留每个单元的真实页码。"""
    units: list[dict[str, Any]] = []
    current_section = "Unknown"
    for page in pages:
        page_number = int(_page_value(page, "page_number", 0) or 0)
        if page_number <= 0:
            continue
        lines: list[str] = []
        for raw_line in _display_text(page).splitlines():
            line = _normalize_line(raw_line)
            if not line:
                if lines:
                    units.append({"section": current_section, "text": " ".join(lines), "start": page_number, "end": page_number})
                    lines = []
                continue
            heading = _heading_kind(line)
            if heading == "excluded":
                if lines:
                    units.append({"section": current_section, "text": " ".join(lines), "start": page_number, "end": page_number})
                    lines = []
                current_section = "Unknown"
                continue
            if heading:
                if lines:
                    units.append({"section": current_section, "text": " ".join(lines), "start": page_number, "end": page_number})
                    lines = []
                current_section = _SECTION_LABELS.get(heading, "Unknown")
                continue
            lines.append(line)
        if lines:
            units.append({"section": current_section, "text": " ".join(lines), "start": page_number, "end": page_number})
    return units


def build_evidence_passages(
    paper_or_pages: Mapping[str, Any] | Iterable[Any],
    *,
    max_chars: int = DEFAULT_PASSAGE_MAX_CHARS,
) -> list[EvidencePassage]:
    """从页面构建稳定、带物理页码的证据段落。"""
    if isinstance(paper_or_pages, Mapping):
        pages = paper_or_pages.get("pages", []) or []
        document_id = paper_or_pages.get("document_id")
    else:
        pages = list(paper_or_pages)
        document_id = None
    if max_chars < 120:
        raise ValueError("max_chars 过小，无法保留可读证据段落")
    if not pages:
        return []
    source_file = str(_page_value(pages[0], "source_file", ""))
    pieces: list[dict[str, Any]] = []
    for unit in _page_units(pages):
        for text in _split_long_text(unit["text"], max_chars):
            pieces.append({**unit, "text": text})

    grouped: list[dict[str, Any]] = []
    for piece in pieces:
        if grouped and grouped[-1]["section"] == piece["section"]:
            candidate = f"{grouped[-1]['text']} {piece['text']}"
            if len(candidate) <= max_chars:
                grouped[-1]["text"] = candidate
                grouped[-1]["end"] = piece["end"]
                continue
        grouped.append(dict(piece))

    passages: list[EvidencePassage] = []
    for ordinal, item in enumerate(grouped, 1):
        text = re.sub(r"\s+", " ", item["text"]).strip()
        if not text:
            continue
        identity = "|".join(
            [
                str(document_id or source_file),
                item["section"],
                str(item["start"]),
                str(item["end"]),
                str(ordinal),
                hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            ]
        )
        passage_id = "passage_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        passages.append(
            EvidencePassage(
                passage_id=passage_id,
                source_file=source_file,
                document_id=str(document_id) if document_id else None,
                section=item["section"],
                text=text,
                pdf_page_start=int(item["start"]),
                pdf_page_end=int(item["end"]),
                ordinal=ordinal,
            )
        )
    return passages
