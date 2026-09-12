"""基于明确标题的保守章节识别。"""

from __future__ import annotations

import re
from typing import Any, Iterable


SECTION_ALIASES = {
    "abstract": ("abstract", "摘要"),
    "introduction": ("introduction", "background", "背景", "引言", "绪论"),
    "methods": ("methods", "method", "materials and methods", "研究方法", "方法"),
    "results": ("results", "结果", "研究结果"),
    "discussion": ("discussion", "讨论"),
    "conclusion": ("conclusion", "conclusions", "结论"),
    "references": ("references", "bibliography", "参考文献"),
}


def _heading_kind(line: str) -> str | None:
    cleaned = re.sub(r"^\s*(?:\d+(?:\.\d+)*[.)、]?|[IVX]+[.)])\s*", "", line).strip(" #*\t")
    if len(cleaned) > 80 or not cleaned:
        return None
    for kind, aliases in SECTION_ALIASES.items():
        if cleaned.casefold() in {alias.casefold() for alias in aliases}:
            return kind
    return None


def parse_sections(pages: Iterable[Any]) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for page in pages:
        page_text = page.get("display_text", page.get("text", "")) if isinstance(page, dict) else page.text
        page_number = page.get("page_number", 0) if isinstance(page, dict) else page.page_number
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            kind = _heading_kind(line)
            if kind:
                if kind == "references" and "references" not in sections:
                    sections[kind] = {"title": line, "text": "", "page_start": page_number, "page_end": page_number}
                    current = sections[kind]
                else:
                    current = sections.setdefault(kind, {"title": line, "text": "", "page_start": page_number, "page_end": page_number})
                    current["page_end"] = page_number
                continue
            if current is not None and line:
                current["text"] = (current["text"] + "\n" + line).strip()
                current["page_end"] = page_number
    return sections
