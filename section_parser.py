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

EXCLUDED_SECTION_ALIASES = (
    "authors' contributions",
    "author contributions",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "availability of data and materials",
    "ethics approval",
    "consent for publication",
    "competing interests",
    "publisher's note",
)


def _heading_kind(line: str) -> str | None:
    cleaned = re.sub(r"^\s*(?:\d+(?:\.\d+)*[.)、]?|[IVX]+[.)])\s*", "", line).strip(" #*\t")
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    if not cleaned:
        return None
    folded = cleaned.casefold()
    for alias in EXCLUDED_SECTION_ALIASES:
        if folded == alias.casefold() or folded.startswith(alias.casefold() + " ") or folded.startswith(alias.casefold() + ":"):
            return "excluded"
    if len(cleaned) > 80:
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
            if kind == "excluded":
                # 署名、资助、伦理和出版声明从这里开始不再进入任何
                # 后续章节，避免污染摘取式摘要和结论候选。
                current = None
                continue
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
