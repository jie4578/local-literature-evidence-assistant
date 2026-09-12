"""完全离线的结构化多论文对比和摘取式摘要。"""

from __future__ import annotations

import re
from typing import Any, Iterable


def compare_papers(papers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    fields = ("title_candidate", "author_candidate", "year_candidate", "doi", "research_object", "sample_size")
    for paper in papers:
        facts = paper.get("facts", [])
        sample_facts = [f["evidence_quote"] for f in facts if f["category"] == "sample_size"]
        def quotes(categories):
            return [fact.get("evidence_quote_display", fact["evidence_quote"]) for fact in facts if fact["category"] in categories]
        row = {"file_name": paper.get("file_name")}
        row.update({field: paper.get(field) for field in fields})
        if not row.get("sample_size") and sample_facts:
            row["sample_size"] = sample_facts[0]
        row.update({"research_methods_keywords": [f["category"] for f in facts if f["category"] in {"randomization", "double_blind", "control_group"}],
                    "experimental_conditions": quotes({"temperature", "time", "concentration", "dose"}),
                    "statistical_information": quotes({"p_value", "confidence_interval", "mean_sd"}),
                    "major_results_original": quotes({"major_result"}),
                    "conclusion_original": quotes({"conclusion"}),
                    "limitations_original": quotes({"limitation"}),
                    "source_pages": sorted({(f["pdf_page_start"], f["pdf_page_end"]) for f in facts}),
                    "missing_fields": [field for field in fields if not paper.get(field)]})
        rows.append(row)
    return rows


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_for_match(value: str) -> str:
    value = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", value)
    return _normalize_space(value)


def _looks_like_subheading(line: str) -> bool:
    value = line.strip()
    words = value.split()
    return bool(1 < len(words) <= 8 and len(value) <= 70 and not re.search(r"[.!?;:。！？；]$", value)
                and re.match(r"^[A-Z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*)+$", value))


def _line_segments(value: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                segments.append(" ".join(current))
                current = []
            continue
        if _looks_like_subheading(line):
            if current:
                segments.append(" ".join(current))
                current = []
            segments.append(line)
        else:
            current.append(line)
    if current:
        segments.append(" ".join(current))
    return segments


def _split_sentences(value: str) -> list[str]:
    """重建段落句子，避免把 PDF 视觉换行误当成句子边界。"""
    sentences: list[str] = []
    for segment in _line_segments(value):
        compact = _normalize_space(segment)
        sentences.extend(part.strip() for part in re.split(r"(?<=[.!?;。！？；])\s+", compact) if part.strip())
    return sentences


def _raw_sentence_parts(value: str) -> list[str]:
    return [part.strip() for part in re.findall(r".+?(?:[.!?。！？](?=\s|$)|$)", value, flags=re.DOTALL) if part.strip()]


def _raw_quote_candidates(value: str) -> list[str]:
    """按原始行窗口重建证据，允许已知的安全断行连字符差异。"""
    lines = value.splitlines(keepends=True)
    candidates = _raw_sentence_parts(value)
    for start in range(len(lines)):
        joined = ""
        for end in range(start, min(len(lines), start + 20)):
            joined += lines[end]
            if re.search(r"[.!?。！？]\s*$", joined):
                candidates.append(joined.strip())
                break
    return candidates


def _is_metadata_candidate(sentence: str) -> bool:
    normalized = _normalize_space(sentence)
    lowered = normalized.casefold()
    blocked = {
        "research article", "open access", "original article", "article",
        "abstract", "introduction", "methods", "materials and methods",
        "results", "discussion", "conclusion", "references", "copyright",
    }
    if lowered in blocked or "doi.org/" in lowered or "@" in sentence:
        return True
    if any(term in lowered for term in ("financial disclosure", "no relevant financial", "competing interest", "funding", "author contributions", "acknowledg")):
        return True
    if normalized[:1].islower():
        return True
    if re.fullmatch(r"[A-Z][A-Z\s,.'-]{5,}", normalized):
        return True
    if any(term in lowered for term in ("university", "department of", "institute", "school of", "corresponding author")):
        return True
    if not re.search(r"[.!?。！？]$", normalized):
        return True
    return len(normalized) < 40


def _candidate_evidence(sentence: str, paper: dict[str, Any], section: dict[str, Any]) -> dict[str, Any] | None:
    target = _normalize_for_match(sentence)
    for page in paper.get("pages", []):
        number = page.get("page_number", 0)
        if not (section.get("page_start", 0) <= number <= section.get("page_end", 0)):
            continue
        raw_text = page.get("raw_text", page.get("text", ""))
        raw_parts = _raw_quote_candidates(raw_text)
        raw_parts.extend(line.strip() for line in raw_text.splitlines() if line.strip())
        for raw_part in raw_parts:
            if _normalize_for_match(raw_part) == target:
                return {
                    "text": sentence,
                    "section": section.get("title", "").strip() or "未标明章节",
                    "pdf_pages": [number],
                    "evidence_quote_raw": raw_part,
                    "evidence_quote_display": sentence,
                    "verified": raw_part in raw_text,
                    "quality_flags": [],
                }
    return None


def _score_sentence(sentence: str, section_name: str) -> int:
    lowered = sentence.casefold()
    terms = {
        "abstract": ("purpose", "objective", "aim", "investigat", "evaluat", "background"),
        "introduction": ("purpose", "objective", "aim", "background", "we sought"),
        "methods": ("random", "double-blind", "controlled", "trial", "sample", "dose", "week", "method", "lc-ms", "immunocapture", "monkey", "pk", "multiple-dose", "single-dose"),
        "results": ("p ", "p=", "p<", "%", "significant", "result", "mean", "improvement", "decreased", "increased", "deamidation", "ptm"),
        "conclusion": ("conclu", "suggest", "greater", "effective", "limitation", "showed", "improved"),
        "discussion": ("conclu", "limitation", "suggest", "result"),
    }
    return sum(1 for term in terms.get(section_name, ()) if term in lowered)


def extractive_summary(paper: dict[str, Any], max_sentences: int = 8) -> dict[str, Any]:
    selected: list[str] = []
    evidence: list[dict[str, Any]] = []
    sections = paper.get("sections", {})
    # 每个章节只使用 section_parser 已经截取的正文，绝不回退到首页前若干行。
    # 先为目的/背景、方法、结果和结论保留名额，再用引言或讨论补足，避免摘要被首页内容占满。
    quotas = (("abstract", 2), ("methods", 2), ("results", 2), ("conclusion", 1), ("introduction", 1), ("discussion", 1))
    for name, quota in quotas:
        section = sections.get(name, {})
        if not section or not section.get("text"):
            continue
        candidates = []
        for sentence in _split_sentences(section["text"]):
            if "�" in sentence or _is_metadata_candidate(sentence):
                continue
            item = _candidate_evidence(sentence, paper, section)
            if item and item["verified"]:
                candidates.append((_score_sentence(sentence, name), item))
        candidates.sort(key=lambda pair: -pair[0])
        for _, item in candidates:
            if item["text"] in selected or _normalize_space(item["text"]) in {_normalize_space(x) for x in selected}:
                continue
            selected.append(item["text"])
            evidence.append(item)
            if len([x for x in evidence if x["section"] == section.get("title", "")]) >= quota or len(selected) >= max_sentences:
                break
        if len(selected) >= max_sentences:
            break
    return {
        "summary_type": "extractive",
        "label": "摘取式摘要",
        "description": "摘取式摘要：仅选取论文原句，不代表 AI 生成或事实核验。",
        "sentences": selected,
        "evidence": evidence,
    }
