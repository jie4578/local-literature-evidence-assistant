"""完全离线的结构化多论文对比和摘取式摘要。"""

from __future__ import annotations

import re
from typing import Any, Iterable

from scientific_facts import repair_visual_word_breaks


LANGUAGE_OPTIONS = ("原文", "中文摘要＋英文证据（推荐）", "中英对照")

_METHOD_SECTION_KEYS = {"methods", "methodology", "materials_and_methods"}
_METHOD_FACT_CATEGORIES = {
    "research_method", "model_fit", "r_square", "randomization", "double_blind",
    "control_group", "group_count", "immunocapture_lc_ms", "affinity_purification_lc_ms",
    "single_dose_pk", "multiple_dose_pk", "randomized_controlled_trial", "dietary_intervention",
}
_METHOD_SEMANTIC_RE = re.compile(
    r"\b(?:used|measured|analy[sz]ed|fitt?(?:ed|ing)?|performed|assigned|collected|estimated|calculated|modeled|modelled)\b",
    re.IGNORECASE,
)
_RESULT_ONLY_RE = re.compile(
    r"\b(?:observed|increased|decreased|higher|lower|retained|improved|reduced|greater|unchanged|remained|unaffected|stable)\b|"
    r"\bp\s*[<=>]|\bR\s*(?:square|squared|2|²)\b",
    re.IGNORECASE,
)
_NON_METHOD_SECTION_RE = re.compile(
    r"^(?:keywords?|correspondence|acknowledg(?:e)?ments?|funding|availability of data and materials|"
    r"ethics approval|consent for publication|competing interests|publisher['’]s note)\b",
    re.IGNORECASE,
)


def _method_sentence_candidates(text: Any) -> list[str]:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", compact) if part.strip()]


def _strip_section_prefix(value: str) -> str:
    return re.sub(r"^(?:Methods?|Methodology|Materials\s+and\s+Methods?)\s+", "", value, flags=re.IGNORECASE).strip()


def _clean_sentence(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = _strip_section_prefix(text)
    text = re.sub(r"(?:\s*[.;；]+)+$", ".", text)
    return text


def _sentence_key(value: Any) -> str:
    text = _clean_sentence(value)
    # Treat common Chinese/English punctuation variants and punctuation
    # spacing as equivalent before comparing sentence-level evidence.
    text = (
        text.replace("；", ";")
        .replace("。", ".")
        .replace("：", ":")
        .replace("，", ",")
        .replace("！", "!")
        .replace("？", "?")
    )
    text = re.sub(r"\s*([,;:.!?])\s*", r"\1", text)
    text = re.sub(r"[.!?。！？；;]+$", "", text).strip()
    return text.casefold()


def _with_section_prefix(sentence: str, section_key: str) -> str:
    prefix = {
        "methods": "Methods",
        "methodology": "Methodology",
        "materials_and_methods": "Materials and Methods",
    }.get(section_key, "Methods")
    if re.match(r"^(?:Methods?|Methodology|Materials\s+and\s+Methods?)\b", sentence, re.IGNORECASE):
        return sentence
    return f"{prefix} {sentence}"


def _method_sentence_is_safe(sentence: str) -> bool:
    sentence = _clean_sentence(sentence)
    if len(sentence) < 12 or not re.search(r"[.!?。！？]$", sentence):
        return False
    if _NON_METHOD_SECTION_RE.match(sentence):
        return False
    # 结果信号或 R²/p 值本身不能把结果句冒充为方法；明确方法语义仍可保留。
    if _RESULT_ONLY_RE.search(sentence) and not _METHOD_SEMANTIC_RE.search(sentence):
        return False
    return True


def _page_contains_method_sentence(paper: dict[str, Any], sentence: str) -> bool:
    pages = paper.get("pages") or []
    if not pages:
        return True
    target = re.sub(r"\s+", " ", sentence).strip()
    for page in pages:
        source = page.get("display_text") or page.get("text") or page.get("raw_text") or ""
        if target in re.sub(r"\s+", " ", str(source)).strip():
            return True
    return False


def _collect_method_evidence(paper: dict[str, Any]) -> list[str]:
    """只从 Methods 原句或已验证方法事实中选择用户可读内容。"""
    ranked: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    order = 0
    for fact in paper.get("facts", []) or []:
        category = fact.get("category")
        section = str(fact.get("section") or "").casefold()
        if category not in _METHOD_FACT_CATEGORIES or fact.get("verified") is not True:
            continue
        if category in {"research_method", "model_fit", "r_square"} and section == "results":
            continue
        value = _clean_sentence(fact.get("evidence_quote_display") or fact.get("evidence_quote") or "")
        if not value or value in seen or not _method_sentence_is_safe(value):
            continue
        key = _sentence_key(value)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((3 if section in _METHOD_SECTION_KEYS else 2, order, value))
        order += 1

    sections = paper.get("sections") or {}
    for key in ("methods", "methodology", "materials_and_methods"):
        section = sections.get(key) or {}
        section_sentences = _method_sentence_candidates(section.get("text"))
        section_values: list[tuple[int, str]] = []
        for index, sentence in enumerate(section_sentences):
            sentence = _clean_sentence(sentence)
            if not _method_sentence_is_safe(sentence):
                continue
            display_sentence = _with_section_prefix(sentence, key) if index == 0 else sentence
            if not _page_contains_method_sentence(paper, display_sentence):
                continue
            if _RESULT_ONLY_RE.search(sentence) and not _METHOD_SEMANTIC_RE.search(sentence):
                continue
            # 明确方法语义优先；纯条件句留作无方法句时的最后保守兜底。
            condition_only = bool(re.search(
                r"\b(?:stored|temperature|degrees?|exposure|concentration|dose|for\s+\d+\s+weeks?)\b",
                sentence,
                re.IGNORECASE,
            )) and not re.search(r"\b(?:used|measured|analy[sz]ed|fitted|performed|assigned)\b", sentence, re.IGNORECASE)
            score = 2 if _METHOD_SEMANTIC_RE.search(sentence) else (0 if condition_only else 1)
            section_values.append((score, display_sentence))
            order += 1
        for score, value in section_values:
            key_value = _sentence_key(value)
            if key_value in seen:
                continue
            seen.add(key_value)
            ranked.append((score, order, value))
            order += 1

    ranked.sort(key=lambda item: (-item[0], item[1]))
    preferred = [item for item in ranked if item[0] >= 2]
    if preferred:
        ranked = preferred
    elif ranked:
        # 只有章节正文、没有显式方法动词时，保留第一条可回查完整句。
        ranked = [ranked[0]]
    return [value for _, _, value in ranked[:3]]


def compare_papers(papers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成不改写原文的对比行，并在回填后再计算 missing_fields。"""
    rows = []
    fields = ("title_candidate", "author_candidate", "year_candidate", "doi", "research_object", "sample_size")
    method_order = ("research_method", "randomization", "double_blind", "control_group", "group_count")

    def normalized(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def sample_text(facts: list[dict[str, Any]]) -> str | None:
        candidates: list[tuple[int, str]] = []
        for index, fact in enumerate(facts):
            if fact.get("category") != "sample_size":
                continue
            value = normalized(fact.get("evidence_quote_display", fact.get("evidence_quote", "")))
            if not value:
                continue
            if fact.get("source_type") == "possible_table":
                continue
            merged = value
            next_index = index + 1
            while (next_index < len(facts)
                   and facts[next_index].get("category") == "sample_size"
                   and facts[next_index].get("pdf_page_start") == fact.get("pdf_page_start")
                   and (merged.endswith((",", ";", ":")) or re.search(r"\b(?:control|intervention)\s*,?$", merged, re.I))):
                continuation = normalized(facts[next_index].get("evidence_quote_display", facts[next_index].get("evidence_quote", "")))
                if not continuation or continuation in merged:
                    break
                merged = f"{merged} {continuation}"
                next_index += 1
            score = (4 if re.search(r"\b(?:enrolled|randomi[sz]ed|allocated)\b", merged, re.I) else 1)
            if re.search(r"\bn\s*=\s*\d", merged, re.I):
                score += 2
            primary = re.search(
                r"\b\d[\d,]*\s+(?:independent\s+)?(?:samples?|participants?|subjects?|patients?)\b",
                merged,
                re.IGNORECASE,
            )
            compact = primary.group(0) if primary else None
            if compact:
                group_counts = re.findall(r"\b(?:n\s*=\s*\d[\d,]*)\b", merged, re.IGNORECASE)
                for count in group_counts:
                    if count.casefold() not in compact.casefold():
                        compact += f"; {count}"
                merged = compact
            candidates.append((score, merged))
        if not candidates:
            return None
        unique = list(dict.fromkeys(candidates))
        return max(unique, key=lambda item: (item[0], -len(item[1])))[1]

    def quotes(facts: list[dict[str, Any]], categories: set[str], sections: set[str] | None = None) -> list[str]:
        values = []
        for fact in facts:
            if fact.get("category") not in categories:
                continue
            if sections is not None and fact.get("section") not in sections:
                continue
            value = normalized(fact.get("evidence_quote_display", fact.get("evidence_quote", "")))
            if value and value not in values:
                values.append(value)
        return values

    for paper in papers:
        facts = paper.get("facts", [])
        row = {
            "file_name": paper.get("file_name"),
            "document_id": paper.get("document_id"),
            "content_sha256": paper.get("content_sha256"),
            "source_alias": paper.get("source_alias"),
            "selection_strategy": paper.get("selection_strategy"),
            "included_in_word": paper.get("included_in_word", False),
            "selection_rank": paper.get("selection_rank"),
            "selection_score": paper.get("selection_score"),
            "selection_reason": paper.get("selection_reason"),
        }
        row.update({field: paper.get(field) for field in fields})
        if not row.get("sample_size"):
            row["sample_size"] = sample_text(facts)
        method_keywords = []
        categories = {fact.get("category") for fact in facts}
        for category in method_order:
            if category in categories and category not in method_keywords:
                method_keywords.append(category)
        method_text = " ".join(
            str(paper.get("sections", {}).get(name, {}).get("text", ""))
            for name in ("abstract", "methods")
        )
        method_patterns = (
            ("immunocapture_lc_ms", r"immunocapture[- ](?:liquid chromatography/)?mass spectrometry|immunocapture[- ]LC/?MS"),
            ("affinity_purification_lc_ms", r"affinity purification.{0,80}LC[-/]MS"),
            ("single_dose_pk", r"single[- ]dose.{0,60}(?:monkey )?P\.?K\.?|single[- ]dose PK"),
            ("multiple_dose_pk", r"multiple[- ]dose.{0,60}(?:monkey )?P\.?K\.?|multiple[- ]dose PK"),
            ("randomized_controlled_trial", r"randomi[sz]ed controlled trial|randomi[sz]ed controlled"),
            ("dietary_intervention", r"dietary intervention|dietary improvement program"),
        )
        for keyword, pattern in method_patterns:
            if re.search(pattern, method_text, re.IGNORECASE) and keyword not in method_keywords:
                method_keywords.append(keyword)
        method_evidence = _collect_method_evidence(paper)
        condition_values: list[str] = []
        for fact in facts:
            if fact.get("category") not in {"temperature", "time", "concentration", "dose"}:
                continue
            value = _clean_sentence(fact.get("evidence_quote_display", fact.get("evidence_quote", "")))
            if fact.get("section") not in {"abstract", "methods", "methodology", "materials_and_methods"} or not value:
                continue
            # 含参与者和研究时长的复合句归入样本/方法，不冒充纯实验条件。
            if fact.get("category") == "time" and re.search(r"\b(?:included|enrolled|participants?|subjects?|patients?|sample)\b", value, re.IGNORECASE):
                continue
            if _sentence_key(value) in {_sentence_key(item) for item in method_evidence}:
                continue
            if _sentence_key(value) not in {_sentence_key(item) for item in condition_values}:
                condition_values.append(value)
        experimental_conditions = condition_values
        row.update({
            "research_methods_keywords": method_keywords,
            "research_methods_original": method_evidence,
            "experimental_conditions": experimental_conditions,
            "statistical_information": quotes(facts, {"p_value", "confidence_interval", "mean_sd"}, {"abstract", "methods", "results"}),
            "major_results_original": quotes(facts, {"major_result"}, {"results"}),
            "conclusion_original": quotes(facts, {"conclusion"}, {"abstract", "conclusion", "discussion"}),
            "limitations_original": quotes(facts, {"limitation"}, {"discussion", "conclusion", "results"}),
            "source_pages": sorted({(f["pdf_page_start"], f["pdf_page_end"]) for f in facts if f.get("pdf_page_start")}),
            "missing_fields": [],
        })
        row["missing_fields"] = [field for field in fields if not row.get(field)]
        rows.append(row)
    return rows


def _normalize_space(value: str) -> str:
    return repair_visual_word_breaks(re.sub(r"\s+", " ", value).strip())


def _normalize_for_match(value: str) -> str:
    value = re.sub(r"(?<=[A-Za-z])-[ \t\r\n]+(?=[a-z])", "", value)
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
    """按原始行窗口重建证据，保留原始换行和连字符。"""
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
    if re.match(r"^(?:keywords?|correspondence|doi|received|accepted|trial registration|registered on|author details|authors?['’] contributions|competing interests?)\b", normalized, re.IGNORECASE):
        return True
    if any(term in lowered for term in ("all authors read and approved", "financial disclosure", "no relevant financial", "competing interest", "funding", "author contributions", "acknowledg", "ethics approval", "consent for publication", "creative commons", "article is distributed")):
        return True
    if normalized[:1].islower():
        return True
    if re.fullmatch(r"[A-Z][A-Z\s,.'-]{5,}", normalized):
        return True
    if any(term in lowered for term in ("university", "department of", "institute", "school of", "corresponding author", "hospital", "author details")):
        return True
    if not re.search(r"[.!?。！？]$", normalized):
        return True
    if len(normalized) < 40 and not re.search(r"(?i)\b(?:result|study|trial|group|treatment|participant|significant|conclusion|method)\b", normalized):
        return True
    return False


def _looks_incomplete_fragment(sentence: str) -> bool:
    value = _normalize_space(sentence)
    lowered = value.casefold()
    if re.search(r"(?:[-–—]|\b(?:significantly|compared|than|of|and|or|with|to|from|in|for|the|a|an|on|by|as))\s*$", lowered):
        return True
    return bool(re.search(r"\bfig(?:s?\.?|ures?)\s*\d+\s*[-–—]\s*$", lowered))


def _find_raw_match(target: str, raw_text: str) -> str | None:
    target_forms = [_normalize_for_match(target)]
    for fixed, broken in (("increased", "i ncreased"), ("pattern", "p attern"), ("site", "s ite"), ("in", "i n"), ("to", "t o"), ("recommendations", "r ecommendations"), ("12-week", "12-w eek"), ("support", "s upport")):
        if fixed in target.casefold():
            target_forms.append(_normalize_for_match(re.sub(fixed, broken, target, flags=re.IGNORECASE)))
    for normalized_target in dict.fromkeys(target_forms):
        tokens = [token for token in re.split(r"\s+", normalized_target) if token]
        if tokens:
            pattern = r"\s+".join(re.escape(token) for token in tokens)
            match = re.search(pattern, raw_text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0)
    for raw_part in _raw_quote_candidates(raw_text):
        if _normalize_for_match(raw_part) == normalized_target:
            return raw_part
    return None


def _candidate_evidence(sentence: str, paper: dict[str, Any], section: dict[str, Any], section_key: str) -> dict[str, Any] | None:
    for page in paper.get("pages", []):
        number = page.get("page_number", 0)
        if not (section.get("page_start", 0) <= number <= section.get("page_end", 0)):
            continue
        raw_text = page.get("raw_text", page.get("text", ""))
        raw_part = _find_raw_match(sentence, raw_text)
        if raw_part:
            flags = ["incomplete_fragment"] if _looks_incomplete_fragment(sentence) else []
            return {
                "text": sentence,
                "section": section.get("title", "").strip() or "未标明章节",
                "section_key": section_key,
                "pdf_pages": [number],
                "evidence_quote_raw": raw_part,
                "evidence_quote_display": sentence,
                "evidence_quote_zh": None,
                "translation_status": "not_requested",
                "verified": raw_part in raw_text,
                "quality_flags": flags,
            }
    return None


def _score_sentence(sentence: str, section_name: str) -> int:
    lowered = sentence.casefold()
    terms = {
        "abstract": ("purpose", "objective", "aim", "investigat", "evaluat", "developed", "quantif", "background", "randomi"),
        "introduction": ("purpose", "objective", "aim", "background", "we sought"),
        "methods": ("random", "double-blind", "controlled", "trial", "sample", "dose", "week", "method", "lc-ms", "immunocapture", "monkey", "pk", "multiple-dose", "single-dose"),
        "results": ("p ", "p=", "p<", "%", "significant", "result", "mean", "improvement", "decreased", "increased", "deamidation", "ptm", "effect size", "unchanged", "no change", "remained", "unaffected", "stable", "oxidation", "lysine"),
        "conclusion": ("conclu", "suggest", "greater", "effective", "limitation", "showed", "improved"),
        "discussion": ("conclu", "limitation", "suggest", "result"),
    }
    return sum(1 for term in terms.get(section_name, ()) if term in lowered)


def extractive_summary(paper: dict[str, Any], max_sentences: int = 6) -> dict[str, Any]:
    selected: list[str] = []
    evidence: list[dict[str, Any]] = []
    sections = paper.get("sections", {})
    quotas = (("abstract", 1), ("methods", 1), ("results", 2), ("conclusion", 1), ("discussion", 1))
    for name, quota in quotas:
        section = sections.get(name, {})
        if not section or not section.get("text"):
            continue
        candidates = []
        for sentence in _split_sentences(section["text"]):
            if "�" in sentence or _is_metadata_candidate(sentence) or _looks_incomplete_fragment(sentence):
                continue
            item = _candidate_evidence(sentence, paper, section, name)
            if item and item["verified"] and not item["quality_flags"]:
                candidates.append((_score_sentence(sentence, name), item))
        candidates.sort(key=lambda pair: (-pair[0], len(pair[1]["text"])))
        for _, item in candidates:
            if _normalize_space(item["text"]) in {_normalize_space(x) for x in selected}:
                continue
            selected.append(item["text"])
            evidence.append(item)
            if sum(1 for x in evidence if x.get("section_key") == name) >= quota or len(selected) >= max_sentences:
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


def offline_language_notice(language: str) -> str:
    if language in {"中文摘要＋英文证据（推荐）", "中英对照"}:
        return "本地离线模式未启用语义翻译，英文证据原文仍保留。"
    return "当前显示原文和程序验证信息。"
