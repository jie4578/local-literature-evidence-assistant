"""从原文句子中做保守、摘取式科研信息提取。"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


RULES = {
    "sample_size": r"\b(?:n\s*=\s*\d[\d,]*|\d[\d,]*\s+(?:independent\s+)?samples?|sample size\s*(?:was|of|=)\s*\d[\d,]*)\b",
    "group_count": r"\b(?:two|three|\d+)\s+(?:groups?|arms?)\b|(?:assigned|randomized|allocated)\s+to\s+the\s+[^.]{0,100}\s+and\s+[^.]{0,100}\s+groups?",
    "temperature": r"\b(?:at|to|stored at)\s*-?\d+(?:\.\d+)?\s*(?:degrees?\s*)?(?:C|F|Celsius|Fahrenheit)\b|\b\d+(?:\.\d+)?\s*°\s*[CF]\b",
    "time": r"\b(?:for|over|lasted?)\s+(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:days?|weeks?|months?|hours?|years?)\b",
    "concentration": r"\b\d+(?:\.\d+)?\s*(?:mg/mL|mg/L|[µμ]g/mL|ug/mL|mM|[µμ]M|uM|%)\b",
    "dose": r"\b(?:dose|dosed|dosage)\w*\s*(?:of|was|=)?\s*\d+(?:\.\d+)?\s*(?:mg/kg|mg|g|µg|ug)\b",
    "p_value": r"\bp\s*[<=>]\s*0?\.\d+\b",
    "confidence_interval": r"\b(?:95%\s*)?(?:CI|confidence interval)\s*[:=]?\s*\(?\s*[-\d.]+\s*(?:to|–|-|,)\s*[-\d.]+\s*\)?",
    "mean_sd": r"\b(?:mean|average)\s*(?:±|\+/-)\s*|\b\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?",
    "research_method": r"\b(?:model\s+fit(?:ting)?|goodness[- ]of[- ]fit|fit(?:ting)?\s+criteria|convergence\s+criteria|cut[- ]off\s+thresholds?|nonlinear\s+mixed[- ]effects|two[- ]compartment\s+model|r\s*(?:square|squared|2|²)(?:\s*[<>=]\s*[-+]?\d*\.?\d+)?)",
    "randomization": r"\b(?:randomized|randomised|randomization|randomisation)\b",
    "double_blind": r"\bdouble[- ]blind(?:ed)?\b",
    "control_group": r"\bcontrol\s+group\b|\bplacebo\s+group\b",
    "major_result": r"\b(?:significant|increased|decreased|higher|lower|retained|improved|reduced|greater|unchanged|no change|remained|unaffected|stable)\b",
    "limitation": r"\b(?:limitation|limited by|preliminary|replication|larger sample|single laboratory|single[- ]center|small sample|short duration)\b",
    "conclusion": r"\b(?:we conclude|in conclusion|concluded that|showed greater|demonstrated)\b",
}


VISUAL_WORD_REPAIRS = (
    (re.compile(r"\bi\s+ncreased\b", re.IGNORECASE), "increased"),
    (re.compile(r"\bp\s+attern\b", re.IGNORECASE), "pattern"),
    (re.compile(r"\bs\s+ite\b", re.IGNORECASE), "site"),
    (re.compile(r"\bi\s+n\b", re.IGNORECASE), "in"),
    (re.compile(r"\bt\s+o\b", re.IGNORECASE), "to"),
    (re.compile(r"\br\s+ecommendations\b", re.IGNORECASE), "recommendations"),
    (re.compile(r"\b12-w\s+eek\b", re.IGNORECASE), "12-week"),
    (re.compile(r"\bs\s+upport\b", re.IGNORECASE), "support"),
)


def repair_visual_word_breaks(value: str) -> str:
    """仅修复已知的视觉断词；不改写 raw_text。"""
    for pattern, replacement in VISUAL_WORD_REPAIRS:
        value = pattern.sub(replacement, value)
    return value


def _sentences(text: str) -> Iterable[str]:
    # PyMuPDF 的视觉换行不一定代表句子边界；先重建连续文本，
    # 再按真实标点切句，避免样本量或统计句子被拆成残片。
    joined = re.sub(r"[ \t]*\r?\n[ \t]*", " ", text)
    for part in re.split(r"(?<=[.!?。！？])\s+", joined):
        value = part.strip()
        if value:
            yield value


def _raw_quote_for_display(display_sentence: str, raw_text: str) -> str | None:
    normalized_target = re.sub(r"\s+", " ", display_sentence).strip()
    # 返回 raw_text 中的真实子串，允许只在定位时跨越原始换行。
    tokens = [token for token in re.split(r"\s+", normalized_target) if token]
    if tokens:
        pattern = r"\s+".join(re.escape(token) for token in tokens)
        match = re.search(pattern, raw_text, flags=re.DOTALL)
        if match:
            return match.group(0)
    for candidate in _sentences(raw_text):
        normalized_candidate = re.sub(r"\s+", " ", candidate).strip()
        # NFKC 只用于定位 Unicode 等价字符（例如 µ/μ），返回值仍是 raw 原文。
        target_nfkc = unicodedata.normalize("NFKC", normalized_target)
        candidate_nfkc = unicodedata.normalize("NFKC", normalized_candidate)
        if target_nfkc in candidate_nfkc or candidate_nfkc in target_nfkc:
            return candidate
    return display_sentence if display_sentence in raw_text else None


def extract_scientific_facts(pages: Iterable[Any]) -> list[dict[str, Any]]:
    facts = []
    front_matter_prefix = re.compile(
        r"^(?:research article|open access|original article|article|plos one)\b",
        re.IGNORECASE,
    )
    for page in pages:
        display_text = page.get("display_text", page.get("text", "")) if isinstance(page, dict) else page.text
        raw_text = page.get("raw_text", page.get("text", "")) if isinstance(page, dict) else page.text
        for sentence in _sentences(display_text):
            display_sentence = repair_visual_word_breaks(sentence)
            normalized_sentence = re.sub(r"\s+", " ", display_sentence).strip()
            is_model_fit = bool(re.search(RULES["research_method"], display_sentence, flags=re.IGNORECASE))
            # 出版标签、标题、作者和单位可能与下一段连成一个视觉句子；
            # 不把这类首屏前置信息当作科研事实。原始页面仍完整保留。
            if front_matter_prefix.match(normalized_sentence):
                continue
            for category, pattern in RULES.items():
                if category == "major_result" and is_model_fit:
                    continue
                if re.search(pattern, display_sentence, flags=re.IGNORECASE):
                    raw_quote = _raw_quote_for_display(sentence, raw_text)
                    if not raw_quote:
                        continue
                    quality_flags = ["replacement_character"] if "�" in raw_quote else []
                    if display_sentence.rstrip().endswith("-"):
                        quality_flags.append("unresolved_line_break_hyphen")
                    if re.match(r"(?i)^(?:table|fig(?:ure)?|图|表)\s*\d", normalized_sentence):
                        source_type = "possible_table"
                    elif display_sentence.strip() == display_sentence.strip().title() and len(display_sentence.split()) <= 8:
                        source_type = "heading"
                    else:
                        source_type = "unknown"
                    facts.append({"category": category, "evidence_quote": display_sentence,
                                  "evidence_quote_display": display_sentence, "evidence_quote_raw": raw_quote,
                                  "source_file": page.get("source_file") if isinstance(page, dict) else page.source_file,
                                  "pdf_page_start": page.get("page_number") if isinstance(page, dict) else page.page_number,
                                  "pdf_page_end": page.get("page_number") if isinstance(page, dict) else page.page_number,
                                  "extraction_method": "rule", "source_type": source_type,
                                  "verified": raw_quote in raw_text, "quality_flags": quality_flags})
    return facts
