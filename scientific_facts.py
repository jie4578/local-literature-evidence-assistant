"""从原文句子中做保守、摘取式科研信息提取。"""

from __future__ import annotations

import re
from typing import Any, Iterable


RULES = {
    "sample_size": r"\b(?:n\s*=\s*\d[\d,]*|\d[\d,]*\s+(?:independent\s+)?samples?|sample size\s*(?:was|of|=)\s*\d[\d,]*)\b",
    "group_count": r"\b(?:two|three|\d+)\s+(?:groups?|arms?)\b|(?:assigned|randomized|allocated)\s+to\s+the\s+[^.]{0,100}\s+and\s+[^.]{0,100}\s+groups?",
    "temperature": r"\b(?:at|to|stored at)\s*-?\d+(?:\.\d+)?\s*(?:degrees?\s*)?(?:C|F|Celsius|Fahrenheit)\b|\b\d+(?:\.\d+)?\s*°\s*[CF]\b",
    "time": r"\b(?:for|over|lasted?)\s+(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:days?|weeks?|months?|hours?|years?)\b",
    "concentration": r"\b\d+(?:\.\d+)?\s*(?:mg/mL|mg/L|µg/mL|ug/mL|mM|µM|uM|%)\b",
    "dose": r"\b(?:dose|dosed|dosage)\w*\s*(?:of|was|=)?\s*\d+(?:\.\d+)?\s*(?:mg/kg|mg|g|µg|ug)\b",
    "p_value": r"\bp\s*[<=>]\s*0?\.\d+\b",
    "confidence_interval": r"\b(?:95%\s*)?(?:CI|confidence interval)\s*[:=]?\s*\(?\s*[-\d.]+\s*(?:to|–|-|,)\s*[-\d.]+\s*\)?",
    "mean_sd": r"\b(?:mean|average)\s*(?:±|\+/-)\s*|\b\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?",
    "randomization": r"\b(?:randomized|randomised|randomization|randomisation)\b",
    "double_blind": r"\bdouble[- ]blind(?:ed)?\b",
    "control_group": r"\bcontrol\s+group\b|\bplacebo\s+group\b",
    "major_result": r"\b(?:significant|increased|decreased|higher|lower|retained|improved|reduced|greater)\b",
    "limitation": r"\b(?:limitation|limited by|single laboratory|single[- ]center|small sample|short duration)\b",
    "conclusion": r"\b(?:we conclude|in conclusion|concluded that|showed greater|demonstrated)\b",
}


def _sentences(text: str) -> Iterable[str]:
    for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text):
        value = part.strip()
        if value:
            yield value


def extract_scientific_facts(pages: Iterable[Any]) -> list[dict[str, Any]]:
    facts = []
    for page in pages:
        for sentence in _sentences(page.text):
            for category, pattern in RULES.items():
                if re.search(pattern, sentence, flags=re.IGNORECASE):
                    facts.append({"category": category, "evidence_quote": sentence,
                                  "source_file": page.source_file, "pdf_page_start": page.page_number,
                                  "pdf_page_end": page.page_number, "extraction_method": "rule",
                                  "verified": True})
    return facts
