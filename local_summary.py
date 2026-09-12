"""完全离线的结构化多论文对比和摘取式摘要。"""

from __future__ import annotations

from typing import Any, Iterable


def compare_papers(papers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    fields = ("title_candidate", "author_candidate", "year_candidate", "doi", "research_object", "sample_size")
    for paper in papers:
        facts = paper.get("facts", [])
        sample_facts = [f["evidence_quote"] for f in facts if f["category"] == "sample_size"]
        def quotes(categories):
            return [fact["evidence_quote"] for fact in facts if fact["category"] in categories]
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


def extractive_summary(paper: dict[str, Any], max_sentences: int = 8) -> dict[str, Any]:
    selected = []
    evidence = []
    sections = paper.get("sections", {})
    pages = paper.get("pages", [])
    for name in ("abstract", "results", "conclusion"):
        section = sections.get(name, {})
        for page in pages:
            if not (section.get("page_start") and section.get("page_start") <= page.get("page_number", 0) <= section.get("page_end", 0)):
                continue
            for sentence in page.get("text", "").splitlines():
                sentence = sentence.strip()
                if sentence and sentence not in selected:
                    selected.append(sentence)
                    evidence.append({"evidence_quote": sentence, "source_file": paper.get("file_name"),
                                      "pdf_page_start": page.get("page_number"), "pdf_page_end": page.get("page_number"),
                                      "verified": sentence in page.get("text", "")})
                if len(selected) >= max_sentences:
                    break
            if len(selected) >= max_sentences:
                break
        if len(selected) >= max_sentences:
            break
    return {"summary_type": "extractive", "label": "摘取式摘要", "sentences": selected, "evidence": evidence}
