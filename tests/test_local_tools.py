import json
from pathlib import Path

import fitz
import pytest

from exporters import export_csv, export_excel, export_json, export_word
from local_extractor import extract_local_paper
from local_search import LocalSearchIndex
from local_summary import compare_papers, extractive_summary
from paper_pipeline import PageText
from section_parser import parse_sections
from scientific_facts import extract_scientific_facts


def make_local_pdf(path: Path) -> Path:
    doc = fitz.open()
    pages = [
        "A Controlled Study\nAlice Author\n2024\ndoi: 10.1234/example.doi\nAbstract\nWe evaluated stability.",
        "Research purpose: Evaluate short-term stability.\nMethods\nThe study included 120 independent samples.\nThe randomized, double-blind, controlled design stored samples at 4 degrees Celsius for eight weeks.",
        "Results\nThe result was statistically significant with p = 0.003.\nThe mean was 92.4 ± 2.1.",
        "Discussion\nThe limitation was a single laboratory and short duration.\nConclusion\nABX-17 showed greater stability.",
        "References\nA. Author. 2024.",
    ]
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_local_extraction_sections_metadata_and_facts(tmp_path):
    result = extract_local_paper(make_local_pdf(tmp_path / "中文论文.pdf"))
    assert len(result["pages"]) == 5
    assert result["title_candidate"] == "A Controlled Study"
    assert result["doi"] == "10.1234/example.doi"
    assert "abstract" in result["sections"] and "methods" in result["sections"]
    categories = {fact["category"] for fact in result["facts"]}
    assert {"sample_size", "temperature", "time", "p_value", "mean_sd", "randomization", "double_blind", "limitation", "conclusion"} <= categories
    assert all(fact["verified"] is True for fact in result["facts"])
    assert all(fact["pdf_page_start"] == fact["pdf_page_end"] for fact in result["facts"])


def test_section_parser_requires_explicit_heading():
    pages = [PageText("a.pdf", 1, "The results were discussed.\nNot a heading", 0, False)]
    assert parse_sections(pages) == {}


def test_local_search_fts_duplicate_update_phrase_and_clear(tmp_path):
    pages = [PageText("a.pdf", 1, "alpha beta phrase", 17, False), PageText("a.pdf", 2, "gamma", 5, False)]
    index = LocalSearchIndex(tmp_path / "index.sqlite")
    assert index.index_pages(pages) == 2
    assert index.index_pages(pages) == 0
    assert index.search('"alpha beta"')[0]["page_number"] == 1
    assert index.search("alpha gamma", limit=5) == []
    updated = [PageText("a.pdf", 1, "updated text", 12, False)]
    assert index.index_pages(updated) == 1
    assert index.search("updated")[0]["snippet"] == "updated text"
    with pytest.raises(ValueError):
        index.clear()
    index.clear(confirm=True)


def test_offline_summary_and_exports_are_structured(tmp_path):
    paper = extract_local_paper(make_local_pdf(tmp_path / "paper.pdf"))
    rows = compare_papers([paper])
    assert rows[0]["file_name"] == "paper.pdf"
    summary = extractive_summary(paper)
    assert summary["summary_type"] == "extractive"
    assert all(sentence in "\n".join(page["text"] for page in paper["pages"]) for sentence in summary["sentences"])
    json_path = export_json(rows, tmp_path / "中文.json")
    csv_path = export_csv(rows, tmp_path / "comparison.csv")
    xlsx_path = export_excel(rows, tmp_path / "comparison.xlsx")
    docx_path = export_word(rows, tmp_path / "comparison.docx")
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["file_name"] == "paper.pdf"
    assert all(path.exists() for path in (csv_path, xlsx_path, docx_path))


def test_local_mode_does_not_create_deepseek_client(monkeypatch, tmp_path):
    import paper_claude

    monkeypatch.setattr(paper_claude, "get_client", lambda *_: pytest.fail("Local Offline 不应创建 DeepSeek 客户端"))
    file_path = make_local_pdf(tmp_path / "offline.pdf")
    result, output = paper_claude.process_local_papers([str(file_path)])
    assert "Local Offline" in result
    assert output and Path(output).exists()
