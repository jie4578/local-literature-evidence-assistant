import json
from pathlib import Path
from zipfile import ZipFile

import fitz
import pytest
from docx import Document
from openpyxl import load_workbook

from exporters import export_csv, export_excel, export_json, export_word
from bilingual import translate_paper_for_display
from local_extractor import _clean_page_text, extract_local_paper
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


def test_section_parser_recognizes_introduction_variants_and_background_heading():
    pages = [PageText("a.pdf", 1, "1. Introduction\nintro\n2. Methods\nmethod", 0, False)]
    sections = parse_sections(pages)
    assert "introduction" in sections and "methods" in sections
    background = parse_sections([PageText("a.pdf", 1, "Background\ncontext", 0, False)])
    assert "introduction" in background


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


def test_local_search_supports_statistical_p_value_variants(tmp_path):
    pages = [PageText("stats.pdf", 1, "The p = 0.003 and P<0.05 findings were significant.", 48, False),
             PageText("stats.pdf", 2, "The p-value was reported with p\u202f<\u202f0.001.", 45, False),
             PageText("stats.pdf", 3, "统计结果 p值显著。", 12, False)]
    index = LocalSearchIndex(tmp_path / "stats.sqlite")
    index.index_pages(pages)
    assert index.search("p value")
    assert index.search("p-value")
    assert index.search("p值")
    assert index.search("p = 0.003")
    assert index.search("p < 0.001")
    assert index.search("P<0.05")
    assert index.search("alpha") == []


def test_offline_summary_and_exports_are_structured(tmp_path):
    paper = extract_local_paper(make_local_pdf(tmp_path / "paper.pdf"))
    rows = compare_papers([paper])
    assert rows[0]["file_name"] == "paper.pdf"
    summary = extractive_summary(paper)
    assert summary["summary_type"] == "extractive"
    assert summary["description"].startswith("摘取式摘要")
    assert all(sentence in "\n".join(page["text"] for page in paper["pages"]) for sentence in summary["sentences"])
    json_path = export_json(rows, tmp_path / "中文.json")
    csv_path = export_csv(rows, tmp_path / "comparison.csv")
    xlsx_path = export_excel(rows, tmp_path / "comparison.xlsx")
    docx_path = export_word(rows, tmp_path / "comparison.docx")
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["file_name"] == "paper.pdf"
    assert all(path.exists() for path in (csv_path, xlsx_path, docx_path))


def test_extractive_summary_uses_sections_and_excludes_front_matter():
    paper = {
        "file_name": "paper.pdf",
        "pages": [{"page_number": 1, "text": "RESEARCH ARTICLE\nOpen Access\nSplit\nTitle\nAlice Author\nUniversity\nAbstract\nWe evaluated the treatment purpose in adults."},
                   {"page_number": 2, "text": "Methods\nThis randomized controlled trial included 120 participants.\nResults\nThe result was significant with p = 0.003.\nConclusion\nThe treatment improved outcomes."}],
        "sections": {
            "abstract": {"title": "Abstract", "text": "We evaluated the treatment purpose in adults.", "page_start": 1, "page_end": 1},
            "methods": {"title": "Methods", "text": "This randomized controlled trial included 120 participants.", "page_start": 2, "page_end": 2},
            "results": {"title": "Results", "text": "The result was significant with p = 0.003.", "page_start": 2, "page_end": 2},
            "conclusion": {"title": "Conclusion", "text": "The treatment improved outcomes.", "page_start": 2, "page_end": 2},
        },
    }
    summary = extractive_summary(paper)
    joined = " ".join(summary["sentences"])
    assert "RESEARCH ARTICLE" not in joined and "Open Access" not in joined
    assert "Alice Author" not in joined and "University" not in joined
    assert any("purpose" in sentence for sentence in summary["sentences"])
    assert any("randomized" in sentence for sentence in summary["sentences"])
    assert any("p = 0.003" in sentence for sentence in summary["sentences"])
    assert all(item["verified"] and item["pdf_pages"] for item in summary["evidence"])


def test_publication_front_matter_is_not_rule_evidence():
    pages = [{
        "source_file": "paper.pdf", "page_number": 1,
        "raw_text": "RESEARCH ARTICLE\nOpen Access\nA randomised controlled trial of diet improvement.\nFelice N. Jacka\nAbstract\nWe randomised 67 participants.",
        "text": "RESEARCH ARTICLE\nOpen Access\nA randomised controlled trial of diet improvement.\nFelice N. Jacka\nAbstract\nWe randomised 67 participants.",
        "display_text": "RESEARCH ARTICLE\nOpen Access\nA randomised controlled trial of diet improvement.\nFelice N. Jacka\nAbstract\nWe randomised 67 participants.",
    }]
    facts = extract_scientific_facts(pages)
    assert all("RESEARCH ARTICLE" not in fact["evidence_quote_display"] for fact in facts)
    assert all("Open Access" not in fact["evidence_quote_display"] for fact in facts)


def test_raw_and_display_evidence_are_separate_and_raw_is_exact():
    pages = [{"source_file": "a.pdf", "page_number": 1, "text": "admin-\nistration result", "raw_text": "admin-\nistration result", "display_text": "administration result"}]
    facts = extract_scientific_facts(pages)
    assert facts == [] or all(fact["evidence_quote_raw"] in pages[0]["raw_text"] for fact in facts)


def test_display_removes_controls_but_raw_is_preserved_and_flagged(tmp_path):
    display, flags = _clean_page_text("value\x01 with µ °C ± α β ≤ ≥", set())
    assert "\x01" not in display
    assert "raw_control_character_removed_for_display" in flags


def test_unicode_exports_round_trip(tmp_path):
    value = [{"evidence_quote_raw": "µ °C ± α β ≤ ≥ – − ²", "evidence_quote_display": "µ °C ± α β ≤ ≥ – − ²"}]
    json_path = export_json(value, tmp_path / "unicode.json")
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["evidence_quote_display"] == value[0]["evidence_quote_display"]
    csv_path = export_csv(value, tmp_path / "unicode.csv")
    assert "µ" in csv_path.read_text(encoding="utf-8-sig")
    xlsx_path = export_excel(value, tmp_path / "unicode.xlsx")
    assert load_workbook(xlsx_path, read_only=True).active.cell(2, 1).value == value[0]["evidence_quote_raw"]
    docx_path = export_word(value, tmp_path / "unicode.docx")
    assert "µ" in "\n".join(paragraph.text for paragraph in Document(docx_path).paragraphs)


def test_safe_line_break_hyphen_cleaning_and_compound_preservation():
    display, flags = _clean_page_text("admin-\nistration\ndouble-blind\nsingle-dose\nLC-MS", set())
    assert "administration" in display
    assert "double-blind" in display and "single-dose" in display and "LC-MS" in display
    assert "unresolved_line_break_hyphen" not in flags


def test_unsafe_line_break_hyphen_is_not_guessed():
    pages = [{"source_file": "a.pdf", "page_number": 1,
              "raw_text": "The Modi-\nMedDiet result was observed.",
              "display_text": "The Modi-\nMedDiet result was observed.",
              "text": "The Modi-\nMedDiet result was observed."}]
    facts = extract_scientific_facts(pages)
    assert facts == [] or all("unresolved_line_break_hyphen" in fact.get("quality_flags", []) for fact in facts)


def test_local_mode_does_not_create_deepseek_client(monkeypatch, tmp_path):
    import paper_claude

    monkeypatch.setattr(paper_claude, "get_client", lambda *_: pytest.fail("Local Offline 不应创建 DeepSeek 客户端"))
    file_path = make_local_pdf(tmp_path / "offline.pdf")
    result, output = paper_claude.process_local_papers([str(file_path)])
    assert "Local Offline" in result
    assert output and Path(output).exists()


def test_research_object_is_conservative_and_keeps_page_evidence(tmp_path):
    path = tmp_path / "objects.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "A monkey pharmacokinetic (PK) studies report on MAB1.")
    doc.new_page().insert_text((72, 72), "Adults with major depression joined the trial.")
    doc.save(path)
    doc.close()
    first = extract_local_paper(path)
    assert first["research_object"] == "monkey pharmacokinetic (PK) studies"
    assert first["research_object_pdf_page"] == 1


def test_compare_repairs_cross_line_sample_and_deduplicates_methods():
    paper = {
        "file_name": "trial.pdf", "title_candidate": "Trial", "author_candidate": "Author",
        "year_candidate": 2024, "doi": "10.1234/trial", "research_object": "adults",
        "facts": [
            {"category": "sample_size", "evidence_quote_display": "We enrolled 67 participants (intervention, n = 33; control,", "pdf_page_start": 1, "pdf_page_end": 1, "source_type": "unknown"},
            {"category": "sample_size", "evidence_quote_display": "n = 34).", "pdf_page_start": 1, "pdf_page_end": 1, "source_type": "unknown"},
            {"category": "sample_size", "evidence_quote_display": "Table 1 Total (n = 67) intervention (n = 33) control (n = 34)", "pdf_page_start": 1, "pdf_page_end": 1, "source_type": "possible_table"},
            {"category": "randomization"}, {"category": "randomization"}, {"category": "control_group"},
        ],
        "sections": {"methods": {"text": "A randomized controlled trial."}},
    }
    row = compare_papers([paper])[0]
    assert "67" in row["sample_size"] and "n = 33" in row["sample_size"] and "n = 34" in row["sample_size"]
    assert "sample_size" not in row["missing_fields"]
    assert row["research_methods_keywords"].count("randomization") == 1


def test_summary_excludes_incomplete_fragments_and_keeps_verified_evidence():
    paper = {
        "pages": [{"page_number": 1, "raw_text": "Results\nComplete result was significant.\nThe dietary support group demonstrated significantly\nFigs 4–", "text": "Results\nComplete result was significant.\nThe dietary support group demonstrated significantly\nFigs 4–", "display_text": "Results\nComplete result was significant.\nThe dietary support group demonstrated significantly\nFigs 4–"}],
        "sections": {"results": {"title": "Results", "text": "Complete result was significant.\nThe dietary support group demonstrated significantly\nFigs 4–", "page_start": 1, "page_end": 1}},
    }
    summary = __import__("local_summary").extractive_summary(paper)
    assert summary["sentences"] == ["Complete result was significant."]
    assert summary["evidence"][0]["verified"] is True


class TranslationProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(kwargs)
        from providers import ProviderResponse
        return ProviderResponse.from_legacy_string(self.response)


def test_translation_is_bounded_and_preserves_statistics():
    paper = {"title_candidate": "Trial", "sample_size": "67 participants, p = 0.003", "evidence": [{"evidence_quote_display": "67 participants, p = 0.003", "verified": True, "pdf_page_start": 2}]}
    provider = TranslationProvider('{"translations":{"title_candidate":"试验","sample_size":"67名参与者，p = 0.003","evidence_0":"67名参与者，p = 0.003"}}')
    outcome = translate_paper_for_display(provider, paper)
    assert outcome["status"] == "success"
    assert paper["sample_size_zh"] == "67名参与者，p = 0.003"
    assert paper["evidence"][0]["translation_status"] == "success"
    assert provider.calls[0]["max_output_tokens"] > 0


def test_translation_failure_keeps_english_fallback():
    paper = {"sample_size": "67 participants, p = 0.003", "evidence": [{"evidence_quote_display": "67 participants, p = 0.003", "verified": True}]}
    provider = TranslationProvider('{"translations":{"sample_size":"六十七名参与者"}}')
    outcome = translate_paper_for_display(provider, paper)
    assert outcome["status"] == "failed"
    assert "sample_size_zh" not in paper
    assert paper["evidence"][0]["translation_status"] == "failed"


def test_word_report_has_comparison_and_evidence_tables_without_internal_fields(tmp_path):
    paper = {"file_name": "trial.pdf", "title_candidate": "Trial", "author_candidate": "Author", "year_candidate": 2024, "doi": "10.1234/trial", "research_object": "adults", "sample_size": "67 participants", "facts": [{"category": "sample_size", "evidence_quote_raw": "67 participants", "evidence_quote_display": "67 participants", "pdf_page_start": 2, "pdf_page_end": 2, "verified": True, "source_type": "body", "quality_flags": []}], "extractive_summary": {"sentences": ["67 participants"]}}
    rows = compare_papers([paper])
    path = export_word(rows, tmp_path / "report.docx", papers=[paper])
    document = Document(path)
    xml_text = "\n".join(cell.text for table in document.tables for row in table.rows for cell in row.cells)
    assert len(document.tables) >= 3
    assert "类型与页码" in xml_text and "PDF 第 2 页" in xml_text
    assert "research_object" not in xml_text and "None" not in xml_text and "[]" not in xml_text


def test_word_report_uses_compact_columns_and_merges_duplicate_evidence(tmp_path):
    shared = {"evidence_quote_raw": "The study included 67 participants.", "evidence_quote_display": "The study included 67 participants.", "pdf_page_start": 2, "pdf_page_end": 2, "verified": True, "source_type": "body", "quality_flags": []}
    first = {"file_name": "one.pdf", "title_candidate": "One", "facts": [dict(shared, category="sample_size"), dict(shared, category="major_result")], "extractive_summary": {"evidence": []}}
    second = {"file_name": "two.pdf", "title_candidate": "Two", "facts": [dict(shared, category="sample_size")], "extractive_summary": {"evidence": []}}
    rows = compare_papers([first, second])
    path = export_word(rows, tmp_path / "compact.docx", papers=[first, second])
    document = Document(path)
    assert len(document.tables[0].columns) == 3
    assert [len(table.columns) for table in document.tables[2::2]] == [4, 4]
    assert document.tables[2].rows.__len__() == 2  # header + one merged evidence row
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    assert b"w:wordWrap" in xml and b"w:eastAsia" in xml
