from __future__ import annotations

from pathlib import Path

import fitz

import paper_claude
from local_search import LocalSearchIndex
from paper_pipeline import PageText


def _pages(source: str, text: str, page: int = 1) -> list[PageText]:
    return [PageText(source, page, text, len(text), False)]


def _index(path: Path, source: str, text: str) -> LocalSearchIndex:
    index = LocalSearchIndex(path)
    index.index_pages(_pages(source, text))
    return index


def test_search_isolated_between_batch_databases(tmp_path):
    batch_a = _index(tmp_path / "batch_a" / "local_index.sqlite", "a.pdf", "alpha-only finding")
    batch_b = _index(tmp_path / "batch_b" / "local_index.sqlite", "b.pdf", "beta-only finding")
    try:
        assert batch_a.search("alpha-only")
        assert batch_a.search("beta-only") == []
        assert batch_b.search("beta-only")
        assert batch_b.search("alpha-only") == []
    finally:
        batch_a.close()
        batch_b.close()


def test_source_filter_limits_results_within_a_batch(tmp_path):
    index = LocalSearchIndex(tmp_path / "batch" / "local_index.sqlite")
    index.index_pages(_pages("a.pdf", "shared phrase from A"))
    index.index_pages(_pages("b.pdf", "shared phrase from B"))
    try:
        rows = index.search("shared phrase", source_files=["b.pdf"])
        assert [row["source_file"] for row in rows] == ["b.pdf"]
        assert index.search("shared phrase", source_files=["missing.pdf"]) == []
    finally:
        index.close()


def test_fts_and_fallback_search_have_the_same_scope_behavior(tmp_path):
    fts = LocalSearchIndex(tmp_path / "fts.sqlite")
    fts.index_pages(_pages("a.pdf", "shared phrase A"))
    fallback = LocalSearchIndex(tmp_path / "fallback.sqlite")
    fallback.conn.execute("DROP TABLE pages_fts")
    fallback.conn.execute(
        "CREATE TABLE pages_fallback(source_file TEXT, page_number INTEGER, text TEXT, content_hash TEXT, UNIQUE(source_file, page_number))"
    )
    fallback.conn.commit()
    fallback.fts5_available = False
    fallback.index_pages(_pages("b.pdf", "shared phrase B"))
    try:
        assert fts.search("shared phrase", source_files=["a.pdf"])[0]["source_file"] == "a.pdf"
        assert fallback.search("shared phrase", source_files=["b.pdf"])[0]["source_file"] == "b.pdf"
        assert fallback.search("shared phrase", source_files=["a.pdf"]) == []
    finally:
        fts.close()
        fallback.close()


def test_no_context_does_not_fall_back_to_legacy_global_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = _index(tmp_path / "output" / "local_index.sqlite", "legacy.pdf", "legacy-only phrase")
    legacy.close()
    assert paper_claude.search_local_index("legacy-only") == "请先完成一次 Local Offline 文献处理。"


def test_stale_context_returns_clear_error_without_fallback(tmp_path):
    context = {
        "batch_id": "missing-batch",
        "task_dir": str(tmp_path / "missing-batch"),
        "db_path": str(tmp_path / "missing-batch" / "local_index.sqlite"),
        "document_sources": ["old.pdf"],
        "document_count": 1,
    }
    assert "索引不存在" in paper_claude.search_local_index("old", context)


def test_reindex_same_source_does_not_duplicate_rows(tmp_path):
    index = LocalSearchIndex(tmp_path / "batch" / "local_index.sqlite")
    pages = _pages("same.pdf", "one stable phrase")
    try:
        assert index.index_pages(pages) == 1
        assert index.index_pages(pages) == 0
        assert len(index.search("stable phrase")) == 1
    finally:
        index.close()


def test_local_processing_creates_batch_scoped_index_and_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = _index(tmp_path / "output" / "local_index.sqlite", "legacy.pdf", "legacy-only phrase")
    legacy.close()
    pdf_path = tmp_path / "batch_input.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Scoped batch title\nAbstract\nA local finding.")
        document.save(pdf_path)

    result, _, context = paper_claude.process_local_papers([str(pdf_path)], return_search_context=True)
    assert "Local Offline" in result
    assert context["db_path"].endswith("local_index.sqlite")
    assert Path(context["db_path"]).parent == Path(context["task_dir"])
    assert not (tmp_path / "output" / "local_index.sqlite").samefile(Path(context["db_path"]))
    assert paper_claude.search_local_index("legacy-only", context) == "未找到匹配内容。"


def test_failed_document_is_not_added_to_current_batch_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    valid_path = tmp_path / "valid.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "VALID_BATCH_ONLY_PHRASE")
        document.save(valid_path)
    failed_path = tmp_path / "failed.pdf"
    failed_path.write_bytes(b"not a valid PDF")

    _, _, context = paper_claude.process_local_papers(
        [str(valid_path), str(failed_path)],
        return_search_context=True,
    )
    assert "valid.pdf" in context["document_sources"]
    assert "failed.pdf" not in context["document_sources"]
    assert "VALID_BATCH_ONLY_PHRASE" in paper_claude.search_local_index("VALID_BATCH_ONLY_PHRASE", context)
    assert paper_claude.search_local_index("FAILED_DOCUMENT_ONLY_PHRASE", context) == "未找到匹配内容。"
