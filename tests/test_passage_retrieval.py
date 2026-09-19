from __future__ import annotations

from pathlib import Path

from local_search import LocalSearchIndex
from passage_retrieval import build_evidence_passages
from paper_pipeline import PageText


def _page(source: str, number: int, text: str) -> dict:
    return {
        "source_file": source,
        "page_number": number,
        "text": text,
        "display_text": text,
    }


def _index_passages(path: Path, passages):
    index = LocalSearchIndex(path)
    index.index_passages(passages)
    return index


def test_single_page_passage_keeps_section_and_physical_page():
    passages = build_evidence_passages([_page("paper.pdf", 2, "Methods\nThe study used a controlled design.")])
    method = next(item for item in passages if item.section == "Methods")
    assert method.pdf_page_start == 2
    assert method.pdf_page_end == 2
    assert "controlled design" in method.text


def test_multi_page_passage_records_real_page_range():
    passages = build_evidence_passages(
        [
            _page("paper.pdf", 2, "Methods\nThe first part of the protocol was described here."),
            _page("paper.pdf", 3, "The second part of the protocol continued on the next page."),
        ],
        max_chars=200,
    )
    cross_page = next(item for item in passages if item.pdf_page_start == 2 and item.pdf_page_end == 3)
    assert "second part" in cross_page.text


def test_section_boundary_never_merges_methods_and_results():
    passages = build_evidence_passages(
        [_page("paper.pdf", 2, "Methods\nThe method was randomized."), _page("paper.pdf", 3, "Results\nThe result improved.")],
        max_chars=500,
    )
    assert {item.section for item in passages} == {"Methods", "Results"}
    assert all("The method" not in item.text or item.section == "Methods" for item in passages)
    assert all("The result" not in item.text or item.section == "Results" for item in passages)


def test_very_long_paragraph_is_split_into_nonempty_bounded_passages():
    text = "Methods\n" + " ".join(f"The protocol sentence {number} was recorded." for number in range(80))
    passages = build_evidence_passages([_page("paper.pdf", 4, text)], max_chars=300)
    assert len(passages) > 1
    assert all(0 < len(item.text) <= 300 for item in passages)
    assert all(item.pdf_page_start == item.pdf_page_end == 4 for item in passages)


def test_short_paragraphs_can_merge_within_one_section():
    passages = build_evidence_passages(
        [_page("paper.pdf", 2, "Methods\nShort paragraph one.\n\nShort paragraph two.")],
        max_chars=200,
    )
    methods = [item for item in passages if item.section == "Methods"]
    assert len(methods) == 1
    assert "paragraph one" in methods[0].text and "paragraph two" in methods[0].text


def test_deterministic_lexical_ranking_prefers_more_matching_terms(tmp_path):
    passages = [
        {
            "passage_id": "a",
            "source_file": "a.pdf",
            "document_id": "doc-a",
            "section": "Results",
            "text": "The antibody aggregation increased significantly after incubation.",
            "pdf_page_start": 2,
            "pdf_page_end": 2,
            "ordinal": 1,
        },
        {
            "passage_id": "b",
            "source_file": "b.pdf",
            "document_id": "doc-b",
            "section": "Results",
            "text": "Aggregation was measured after incubation.",
            "pdf_page_start": 3,
            "pdf_page_end": 3,
            "ordinal": 1,
        },
    ]
    with _index_passages(tmp_path / "ranking.sqlite", passages) as index:
        rows = index.search_passages("antibody aggregation incubation")
    assert [row["source_file"] for row in rows[:2]] == ["a.pdf", "b.pdf"]
    assert [row["rank"] for row in rows[:2]] == [1, 2]


def test_section_and_source_filters_are_parameterized(tmp_path):
    passages = [
        {
            "passage_id": "methods",
            "source_file": "a.pdf",
            "document_id": "doc-a",
            "section": "Methods",
            "text": "The p = 0.03 method threshold was configured.",
            "pdf_page_start": 2,
            "pdf_page_end": 2,
            "ordinal": 1,
        },
        {
            "passage_id": "results",
            "source_file": "a.pdf",
            "document_id": "doc-a",
            "section": "Results",
            "text": "The result was significant with p = 0.03.",
            "pdf_page_start": 3,
            "pdf_page_end": 3,
            "ordinal": 2,
        },
        {
            "passage_id": "other",
            "source_file": "b.pdf",
            "document_id": "doc-b",
            "section": "Results",
            "text": "The result was significant with p = 0.03.",
            "pdf_page_start": 4,
            "pdf_page_end": 4,
            "ordinal": 1,
        },
    ]
    with _index_passages(tmp_path / "filters.sqlite", passages) as index:
        rows = index.search_passages("p = 0.03", source_files=["a.pdf"], sections=["Results"])
    assert len(rows) == 1
    assert rows[0]["source_file"] == "a.pdf"
    assert rows[0]["section"] == "Results"
    assert rows[0]["pdf_page_start"] == 3


def test_reference_exclusion_defaults_to_true(tmp_path):
    passages = [
        {
            "passage_id": "ref",
            "source_file": "paper.pdf",
            "document_id": "doc",
            "section": "References",
            "text": "The reference reports a significant result.",
            "pdf_page_start": 5,
            "pdf_page_end": 5,
            "ordinal": 1,
        },
        {
            "passage_id": "result",
            "source_file": "paper.pdf",
            "document_id": "doc",
            "section": "Results",
            "text": "The study reports a significant result.",
            "pdf_page_start": 3,
            "pdf_page_end": 3,
            "ordinal": 2,
        },
    ]
    with _index_passages(tmp_path / "references.sqlite", passages) as index:
        default_rows = index.search_passages("significant result")
        all_rows = index.search_passages("significant result", exclude_references=False)
    assert [row["section"] for row in default_rows] == ["Results"]
    assert len(all_rows) == 2


def test_statistical_aliases_return_short_passages(tmp_path):
    passages = [
        {
            "passage_id": "stats",
            "source_file": "stats.pdf",
            "document_id": "doc",
            "section": "Results",
            "text": "The result was significant with p = 0.03.",
            "pdf_page_start": 4,
            "pdf_page_end": 4,
            "ordinal": 1,
        }
    ]
    with _index_passages(tmp_path / "stats.sqlite", passages) as index:
        for query in ("p value", "p-value", "p值", "p = 0.03"):
            rows = index.search_passages(query)
            assert rows and rows[0]["text"] == passages[0]["text"]
            assert rows[0]["pdf_page_start"] == 4


def test_fallback_passage_search_preserves_filters_and_scope(tmp_path):
    index = LocalSearchIndex(tmp_path / "fallback.sqlite")
    index.conn.execute("DROP TABLE passages_fts")
    index.conn.execute(
        "CREATE TABLE passages_fallback(passage_id TEXT PRIMARY KEY, text TEXT NOT NULL)"
    )
    index.conn.commit()
    index.passage_fts5_available = False
    index.index_passages(
        [
            {
                "passage_id": "a",
                "source_file": "a.pdf",
                "document_id": "a",
                "section": "Results",
                "text": "fallback alpha phrase",
                "pdf_page_start": 2,
                "pdf_page_end": 2,
                "ordinal": 1,
            },
            {
                "passage_id": "b",
                "source_file": "b.pdf",
                "document_id": "b",
                "section": "Methods",
                "text": "fallback alpha phrase",
                "pdf_page_start": 3,
                "pdf_page_end": 3,
                "ordinal": 1,
            },
        ]
    )
    try:
        rows = index.search_passages("fallback alpha", source_files=["a.pdf"], sections=["Results"])
        assert len(rows) == 1 and rows[0]["source_file"] == "a.pdf"
    finally:
        index.close()


def test_passage_database_can_be_deleted_after_close(tmp_path):
    path = tmp_path / "close.sqlite"
    index = LocalSearchIndex(path)
    index.close()
    path.unlink()
    assert not path.exists()
