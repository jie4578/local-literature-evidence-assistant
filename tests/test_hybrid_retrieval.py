from __future__ import annotations

from dataclasses import replace

from local_search import LocalSearchIndex
from retrieval import HybridRetriever, NullSemanticRetriever, RetrievalResult
from retrieval.lexical import LexicalRetriever
from retrieval.fusion import reciprocal_rank_fusion


def _result(passage_id: str, rank: int, source: str = "lexical", **overrides) -> RetrievalResult:
    value = RetrievalResult(
        passage_id=passage_id,
        source_file=f"{passage_id}.pdf",
        document_id=f"doc-{passage_id}",
        section="Results",
        pdf_page_start=rank,
        pdf_page_end=rank,
        text=f"text for {passage_id}",
        rank=rank,
    )
    if source == "lexical":
        value.lexical_rank = rank
    else:
        value.semantic_rank = rank
    for key, item in overrides.items():
        setattr(value, key, item)
    return value


class FakeRetriever:
    def __init__(self, results, error: Exception | None = None):
        self.results = results
        self.error = error
        self.calls = []

    def search(self, query, limit=10, source_files=None, sections=None, exclude_references=True):
        self.calls.append({
            "query": query,
            "limit": limit,
            "source_files": source_files,
            "sections": sections,
            "exclude_references": exclude_references,
        })
        if self.error:
            raise self.error
        return list(self.results)[:limit]


class FakePassageBackend:
    def __init__(self, results):
        self.results = results

    def search_passages(self, query, limit=10, source_files=None, sections=None, exclude_references=True):
        return [result.to_dict() for result in self.results[:limit]]


def test_rrf_deduplicates_and_records_both_sources():
    lexical = [_result("P1", 1), _result("P2", 2), _result("P3", 3)]
    semantic = [_result("P3", 1, "semantic"), _result("P2", 2, "semantic"), _result("P4", 3, "semantic")]
    fused = reciprocal_rank_fusion(lexical, semantic, limit=10, allowed_passage_ids={"P1", "P2", "P3", "P4"})
    assert {item.passage_id for item in fused[:2]} == {"P2", "P3"}
    assert len({item.passage_id for item in fused}) == 4
    p2 = next(item for item in fused if item.passage_id == "P2")
    assert p2.retrieval_sources == ["lexical", "semantic"]
    assert p2.lexical_rank == 2 and p2.semantic_rank == 2


def test_lexical_only_hybrid_preserves_lexical_order():
    backend = FakePassageBackend([_result("P1", 1), _result("P2", 2)])
    expected = LexicalRetriever(backend).search("query", limit=10)
    response = HybridRetriever(LexicalRetriever(backend), semantic_retriever=None).search("query", limit=10)
    assert response.mode == "lexical"
    assert [item.passage_id for item in response.results] == [item.passage_id for item in expected]
    assert response.semantic_status == "disabled"


def test_semantic_only_candidate_is_kept_when_in_current_batch():
    lexical = FakeRetriever([_result("P1", 1)])
    semantic = FakeRetriever([_result("P4", 1, "semantic")])
    response = HybridRetriever(
        lexical,
        semantic,
        allowed_passage_ids={"P1", "P4"},
    ).search("query", limit=5)
    p4 = next(item for item in response.results if item.passage_id == "P4")
    assert p4.semantic_rank == 1
    assert p4.lexical_rank is None
    assert p4.retrieval_sources == ["semantic"]


def test_duplicate_passage_is_returned_once():
    lexical = [_result("P1", 1)]
    semantic = [_result("P1", 1, "semantic")]
    fused = reciprocal_rank_fusion(lexical, semantic, allowed_passage_ids={"P1"})
    assert [item.passage_id for item in fused] == ["P1"]


def test_filters_are_forwarded_to_both_retrievers():
    lexical = FakeRetriever([_result("P1", 1)])
    semantic = FakeRetriever([_result("P1", 1, "semantic")])
    HybridRetriever(lexical, semantic, allowed_passage_ids={"P1"}).search(
        "query", limit=4, source_files=["paper_a.pdf"], sections=["Results"], exclude_references=False
    )
    assert lexical.calls[0] == semantic.calls[0]
    assert lexical.calls[0]["limit"] == 12


def test_unknown_semantic_passage_is_dropped():
    lexical = FakeRetriever([_result("P1", 1)])
    semantic = FakeRetriever([_result("UNKNOWN_PASSAGE_ID", 1, "semantic")])
    response = HybridRetriever(lexical, semantic, allowed_passage_ids={"P1"}).search("query")
    assert [item.passage_id for item in response.results] == ["P1"]


def test_semantic_failure_falls_back_to_lexical_with_warning():
    lexical = FakeRetriever([_result("P1", 1)])
    semantic = FakeRetriever([], error=TimeoutError("offline fake"))
    response = HybridRetriever(lexical, semantic).search("query")
    assert response.mode == "lexical"
    assert response.semantic_status == "error"
    assert response.results[0].passage_id == "P1"
    assert response.warnings


def test_rrf_ties_are_stable_across_runs():
    lexical = [_result("P1", 1), _result("P2", 2)]
    semantic = [_result("P2", 1, "semantic"), _result("P1", 2, "semantic")]
    runs = [
        [item.passage_id for item in reciprocal_rank_fusion(lexical, semantic, allowed_passage_ids={"P1", "P2"})]
        for _ in range(10)
    ]
    assert all(run == runs[0] for run in runs)


def test_candidate_limit_uses_multiplier_and_cap():
    lexical = FakeRetriever([_result("P1", 1)])
    semantic = FakeRetriever([_result("P1", 1, "semantic")])
    HybridRetriever(lexical, semantic, allowed_passage_ids={"P1"}).search("query", limit=5)
    assert lexical.calls[0]["limit"] == 15
    assert semantic.calls[0]["limit"] == 15
    lexical.calls.clear()
    semantic.calls.clear()
    HybridRetriever(lexical, semantic, max_candidate_limit=20, allowed_passage_ids={"P1"}).search("query", limit=50)
    assert lexical.calls[0]["limit"] == 20
    assert semantic.calls[0]["limit"] == 20


def test_fusion_preserves_page_section_and_text_metadata():
    original = _result("P1", 1, source="lexical", source_file="source.pdf", section="Methods", pdf_page_start=2, pdf_page_end=3, text="original evidence")
    semantic = replace(original, rank=1, lexical_rank=None, semantic_rank=1, retrieval_sources=["semantic"])
    fused = reciprocal_rank_fusion([original], [semantic], allowed_passage_ids={"P1"})
    assert fused[0].source_file == "source.pdf"
    assert fused[0].section == "Methods"
    assert (fused[0].pdf_page_start, fused[0].pdf_page_end) == (2, 3)
    assert fused[0].text == "original evidence"


def test_null_semantic_backend_is_explicitly_empty():
    assert NullSemanticRetriever().search("anything") == []


def test_lexical_adapter_keeps_statistical_query_compatibility(tmp_path):
    with LocalSearchIndex(tmp_path / "stats.sqlite") as index:
        index.index_passages([
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
        ])
        retriever = LexicalRetriever(index)
        for query in ("p = 0.03", "p value", "p-value", "p值"):
            rows = retriever.search(query)
            assert rows and rows[0].passage_id == "stats"
