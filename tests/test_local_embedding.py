from __future__ import annotations

import hashlib
import sys

import numpy as np
import pytest

from local_search import LocalSearchIndex, SearchContext
from retrieval import (
    HybridRetriever,
    LocalEmbeddingSemanticRetriever,
    LocalModelPathError,
    LocalSentenceTransformerEncoder,
    OptionalSemanticDependencyError,
    build_semantic_index,
)
from retrieval.lexical import LexicalRetriever


class DeterministicTestEncoder:
    """测试专用语义空间：同义科研词映射到固定维度，不使用随机数。"""

    dimension = 3
    batch_size = 2
    model_fingerprint = "deterministic-test-encoder-v1"

    def __init__(self, fingerprint: str | None = None):
        if fingerprint:
            self.model_fingerprint = fingerprint
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        thermal = any(word in lowered for word in ("thermal", "heat", "temperature"))
        stability = any(word in lowered for word in ("protein", "aggregation", "stability", "instability"))
        unrelated = any(word in lowered for word in ("cell", "culture", "medium", "unrelated"))
        return [float(thermal), float(stability), float(unrelated)]

    def _encode(self, texts):
        values = np.asarray([self._vector(text) for text in texts], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1)
        norms[norms == 0] = 1
        return values / norms[:, None]

    def encode_queries(self, texts):
        values = list(texts)
        self.query_calls += len(values)
        return self._encode(values)

    def encode_documents(self, texts):
        values = list(texts)
        self.document_calls += len(values)
        return self._encode(values)


def _passage(passage_id, source, section, text, page):
    return {
        "passage_id": passage_id,
        "source_file": source,
        "document_id": f"doc-{source}",
        "section": section,
        "text": text,
        "pdf_page_start": page,
        "pdf_page_end": page,
        "ordinal": page,
    }


def _batch(tmp_path, name, passages):
    task_dir = tmp_path / name
    task_dir.mkdir()
    db_path = task_dir / "local_index.sqlite"
    with LocalSearchIndex(db_path) as index:
        index.index_passages(passages)
    return SearchContext(
        batch_id=name,
        task_dir=str(task_dir),
        db_path=str(db_path),
        document_sources=tuple(sorted({item["source_file"] for item in passages})),
        document_count=len({item["source_file"] for item in passages}),
    ).to_dict()


def _corpus():
    return [
        _passage("P1", "a.pdf", "Results", "Thermal stress increased protein aggregation.", 2),
        _passage("P2", "a.pdf", "Methods", "Cell culture medium was prepared.", 3),
        _passage("P3", "a.pdf", "Discussion", "Heat exposure reduced antibody stability.", 4),
    ]


def test_core_imports_without_optional_sentence_transformers(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    import paper_claude

    assert paper_claude.build_ui() is not None
    with pytest.raises(OptionalSemanticDependencyError):
        LocalSentenceTransformerEncoder(tmp_path)


def test_remote_model_id_is_rejected_before_optional_import(monkeypatch):
    def fail_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise AssertionError("optional model import must not happen for a non-directory path")
        return original_import(name, *args, **kwargs)

    original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    monkeypatch.setattr("builtins.__import__", fail_import)
    with pytest.raises(LocalModelPathError):
        LocalSentenceTransformerEncoder("BAAI/bge-small-en-v1.5")


def test_embedding_cache_first_build_and_second_cache_hit(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    encoder = DeterministicTestEncoder()
    first = build_semantic_index(context, encoder)
    second = build_semantic_index(context, encoder)
    assert (first["total_passages"], first["encoded"], first["cached"]) == (3, 3, 0)
    assert (second["encoded"], second["cached"], second["failed"]) == (0, 3, 0)
    assert encoder.document_calls == 3


def test_text_hash_invalidation_reencodes_only_changed_passage(tmp_path):
    passages = _corpus()
    context = _batch(tmp_path, "batch", passages)
    encoder = DeterministicTestEncoder()
    build_semantic_index(context, encoder)
    changed = [dict(item) for item in passages]
    changed[1]["text"] = "Cell culture medium was heated for thermal stability testing."
    with LocalSearchIndex(context["db_path"]) as index:
        index.index_passages(changed)
    result = build_semantic_index(context, encoder)
    assert result["cached"] == 2
    assert result["encoded"] == 1


def test_model_fingerprint_isolation(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    encoder_a = DeterministicTestEncoder("encoder-a")
    encoder_b = DeterministicTestEncoder("encoder-b")
    first = build_semantic_index(context, encoder_a)
    second = build_semantic_index(context, encoder_b)
    assert first["encoded"] == 3 and second["encoded"] == 3
    assert first["model_fingerprint"] != second["model_fingerprint"]


def test_embedding_cache_schema_and_float32_blob(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    encoder = DeterministicTestEncoder()
    build_semantic_index(context, encoder)
    with LocalSearchIndex(context["db_path"]) as index:
        columns = {
            row[1]
            for row in index.conn.execute("PRAGMA table_info(passage_embeddings)").fetchall()
        }
        row = index.conn.execute(
            "SELECT dimension, vector_blob FROM passage_embeddings WHERE passage_id = ?",
            ("P1",),
        ).fetchone()
    assert columns == {"passage_id", "model_fingerprint", "text_hash", "dimension", "vector_blob"}
    assert row[0] == 3 and len(row[1]) == 3 * 4


def test_embedding_passage_cap_is_enforced_before_encoding(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    encoder = DeterministicTestEncoder()
    with pytest.raises(ValueError, match="semantic 上限"):
        build_semantic_index(context, encoder, max_passages=2)
    assert encoder.document_calls == 0


def test_explicit_empty_source_filter_does_not_fall_back_to_batch_sources(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    encoder = DeterministicTestEncoder()
    result = build_semantic_index(context, encoder, source_files=[])
    assert result["total_passages"] == 0
    assert result["encoded"] == 0


def test_semantic_ranking_is_deterministic_and_not_lexical_only(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    encoder = DeterministicTestEncoder()
    build_semantic_index(context, encoder)
    retriever = LocalEmbeddingSemanticRetriever(context, encoder)
    rows = retriever.search("temperature-related protein instability", limit=3)
    assert [row.passage_id for row in rows[:2]] == ["P1", "P3"]
    assert rows[0].semantic_score is not None
    assert rows[0].retrieval_sources == ["semantic"]


def test_semantic_source_and_section_filters(tmp_path):
    passages = _corpus() + [_passage("P4", "b.pdf", "Results", "Heat exposure reduced protein stability.", 5)]
    context = _batch(tmp_path, "batch", passages)
    encoder = DeterministicTestEncoder()
    build_semantic_index(context, encoder)
    retriever = LocalEmbeddingSemanticRetriever(context, encoder)
    rows = retriever.search("temperature protein stability", source_files=["b.pdf"], sections=["Results"])
    assert rows and {row.source_file for row in rows} == {"b.pdf"}
    assert {row.section for row in rows} == {"Results"}


def test_semantic_batch_isolation(tmp_path):
    context_a = _batch(tmp_path, "batch_a", [_passage("A1", "a.pdf", "Results", "ALPHA_SEMANTIC_ONLY", 2)])
    context_b = _batch(tmp_path, "batch_b", [_passage("B1", "b.pdf", "Results", "BETA_SEMANTIC_ONLY", 2)])
    encoder = DeterministicTestEncoder()
    build_semantic_index(context_a, encoder)
    build_semantic_index(context_b, encoder)
    rows = LocalEmbeddingSemanticRetriever(context_b, encoder).search("ALPHA_SEMANTIC_ONLY")
    assert all(row.source_file != "a.pdf" for row in rows)


def test_corrupt_vector_is_skipped_with_warning(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    encoder = DeterministicTestEncoder()
    build_semantic_index(context, encoder)
    with LocalSearchIndex(context["db_path"]) as index:
        index.conn.execute(
            "UPDATE passage_embeddings SET dimension = 99, vector_blob = ? WHERE passage_id = ?",
            (b"bad", "P1"),
        )
        index.conn.commit()
    retriever = LocalEmbeddingSemanticRetriever(context, encoder)
    rows = retriever.search("temperature protein instability")
    assert all(row.passage_id != "P1" for row in rows)
    assert retriever.last_warnings


def test_hybrid_integration_preserves_metadata_and_uses_rrf(tmp_path):
    passages = _corpus()
    context = _batch(tmp_path, "batch", passages)
    encoder = DeterministicTestEncoder()
    build_semantic_index(context, encoder)
    with LocalSearchIndex(context["db_path"]) as index:
        response = HybridRetriever(
            LexicalRetriever(index),
            LocalEmbeddingSemanticRetriever(context, encoder),
            allowed_passage_ids={item["passage_id"] for item in passages},
        ).search("temperature protein instability", limit=3)
    assert response.mode == "hybrid"
    assert response.results
    assert all(item.source_file == "a.pdf" for item in response.results)
    assert all(item.pdf_page_start > 0 and item.section for item in response.results)


def test_corrupted_or_missing_index_reports_not_indexed(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    encoder = DeterministicTestEncoder()
    response = HybridRetriever(
        LexicalRetriever(type("EmptyBackend", (), {"search_passages": lambda self, *args, **kwargs: []})()),
        LocalEmbeddingSemanticRetriever(context, encoder),
        allowed_passage_ids=set(),
    ).search("temperature")
    assert response.mode == "lexical"
    assert response.semantic_status == "not_indexed"
    assert response.warnings


def test_model_path_is_not_written_to_sqlite(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    secret_model_path = r"X:\redacted\secret_model"
    fingerprint = hashlib.sha256(secret_model_path.encode("utf-8")).hexdigest()
    build_semantic_index(context, DeterministicTestEncoder(fingerprint))
    database_bytes = open(context["db_path"], "rb").read()
    assert secret_model_path.encode("utf-8") not in database_bytes


def test_embedding_database_can_be_deleted_after_retriever_use(tmp_path):
    context = _batch(tmp_path, "batch", _corpus())
    encoder = DeterministicTestEncoder()
    build_semantic_index(context, encoder)
    LocalEmbeddingSemanticRetriever(context, encoder).search("temperature")
    path = tmp_path / "batch" / "local_index.sqlite"
    path.unlink()
    assert not path.exists()
