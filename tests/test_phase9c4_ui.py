from __future__ import annotations

from pathlib import Path

import numpy as np

import paper_claude
from local_search import LocalSearchIndex, SearchContext
from retrieval.ui_runtime import LocalModelCache, profile_config


class FakeEncoder:
    dimension = 3
    batch_size = 8

    def __init__(self, fingerprint: str = "fake-ui-model", *, fail_queries: bool = False, **kwargs):
        self.model_fingerprint = fingerprint
        self.fail_queries = fail_queries
        self.query_prefix = kwargs.get("query_prefix", "")
        self.document_prefix = kwargs.get("document_prefix", "")
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        value = text.casefold()
        return [
            float(any(term in value for term in ("thermal", "heat", "temperature"))),
            float(any(term in value for term in ("stability", "aggregation", "protein"))),
            float(any(term in value for term in ("control", "unrelated"))),
        ]

    def _encode(self, texts):
        values = np.asarray([self._vector(text) for text in texts], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1)
        norms[norms == 0] = 1
        return values / norms[:, None]

    def encode_documents(self, texts):
        values = list(texts)
        self.document_calls += len(values)
        return self._encode(values)

    def encode_queries(self, texts):
        if self.fail_queries:
            raise RuntimeError("offline test failure")
        values = list(texts)
        self.query_calls += len(values)
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


def _context(tmp_path, name="batch", source="synthetic.pdf"):
    task_dir = tmp_path / name
    task_dir.mkdir()
    db_path = task_dir / "local_index.sqlite"
    passages = [
        _passage("P1", source, "Methods", "The study used thermal exposure for stability testing.", 1),
        _passage("P2", source, "Results", "Heat exposure reduced protein stability.", 2),
        _passage("P3", source, "Discussion", "The control condition remained unchanged.", 3),
    ]
    with LocalSearchIndex(db_path) as index:
        index.index_passages(passages)
    return SearchContext(
        batch_id=name,
        task_dir=str(task_dir),
        db_path=str(db_path),
        document_sources=(source,),
        document_count=1,
    ).to_dict()


def test_default_ui_build_does_not_load_a_model(monkeypatch):
    class NoLoadCache:
        def get_or_load(self, *args, **kwargs):
            raise AssertionError("UI construction must not load a local model")

        def get(self, *args, **kwargs):
            raise AssertionError("UI construction must not access the model cache")

    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", NoLoadCache())
    demo = paper_claude.build_ui()
    assert demo is not None


def test_hybrid_toggle_and_profile_change_only_reset_state(tmp_path):
    state, status = paper_claude.reset_semantic_ui_state("配置已变化")
    assert state["ready"] is False
    assert "配置已变化" in status
    assert profile_config("E5 Small v2")["query_prefix"] == "query: "
    assert profile_config("E5 Small v2")["document_prefix"] == "passage: "


def test_build_without_context_does_not_call_model_factory(monkeypatch, tmp_path):
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeEncoder()

    cache = LocalModelCache(encoder_factory=factory)
    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", cache)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    state, message = paper_claude.build_semantic_index_for_ui(None, "PubMedBERT", str(model_dir))
    assert calls == []
    assert state["ready"] is False
    assert "先完成一次 Local Offline" in message


def test_explicit_build_is_ready_and_keeps_state_small(monkeypatch, tmp_path):
    calls = []

    def factory(model_path, **kwargs):
        calls.append((Path(model_path), kwargs))
        return FakeEncoder(**kwargs)

    cache = LocalModelCache(encoder_factory=factory)
    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", cache)
    context = _context(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    state, message = paper_claude.build_semantic_index_for_ui(context, "PubMedBERT", str(model_dir))
    assert state["ready"] is True
    assert state["batch_id"] == "batch"
    assert state["indexed_passages"] == 3
    assert state["encoded"] == 3
    assert calls
    assert str(model_dir) not in repr(state)
    assert str(model_dir) not in message
    assert "Hybrid Local 已就绪" in message


def test_e5_ui_profile_passes_query_and_document_prefixes(monkeypatch, tmp_path):
    captured = {}

    def factory(model_path, **kwargs):
        captured.update(kwargs)
        return FakeEncoder(**kwargs)

    cache = LocalModelCache(encoder_factory=factory)
    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", cache)
    context = _context(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    state, _ = paper_claude.build_semantic_index_for_ui(context, "E5 Small v2", str(model_dir))
    assert state["ready"] is True
    assert captured["query_prefix"] == "query: "
    assert captured["document_prefix"] == "passage: "


def test_missing_optional_dependency_has_friendly_fallback(monkeypatch, tmp_path):
    class DependencyMissingCache:
        def get_or_load(self, *args, **kwargs):
            from retrieval.local_embedding import OptionalSemanticDependencyError

            raise OptionalSemanticDependencyError("do not expose implementation details")

    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", DependencyMissingCache())
    context = _context(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    state, message = paper_claude.build_semantic_index_for_ui(context, "PubMedBERT", str(model_dir))
    assert state["ready"] is False
    assert "requirements-semantic.txt" in message
    assert "do not expose" not in message


def test_second_build_uses_model_and_embedding_caches(monkeypatch, tmp_path):
    calls = []

    def factory(model_path, **kwargs):
        calls.append(model_path)
        return FakeEncoder(**kwargs)

    cache = LocalModelCache(encoder_factory=factory)
    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", cache)
    context = _context(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    first, _ = paper_claude.build_semantic_index_for_ui(context, "PubMedBERT", str(model_dir))
    second, _ = paper_claude.build_semantic_index_for_ui(context, "PubMedBERT", str(model_dir))
    assert first["encoded"] == 3 and first["cached"] == 0
    assert second["encoded"] == 0 and second["cached"] == 3
    assert len(calls) == 1


def test_hybrid_search_success_preserves_source_section_and_page(monkeypatch, tmp_path):
    cache = LocalModelCache(encoder_factory=lambda *args, **kwargs: FakeEncoder(**kwargs))
    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", cache)
    context = _context(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    state, _ = paper_claude.build_semantic_index_for_ui(context, "PubMedBERT", str(model_dir))
    result = paper_claude.search_local_index_for_ui(
        "protein became less stable after heating",
        "Hybrid Local",
        context,
        state,
    )
    assert "Hybrid Local / RRF" in result
    assert "synthetic.pdf" in result
    assert "PDF 第 2 页" in result
    assert "Results" in result


def test_hybrid_not_ready_and_new_batch_fall_back_to_current_batch(monkeypatch, tmp_path):
    cache = LocalModelCache(encoder_factory=lambda *args, **kwargs: FakeEncoder(**kwargs))
    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", cache)
    context_a = _context(tmp_path, "batch_a", "a.pdf")
    context_b = _context(tmp_path, "batch_b", "b.pdf")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    state_a, _ = paper_claude.build_semantic_index_for_ui(context_a, "PubMedBERT", str(model_dir))
    result = paper_claude.search_local_index_for_ui("stability", "Hybrid Local", context_b, state_a)
    assert "回退为 Lexical / FTS5" in result
    assert "b.pdf" in result
    assert "a.pdf" not in result
    reset, _ = paper_claude.reset_semantic_ui_state("新批次")
    assert reset["batch_id"] is None and reset["ready"] is False


def test_semantic_failure_falls_back_without_traceback_or_path(monkeypatch, tmp_path):
    cache = LocalModelCache(encoder_factory=lambda *args, **kwargs: FakeEncoder(fail_queries=True, **kwargs))
    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", cache)
    context = _context(tmp_path)
    model_dir = tmp_path / "private-model-path"
    model_dir.mkdir()
    state, _ = paper_claude.build_semantic_index_for_ui(context, "PubMedBERT", str(model_dir))
    result = paper_claude.search_local_index_for_ui("stability", "Hybrid Local", context, state)
    assert "回退为 Lexical / FTS5" in result
    assert "Traceback" not in result
    assert str(model_dir) not in result


def test_invalid_model_path_is_friendly_and_private(monkeypatch, tmp_path):
    cache = LocalModelCache(encoder_factory=lambda *args, **kwargs: FakeEncoder(**kwargs))
    monkeypatch.setattr(paper_claude, "LOCAL_MODEL_CACHE", cache)
    context = _context(tmp_path)
    private_path = tmp_path / "private-secret-model"
    state, message = paper_claude.build_semantic_index_for_ui(context, "PubMedBERT", str(private_path))
    assert state["ready"] is False
    assert "无效" in message
    assert str(private_path) not in message
    assert str(private_path) not in repr(state)


def test_model_cache_is_bounded_and_evicts_oldest(tmp_path):
    calls = []

    def factory(model_path, **kwargs):
        calls.append(str(model_path))
        return FakeEncoder(fingerprint=Path(model_path).name, **kwargs)

    cache = LocalModelCache(max_models=2, encoder_factory=factory)
    dirs = []
    for index in range(3):
        model_dir = tmp_path / f"model-{index}"
        model_dir.mkdir()
        dirs.append(model_dir)
        cache.get_or_load("PubMedBERT", model_dir)
    assert len(cache.fingerprints()) == 2
    assert cache.get("model-0") is None
    assert cache.get("model-1") is not None
    assert cache.get("model-2") is not None
    assert len(calls) == 3


def test_legacy_lexical_search_signature_stays_unchanged(monkeypatch, tmp_path):
    context = _context(tmp_path)
    result = paper_claude.search_local_index("stability", context)
    assert "synthetic.pdf" in result
    assert "Hybrid Local / RRF" not in result
