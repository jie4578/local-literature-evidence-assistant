from __future__ import annotations

import json
import sys
import types

import numpy as np

from retrieval.local_embedding import LocalSentenceTransformerEncoder
from scripts.benchmark_retrieval import CATEGORIES, evaluate_results, load_benchmark, write_result


BENCHMARK = "benchmarks/retrieval_v1.json"


def test_controlled_benchmark_integrity():
    data = load_benchmark(BENCHMARK)
    assert "SYNTHETIC" in data["disclosure"]
    assert len(data["passages"]) == 80
    assert len(data["queries"]) == 60
    assert {item["category"] for item in data["queries"]} == set(CATEGORIES)
    assert all(item["relevant_passage_ids"] for item in data["queries"])


def test_benchmark_metrics_recall_and_mrr():
    data = {
        "queries": [
            {
                "query_id": "q1",
                "category": "exact",
                "relevant_passage_ids": ["p1"],
                "primary_relevant_passage_id": "p1",
            },
            {
                "query_id": "q2",
                "category": "exact",
                "relevant_passage_ids": ["p2"],
                "primary_relevant_passage_id": "p2",
            },
        ]
    }
    ranked = {"q1": ["p1", "other"], "q2": ["other", "p2"]}
    metrics = evaluate_results(data, ranked)["overall"]
    assert metrics["Recall@1"] == 0.5
    assert metrics["Recall@3"] == 1.0
    assert metrics["MRR@10"] == 0.75


def test_e5_prefixes_are_applied_to_queries_and_documents(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    class FakeSentenceTransformer:
        max_seq_length = 128

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            return np.ones((len(texts), 2), dtype=np.float32)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    fake_module.__version__ = "test"
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    encoder = LocalSentenceTransformerEncoder(
        tmp_path,
        query_prefix="query: ",
        document_prefix="passage: ",
    )
    encoder.encode_queries(["alpha"])
    encoder.encode_documents(["beta"])
    assert calls == [["query: alpha"], ["passage: beta"]]
    assert encoder.query_prefix == "query: "
    assert encoder.document_prefix == "passage: "


def test_prefix_configuration_isolated_in_model_fingerprint(monkeypatch, tmp_path):
    class FakeSentenceTransformer:
        max_seq_length = 128

        def __init__(self, *args, **kwargs):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    fake_module.__version__ = "test"
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    plain = LocalSentenceTransformerEncoder(tmp_path)
    e5 = LocalSentenceTransformerEncoder(tmp_path, query_prefix="query: ", document_prefix="passage: ")
    assert plain.model_fingerprint != e5.model_fingerprint


def test_result_serialization_round_trip_has_no_absolute_path(tmp_path):
    result = {
        "benchmark_id": "retrieval_v1",
        "disclosure": "SYNTHETIC / CONTROLLED BENCHMARK",
        "models": {"BGE": {"model_fingerprint": "st_test", "dimension": 384}},
    }
    output = write_result(result, tmp_path / "summary.json")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == result
    assert "C:\\Users\\" not in output.read_text(encoding="utf-8")
