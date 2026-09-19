"""Offline benchmark runner for the controlled local retrieval corpus.

The runner never downloads models. Model directories are supplied by the caller
and are loaded through the local-only SentenceTransformer adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_search import LocalSearchIndex, SearchContext
from retrieval import (
    HybridRetriever,
    LocalEmbeddingSemanticRetriever,
    LocalSentenceTransformerEncoder,
    build_semantic_index,
)
from retrieval.lexical import LexicalRetriever


CATEGORIES = ("exact", "paraphrase", "biomedical", "hard_negative")
MODEL_CONFIGS = {
    "bge-small-en-v1.5": {
        "label": "BGE",
        "query_prefix": "",
        "document_prefix": "",
    },
    "e5-small-v2": {
        "label": "E5",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "pubmedbert-base-embeddings": {
        "label": "PubMedBERT",
        "query_prefix": "",
        "document_prefix": "",
    },
}


def load_benchmark(path: str | Path) -> dict[str, Any]:
    benchmark_path = Path(path)
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    disclosure = str(data.get("disclosure", ""))
    if "SYNTHETIC" not in disclosure.upper() or "CONTROLLED" not in disclosure.upper():
        raise ValueError("benchmark 必须明确声明 SYNTHETIC / CONTROLLED")
    passages = data.get("passages")
    queries = data.get("queries")
    if not isinstance(passages, list) or not isinstance(queries, list):
        raise ValueError("benchmark 必须包含 passages 和 queries 数组")
    passage_ids = [item.get("passage_id") for item in passages]
    if len(passage_ids) != len(set(passage_ids)) or not all(isinstance(item, str) and item for item in passage_ids):
        raise ValueError("passage_id 必须唯一且非空")
    passage_id_set = set(passage_ids)
    query_ids = [item.get("query_id") for item in queries]
    if len(query_ids) != len(set(query_ids)) or not all(isinstance(item, str) and item for item in query_ids):
        raise ValueError("query_id 必须唯一且非空")
    suspicious = re.compile(
        r"(?i)(?:[A-Z]:[\\/]|"
        + chr(47)
        + "Users"
        + chr(47)
        + "|"
        + chr(47)
        + "home"
        + chr(47)
        + "|sk-"
        + r"[A-Za-z0-9]{20,})"
    )
    for passage in passages:
        if not isinstance(passage.get("text"), str) or not passage["text"].strip():
            raise ValueError("每个 passage 必须有非空 text")
        if suspicious.search(passage["text"]):
            raise ValueError("benchmark passage 包含绝对路径或疑似密钥")
    for query in queries:
        if query.get("category") not in CATEGORIES:
            raise ValueError(f"非法 query category：{query.get('category')}")
        relevant = query.get("relevant_passage_ids")
        if not isinstance(relevant, list) or not relevant or not set(relevant).issubset(passage_id_set):
            raise ValueError(f"query {query.get('query_id')} 的 relevant_passage_ids 无效")
        if query.get("primary_relevant_passage_id") not in relevant:
            raise ValueError(f"query {query.get('query_id')} 缺少有效 primary relevant passage")
    return data


def make_passages(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "passage_id": item["passage_id"],
            "source_file": "synthetic_benchmark.pdf",
            "document_id": "synthetic-retrieval-v1",
            "section": item.get("section", "Results"),
            "text": item["text"],
            "pdf_page_start": index + 1,
            "pdf_page_end": index + 1,
            "ordinal": index,
        }
        for index, item in enumerate(data["passages"])
    ]


def metric_values(ranked_ids: list[str], relevant_ids: set[str], k: int) -> tuple[float, float, float]:
    hit_rank = next((rank for rank, passage_id in enumerate(ranked_ids[:k], 1) if passage_id in relevant_ids), None)
    recall = 1.0 if hit_rank is not None else 0.0
    reciprocal = 1.0 / hit_rank if hit_rank is not None else 0.0
    gains = [1.0 if passage_id in relevant_ids else 0.0 for passage_id in ranked_ids[:k]]
    dcg = sum(gain / __import__("math").log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / __import__("math").log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return recall, reciprocal, ndcg


def evaluate_results(data: dict[str, Any], ranked: dict[str, list[str]]) -> dict[str, Any]:
    queries = data["queries"]
    metrics: dict[str, float] = {}
    all_values = [
        metric_values(ranked[item["query_id"]], set(item["relevant_passage_ids"]), 10)
        for item in queries
    ]
    recall1 = [metric_values(ranked[item["query_id"]], set(item["relevant_passage_ids"]), 1)[0] for item in queries]
    recall3 = [metric_values(ranked[item["query_id"]], set(item["relevant_passage_ids"]), 3)[0] for item in queries]
    recall5 = [metric_values(ranked[item["query_id"]], set(item["relevant_passage_ids"]), 5)[0] for item in queries]
    metrics["Recall@1"] = _mean(recall1)
    metrics["Recall@3"] = _mean(recall3)
    metrics["Recall@5"] = _mean(recall5)
    metrics["MRR@10"] = _mean([item[1] for item in all_values])
    metrics["nDCG@10"] = _mean([item[2] for item in all_values])
    return {"overall": metrics, "by_category": _category_metrics(data, ranked)}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else 0.0


def _category_metrics(data: dict[str, Any], ranked: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        category_queries = [item for item in data["queries"] if item["category"] == category]
        result[category] = {
            "Recall@1": _mean([
                metric_values(ranked[item["query_id"]], set(item["relevant_passage_ids"]), 1)[0]
                for item in category_queries
            ]),
            "Recall@3": _mean([
                metric_values(ranked[item["query_id"]], set(item["relevant_passage_ids"]), 3)[0]
                for item in category_queries
            ]),
            "Recall@5": _mean([
                metric_values(ranked[item["query_id"]], set(item["relevant_passage_ids"]), 5)[0]
                for item in category_queries
            ]),
            "MRR@10": _mean([
                metric_values(ranked[item["query_id"]], set(item["relevant_passage_ids"]), 10)[1]
                for item in category_queries
            ]),
            "nDCG@10": _mean([
                metric_values(ranked[item["query_id"]], set(item["relevant_passage_ids"]), 10)[2]
                for item in category_queries
            ]),
        }
    return result


def _ranked_ids(results: Iterable[Any]) -> list[str]:
    return [result.passage_id for result in results]


def _run_queries(queries: list[dict[str, Any]], search: Callable[[str], Iterable[Any]]) -> tuple[dict[str, list[str]], dict[str, float]]:
    ranked: dict[str, list[str]] = {}
    timings: list[float] = []
    for query in queries:
        start = time.perf_counter()
        ranked[query["query_id"]] = _ranked_ids(search(query["query"]))
        timings.append(time.perf_counter() - start)
    cold_count = min(5, len(timings))
    return ranked, {
        "cold_query_median_ms": round(median(timings[:cold_count]) * 1000, 3) if timings else 0.0,
        "warm_query_median_ms": round(median(timings[cold_count:]) * 1000, 3) if len(timings) > cold_count else 0.0,
    }


def _model_size_bytes(model_dir: Path) -> int:
    return sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())


def _count_truncated(encoder: LocalSentenceTransformerEncoder, passages: list[dict[str, Any]]) -> int:
    max_length = encoder.max_seq_length
    tokenizer = getattr(getattr(encoder, "_model", None), "tokenizer", None)
    if not max_length or tokenizer is None:
        return 0
    count = 0
    for passage in passages:
        encoded = tokenizer(
            f"{encoder.document_prefix}{passage['text']}",
            add_special_tokens=True,
            truncation=False,
            max_length=None,
        )
        if len(encoded["input_ids"]) > int(max_length):
            count += 1
    return count


def _make_context(root: Path, name: str, passages: list[dict[str, Any]]) -> SearchContext:
    task_dir = root / name
    task_dir.mkdir(parents=True, exist_ok=True)
    db_path = task_dir / "local_index.sqlite"
    with LocalSearchIndex(db_path) as index:
        index.index_passages(passages)
    return SearchContext(
        batch_id=name,
        task_dir=str(task_dir),
        db_path=str(db_path),
        document_sources=tuple(sorted({item["source_file"] for item in passages})),
        document_count=len({item["source_file"] for item in passages}),
    )


def _run_model(data: dict[str, Any], root: Path, model_name: str, model_dir: Path) -> dict[str, Any]:
    config = MODEL_CONFIGS[model_name]
    passages = make_passages(data)
    context = _make_context(root, model_name, passages)
    load_start = time.perf_counter()
    encoder = LocalSentenceTransformerEncoder(
        model_dir,
        device="cpu",
        query_prefix=config["query_prefix"],
        document_prefix=config["document_prefix"],
    )
    load_time = time.perf_counter() - load_start
    build_start = time.perf_counter()
    first_cache = build_semantic_index(context, encoder)
    build_time = time.perf_counter() - build_start
    reuse_start = time.perf_counter()
    second_cache = build_semantic_index(context, encoder)
    reuse_time = time.perf_counter() - reuse_start
    if first_cache["failed"] or second_cache["failed"]:
        raise RuntimeError(f"{model_name} semantic index failed")
    queries = data["queries"]
    with LocalSearchIndex(context.db_path) as index:
        lexical_retriever = LexicalRetriever(index)
        semantic_retriever = LocalEmbeddingSemanticRetriever(context, encoder)
        semantic_ranked, semantic_latency = _run_queries(
            queries,
            lambda query: semantic_retriever.search(query, limit=10),
        )
        hybrid_retriever = HybridRetriever(
            lexical_retriever,
            semantic_retriever,
            allowed_passage_ids={passage["passage_id"] for passage in passages},
        )
        hybrid_ranked, hybrid_latency = _run_queries(
            queries,
            lambda query: hybrid_retriever.search(query, limit=10).results,
        )
    isolated = _make_context(
        root,
        f"{model_name}_isolated",
        [
            {
                **passages[0],
                "passage_id": "ISOLATED-1",
                "source_file": "isolated.pdf",
                "document_id": "isolated",
                "text": "Isolated batch thermal control passage.",
            }
        ],
    )
    build_semantic_index(isolated, encoder)
    isolated_results = LocalEmbeddingSemanticRetriever(isolated, encoder).search("thermal protein stability")
    batch_isolation_pass = bool(isolated_results) and all(item.source_file == "isolated.pdf" for item in isolated_results)
    base = {
        "model_alias": model_name,
        "label": config["label"],
        "dimension": encoder.dimension,
        "max_seq_length": encoder.max_seq_length,
        "model_fingerprint": encoder.model_fingerprint,
        "query_prefix": encoder.query_prefix,
        "document_prefix": encoder.document_prefix,
        "model_size_bytes": _model_size_bytes(model_dir),
        "truncated_passages_count": _count_truncated(encoder, passages),
        "passage_count": len(passages),
        "query_count": len(queries),
        "model_load_time_s": round(load_time, 6),
        "embedding_index_build_time_s": round(build_time, 6),
        "cache_reuse_time_s": round(reuse_time, 6),
        "cache_first": {key: value for key, value in first_cache.items() if key != "warnings"},
        "cache_second": {key: value for key, value in second_cache.items() if key != "warnings"},
        "batch_isolation_pass": batch_isolation_pass,
        "semantic_latency": semantic_latency,
        "hybrid_latency": hybrid_latency,
        "semantic_metrics": evaluate_results(data, semantic_ranked),
        "hybrid_metrics": evaluate_results(data, hybrid_ranked),
    }
    if not batch_isolation_pass or not semantic_ranked or not hybrid_ranked:
        raise RuntimeError(f"{model_name} runtime validation failed")
    return base


def _run_lexical(data: dict[str, Any], root: Path) -> dict[str, Any]:
    passages = make_passages(data)
    context = _make_context(root, "lexical", passages)
    with LocalSearchIndex(context.db_path) as index:
        ranked, latency = _run_queries(data["queries"], lambda query: LexicalRetriever(index).search(query, limit=10))
    metrics = evaluate_results(data, ranked)
    return {
        "label": "FTS5",
        "passage_count": len(passages),
        "query_count": len(data["queries"]),
        "dimension": None,
        "model_load_time_s": 0.0,
        "embedding_index_build_time_s": 0.0,
        "cache_reuse_time_s": 0.0,
        "latency": latency,
        "metrics": metrics,
    }


def run_benchmark(
    benchmark_path: str | Path,
    models_root: str | Path,
    output_path: str | Path,
    model_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    data = load_benchmark(benchmark_path)
    names = list(model_names or MODEL_CONFIGS)
    unknown = [name for name in names if name not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"未知模型别名：{unknown}")
    root = Path(models_root)
    with tempfile.TemporaryDirectory(prefix="retrieval_v1_benchmark_") as temp:
        temp_root = Path(temp)
        modes: dict[str, Any] = {"FTS5": _run_lexical(data, temp_root)}
        model_results: dict[str, Any] = {}
        for name in names:
            model_results[name] = _run_model(data, temp_root, name, root / name)
            modes[f"{MODEL_CONFIGS[name]['label']} semantic"] = {
                **model_results[name]["semantic_metrics"],
                "latency": model_results[name]["semantic_latency"],
                "dimension": model_results[name]["dimension"],
            }
            modes[f"{MODEL_CONFIGS[name]['label']} hybrid"] = {
                **model_results[name]["hybrid_metrics"],
                "latency": model_results[name]["hybrid_latency"],
                "dimension": model_results[name]["dimension"],
            }
    result = {
        "benchmark_id": data["benchmark_id"],
        "disclosure": data["disclosure"],
        "benchmark": {
            "passages": len(data["passages"]),
            "queries": len(data["queries"]),
            "category_counts": {category: sum(item["category"] == category for item in data["queries"]) for category in CATEGORIES},
            "device": "cpu",
            "network_during_benchmark": False,
        },
        "modes": modes,
        "models": model_results,
    }
    write_result(result, output_path)
    return result


def write_result(result: dict[str, Any], output_path: str | Path) -> Path:
    """序列化只含汇总指标的结果，不写入模型路径或原始缓存。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline synthetic retrieval benchmark.")
    parser.add_argument("--benchmark", default="benchmarks/retrieval_v1.json")
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--model-name", choices=[*MODEL_CONFIGS, "all"], default="all")
    parser.add_argument("--output", default="benchmarks/results/retrieval_v1_summary.json")
    args = parser.parse_args()
    names = list(MODEL_CONFIGS) if args.model_name == "all" else [args.model_name]
    result = run_benchmark(args.benchmark, args.models_root, args.output, names)
    print(f"BENCHMARK_PASS passages={result['benchmark']['passages']} queries={result['benchmark']['queries']}")
    for name, model in result["models"].items():
        print(f"MODEL_PASS {name} dimension={model['dimension']} fingerprint={model['model_fingerprint']}")
    print(f"RESULT_WRITTEN {Path(args.output).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
