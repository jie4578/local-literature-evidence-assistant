"""确定性 Reciprocal Rank Fusion。"""

from __future__ import annotations

from math import inf
from typing import Iterable

from .models import RetrievalResult


def reciprocal_rank_fusion(
    lexical_results: Iterable[RetrievalResult],
    semantic_results: Iterable[RetrievalResult],
    *,
    limit: int = 10,
    rrf_k: int = 60,
    lexical_weight: float = 1.0,
    semantic_weight: float = 1.0,
    allowed_passage_ids: set[str] | None = None,
) -> list[RetrievalResult]:
    """按 rank 融合，绝不直接相加 BM25 与 semantic 原始分数。"""
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须为正数")
    grouped: dict[str, RetrievalResult] = {}
    fused_scores: dict[str, float] = {}
    for source, values, weight in (
        ("lexical", lexical_results, lexical_weight),
        ("semantic", semantic_results, semantic_weight),
    ):
        for result in values:
            if allowed_passage_ids is not None and result.passage_id not in allowed_passage_ids:
                continue
            if not result.passage_id or result.rank <= 0:
                continue
            current = grouped.get(result.passage_id)
            if current is None:
                current = RetrievalResult(
                    passage_id=result.passage_id,
                    source_file=result.source_file,
                    document_id=result.document_id,
                    section=result.section,
                    pdf_page_start=result.pdf_page_start,
                    pdf_page_end=result.pdf_page_end,
                    text=result.text,
                )
                grouped[result.passage_id] = current
            current.retrieval_sources = sorted(set(current.retrieval_sources).union({source}))
            if source == "lexical":
                current.lexical_rank = result.rank
                current.lexical_score = result.lexical_score
            else:
                current.semantic_rank = result.rank
                current.semantic_score = result.semantic_score
            fused_scores[result.passage_id] = fused_scores.get(result.passage_id, 0.0) + weight / (rrf_k + result.rank)

    def sort_key(item: RetrievalResult) -> tuple[float, float, float, float, str]:
        lexical_rank = item.lexical_rank if item.lexical_rank is not None else inf
        semantic_rank = item.semantic_rank if item.semantic_rank is not None else inf
        best_rank = min(lexical_rank, semantic_rank)
        return (-fused_scores[item.passage_id], best_rank, lexical_rank, semantic_rank, item.passage_id)

    ordered = sorted(grouped.values(), key=sort_key)
    results: list[RetrievalResult] = []
    for rank, result in enumerate(ordered[: max(1, int(limit))], 1):
        result.rank = rank
        result.fused_score = fused_scores[result.passage_id]
        results.append(result)
    return results
