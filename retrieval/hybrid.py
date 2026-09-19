"""Lexical + semantic 检索编排；Phase 9C1 默认 lexical-only。"""

from __future__ import annotations

from typing import Iterable

from .fusion import reciprocal_rank_fusion
from .lexical import LexicalRetriever
from .models import HybridSearchResponse, RetrievalResult
from .semantic import SemanticRetriever


class HybridRetriever:
    def __init__(
        self,
        lexical_retriever: LexicalRetriever,
        semantic_retriever: SemanticRetriever | None = None,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
        max_candidate_limit: int = 100,
        allowed_passage_ids: Iterable[str] | None = None,
    ):
        if candidate_multiplier < 1 or max_candidate_limit < 1:
            raise ValueError("candidate_multiplier 和 max_candidate_limit 必须为正数")
        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.max_candidate_limit = max_candidate_limit
        self.allowed_passage_ids = set(allowed_passage_ids) if allowed_passage_ids is not None else None

    def search(
        self,
        query: str,
        limit: int = 10,
        source_files: Iterable[str] | None = None,
        sections: Iterable[str] | None = None,
        exclude_references: bool = True,
    ) -> HybridSearchResponse:
        limit = max(1, int(limit))
        candidate_limit = min(max(limit * self.candidate_multiplier, limit), self.max_candidate_limit)
        lexical = self.lexical_retriever.search(
            query,
            limit=candidate_limit,
            source_files=source_files,
            sections=sections,
            exclude_references=exclude_references,
        )
        if self.semantic_retriever is None:
            return HybridSearchResponse(
                results=lexical[:limit],
                mode="lexical",
                semantic_status="disabled",
                warnings=["语义检索后端未启用，当前使用 lexical-only。"],
            )

        try:
            semantic = self.semantic_retriever.search(
                query,
                limit=candidate_limit,
                source_files=source_files,
                sections=sections,
                exclude_references=exclude_references,
            )
        except Exception as exc:  # semantic backend must not break local lexical search
            return HybridSearchResponse(
                results=lexical[:limit],
                mode="lexical",
                semantic_status="error",
                warnings=[f"语义检索不可用，已回退 lexical：{type(exc).__name__}"],
            )

        semantic_status = getattr(self.semantic_retriever, "last_status", "available")
        semantic_warnings = list(getattr(self.semantic_retriever, "last_warnings", []) or [])
        if semantic_status in {"unavailable", "not_indexed", "error"}:
            return HybridSearchResponse(
                results=lexical[:limit],
                mode="lexical",
                semantic_status=semantic_status,
                warnings=semantic_warnings or ["语义检索未能提供可用结果，已回退 lexical。"],
            )

        allowed = self.allowed_passage_ids
        if allowed is None:
            # 未提供当前批次 passage store 时，安全地只接受 lexical 已知 passage。
            allowed = {result.passage_id for result in lexical}
        fused = reciprocal_rank_fusion(
            lexical,
            semantic,
            limit=limit,
            rrf_k=self.rrf_k,
            allowed_passage_ids=allowed,
        )
        return HybridSearchResponse(
            results=fused,
            mode="hybrid",
            semantic_status="available",
            warnings=semantic_warnings,
        )
