"""Semantic retriever interface only; no model or network implementation exists here."""

from __future__ import annotations

from typing import Iterable, Protocol

from .models import RetrievalResult


class SemanticRetriever(Protocol):
    def search(
        self,
        query: str,
        limit: int = 10,
        source_files: Iterable[str] | None = None,
        sections: Iterable[str] | None = None,
        exclude_references: bool = True,
    ) -> list[RetrievalResult]: ...


class NullSemanticRetriever:
    """默认空 semantic backend，明确表示未启用语义检索。"""

    def search(
        self,
        query: str,
        limit: int = 10,
        source_files: Iterable[str] | None = None,
        sections: Iterable[str] | None = None,
        exclude_references: bool = True,
    ) -> list[RetrievalResult]:
        return []
