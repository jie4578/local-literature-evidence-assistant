"""Phase 9B passage search 的统一 adapter。"""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from .models import RetrievalResult


class PassageSearchBackend(Protocol):
    def search_passages(
        self,
        query: str,
        limit: int = 10,
        source_files: Iterable[str] | None = None,
        sections: Iterable[str] | None = None,
        exclude_references: bool = True,
    ) -> list[dict[str, Any]]: ...


class LexicalRetriever:
    """适配现有 LocalSearchIndex，不改变其公开 search_passages API。"""

    def __init__(self, backend: PassageSearchBackend):
        self.backend = backend

    def search(
        self,
        query: str,
        limit: int = 10,
        source_files: Iterable[str] | None = None,
        sections: Iterable[str] | None = None,
        exclude_references: bool = True,
    ) -> list[RetrievalResult]:
        rows = self.backend.search_passages(
            query,
            limit=limit,
            source_files=source_files,
            sections=sections,
            exclude_references=exclude_references,
        )
        return [RetrievalResult.from_mapping(row, source="lexical") for row in rows]
