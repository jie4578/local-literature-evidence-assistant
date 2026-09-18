"""检索结果的统一数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class RetrievalResult:
    passage_id: str
    source_file: str
    document_id: str | None
    section: str
    pdf_page_start: int
    pdf_page_end: int
    text: str
    rank: int = 0
    retrieval_sources: list[str] = field(default_factory=list)
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    fused_score: float | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, source: str) -> "RetrievalResult":
        sources = list(value.get("retrieval_sources") or [source])
        if source not in sources:
            sources.append(source)
        lexical_rank = value.get("lexical_rank")
        semantic_rank = value.get("semantic_rank")
        if source == "lexical" and lexical_rank is None:
            lexical_rank = value.get("rank")
        if source == "semantic" and semantic_rank is None:
            semantic_rank = value.get("rank")
        return cls(
            passage_id=str(value["passage_id"]),
            source_file=str(value.get("source_file", "")),
            document_id=value.get("document_id"),
            section=str(value.get("section", "Unknown")),
            pdf_page_start=int(value["pdf_page_start"]),
            pdf_page_end=int(value["pdf_page_end"]),
            text=str(value["text"]),
            rank=int(value.get("rank", 0) or 0),
            retrieval_sources=sources,
            lexical_rank=lexical_rank,
            semantic_rank=semantic_rank,
            fused_score=value.get("fused_score"),
            lexical_score=value.get("lexical_score"),
            semantic_score=value.get("semantic_score"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridSearchResponse:
    results: list[RetrievalResult]
    mode: str
    semantic_status: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "mode": self.mode,
            "semantic_status": self.semantic_status,
            "warnings": list(self.warnings),
        }
