"""可插拔的本地检索抽象；Phase 9C1 默认仍为 lexical-only。"""

from .hybrid import HybridRetriever
from .local_embedding import (
    EmbeddingEncoder,
    LocalEmbeddingSemanticRetriever,
    LocalModelPathError,
    LocalSentenceTransformerEncoder,
    OptionalSemanticDependencyError,
    build_semantic_index,
)
from .models import HybridSearchResponse, RetrievalResult
from .semantic import NullSemanticRetriever, SemanticRetriever

__all__ = [
    "HybridRetriever",
    "HybridSearchResponse",
    "EmbeddingEncoder",
    "LocalEmbeddingSemanticRetriever",
    "LocalModelPathError",
    "LocalSentenceTransformerEncoder",
    "NullSemanticRetriever",
    "OptionalSemanticDependencyError",
    "RetrievalResult",
    "SemanticRetriever",
    "build_semantic_index",
]
