"""可插拔的本地检索抽象；Phase 9C1 默认仍为 lexical-only。"""

from .hybrid import HybridRetriever
from .models import HybridSearchResponse, RetrievalResult
from .semantic import NullSemanticRetriever, SemanticRetriever

__all__ = [
    "HybridRetriever",
    "HybridSearchResponse",
    "NullSemanticRetriever",
    "RetrievalResult",
    "SemanticRetriever",
]
