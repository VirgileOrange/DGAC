"""
Search module for FTS5 full-text search with BM25 ranking.

Provides query parsing, search execution, and result models
for the PDF search engine. Includes semantic and hybrid search.
"""

from .models import SearchResult, SearchQuery, SearchStats
from .query_parser import QueryParser
from .bm25_engine import BM25Engine
from .embedding_service import EmbeddingService, get_embedding_service
from .semantic_engine import SemanticEngine, SemanticSearchResult, SemanticSearchStats
from .hybrid_engine import HybridEngine, HybridSearchResult, HybridSearchStats, SearchMode

__all__ = [
    "SearchResult",
    "SearchQuery",
    "SearchStats",
    "QueryParser",
    "BM25Engine",
    "EmbeddingService",
    "get_embedding_service",
    "SemanticEngine",
    "SemanticSearchResult",
    "SemanticSearchStats",
    "HybridEngine",
    "HybridSearchResult",
    "HybridSearchStats",
    "SearchMode"
]
