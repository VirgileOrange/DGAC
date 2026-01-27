"""
Search module for FTS5 full-text search with BM25 ranking.

Provides query parsing, search execution, and result models
for the PDF search engine.
"""

from .models import SearchResult, SearchQuery
from .query_parser import QueryParser
from .bm25_engine import BM25Engine

__all__ = [
    "SearchResult",
    "SearchQuery",
    "QueryParser",
    "BM25Engine"
]
