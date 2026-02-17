"""
Hybrid search engine combining lexical and semantic search.

Provides unified search interface with three modes (lexical, semantic, hybrid)
and Reciprocal Rank Fusion for combining results.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..core import get_config, get_logger
from .bm25_engine import BM25Engine
from .models import SearchQuery
from .semantic_engine import SemanticEngine

logger = get_logger(__name__)


class SearchMode(Enum):
    """Available search modes."""
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class HybridSearchResult:
    """
    Unified search result from hybrid search.

    Attributes:
        document_id: Document database ID.
        filepath: Path to the source PDF file.
        filename: PDF filename.
        page_num: Source page number.
        relative_path: Path relative to data directory.
        snippet: Text excerpt with context.
        score: Combined/normalized score.
        source: Which search method(s) found this result.
        lexical_rank: Rank in lexical results (None if not found).
        semantic_rank: Rank in semantic results (None if not found).
        similarity: Semantic similarity if available.
    """
    document_id: int
    filepath: str
    filename: str
    page_num: int
    relative_path: str
    snippet: str
    score: float
    source: str
    lexical_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    similarity: Optional[float] = None


@dataclass
class HybridSearchStats:
    """
    Statistics from hybrid search execution.

    Attributes:
        query: Original query text.
        mode: Search mode used.
        total_results: Number of results returned.
        execution_time_ms: Total execution time.
        lexical_results: Count from lexical search.
        semantic_results: Count from semantic search.
        overlap_count: Results found by both methods.
        lexical_time_ms: Time for lexical search.
        semantic_time_ms: Time for semantic search.
        errors: List of any errors encountered.
    """
    query: str
    mode: str
    total_results: int
    execution_time_ms: float
    lexical_results: int = 0
    semantic_results: int = 0
    overlap_count: int = 0
    lexical_time_ms: float = 0.0
    semantic_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class HybridEngine:
    """
    Unified search engine supporting lexical, semantic, and hybrid modes.

    Routes queries to appropriate engine(s) and applies RRF fusion
    for hybrid searches.
    """

    def __init__(self):
        """Initialize the hybrid search engine."""
        self.config = get_config()
        self.bm25_engine = BM25Engine()
        self.semantic_engine = SemanticEngine()

        self.rrf_k = self.config.hybrid.rrf_k
        self.default_mode = SearchMode(self.config.hybrid.default_mode)
        self.default_lexical_weight = self.config.hybrid.default_lexical_weight
        self.default_semantic_weight = self.config.hybrid.default_semantic_weight

    def search(
        self,
        query: str,
        mode: SearchMode = None,
        limit: int = 50,
        lexical_weight: float = None,
        semantic_weight: float = None
    ) -> Tuple[List[HybridSearchResult], HybridSearchStats]:
        """
        Execute a search with the specified mode.

        Args:
            query: Search query text.
            mode: Search mode (defaults to config default).
            limit: Maximum number of results.
            lexical_weight: Weight for lexical results in RRF.
            semantic_weight: Weight for semantic results in RRF.

        Returns:
            Tuple of (list of HybridSearchResult, HybridSearchStats).
        """
        start_time = time.time()

        mode = mode or self.default_mode
        lexical_weight = lexical_weight or self.default_lexical_weight
        semantic_weight = semantic_weight or self.default_semantic_weight

        if not query or not query.strip():
            return [], HybridSearchStats(
                query=query,
                mode=mode.value,
                total_results=0,
                execution_time_ms=0
            )

        stats = HybridSearchStats(query=query, mode=mode.value, total_results=0, execution_time_ms=0)

        if mode == SearchMode.LEXICAL:
            results = self._search_lexical(query, limit, stats)

        elif mode == SearchMode.SEMANTIC:
            results = self._search_semantic(query, limit, stats)

        else:
            results = self._search_hybrid(
                query, limit, lexical_weight, semantic_weight, stats
            )

        stats.total_results = len(results)
        stats.execution_time_ms = round((time.time() - start_time) * 1000, 2)

        logger.debug(
            f"Hybrid search ({mode.value}) '{query}': {len(results)} results "
            f"in {stats.execution_time_ms:.1f}ms"
        )

        return results, stats

    def _search_lexical(
        self,
        query: str,
        limit: int,
        stats: HybridSearchStats
    ) -> List[HybridSearchResult]:
        """Execute lexical-only search."""
        search_start = time.time()

        try:
            search_query = SearchQuery(text=query, limit=limit)
            bm25_results, bm25_stats = self.bm25_engine.search(search_query)
            stats.lexical_results = len(bm25_results)
            stats.lexical_time_ms = round((time.time() - search_start) * 1000, 2)

            results = []
            for rank, r in enumerate(bm25_results, 1):
                results.append(HybridSearchResult(
                    document_id=r.id,
                    filepath=r.filepath,
                    filename=r.filename,
                    page_num=r.page_num,
                    relative_path=r.relative_path,
                    snippet=r.snippet,
                    score=r.display_score,
                    source="lexical",
                    lexical_rank=rank
                ))

            return results

        except Exception as e:
            logger.error(f"Lexical search failed: {e}")
            stats.errors.append(f"Lexical search error: {str(e)}")
            return []

    def _search_semantic(
        self,
        query: str,
        limit: int,
        stats: HybridSearchStats
    ) -> List[HybridSearchResult]:
        """Execute semantic-only search."""
        search_start = time.time()

        try:
            semantic_results, semantic_stats = self.semantic_engine.search(query, limit)
            stats.semantic_results = len(semantic_results)
            stats.semantic_time_ms = round((time.time() - search_start) * 1000, 2)

            results = []
            for rank, r in enumerate(semantic_results, 1):
                results.append(HybridSearchResult(
                    document_id=r.document_id,
                    filepath=r.filepath,
                    filename=r.filename,
                    page_num=r.page_num,
                    relative_path=r.relative_path,
                    snippet=r.snippet,
                    score=r.similarity,
                    source="semantic",
                    semantic_rank=rank,
                    similarity=r.similarity
                ))

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            stats.errors.append(f"Semantic search error: {str(e)}")
            return []

    def _search_hybrid(
        self,
        query: str,
        limit: int,
        lexical_weight: float,
        semantic_weight: float,
        stats: HybridSearchStats
    ) -> List[HybridSearchResult]:
        """Execute hybrid search with RRF fusion."""
        fetch_limit = limit * 2

        lexical_start = time.time()
        lexical_results = []
        try:
            search_query = SearchQuery(text=query, limit=fetch_limit)
            bm25_results, _ = self.bm25_engine.search(search_query)
            lexical_results = bm25_results
            stats.lexical_results = len(lexical_results)
        except Exception as e:
            logger.warning(f"Lexical search failed in hybrid mode: {e}")
            stats.errors.append(f"Lexical search error: {str(e)}")
        stats.lexical_time_ms = round((time.time() - lexical_start) * 1000, 2)

        semantic_start = time.time()
        semantic_results = []
        try:
            sem_results, _ = self.semantic_engine.search(query, fetch_limit)
            semantic_results = sem_results
            stats.semantic_results = len(semantic_results)
        except Exception as e:
            logger.warning(f"Semantic search failed in hybrid mode: {e}")
            stats.errors.append(f"Semantic search error: {str(e)}")
        stats.semantic_time_ms = round((time.time() - semantic_start) * 1000, 2)

        if not lexical_results and not semantic_results:
            return []

        if not semantic_results:
            return self._convert_lexical_results(lexical_results[:limit])

        if not lexical_results:
            return self._convert_semantic_results(semantic_results[:limit])

        fused_results = self._apply_rrf_fusion(
            lexical_results,
            semantic_results,
            lexical_weight,
            semantic_weight,
            limit,
            stats
        )

        return fused_results

    def _apply_rrf_fusion(
        self,
        lexical_results: list,
        semantic_results: list,
        lexical_weight: float,
        semantic_weight: float,
        limit: int,
        stats: HybridSearchStats
    ) -> List[HybridSearchResult]:
        """
        Apply Reciprocal Rank Fusion to combine result sets.

        RRF score = sum(weight * 1/(k + rank)) for each result set.
        """
        scores: Dict[str, float] = {}
        result_data: Dict[str, dict] = {}
        lexical_ranks: Dict[str, int] = {}
        semantic_ranks: Dict[str, int] = {}
        semantic_similarities: Dict[str, float] = {}

        for rank, r in enumerate(lexical_results, 1):
            key = f"{r.filepath}:{r.page_num}"
            rrf_score = lexical_weight * (1.0 / (self.rrf_k + rank))
            scores[key] = scores.get(key, 0) + rrf_score
            lexical_ranks[key] = rank

            if key not in result_data:
                result_data[key] = {
                    "document_id": r.id,
                    "filepath": r.filepath,
                    "filename": r.filename,
                    "page_num": r.page_num,
                    "relative_path": r.relative_path,
                    "snippet": r.snippet
                }

        for rank, r in enumerate(semantic_results, 1):
            key = f"{r.filepath}:{r.page_num}"
            rrf_score = semantic_weight * (1.0 / (self.rrf_k + rank))
            scores[key] = scores.get(key, 0) + rrf_score
            semantic_ranks[key] = rank
            semantic_similarities[key] = r.similarity

            if key not in result_data:
                result_data[key] = {
                    "document_id": r.document_id,
                    "filepath": r.filepath,
                    "filename": r.filename,
                    "page_num": r.page_num,
                    "relative_path": r.relative_path,
                    "snippet": r.snippet
                }

        lexical_keys = set(lexical_ranks.keys())
        semantic_keys = set(semantic_ranks.keys())
        overlap_keys = lexical_keys & semantic_keys
        stats.overlap_count = len(overlap_keys)

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        results = []
        for key in sorted_keys[:limit]:
            data = result_data[key]
            lex_rank = lexical_ranks.get(key)
            sem_rank = semantic_ranks.get(key)

            if lex_rank and sem_rank:
                source = "both"
            elif lex_rank:
                source = "lexical"
            else:
                source = "semantic"

            results.append(HybridSearchResult(
                document_id=data["document_id"],
                filepath=data["filepath"],
                filename=data["filename"],
                page_num=data["page_num"],
                relative_path=data["relative_path"],
                snippet=data["snippet"],
                score=scores[key],
                source=source,
                lexical_rank=lex_rank,
                semantic_rank=sem_rank,
                similarity=semantic_similarities.get(key)
            ))

        return results

    def _convert_lexical_results(self, results: list) -> List[HybridSearchResult]:
        """Convert BM25 results to HybridSearchResult format."""
        return [
            HybridSearchResult(
                document_id=r.id,
                filepath=r.filepath,
                filename=r.filename,
                page_num=r.page_num,
                relative_path=r.relative_path,
                snippet=r.snippet,
                score=r.display_score,
                source="lexical",
                lexical_rank=rank
            )
            for rank, r in enumerate(results, 1)
        ]

    def _convert_semantic_results(self, results: list) -> List[HybridSearchResult]:
        """Convert semantic results to HybridSearchResult format."""
        return [
            HybridSearchResult(
                document_id=r.document_id,
                filepath=r.filepath,
                filename=r.filename,
                page_num=r.page_num,
                relative_path=r.relative_path,
                snippet=r.snippet,
                score=r.similarity,
                source="semantic",
                semantic_rank=rank,
                similarity=r.similarity
            )
            for rank, r in enumerate(results, 1)
        ]

    def get_available_modes(self) -> List[str]:
        """
        Get list of available search modes.

        Returns:
            List of mode names.
        """
        return [mode.value for mode in SearchMode]


if __name__ == "__main__":
    engine = HybridEngine()

    print("Hybrid Engine Configuration:")
    print(f"  Default mode: {engine.default_mode.value}")
    print(f"  RRF k: {engine.rrf_k}")
    print(f"  Lexical weight: {engine.default_lexical_weight}")
    print(f"  Semantic weight: {engine.default_semantic_weight}")
    print(f"\nAvailable modes: {engine.get_available_modes()}")
