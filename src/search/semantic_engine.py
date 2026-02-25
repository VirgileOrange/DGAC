"""
Semantic search engine using vector similarity.

Orchestrates embedding generation and vector search to find
semantically similar documents.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..core import get_config, get_logger
from ..database import get_connection
from ..database.vector_repository import VectorRepository
from .embedding_service import get_embedding_service

logger = get_logger(__name__)


@dataclass
class SemanticSearchResult:
    """
    Result from semantic search.

    Attributes:
        chunk_id: Unique chunk identifier.
        document_id: Parent document ID.
        filepath: Path to the source PDF file.
        filename: PDF filename.
        page_num: Source page number.
        relative_path: Path relative to data directory.
        snippet: Text excerpt from the chunk.
        similarity: Cosine similarity score (0-1, higher is better).
    """
    chunk_id: str
    document_id: int
    filepath: str
    filename: str
    page_num: int
    relative_path: str
    snippet: str
    similarity: float


@dataclass
class SemanticSearchStats:
    """
    Statistics from semantic search execution.

    Attributes:
        query: Original query text.
        total_results: Number of results returned.
        execution_time_ms: Total execution time in milliseconds.
        embedding_time_ms: Time spent generating query embedding.
        search_time_ms: Time spent in vector search.
    """
    query: str
    total_results: int
    execution_time_ms: float
    embedding_time_ms: float
    search_time_ms: float


class SemanticEngine:
    """
    Semantic search engine using vector embeddings.

    Combines EmbeddingService and VectorRepository to perform
    meaning-based search across indexed documents.
    """

    def __init__(self):
        """Initialize the semantic search engine."""
        self.config = get_config()
        self.embedding_service = get_embedding_service()
        self.vector_repo = VectorRepository()
        self.snippet_length = self.config.search.snippet_length
        self._document_cache: Dict[int, dict] = {}

    def search(
        self,
        query: str,
        limit: int = 50,
        min_similarity: float = 0.0
    ) -> Tuple[List[SemanticSearchResult], SemanticSearchStats]:
        """
        Perform semantic search for a query.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            Tuple of (list of SemanticSearchResult, SemanticSearchStats).
        """
        start_time = time.time()

        if not query or not query.strip():
            return [], SemanticSearchStats(
                query=query,
                total_results=0,
                execution_time_ms=0,
                embedding_time_ms=0,
                search_time_ms=0
            )

        embed_start = time.time()
        query_embedding = self.embedding_service.embed_query(query)
        embedding_time = (time.time() - embed_start) * 1000

        search_start = time.time()
        vector_results = self.vector_repo.search_similar(query_embedding, limit)
        search_time = (time.time() - search_start) * 1000

        results = []
        for vr in vector_results:
            if vr.similarity < min_similarity:
                continue

            doc_info = self._get_document_info(vr.document_id)
            if not doc_info:
                continue

            snippet = self._generate_snippet(vr.content)

            results.append(SemanticSearchResult(
                chunk_id=vr.chunk_id,
                document_id=vr.document_id,
                filepath=doc_info["filepath"],
                filename=doc_info["filename"],
                page_num=vr.page_num,
                relative_path=doc_info["relative_path"],
                snippet=snippet,
                similarity=vr.similarity
            ))

        total_time = (time.time() - start_time) * 1000

        stats = SemanticSearchStats(
            query=query,
            total_results=len(results),
            execution_time_ms=round(total_time, 2),
            embedding_time_ms=round(embedding_time, 2),
            search_time_ms=round(search_time, 2)
        )

        logger.debug(
            f"Semantic search '{query}': {len(results)} results in {total_time:.1f}ms "
            f"(embed: {embedding_time:.1f}ms, search: {search_time:.1f}ms)"
        )

        return results, stats

    def _get_document_info(self, document_id: int) -> Optional[dict]:
        """
        Get document metadata, using cache for efficiency.

        Args:
            document_id: Document ID to look up.

        Returns:
            Dictionary with filepath, filename, relative_path or None.
        """
        if document_id in self._document_cache:
            return self._document_cache[document_id]

        with get_connection() as conn:
            row = conn.execute("""
                SELECT filepath, filename, relative_path
                FROM documents
                WHERE id = ?
                LIMIT 1
            """, (document_id,)).fetchone()

            if not row:
                return None

            doc_info = {
                "filepath": row["filepath"],
                "filename": row["filename"],
                "relative_path": row["relative_path"]
            }

            self._document_cache[document_id] = doc_info
            return doc_info

    def _generate_snippet(self, content: str) -> str:
        """
        Generate a display snippet from chunk content.

        Args:
            content: Full chunk text.

        Returns:
            Truncated snippet suitable for display.
        """
        if len(content) <= self.snippet_length:
            return content

        truncated = content[:self.snippet_length]
        last_space = truncated.rfind(" ")

        if last_space > self.snippet_length * 0.7:
            truncated = truncated[:last_space]

        return truncated + "..."

    def get_index_stats(self) -> dict:
        """
        Get statistics about the semantic index.

        Returns:
            Dictionary with index statistics.
        """
        chunk_count = self.vector_repo.get_chunk_count()
        model_info = self.embedding_service.get_model_info()

        return {
            "total_chunks": chunk_count,
            "embedding_model": model_info["model"],
            "embedding_dimensions": model_info["dimensions"],
            "index_ready": chunk_count > 0
        }

    def clear_cache(self) -> None:
        """Clear the document info cache."""
        self._document_cache.clear()


if __name__ == "__main__":
    engine = SemanticEngine()

    print("Semantic Engine Index Stats:")
    stats = engine.get_index_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nNote: Actual search requires indexed documents and valid API credentials.")
