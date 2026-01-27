"""
BM25 search engine using SQLite FTS5.

Executes full-text searches with BM25 ranking, generates snippets
with highlighted matches, and handles pagination.
"""

import time
from typing import List, Optional, Tuple

from ..core import get_config, get_logger, SearchError
from ..database import get_connection
from .models import SearchQuery, SearchResult, SearchStats
from .query_parser import QueryParser

logger = get_logger(__name__)


class BM25Engine:
    """
    Full-text search engine using SQLite FTS5 with BM25 ranking.

    Provides search execution with relevance ranking, snippet generation,
    and pagination support.
    """

    def __init__(self):
        """Initialize the search engine with configuration."""
        self.config = get_config()
        self.parser = QueryParser()

        self.filename_weight = self.config.search.bm25_weights.filename
        self.content_weight = self.config.search.bm25_weights.content
        self.snippet_length = self.config.search.snippet_length
        self.default_limit = self.config.search.default_limit
        self.max_limit = self.config.search.max_limit

    def search(
        self,
        query: SearchQuery,
        include_content: bool = False
    ) -> Tuple[List[SearchResult], SearchStats]:
        """
        Execute a full-text search.

        Args:
            query: SearchQuery object with text, limit, and offset.
            include_content: Whether to include full page content in results.

        Returns:
            Tuple of (list of SearchResult, SearchStats).

        Raises:
            SearchError: If query execution fails.
        """
        start_time = time.time()

        if query.advanced:
            parsed_query = self.parser.parse_advanced(query.text)
        else:
            parsed_query = self.parser.parse(query.text)

        if not parsed_query:
            return [], SearchStats(
                query=query.text,
                total_results=0,
                execution_time_ms=0
            )

        limit = min(query.limit or self.default_limit, self.max_limit)

        try:
            results = self._execute_search(
                parsed_query,
                limit,
                query.offset,
                include_content
            )

            total_count = self._count_results(parsed_query)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(
                f"Search execution failed: {e}",
                query=query.text
            )

        execution_time = (time.time() - start_time) * 1000

        total_pages = (total_count + limit - 1) // limit if limit > 0 else 1
        current_page = (query.offset // limit) + 1 if limit > 0 else 1

        stats = SearchStats(
            query=query.text,
            total_results=total_count,
            execution_time_ms=round(execution_time, 2),
            page=current_page,
            total_pages=total_pages
        )

        logger.debug(
            f"Search '{query.text}': {total_count} results in {execution_time:.1f}ms"
        )

        return results, stats

    def _execute_search(
        self,
        query: str,
        limit: int,
        offset: int,
        include_content: bool
    ) -> List[SearchResult]:
        """Execute the FTS5 search query."""
        content_select = ", d.content" if include_content else ""

        # BM25 weights: first param is filename, second is content
        sql = f"""
            SELECT
                d.id,
                d.filepath,
                d.filename,
                d.page_num,
                d.relative_path,
                snippet(documents_fts, 1, '<mark>', '</mark>', '...', {self.snippet_length // 10}) as snippet,
                bm25(documents_fts, {self.filename_weight}, {self.content_weight}) as score
                {content_select}
            FROM documents_fts
            JOIN documents d ON documents_fts.rowid = d.id
            WHERE documents_fts MATCH ?
            ORDER BY score
            LIMIT ? OFFSET ?
        """

        with get_connection() as conn:
            rows = conn.execute(sql, (query, limit, offset)).fetchall()

        results = []
        for row in rows:
            result = SearchResult(
                id=row["id"],
                filepath=row["filepath"],
                filename=row["filename"],
                page_num=row["page_num"],
                relative_path=row["relative_path"],
                snippet=row["snippet"] or "",
                score=row["score"],
                content=row["content"] if include_content else None
            )
            results.append(result)

        return results

    def _count_results(self, query: str) -> int:
        """Count total matching documents for pagination."""
        sql = """
            SELECT COUNT(*) as count
            FROM documents_fts
            WHERE documents_fts MATCH ?
        """

        with get_connection() as conn:
            row = conn.execute(sql, (query,)).fetchone()
            return row["count"]

    def search_simple(
        self,
        text: str,
        limit: int = None,
        offset: int = 0
    ) -> List[SearchResult]:
        """
        Convenience method for simple searches.

        Args:
            text: Search query text.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of SearchResult objects.
        """
        query = SearchQuery(
            text=text,
            limit=limit or self.default_limit,
            offset=offset
        )
        results, _ = self.search(query)
        return results

    def get_document_content(self, doc_id: int) -> Optional[str]:
        """
        Fetch full content for a document by ID.

        Args:
            doc_id: Document row ID.

        Returns:
            Full text content or None.
        """
        with get_connection() as conn:
            row = conn.execute(
                "SELECT content FROM documents WHERE id = ?",
                (doc_id,)
            ).fetchone()
            return row["content"] if row else None


if __name__ == "__main__":
    from ..database import init_schema, get_cursor

    init_schema()

    with get_cursor() as cur:
        test_docs = [
            ("/test/aviation.pdf", "aviation.pdf", 1, "Règlement de l'aviation civile française", "test/aviation.pdf", "a1"),
            ("/test/aviation.pdf", "aviation.pdf", 2, "Sécurité aérienne et contrôle du trafic", "test/aviation.pdf", "a1"),
            ("/test/maritime.pdf", "maritime.pdf", 1, "Règlement maritime international", "test/maritime.pdf", "m1"),
        ]

        cur.executemany("""
            INSERT OR IGNORE INTO documents
            (filepath, filename, page_num, content, relative_path, file_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, test_docs)

    engine = BM25Engine()

    print("=== Search Test ===")

    query = SearchQuery(text="aviation", limit=10)
    results, stats = engine.search(query)

    print(f"Query: '{stats.query}'")
    print(f"Results: {stats.total_results} in {stats.execution_time_ms}ms")

    for r in results:
        print(f"  - {r.filename} p.{r.page_num} (score: {r.display_score:.2f})")
        print(f"    {r.snippet}")
