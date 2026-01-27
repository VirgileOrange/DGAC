"""
Data models for search functionality.

Defines dataclasses for search queries and results used
throughout the search module.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchQuery:
    """
    Represents a search query with pagination options.

    Attributes:
        text: The search query text.
        limit: Maximum number of results to return.
        offset: Number of results to skip for pagination.
        advanced: Whether to use advanced query syntax.
    """
    text: str
    limit: int = 50
    offset: int = 0
    advanced: bool = False


@dataclass
class SearchResult:
    """
    Represents a single search result.

    Attributes:
        id: Database row ID.
        filepath: Absolute path to the PDF file.
        filename: PDF filename.
        page_num: Page number (1-indexed).
        relative_path: Path relative to data directory.
        snippet: Text excerpt with highlighted matches.
        score: BM25 relevance score (lower is better in SQLite FTS5).
        content: Full page content (optional, loaded on demand).
    """
    id: int
    filepath: str
    filename: str
    page_num: int
    relative_path: str
    snippet: str
    score: float
    content: Optional[str] = None

    @property
    def display_score(self) -> float:
        """
        Convert internal score to display-friendly value.

        FTS5 BM25 returns negative scores where more negative = better match.
        This converts to positive where higher = better.
        """
        return abs(self.score)


@dataclass
class SearchStats:
    """
    Statistics about a search execution.

    Attributes:
        query: The original query text.
        total_results: Total matching documents.
        execution_time_ms: Query execution time in milliseconds.
        page: Current page number.
        total_pages: Total number of pages.
    """
    query: str
    total_results: int
    execution_time_ms: float
    page: int = 1
    total_pages: int = 1


if __name__ == "__main__":
    query = SearchQuery(text="aviation civile", limit=20, offset=0)
    print(f"Query: {query}")

    result = SearchResult(
        id=42,
        filepath="/data/docs/reglement.pdf",
        filename="reglement.pdf",
        page_num=5,
        relative_path="docs/reglement.pdf",
        snippet="...règles de l'<b>aviation</b> <b>civile</b> française...",
        score=-12.5
    )
    print(f"\nResult: {result.filename} page {result.page_num}")
    print(f"Display score: {result.display_score}")

    stats = SearchStats(
        query="aviation civile",
        total_results=157,
        execution_time_ms=23.5,
        page=1,
        total_pages=8
    )
    print(f"\nStats: {stats.total_results} results in {stats.execution_time_ms}ms")
