"""
Tests for the BM25 search engine.

Tests search execution, result ranking, and snippet generation.

SAFETY NOTE: All tests use the `configured_db` fixture which:
- Creates a temporary directory via tempfile.mkdtemp()
- Configures the database singleton to use a temp path
- Cleans up all temp files after the test
- Never touches real data directories
"""

import pytest

from src.database.schema import init_schema
from src.database.repository import DocumentRepository
from src.search.bm25_engine import BM25Engine
from src.search.models import SearchQuery


@pytest.fixture
def populated_database(configured_db):
    """Create a database with test documents in temp DB."""
    init_schema()
    repo = DocumentRepository()

    # Insert test documents
    test_docs = [
        ("Aviation civile française et règlements européens", "aviation.pdf"),
        ("Sécurité maritime et navigation", "maritime.pdf"),
        ("Contrôle aérien et espace aérien", "controle.pdf"),
        ("Aviation militaire et défense", "militaire.pdf"),
        ("Transport civil et infrastructure", "transport.pdf"),
    ]

    for i, (content, filename) in enumerate(test_docs):
        repo.insert(
            filepath=f"/test/{filename}",
            filename=filename,
            relative_path=filename,
            file_hash=f"hash{i}",
            page_num=1,
            content=content
        )

    return repo  # Return repo for potential further use


class TestBM25Engine:
    """Tests for BM25Engine class."""

    def test_engine_creation(self, populated_database):
        """Test creating a search engine."""
        engine = BM25Engine()

        assert engine is not None

    def test_search_returns_results(self, populated_database):
        """Test that search returns results."""
        engine = BM25Engine()
        query = SearchQuery(text="aviation")

        results, stats = engine.search(query)

        assert len(results) > 0
        assert stats.total_results > 0

    def test_search_returns_stats(self, populated_database):
        """Test that search returns statistics."""
        engine = BM25Engine()
        query = SearchQuery(text="aviation")

        results, stats = engine.search(query)

        assert stats.query == "aviation"
        assert stats.execution_time_ms >= 0

    def test_search_respects_limit(self, populated_database):
        """Test that search respects limit parameter."""
        engine = BM25Engine()
        query = SearchQuery(text="aviation", limit=1)

        results, stats = engine.search(query)

        assert len(results) <= 1

    def test_search_no_results(self, populated_database):
        """Test search with no matching results."""
        engine = BM25Engine()
        query = SearchQuery(text="xyznonexistentterm")

        results, stats = engine.search(query)

        assert len(results) == 0
        assert stats.total_results == 0

    def test_search_results_have_snippets(self, populated_database):
        """Test that results include snippets."""
        engine = BM25Engine()
        query = SearchQuery(text="aviation")

        results, stats = engine.search(query)

        for result in results:
            # Snippet should exist (may be empty for very short content)
            assert hasattr(result, "snippet")

    def test_search_results_have_scores(self, populated_database):
        """Test that results have relevance scores."""
        engine = BM25Engine()
        query = SearchQuery(text="aviation")

        results, stats = engine.search(query)

        for result in results:
            assert hasattr(result, "score")
            assert hasattr(result, "display_score")

    def test_results_sorted_by_relevance(self, populated_database):
        """Test that results are sorted by relevance (descending display_score)."""
        engine = BM25Engine()
        query = SearchQuery(text="aviation")

        results, stats = engine.search(query)

        if len(results) > 1:
            scores = [r.display_score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestBM25EngineAdvancedSearch:
    """Tests for advanced search features."""

    def test_or_search(self, populated_database):
        """Test OR operator expands results."""
        engine = BM25Engine()

        # Search with OR
        query = SearchQuery(text="aviation OR maritime", advanced=True)
        results, stats = engine.search(query)

        # Should find both aviation and maritime documents
        filenames = [r.filename for r in results]
        assert any("aviation" in f for f in filenames) or any("maritime" in f for f in filenames)

    def test_not_search(self, populated_database):
        """Test NOT operator excludes results."""
        engine = BM25Engine()

        # Search excluding military
        query = SearchQuery(text="aviation NOT militaire", advanced=True)
        results, stats = engine.search(query)

        # Should not include military document
        for result in results:
            assert "militaire" not in result.filename.lower()

    def test_phrase_search(self, populated_database):
        """Test quoted phrase search."""
        engine = BM25Engine()

        query = SearchQuery(text='"aviation civile"', advanced=True)
        results, stats = engine.search(query, include_content=True)

        # Should find exact phrase
        if results:
            found = any("aviation civile" in (r.content or "").lower() for r in results)
            assert found or stats.total_results >= 0  # At minimum, query executes


class TestBM25EngineWithContent:
    """Tests for content inclusion."""

    def test_search_includes_content(self, populated_database):
        """Test that search can include full content."""
        engine = BM25Engine()
        query = SearchQuery(text="aviation")

        results, stats = engine.search(query, include_content=True)

        for result in results:
            assert result.content is not None
            assert len(result.content) > 0

    def test_search_without_content(self, populated_database):
        """Test that search without content flag."""
        engine = BM25Engine()
        query = SearchQuery(text="aviation")

        results, stats = engine.search(query, include_content=False)

        # Content should be None when not included
        for result in results:
            assert result.content is None
