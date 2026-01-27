"""
Tests for search data models.

Tests SearchQuery, SearchResult, and SearchStats dataclasses.
"""

import pytest

from src.search.models import SearchQuery, SearchResult, SearchStats


class TestSearchQuery:
    """Tests for SearchQuery dataclass."""

    def test_query_creation_minimal(self):
        """Test creating query with just text."""
        query = SearchQuery(text="aviation")

        assert query.text == "aviation"
        assert query.limit > 0  # Has default
        assert query.offset == 0
        assert query.advanced is False

    def test_query_creation_full(self):
        """Test creating query with all parameters."""
        query = SearchQuery(
            text="aviation civile",
            limit=20,
            offset=10,
            advanced=True
        )

        assert query.text == "aviation civile"
        assert query.limit == 20
        assert query.offset == 10
        assert query.advanced is True

    def test_query_text_is_required(self):
        """Test that text parameter is required."""
        with pytest.raises(TypeError):
            SearchQuery()  # Missing text


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            id=1,
            filepath="/path/to/doc.pdf",
            filename="doc.pdf",
            relative_path="doc.pdf",
            page_num=5,
            snippet="...matching **text**...",
            score=-2.5,
            content="Full page content here"
        )

        assert result.id == 1
        assert result.filename == "doc.pdf"
        assert result.page_num == 5
        assert result.score == -2.5

    def test_result_display_score_property(self):
        """Test that display_score converts negative BM25 score to positive."""
        result = SearchResult(
            id=1,
            filepath="/test.pdf",
            filename="test.pdf",
            relative_path="test.pdf",
            page_num=1,
            snippet="test",
            score=-12.5  # BM25 negative score
        )

        # display_score should be positive (absolute value)
        assert result.display_score == 12.5

    def test_result_has_all_fields(self):
        """Test that result has all required fields."""
        result = SearchResult(
            id=1,
            filepath="/test.pdf",
            filename="test.pdf",
            relative_path="test.pdf",
            page_num=1,
            snippet="",
            score=0.0
        )

        # All fields should be accessible
        assert hasattr(result, "id")
        assert hasattr(result, "filepath")
        assert hasattr(result, "filename")
        assert hasattr(result, "relative_path")
        assert hasattr(result, "page_num")
        assert hasattr(result, "snippet")
        assert hasattr(result, "score")
        assert hasattr(result, "display_score")  # Property

    def test_result_content_is_optional(self):
        """Test that content field is optional."""
        result = SearchResult(
            id=1,
            filepath="/test.pdf",
            filename="test.pdf",
            relative_path="test.pdf",
            page_num=1,
            snippet="",
            score=0.0
            # content not provided
        )

        assert result.content is None


class TestSearchStats:
    """Tests for SearchStats dataclass."""

    def test_stats_creation(self):
        """Test creating search statistics."""
        stats = SearchStats(
            query="test query",
            total_results=150,
            execution_time_ms=45.5,
            page=1,
            total_pages=8
        )

        assert stats.query == "test query"
        assert stats.total_results == 150
        assert stats.execution_time_ms == 45.5
        assert stats.page == 1
        assert stats.total_pages == 8

    def test_stats_zero_results(self):
        """Test stats with zero results."""
        stats = SearchStats(
            query="nonexistent",
            total_results=0,
            execution_time_ms=5.0
        )

        assert stats.total_results == 0
        assert stats.page == 1  # Default
        assert stats.total_pages == 1  # Default

    def test_stats_defaults(self):
        """Test stats default values."""
        stats = SearchStats(
            query="test",
            total_results=50,
            execution_time_ms=10.0
        )

        # Check defaults
        assert stats.page == 1
        assert stats.total_pages == 1
