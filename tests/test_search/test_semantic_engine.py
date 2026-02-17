"""
Tests for the semantic search engine module.

Tests semantic search operations including query embedding,
vector search, and result enrichment. Uses mocked services.

SAFETY NOTE: All tests use the `configured_db` fixture which:
- Creates a temporary directory via tempfile.mkdtemp()
- Configures the database singleton to use a temp path
- Cleans up all temp files after the test
- Never touches real data directories
- Uses mocked embedding service to avoid API calls
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.database.schema import init_schema, init_vector_index, reset_vec_extension_cache
from src.database.repository import DocumentRepository
from src.database.vector_repository import VectorRepository, VectorSearchResult
from src.search.semantic_engine import (
    SemanticEngine,
    SemanticSearchResult,
    SemanticSearchStats,
)


class TestSemanticSearchResult:
    """Tests for SemanticSearchResult dataclass."""

    def test_result_creation(self):
        """Test creating a SemanticSearchResult instance."""
        result = SemanticSearchResult(
            chunk_id="chunk001",
            document_id=1,
            filepath="/path/to/doc.pdf",
            filename="doc.pdf",
            page_num=1,
            relative_path="doc.pdf",
            snippet="Aviation safety regulations...",
            similarity=0.92
        )

        assert result.chunk_id == "chunk001"
        assert result.document_id == 1
        assert result.filepath == "/path/to/doc.pdf"
        assert result.filename == "doc.pdf"
        assert result.page_num == 1
        assert result.similarity == 0.92


class TestSemanticSearchStats:
    """Tests for SemanticSearchStats dataclass."""

    def test_stats_creation(self):
        """Test creating SemanticSearchStats instance."""
        stats = SemanticSearchStats(
            query="aviation safety",
            total_results=10,
            execution_time_ms=150.5,
            embedding_time_ms=100.0,
            search_time_ms=50.5
        )

        assert stats.query == "aviation safety"
        assert stats.total_results == 10
        assert stats.execution_time_ms == 150.5
        assert stats.embedding_time_ms == 100.0
        assert stats.search_time_ms == 50.5


class TestSemanticEngine:
    """Tests for SemanticEngine class."""

    @pytest.fixture
    def mock_engine(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """
        Create a SemanticEngine with mocked dependencies.

        Args:
            configured_db: Database configuration fixture.
            reset_embedding_singleton: Singleton reset fixture.
            reset_vec_extension_cache: Vec extension cache reset fixture.

        Returns:
            SemanticEngine with mocked dependencies.
        """
        init_schema()
        init_vector_index()

        # Create mocked embedding service
        mock_embed_service = Mock()
        mock_embed_service.embed_query.return_value = np.random.randn(1024)
        mock_embed_service.get_model_info.return_value = {
            "model": "test-model",
            "dimensions": 1024,
            "endpoint": "https://test.example.com",
            "initialized": True
        }

        # Create mocked vector repository
        mock_vector_repo = Mock()
        mock_vector_repo.get_chunk_count.return_value = 100
        mock_vector_repo.search_similar.return_value = []

        # Patch and create engine
        with patch("src.search.semantic_engine.get_embedding_service") as mock_get_embed:
            mock_get_embed.return_value = mock_embed_service
            engine = SemanticEngine()
            engine.embedding_service = mock_embed_service
            engine.vector_repo = mock_vector_repo

        return engine

    def test_engine_creation(self, mock_engine):
        """Test creating a SemanticEngine instance."""
        assert mock_engine is not None

    def test_search_empty_query(self, mock_engine):
        """Test search with empty query returns empty results."""
        results, stats = mock_engine.search("")

        assert len(results) == 0
        assert stats.total_results == 0
        assert stats.execution_time_ms == 0

    def test_search_whitespace_query(self, mock_engine):
        """Test search with whitespace-only query returns empty results."""
        results, stats = mock_engine.search("   ")

        assert len(results) == 0
        assert stats.total_results == 0

    def test_search_calls_embedding_service(self, mock_engine):
        """Test that search calls embedding service."""
        mock_engine.search("aviation safety")

        mock_engine.embedding_service.embed_query.assert_called_once_with("aviation safety")

    def test_search_calls_vector_repo(self, mock_engine):
        """Test that search calls vector repository."""
        mock_engine.search("aviation safety", limit=20)

        mock_engine.vector_repo.search_similar.assert_called_once()

    def test_search_returns_stats(self, mock_engine):
        """Test that search returns statistics."""
        results, stats = mock_engine.search("aviation")

        assert stats.query == "aviation"
        assert stats.execution_time_ms >= 0
        assert stats.embedding_time_ms >= 0
        assert stats.search_time_ms >= 0

    def test_get_index_stats(self, mock_engine):
        """Test getting index statistics."""
        stats = mock_engine.get_index_stats()

        assert "total_chunks" in stats
        assert "embedding_model" in stats
        assert "embedding_dimensions" in stats
        assert "index_ready" in stats

    def test_clear_cache(self, mock_engine):
        """Test clearing document cache."""
        # Add something to cache
        mock_engine._document_cache[1] = {"filepath": "/test"}

        mock_engine.clear_cache()

        assert len(mock_engine._document_cache) == 0


class TestSemanticEngineWithResults:
    """Tests for SemanticEngine with mock results."""

    @pytest.fixture
    def engine_with_results(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """
        Create a SemanticEngine with mocked results.

        Args:
            configured_db: Database configuration fixture.
            reset_embedding_singleton: Singleton reset fixture.
            reset_vec_extension_cache: Vec extension cache reset fixture.

        Returns:
            SemanticEngine configured to return mock results.
        """
        init_schema()

        # Insert test documents into temp database
        repo = DocumentRepository()
        repo.insert(
            filepath="/test/aviation.pdf",
            filename="aviation.pdf",
            relative_path="aviation.pdf",
            file_hash="hash1",
            page_num=1,
            content="Aviation safety regulations"
        )

        # Get the document ID
        doc = repo.get_by_filepath("/test/aviation.pdf")[0]
        doc_id = doc.id

        # Create mocked services
        mock_embed_service = Mock()
        mock_embed_service.embed_query.return_value = np.random.randn(1024)
        mock_embed_service.get_model_info.return_value = {
            "model": "test-model",
            "dimensions": 1024,
            "endpoint": "https://test.example.com",
            "initialized": True
        }

        mock_vector_repo = Mock()
        mock_vector_repo.get_chunk_count.return_value = 10
        mock_vector_repo.search_similar.return_value = [
            VectorSearchResult(
                chunk_id="chunk001",
                document_id=doc_id,
                page_num=1,
                position=0,
                content="Aviation safety regulations and procedures for civil aviation.",
                similarity=0.95
            ),
            VectorSearchResult(
                chunk_id="chunk002",
                document_id=doc_id,
                page_num=2,
                position=0,
                content="Air traffic control guidelines.",
                similarity=0.85
            ),
        ]

        with patch("src.search.semantic_engine.get_embedding_service") as mock_get_embed:
            mock_get_embed.return_value = mock_embed_service
            engine = SemanticEngine()
            engine.embedding_service = mock_embed_service
            engine.vector_repo = mock_vector_repo

        return engine

    def test_search_returns_enriched_results(self, engine_with_results):
        """Test that search returns enriched results with document info."""
        results, stats = engine_with_results.search("aviation")

        assert len(results) == 2
        assert results[0].filename == "aviation.pdf"
        assert results[0].filepath == "/test/aviation.pdf"

    def test_search_results_have_snippets(self, engine_with_results):
        """Test that results include snippets."""
        results, stats = engine_with_results.search("aviation")

        for result in results:
            assert result.snippet is not None
            assert len(result.snippet) > 0

    def test_search_results_sorted_by_similarity(self, engine_with_results):
        """Test that results are ordered by similarity (descending)."""
        results, stats = engine_with_results.search("aviation")

        if len(results) >= 2:
            similarities = [r.similarity for r in results]
            assert similarities == sorted(similarities, reverse=True)

    def test_search_respects_min_similarity(self, engine_with_results):
        """Test that min_similarity filter is applied."""
        results, stats = engine_with_results.search("aviation", min_similarity=0.90)

        for result in results:
            assert result.similarity >= 0.90


class TestSemanticEngineSnippet:
    """Tests for snippet generation."""

    @pytest.fixture
    def engine_for_snippet(self, configured_db, reset_embedding_singleton):
        """Create engine for snippet testing."""
        init_schema()

        with patch("src.search.semantic_engine.get_embedding_service"):
            engine = SemanticEngine()

        return engine

    def test_generate_snippet_short_content(self, engine_for_snippet):
        """Test snippet generation for short content."""
        content = "Short content."

        snippet = engine_for_snippet._generate_snippet(content)

        assert snippet == content

    def test_generate_snippet_long_content(self, engine_for_snippet):
        """Test snippet generation truncates long content."""
        content = "This is a very long content. " * 20

        snippet = engine_for_snippet._generate_snippet(content)

        assert len(snippet) <= engine_for_snippet.snippet_length + 10  # Allow for "..."
        assert snippet.endswith("...")

    def test_generate_snippet_breaks_at_word(self, engine_for_snippet):
        """Test snippet breaks at word boundary."""
        content = "word " * 50

        snippet = engine_for_snippet._generate_snippet(content)

        # Should not end mid-word
        if snippet.endswith("..."):
            before_dots = snippet[:-3]
            assert before_dots.endswith(" ") or before_dots.endswith("d")
