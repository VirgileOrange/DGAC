"""
Tests for the hybrid search engine module.

Tests unified search interface supporting lexical, semantic, and hybrid modes,
including RRF fusion algorithm. Uses mocked sub-engines.

SAFETY NOTE: All tests use the `configured_db` fixture which:
- Creates a temporary directory via tempfile.mkdtemp()
- Configures the database singleton to use a temp path
- Cleans up all temp files after the test
- Never touches real data directories
- Uses mocked engines to avoid API calls
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.database.schema import init_schema
from src.database.repository import DocumentRepository
from src.search.hybrid_engine import (
    HybridEngine,
    HybridSearchResult,
    HybridSearchStats,
    SearchMode,
)
from src.search.models import SearchResult


class TestSearchMode:
    """Tests for SearchMode enum."""

    def test_lexical_mode(self):
        """Test lexical search mode value."""
        assert SearchMode.LEXICAL.value == "lexical"

    def test_semantic_mode(self):
        """Test semantic search mode value."""
        assert SearchMode.SEMANTIC.value == "semantic"

    def test_hybrid_mode(self):
        """Test hybrid search mode value."""
        assert SearchMode.HYBRID.value == "hybrid"


class TestHybridSearchResult:
    """Tests for HybridSearchResult dataclass."""

    def test_result_creation(self):
        """Test creating a HybridSearchResult instance."""
        result = HybridSearchResult(
            document_id=1,
            filepath="/path/to/doc.pdf",
            filename="doc.pdf",
            page_num=1,
            relative_path="doc.pdf",
            snippet="Test snippet...",
            score=0.85,
            source="both"
        )

        assert result.document_id == 1
        assert result.filepath == "/path/to/doc.pdf"
        assert result.filename == "doc.pdf"
        assert result.score == 0.85
        assert result.source == "both"

    def test_result_with_ranks(self):
        """Test result with lexical and semantic ranks."""
        result = HybridSearchResult(
            document_id=1,
            filepath="/path/doc.pdf",
            filename="doc.pdf",
            page_num=1,
            relative_path="doc.pdf",
            snippet="Snippet",
            score=0.9,
            source="both",
            lexical_rank=1,
            semantic_rank=2,
            similarity=0.95
        )

        assert result.lexical_rank == 1
        assert result.semantic_rank == 2
        assert result.similarity == 0.95


class TestHybridSearchStats:
    """Tests for HybridSearchStats dataclass."""

    def test_stats_creation(self):
        """Test creating HybridSearchStats instance."""
        stats = HybridSearchStats(
            query="test query",
            mode="hybrid",
            total_results=10,
            execution_time_ms=100.0
        )

        assert stats.query == "test query"
        assert stats.mode == "hybrid"
        assert stats.total_results == 10
        assert stats.execution_time_ms == 100.0

    def test_stats_with_details(self):
        """Test stats with detailed metrics."""
        stats = HybridSearchStats(
            query="aviation",
            mode="hybrid",
            total_results=15,
            execution_time_ms=200.0,
            lexical_results=10,
            semantic_results=8,
            overlap_count=3,
            lexical_time_ms=50.0,
            semantic_time_ms=150.0
        )

        assert stats.lexical_results == 10
        assert stats.semantic_results == 8
        assert stats.overlap_count == 3


class TestHybridEngine:
    """Tests for HybridEngine class."""

    @pytest.fixture
    def mock_hybrid_engine(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """
        Create a HybridEngine with mocked sub-engines.

        Args:
            configured_db: Database configuration fixture.
            reset_embedding_singleton: Singleton reset fixture.
            reset_vec_extension_cache: Vec extension cache reset fixture.

        Returns:
            HybridEngine with mocked dependencies.
        """
        init_schema()

        # Create mock BM25 engine
        mock_bm25 = Mock()
        mock_bm25.search.return_value = ([], Mock(total_results=0))

        # Create mock Semantic engine
        mock_semantic = Mock()
        mock_semantic.search.return_value = ([], Mock(total_results=0))

        with patch("src.search.hybrid_engine.BM25Engine") as MockBM25, \
             patch("src.search.hybrid_engine.SemanticEngine") as MockSemantic:
            MockBM25.return_value = mock_bm25
            MockSemantic.return_value = mock_semantic

            engine = HybridEngine()
            engine.bm25_engine = mock_bm25
            engine.semantic_engine = mock_semantic

        return engine

    def test_engine_creation(self, mock_hybrid_engine):
        """Test creating a HybridEngine instance."""
        assert mock_hybrid_engine is not None
        assert mock_hybrid_engine.rrf_k == 60

    def test_get_available_modes(self, mock_hybrid_engine):
        """Test getting list of available modes."""
        modes = mock_hybrid_engine.get_available_modes()

        assert "lexical" in modes
        assert "semantic" in modes
        assert "hybrid" in modes

    def test_search_empty_query(self, mock_hybrid_engine):
        """Test search with empty query returns empty results."""
        results, stats = mock_hybrid_engine.search("")

        assert len(results) == 0
        assert stats.total_results == 0


class TestHybridEngineModes:
    """Tests for different search modes."""

    @pytest.fixture
    def engine_with_mock_results(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """
        Create engine with mock results from both sub-engines.

        Args:
            configured_db: Database configuration fixture.
            reset_embedding_singleton: Singleton reset fixture.
            reset_vec_extension_cache: Vec extension cache reset fixture.

        Returns:
            HybridEngine with mocked results.
        """
        init_schema()

        # Create mock BM25 results
        mock_bm25_results = [
            Mock(
                id=1,
                filepath="/test/doc1.pdf",
                filename="doc1.pdf",
                page_num=1,
                relative_path="doc1.pdf",
                snippet="BM25 result 1",
                score=-5.0,
                display_score=0.95,
                content="Content 1"
            ),
            Mock(
                id=2,
                filepath="/test/doc2.pdf",
                filename="doc2.pdf",
                page_num=1,
                relative_path="doc2.pdf",
                snippet="BM25 result 2",
                score=-6.0,
                display_score=0.90,
                content="Content 2"
            ),
        ]
        mock_bm25_stats = Mock(total_results=2)

        mock_bm25 = Mock()
        mock_bm25.search.return_value = (mock_bm25_results, mock_bm25_stats)

        # Create mock Semantic results
        mock_semantic_results = [
            Mock(
                document_id=2,
                filepath="/test/doc2.pdf",
                filename="doc2.pdf",
                page_num=1,
                relative_path="doc2.pdf",
                snippet="Semantic result 1",
                similarity=0.92
            ),
            Mock(
                document_id=3,
                filepath="/test/doc3.pdf",
                filename="doc3.pdf",
                page_num=1,
                relative_path="doc3.pdf",
                snippet="Semantic result 2",
                similarity=0.88
            ),
        ]
        mock_semantic_stats = Mock(total_results=2)

        mock_semantic = Mock()
        mock_semantic.search.return_value = (mock_semantic_results, mock_semantic_stats)

        with patch("src.search.hybrid_engine.BM25Engine") as MockBM25, \
             patch("src.search.hybrid_engine.SemanticEngine") as MockSemantic:
            MockBM25.return_value = mock_bm25
            MockSemantic.return_value = mock_semantic

            engine = HybridEngine()
            engine.bm25_engine = mock_bm25
            engine.semantic_engine = mock_semantic

        return engine

    def test_lexical_mode_only_calls_bm25(self, engine_with_mock_results):
        """Test lexical mode only uses BM25 engine."""
        results, stats = engine_with_mock_results.search(
            "aviation",
            mode=SearchMode.LEXICAL
        )

        engine_with_mock_results.bm25_engine.search.assert_called_once()
        engine_with_mock_results.semantic_engine.search.assert_not_called()
        assert stats.mode == "lexical"

    def test_semantic_mode_only_calls_semantic(self, engine_with_mock_results):
        """Test semantic mode only uses semantic engine."""
        results, stats = engine_with_mock_results.search(
            "aviation",
            mode=SearchMode.SEMANTIC
        )

        engine_with_mock_results.semantic_engine.search.assert_called_once()
        engine_with_mock_results.bm25_engine.search.assert_not_called()
        assert stats.mode == "semantic"

    def test_hybrid_mode_calls_both(self, engine_with_mock_results):
        """Test hybrid mode calls both engines."""
        results, stats = engine_with_mock_results.search(
            "aviation",
            mode=SearchMode.HYBRID
        )

        engine_with_mock_results.bm25_engine.search.assert_called_once()
        engine_with_mock_results.semantic_engine.search.assert_called_once()
        assert stats.mode == "hybrid"

    def test_lexical_results_have_source(self, engine_with_mock_results):
        """Test lexical results are marked with correct source."""
        results, stats = engine_with_mock_results.search(
            "aviation",
            mode=SearchMode.LEXICAL
        )

        for result in results:
            assert result.source == "lexical"

    def test_semantic_results_have_source(self, engine_with_mock_results):
        """Test semantic results are marked with correct source."""
        results, stats = engine_with_mock_results.search(
            "aviation",
            mode=SearchMode.SEMANTIC
        )

        for result in results:
            assert result.source == "semantic"


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion algorithm."""

    @pytest.fixture
    def engine_for_fusion(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Create engine for testing RRF fusion."""
        init_schema()

        # Create overlapping results
        mock_bm25_results = [
            Mock(
                id=1,
                filepath="/test/common.pdf",
                filename="common.pdf",
                page_num=1,
                relative_path="common.pdf",
                snippet="Common doc - BM25",
                score=-5.0,
                display_score=0.95,
                content="Content"
            ),
            Mock(
                id=2,
                filepath="/test/lexical_only.pdf",
                filename="lexical_only.pdf",
                page_num=1,
                relative_path="lexical_only.pdf",
                snippet="Lexical only",
                score=-6.0,
                display_score=0.90,
                content="Content"
            ),
        ]

        mock_semantic_results = [
            Mock(
                document_id=1,
                filepath="/test/common.pdf",
                filename="common.pdf",
                page_num=1,
                relative_path="common.pdf",
                snippet="Common doc - Semantic",
                similarity=0.92
            ),
            Mock(
                document_id=3,
                filepath="/test/semantic_only.pdf",
                filename="semantic_only.pdf",
                page_num=1,
                relative_path="semantic_only.pdf",
                snippet="Semantic only",
                similarity=0.88
            ),
        ]

        mock_bm25 = Mock()
        mock_bm25.search.return_value = (mock_bm25_results, Mock(total_results=2))

        mock_semantic = Mock()
        mock_semantic.search.return_value = (mock_semantic_results, Mock(total_results=2))

        with patch("src.search.hybrid_engine.BM25Engine") as MockBM25, \
             patch("src.search.hybrid_engine.SemanticEngine") as MockSemantic:
            MockBM25.return_value = mock_bm25
            MockSemantic.return_value = mock_semantic

            engine = HybridEngine()
            engine.bm25_engine = mock_bm25
            engine.semantic_engine = mock_semantic

        return engine

    def test_fusion_detects_overlap(self, engine_for_fusion):
        """Test that RRF fusion detects overlapping results."""
        results, stats = engine_for_fusion.search(
            "aviation",
            mode=SearchMode.HYBRID
        )

        assert stats.overlap_count == 1

    def test_fusion_marks_common_results(self, engine_for_fusion):
        """Test that common results are marked as 'both'."""
        results, stats = engine_for_fusion.search(
            "aviation",
            mode=SearchMode.HYBRID
        )

        sources = {r.filename: r.source for r in results}
        assert sources.get("common.pdf") == "both"

    def test_fusion_ranks_common_higher(self, engine_for_fusion):
        """Test that documents found by both methods rank higher."""
        results, stats = engine_for_fusion.search(
            "aviation",
            mode=SearchMode.HYBRID
        )

        # Common document should be first (highest RRF score)
        if results:
            assert results[0].filename == "common.pdf"

    def test_fusion_includes_all_results(self, engine_for_fusion):
        """Test that fusion includes results from both engines."""
        results, stats = engine_for_fusion.search(
            "aviation",
            mode=SearchMode.HYBRID,
            limit=10
        )

        filenames = {r.filename for r in results}
        assert "common.pdf" in filenames
        assert "lexical_only.pdf" in filenames
        assert "semantic_only.pdf" in filenames


class TestHybridEngineWeights:
    """Tests for configurable weights."""

    @pytest.fixture
    def engine_with_weights(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Create engine for testing weight parameters."""
        init_schema()

        mock_bm25 = Mock()
        mock_bm25.search.return_value = ([], Mock(total_results=0))

        mock_semantic = Mock()
        mock_semantic.search.return_value = ([], Mock(total_results=0))

        with patch("src.search.hybrid_engine.BM25Engine") as MockBM25, \
             patch("src.search.hybrid_engine.SemanticEngine") as MockSemantic:
            MockBM25.return_value = mock_bm25
            MockSemantic.return_value = mock_semantic

            engine = HybridEngine()
            engine.bm25_engine = mock_bm25
            engine.semantic_engine = mock_semantic

        return engine

    def test_default_weights(self, engine_with_weights):
        """Test that default weights are applied."""
        assert engine_with_weights.default_lexical_weight == 1.0
        assert engine_with_weights.default_semantic_weight == 1.0

    def test_custom_weights_accepted(self, engine_with_weights):
        """Test that custom weights can be passed."""
        # Should not raise
        results, stats = engine_with_weights.search(
            "aviation",
            mode=SearchMode.HYBRID,
            lexical_weight=2.0,
            semantic_weight=0.5
        )

        assert isinstance(results, list)


class TestHybridEngineErrorHandling:
    """Tests for error handling and graceful degradation."""

    @pytest.fixture
    def engine_with_failing_semantic(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Create engine where semantic search fails."""
        init_schema()

        mock_bm25_results = [
            Mock(
                id=1,
                filepath="/test/doc.pdf",
                filename="doc.pdf",
                page_num=1,
                relative_path="doc.pdf",
                snippet="Result",
                score=-5.0,
                display_score=0.95,
                content="Content"
            ),
        ]

        mock_bm25 = Mock()
        mock_bm25.search.return_value = (mock_bm25_results, Mock(total_results=1))

        mock_semantic = Mock()
        mock_semantic.search.side_effect = Exception("API Error")

        with patch("src.search.hybrid_engine.BM25Engine") as MockBM25, \
             patch("src.search.hybrid_engine.SemanticEngine") as MockSemantic:
            MockBM25.return_value = mock_bm25
            MockSemantic.return_value = mock_semantic

            engine = HybridEngine()
            engine.bm25_engine = mock_bm25
            engine.semantic_engine = mock_semantic

        return engine

    def test_hybrid_degrades_gracefully(self, engine_with_failing_semantic):
        """Test that hybrid mode returns lexical results if semantic fails."""
        results, stats = engine_with_failing_semantic.search(
            "aviation",
            mode=SearchMode.HYBRID
        )

        # Should still return lexical results
        assert len(results) > 0
        assert len(stats.errors) > 0
        assert "Semantic" in stats.errors[0]
