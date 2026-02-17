"""
Tests for the semantic indexer module.

Tests the semantic indexing pipeline including chunking, embedding generation,
and vector storage. Uses mocked embedding service to avoid API calls.

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
from typing import List, Tuple

from src.database.schema import init_schema, init_vector_index, reset_vec_extension_cache
from src.database.repository import DocumentRepository
from src.database.vector_repository import VectorRepository
from src.indexer.semantic_indexer import (
    SemanticIndexer,
    SemanticIndexingStats,
    semantic_progress_printer,
)


class TestSemanticIndexingStats:
    """Tests for SemanticIndexingStats dataclass."""

    def test_stats_creation(self):
        """Test creating SemanticIndexingStats with defaults."""
        stats = SemanticIndexingStats()

        assert stats.documents_processed == 0
        assert stats.documents_skipped == 0
        assert stats.documents_failed == 0
        assert stats.chunks_created == 0
        assert stats.embeddings_generated == 0
        assert stats.errors == []

    def test_stats_with_values(self):
        """Test creating SemanticIndexingStats with values."""
        stats = SemanticIndexingStats(
            documents_processed=10,
            documents_skipped=2,
            documents_failed=1,
            chunks_created=50,
            embeddings_generated=50,
            errors=["Error 1", "Error 2"]
        )

        assert stats.documents_processed == 10
        assert stats.documents_skipped == 2
        assert stats.documents_failed == 1
        assert stats.chunks_created == 50
        assert len(stats.errors) == 2


class TestSemanticIndexer:
    """Tests for SemanticIndexer class."""

    @pytest.fixture
    def mock_indexer(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """
        Create a SemanticIndexer with mocked embedding service.

        Args:
            configured_db: Database configuration fixture.
            reset_embedding_singleton: Singleton reset fixture.
            reset_vec_extension_cache: Vec extension cache reset fixture.

        Returns:
            SemanticIndexer with mocked dependencies.
        """
        init_schema()
        init_vector_index()

        # Create mock embedding service
        mock_embed_service = Mock()
        mock_embed_service.embed_passages.return_value = np.random.randn(1, 1024)
        mock_embed_service.get_model_info.return_value = {
            "model": "test-model",
            "dimensions": 1024,
            "endpoint": "https://test.example.com",
            "initialized": True
        }

        with patch("src.indexer.semantic_indexer.get_embedding_service") as mock_get:
            mock_get.return_value = mock_embed_service
            indexer = SemanticIndexer()
            indexer.embedding_service = mock_embed_service

        return indexer

    def test_indexer_creation(self, mock_indexer):
        """Test creating a SemanticIndexer instance."""
        assert mock_indexer is not None
        assert mock_indexer.enabled is True

    def test_indexer_disabled_by_param(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Test creating disabled indexer via parameter."""
        init_schema()

        with patch("src.indexer.semantic_indexer.get_embedding_service"):
            indexer = SemanticIndexer(enabled=False)

        assert indexer.enabled is False

    def test_ensure_schema(self, mock_indexer):
        """Test that ensure_schema completes without error."""
        # Should not raise
        mock_indexer.ensure_schema()

    def test_get_stats(self, mock_indexer):
        """Test getting indexer statistics."""
        stats = mock_indexer.get_stats()

        assert "total_chunks" in stats
        assert "embedding_model" in stats
        assert "embedding_dimensions" in stats
        assert "semantic_enabled" in stats


class TestSemanticIndexerIndexDocument:
    """Tests for single document indexing."""

    @pytest.fixture
    def indexer_with_mock(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Create indexer with controllable mock."""
        init_schema()
        init_vector_index()

        mock_embed_service = Mock()

        def mock_embed(texts):
            return np.random.randn(len(texts), 1024)

        mock_embed_service.embed_passages.side_effect = mock_embed
        mock_embed_service.get_model_info.return_value = {
            "model": "test-model",
            "dimensions": 1024,
            "endpoint": "https://test.example.com",
            "initialized": True
        }

        with patch("src.indexer.semantic_indexer.get_embedding_service") as mock_get:
            mock_get.return_value = mock_embed_service
            indexer = SemanticIndexer()
            indexer.embedding_service = mock_embed_service

        return indexer

    def test_index_document_single_page(self, indexer_with_mock):
        """Test indexing a document with single page."""
        pages = [(1, "This is the content of page one.")]

        chunks_count = indexer_with_mock.index_document(
            doc_id=1,
            pages=pages,
            filename="test.pdf"
        )

        assert chunks_count > 0
        indexer_with_mock.embedding_service.embed_passages.assert_called_once()

    def test_index_document_multiple_pages(self, indexer_with_mock):
        """Test indexing a document with multiple pages."""
        pages = [
            (1, "Content of page one."),
            (2, "Content of page two."),
            (3, "Content of page three."),
        ]

        chunks_count = indexer_with_mock.index_document(
            doc_id=1,
            pages=pages,
            filename="multi_page.pdf"
        )

        assert chunks_count == 3

    def test_index_document_empty_pages(self, indexer_with_mock):
        """Test indexing document with empty pages returns zero."""
        pages = [
            (1, ""),
            (2, "   "),
        ]

        chunks_count = indexer_with_mock.index_document(
            doc_id=1,
            pages=pages,
            filename="empty.pdf"
        )

        assert chunks_count == 0

    def test_index_document_disabled(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Test that disabled indexer returns zero."""
        init_schema()

        with patch("src.indexer.semantic_indexer.get_embedding_service"):
            indexer = SemanticIndexer(enabled=False)

        pages = [(1, "Content")]
        chunks_count = indexer.index_document(1, pages, "test.pdf")

        assert chunks_count == 0


class TestSemanticIndexerBatch:
    """Tests for batch document indexing."""

    @pytest.fixture
    def batch_indexer(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Create indexer for batch testing."""
        init_schema()
        init_vector_index()

        mock_embed_service = Mock()

        def mock_embed(texts):
            return np.random.randn(len(texts), 1024)

        mock_embed_service.embed_passages.side_effect = mock_embed
        mock_embed_service.get_model_info.return_value = {
            "model": "test-model",
            "dimensions": 1024
        }

        with patch("src.indexer.semantic_indexer.get_embedding_service") as mock_get:
            mock_get.return_value = mock_embed_service
            indexer = SemanticIndexer()
            indexer.embedding_service = mock_embed_service

        return indexer

    def test_index_documents_batch(self, batch_indexer):
        """Test indexing multiple documents in batch."""
        documents = [
            {
                "doc_id": 1,
                "pages": [(1, "Document one content.")],
                "filename": "doc1.pdf"
            },
            {
                "doc_id": 2,
                "pages": [(1, "Document two content.")],
                "filename": "doc2.pdf"
            },
        ]

        stats = batch_indexer.index_documents_batch(documents)

        assert stats.documents_processed == 2
        assert stats.chunks_created == 2

    def test_index_documents_batch_with_callback(self, batch_indexer):
        """Test batch indexing with progress callback."""
        progress_calls = []

        def callback(current, total, message):
            progress_calls.append((current, total, message))

        batch_indexer.progress_callback = callback

        documents = [
            {"doc_id": 1, "pages": [(1, "Content 1")], "filename": "doc1.pdf"},
            {"doc_id": 2, "pages": [(1, "Content 2")], "filename": "doc2.pdf"},
        ]

        batch_indexer.index_documents_batch(documents)

        assert len(progress_calls) == 2
        assert progress_calls[0][0] == 1
        assert progress_calls[1][0] == 2

    def test_index_documents_batch_disabled(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Test that disabled indexer returns empty stats."""
        init_schema()

        with patch("src.indexer.semantic_indexer.get_embedding_service"):
            indexer = SemanticIndexer(enabled=False)

        documents = [
            {"doc_id": 1, "pages": [(1, "Content")], "filename": "doc.pdf"}
        ]

        stats = indexer.index_documents_batch(documents)

        assert stats.documents_processed == 0


class TestSemanticIndexerDelete:
    """Tests for document deletion."""

    @pytest.fixture
    def populated_indexer(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Create indexer with indexed documents."""
        init_schema()
        init_vector_index()

        mock_embed_service = Mock()

        def mock_embed(texts):
            return np.random.randn(len(texts), 1024)

        mock_embed_service.embed_passages.side_effect = mock_embed
        mock_embed_service.get_model_info.return_value = {
            "model": "test-model",
            "dimensions": 1024
        }

        with patch("src.indexer.semantic_indexer.get_embedding_service") as mock_get:
            mock_get.return_value = mock_embed_service
            indexer = SemanticIndexer()
            indexer.embedding_service = mock_embed_service

            # Index some documents
            indexer.index_document(1, [(1, "Content one")], "doc1.pdf")
            indexer.index_document(2, [(1, "Content two")], "doc2.pdf")

        return indexer

    def test_delete_document(self, populated_indexer):
        """Test deleting document from semantic index."""
        # Verify chunks exist
        initial_count = populated_indexer.vector_repo.get_document_chunk_count(1)
        assert initial_count > 0

        # Delete
        deleted = populated_indexer.delete_document(1)

        assert deleted > 0
        assert populated_indexer.vector_repo.get_document_chunk_count(1) == 0

    def test_delete_nonexistent_document(self, populated_indexer):
        """Test deleting nonexistent document returns zero."""
        deleted = populated_indexer.delete_document(999)

        assert deleted == 0


class TestSemanticIndexerReindex:
    """Tests for full reindex operation."""

    @pytest.fixture
    def reindex_setup(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Set up database with documents for reindexing."""
        init_schema()
        init_vector_index()

        # Insert documents into FTS5 index
        repo = DocumentRepository()
        repo.insert(
            filepath="/test/doc1.pdf",
            filename="doc1.pdf",
            relative_path="doc1.pdf",
            file_hash="hash1",
            page_num=1,
            content="Content of document one for reindexing."
        )
        repo.insert(
            filepath="/test/doc2.pdf",
            filename="doc2.pdf",
            relative_path="doc2.pdf",
            file_hash="hash2",
            page_num=1,
            content="Content of document two for reindexing."
        )

        mock_embed_service = Mock()

        def mock_embed(texts):
            return np.random.randn(len(texts), 1024)

        mock_embed_service.embed_passages.side_effect = mock_embed
        mock_embed_service.get_model_info.return_value = {
            "model": "test-model",
            "dimensions": 1024
        }

        with patch("src.indexer.semantic_indexer.get_embedding_service") as mock_get:
            mock_get.return_value = mock_embed_service
            indexer = SemanticIndexer()
            indexer.embedding_service = mock_embed_service

        return indexer

    def test_reindex_all(self, reindex_setup):
        """Test reindexing all documents."""
        stats = reindex_setup.reindex_all()

        assert stats.documents_processed == 2
        assert stats.chunks_created > 0

    def test_reindex_all_disabled(
        self,
        configured_db,
        reset_embedding_singleton,
        reset_vec_extension_cache
    ):
        """Test that disabled indexer returns empty stats on reindex."""
        init_schema()

        with patch("src.indexer.semantic_indexer.get_embedding_service"):
            indexer = SemanticIndexer(enabled=False)

        stats = indexer.reindex_all()

        assert stats.documents_processed == 0


class TestProgressPrinter:
    """Tests for progress printer utility."""

    def test_progress_printer_output(self, capsys):
        """Test that progress printer produces output."""
        semantic_progress_printer(5, 10, "Processing doc.pdf")

        captured = capsys.readouterr()
        assert "50.0%" in captured.out
        assert "5/10" in captured.out
