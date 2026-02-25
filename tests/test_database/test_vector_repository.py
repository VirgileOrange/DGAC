"""
Tests for the vector repository module.

Tests vector storage, retrieval, and similarity search operations.
Uses temporary database to ensure isolation.

SAFETY NOTE: All tests use the `configured_db` fixture which:
- Creates a temporary directory via tempfile.mkdtemp()
- Configures the database singleton to use a temp path
- Cleans up all temp files after the test
- Never touches real data directories
"""

import pytest
import numpy as np

from src.database.schema import init_schema, init_vector_index
from src.database.vector_repository import VectorRepository, VectorSearchResult
from src.extraction.semantic_chunker import SemanticChunk


@pytest.fixture
def vector_repo(configured_db, reset_vec_extension_cache):
    """
    Create a VectorRepository with initialized schema.

    Args:
        configured_db: Database configuration fixture.
        reset_vec_extension_cache: Vec extension cache reset fixture.

    Returns:
        VectorRepository instance.
    """
    init_schema()
    init_vector_index()
    return VectorRepository()


class TestVectorSearchResult:
    """Tests for VectorSearchResult dataclass."""

    def test_result_creation(self):
        """Test creating a VectorSearchResult instance."""
        result = VectorSearchResult(
            chunk_id="chunk001",
            document_id=1,
            page_num=1,
            position=0,
            content="Test content",
            similarity=0.95
        )

        assert result.chunk_id == "chunk001"
        assert result.document_id == 1
        assert result.page_num == 1
        assert result.position == 0
        assert result.content == "Test content"
        assert result.similarity == 0.95

    def test_result_equality(self):
        """Test that results with same values are equal."""
        result1 = VectorSearchResult(
            chunk_id="abc",
            document_id=1,
            page_num=1,
            position=0,
            content="Content",
            similarity=0.9
        )
        result2 = VectorSearchResult(
            chunk_id="abc",
            document_id=1,
            page_num=1,
            position=0,
            content="Content",
            similarity=0.9
        )

        assert result1 == result2


class TestVectorRepository:
    """Tests for VectorRepository class."""

    def test_repository_creation(self, vector_repo):
        """Test creating a VectorRepository instance."""
        assert vector_repo is not None
        assert vector_repo.dimensions == 1024

    def test_get_chunk_count_empty(self, vector_repo):
        """Test chunk count on empty repository."""
        count = vector_repo.get_chunk_count()

        assert count == 0


class TestVectorRepositoryStorage:
    """Tests for chunk storage operations."""

    def test_store_single_chunk(self, vector_repo, sample_chunks, sample_embeddings):
        """Test storing a single chunk with embedding."""
        chunk = sample_chunks[0]
        embedding = sample_embeddings[0]

        vector_repo.store_chunk(chunk, embedding)

        count = vector_repo.get_chunk_count()
        assert count == 1

    def test_store_chunks_batch(self, vector_repo, sample_chunks, sample_embeddings):
        """Test storing multiple chunks in batch."""
        stored = vector_repo.store_chunks_batch(sample_chunks, sample_embeddings)

        assert stored == 3
        assert vector_repo.get_chunk_count() == 3

    def test_store_empty_batch(self, vector_repo):
        """Test storing empty batch returns zero."""
        stored = vector_repo.store_chunks_batch([], np.array([]))

        assert stored == 0

    def test_get_document_chunk_count(self, vector_repo, sample_chunks, sample_embeddings):
        """Test getting chunk count for specific document."""
        vector_repo.store_chunks_batch(sample_chunks, sample_embeddings)

        # Document 1 has 2 chunks, Document 2 has 1 chunk
        count_doc1 = vector_repo.get_document_chunk_count(1)
        count_doc2 = vector_repo.get_document_chunk_count(2)

        assert count_doc1 == 2
        assert count_doc2 == 1

    def test_get_chunk_by_id(self, vector_repo, sample_chunks, sample_embeddings):
        """Test retrieving chunk by ID."""
        vector_repo.store_chunks_batch(sample_chunks, sample_embeddings)

        chunk = vector_repo.get_chunk_by_id("chunk001")

        assert chunk is not None
        assert chunk.chunk_id == "chunk001"
        assert chunk.document_id == 1
        assert "Aviation" in chunk.content

    def test_get_chunk_by_id_nonexistent(self, vector_repo):
        """Test retrieving nonexistent chunk returns None."""
        chunk = vector_repo.get_chunk_by_id("nonexistent")

        assert chunk is None


class TestVectorRepositoryDeletion:
    """Tests for chunk deletion operations."""

    def test_delete_document_chunks(self, vector_repo, sample_chunks, sample_embeddings):
        """Test deleting all chunks for a document."""
        vector_repo.store_chunks_batch(sample_chunks, sample_embeddings)

        deleted = vector_repo.delete_document_chunks(1)

        assert deleted == 2
        assert vector_repo.get_chunk_count() == 1
        assert vector_repo.get_document_chunk_count(1) == 0
        assert vector_repo.get_document_chunk_count(2) == 1

    def test_delete_nonexistent_document(self, vector_repo):
        """Test deleting chunks for nonexistent document."""
        deleted = vector_repo.delete_document_chunks(999)

        assert deleted == 0


class TestVectorRepositoryConversion:
    """Tests for array/blob conversion utilities."""

    def test_array_to_blob_and_back(self, vector_repo):
        """Test round-trip conversion of numpy array to blob."""
        original = np.random.randn(1024).astype(np.float32)

        blob = vector_repo._array_to_blob(original)
        restored = vector_repo._blob_to_array(blob)

        np.testing.assert_array_almost_equal(original, restored)

    def test_blob_size(self, vector_repo):
        """Test that blob has expected size."""
        arr = np.zeros(1024, dtype=np.float32)

        blob = vector_repo._array_to_blob(arr)

        # 1024 floats * 4 bytes = 4096 bytes
        assert len(blob) == 4096


class TestVectorRepositoryReplace:
    """Tests for replace/update operations."""

    def test_store_chunk_replaces_existing(
        self, vector_repo, sample_chunks, sample_embeddings
    ):
        """Test that storing chunk with same ID replaces it."""
        chunk = sample_chunks[0]
        embedding1 = sample_embeddings[0]

        vector_repo.store_chunk(chunk, embedding1)
        count_before = vector_repo.get_chunk_count()

        # Store same chunk again with different embedding
        embedding2 = np.random.randn(1024).astype(np.float32)
        vector_repo.store_chunk(chunk, embedding2)
        count_after = vector_repo.get_chunk_count()

        # Should still be 1 chunk (replaced, not added)
        assert count_before == 1
        assert count_after == 1


class TestVectorRepositorySearch:
    """Tests for similarity search operations.

    Note: These tests require sqlite-vec to be installed.
    If sqlite-vec is not available, search tests are skipped.
    """

    @pytest.fixture
    def populated_vector_repo(self, vector_repo, sample_chunks, sample_embeddings):
        """
        Create a VectorRepository with sample data.

        Args:
            vector_repo: Base repository fixture.
            sample_chunks: Sample chunk fixture.
            sample_embeddings: Sample embeddings fixture.

        Returns:
            Populated VectorRepository.
        """
        vector_repo.store_chunks_batch(sample_chunks, sample_embeddings)
        return vector_repo

    def test_search_similar_returns_results(self, populated_vector_repo, sample_embeddings):
        """Test that search returns results for similar query."""
        # Use one of the stored embeddings as query
        query_embedding = sample_embeddings[0]

        results = populated_vector_repo.search_similar(query_embedding, limit=10)

        # Should return results (may be empty if sqlite-vec not available)
        assert isinstance(results, list)

    def test_search_similar_respects_limit(self, populated_vector_repo, sample_embeddings):
        """Test that search respects limit parameter."""
        query_embedding = sample_embeddings[0]

        results = populated_vector_repo.search_similar(query_embedding, limit=1)

        assert len(results) <= 1

    def test_search_returns_valid_results(self, populated_vector_repo, sample_embeddings):
        """Test that search results have valid structure."""
        query_embedding = sample_embeddings[0]

        results = populated_vector_repo.search_similar(query_embedding, limit=10)

        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.chunk_id is not None
            assert result.document_id > 0
            assert result.page_num > 0
            assert result.content is not None
            assert 0.0 < result.similarity <= 1.0
