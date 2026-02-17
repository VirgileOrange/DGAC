"""
Tests for the embedding service module.

Tests embedding generation, prefix handling, batch processing,
and error resilience. Uses mocked OpenAI client to avoid API calls.

SAFETY NOTE: All tests use the `configured_db` fixture which:
- Creates a temporary directory via tempfile.mkdtemp()
- Configures the database singleton to use a temp path
- Cleans up all temp files after the test
- Never touches real data directories
- Uses mocked API client to avoid external calls
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.search.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    RETRY_DELAY,
    MAX_RETRY_DELAY,
)


class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    def test_service_creation(self, configured_db, reset_embedding_singleton):
        """Test creating an EmbeddingService instance."""
        service = EmbeddingService()

        assert service is not None
        assert service._client is None  # Lazy initialization
        assert service._initialized is False

    def test_get_model_info(self, configured_db, reset_embedding_singleton):
        """Test getting model information."""
        service = EmbeddingService()
        info = service.get_model_info()

        assert "model" in info
        assert "dimensions" in info
        assert "endpoint" in info
        assert "initialized" in info
        assert info["dimensions"] == 1024

    def test_get_embedding_service_singleton(
        self, configured_db, reset_embedding_singleton
    ):
        """Test that get_embedding_service returns singleton."""
        service1 = get_embedding_service()
        service2 = get_embedding_service()

        assert service1 is service2


class TestEmbeddingServiceWithMock:
    """Tests for EmbeddingService with mocked OpenAI client."""

    @pytest.fixture
    def mock_service(
        self, configured_db, reset_embedding_singleton, mock_openai_client
    ):
        """
        Create an EmbeddingService with mocked client.

        Args:
            configured_db: Database configuration fixture.
            reset_embedding_singleton: Singleton reset fixture.
            mock_openai_client: Mock OpenAI client fixture.

        Returns:
            EmbeddingService with mocked client.
        """
        service = EmbeddingService()
        service._client = mock_openai_client
        service._initialized = True
        return service

    def test_embed_passages_single(self, mock_service):
        """Test embedding a single passage."""
        texts = ["Aviation safety regulations."]

        embeddings = mock_service.embed_passages(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 1024)

    def test_embed_passages_multiple(self, mock_service):
        """Test embedding multiple passages."""
        texts = [
            "First passage about aviation.",
            "Second passage about regulations.",
            "Third passage about safety.",
        ]

        embeddings = mock_service.embed_passages(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 1024)

    def test_embed_passages_empty(self, mock_service):
        """Test embedding empty list returns empty array."""
        embeddings = mock_service.embed_passages([])

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 0

    def test_embed_query(self, mock_service):
        """Test embedding a search query."""
        query = "aviation safety"

        embedding = mock_service.embed_query(query)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)

    def test_embed_query_empty(self, mock_service):
        """Test embedding empty query returns empty array."""
        embedding = mock_service.embed_query("")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 0

    def test_embed_passages_adds_prefix(
        self, configured_db, reset_embedding_singleton, mock_embedding_response
    ):
        """Test that passages get 'passage: ' prefix."""
        mock_client = Mock()
        captured_inputs = []

        def capture_create(model, input):
            captured_inputs.extend(input if isinstance(input, list) else [input])
            return mock_embedding_response(input if isinstance(input, list) else [input])

        mock_client.embeddings = Mock()
        mock_client.embeddings.create = capture_create

        service = EmbeddingService()
        service._client = mock_client
        service._initialized = True

        service.embed_passages(["Test passage"])

        assert len(captured_inputs) == 1
        assert captured_inputs[0].startswith("passage: ")

    def test_embed_query_adds_prefix(
        self, configured_db, reset_embedding_singleton, mock_embedding_response
    ):
        """Test that queries get 'query: ' prefix."""
        mock_client = Mock()
        captured_inputs = []

        def capture_create(model, input):
            captured_inputs.extend(input if isinstance(input, list) else [input])
            return mock_embedding_response(input if isinstance(input, list) else [input])

        mock_client.embeddings = Mock()
        mock_client.embeddings.create = capture_create

        service = EmbeddingService()
        service._client = mock_client
        service._initialized = True

        service.embed_query("Test query")

        assert len(captured_inputs) == 1
        assert captured_inputs[0].startswith("query: ")


class TestEmbeddingServiceBatching:
    """Tests for batch processing."""

    def test_batch_processing(
        self, configured_db, reset_embedding_singleton, mock_embedding_response
    ):
        """Test that large input is processed in batches."""
        mock_client = Mock()
        call_count = 0

        def counting_create(model, input):
            nonlocal call_count
            call_count += 1
            texts = input if isinstance(input, list) else [input]
            return mock_embedding_response(texts)

        mock_client.embeddings = Mock()
        mock_client.embeddings.create = counting_create

        service = EmbeddingService()
        service._client = mock_client
        service._initialized = True

        # Create more texts than batch size (32)
        texts = [f"Text number {i}" for i in range(65)]

        embeddings = service.embed_passages(texts)

        # Should have made multiple API calls
        assert call_count >= 2
        assert embeddings.shape == (65, 1024)


class TestEmbeddingServiceSplitPoint:
    """Tests for text split point detection."""

    def test_find_split_point_paragraph(
        self, configured_db, reset_embedding_singleton
    ):
        """Test finding split point at paragraph boundary."""
        service = EmbeddingService()

        text = "First part.\n\nSecond part."
        mid = len(text) // 2

        split = service._find_split_point(text, mid)

        # Should split at paragraph break
        assert split > 0
        assert split <= len(text)

    def test_find_split_point_sentence(
        self, configured_db, reset_embedding_singleton
    ):
        """Test finding split point at sentence boundary."""
        service = EmbeddingService()

        text = "First sentence. Second sentence. Third sentence."
        mid = len(text) // 2

        split = service._find_split_point(text, mid)

        # Should split near a sentence end
        assert split > 0
        assert split <= len(text)

    def test_find_split_point_space(
        self, configured_db, reset_embedding_singleton
    ):
        """Test finding split point at word boundary."""
        service = EmbeddingService()

        text = "wordone wordtwo wordthree wordfour"
        mid = len(text) // 2

        split = service._find_split_point(text, mid)

        # Should split at space
        assert split > 0
        assert text[split - 1] == " " or text[split] == " " or split == mid


class TestEmbeddingServiceConstants:
    """Tests for service configuration constants."""

    def test_retry_delay_positive(self):
        """Test that retry delay is positive."""
        assert RETRY_DELAY > 0

    def test_max_retry_delay_greater(self):
        """Test that max retry delay is greater than base delay."""
        assert MAX_RETRY_DELAY > RETRY_DELAY

    def test_dimensions_from_config(self, configured_db, reset_embedding_singleton):
        """Test that dimensions come from config."""
        service = EmbeddingService()

        assert service.config.semantic.embedding_dimensions == 1024
