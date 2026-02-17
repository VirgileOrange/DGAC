"""
Embedding service for generating vector representations of text.

Wraps the multilingual-e5-large model via OpenAI-compatible API
with proper prefix handling for documents and queries.

Includes resilience features:
- Unlimited retry on server errors (500) with exponential backoff
- Auto-split chunks that exceed token limits
"""

import time
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI, APIError, APIStatusError

from ..core import get_config, get_logger

logger = get_logger(__name__)

_embedding_service: Optional["EmbeddingService"] = None

# Retry configuration for server errors (500)
RETRY_DELAY = 60  # Base delay between retries (seconds)
MAX_RETRY_DELAY = 300  # Max delay (5 minutes)
STATUS_LOG_INTERVAL = 5  # Log status every N retries


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI-compatible API.

    Implements singleton pattern to avoid multiple client initializations.
    Handles E5-specific prefixes for optimal embedding quality.

    Resilience features:
    - Unlimited retries on 500 errors (waits for server to recover)
    - Auto-splits texts that exceed token limits
    """

    def __init__(self):
        """Initialize the embedding service with configuration."""
        self.config = get_config()
        self._client: Optional[OpenAI] = None
        self._initialized = False

    def _ensure_client(self) -> None:
        """Lazily initialize the OpenAI client."""
        if self._client is not None:
            return

        self._client = OpenAI(
            base_url=self.config.semantic.endpoint,
            api_key=self.config.semantic.api_key
        )
        self._initialized = True
        logger.info(f"Embedding service initialized with model: {self.config.semantic.embedding_model}")

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for document passages.

        Adds 'passage: ' prefix as required by E5 models.

        Args:
            texts: List of text passages to embed.

        Returns:
            numpy array of shape (n, embedding_dimensions).
        """
        if not texts:
            return np.array([])

        self._ensure_client()

        prefixed_texts = [f"passage: {text}" for text in texts]
        embeddings = self._embed_batch(prefixed_texts)

        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.

        Adds 'query: ' prefix as required by E5 models.

        Args:
            query: Search query text.

        Returns:
            numpy array of shape (embedding_dimensions,).
        """
        if not query:
            return np.array([])

        self._ensure_client()

        prefixed_query = f"query: {query}"
        embeddings = self._embed_batch([prefixed_query])

        return np.array(embeddings[0])

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Handles batching according to configured batch size.
        Includes retry logic for server errors and auto-split for token limits.

        Args:
            texts: List of texts to embed (already prefixed).

        Returns:
            List of embedding vectors.
        """
        batch_size = self.config.semantic.embedding_batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch_with_resilience(batch)
            all_embeddings.extend(batch_embeddings)

            if len(texts) > batch_size:
                logger.debug(f"Embedded batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

        return all_embeddings

    def _embed_batch_with_resilience(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch with unlimited retry on server errors and auto-split on token limits.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        retry_count = 0
        start_time = time.time()

        while True:
            try:
                response = self._client.embeddings.create(
                    model=self.config.semantic.embedding_model,
                    input=texts
                )
                return [item.embedding for item in response.data]

            except APIStatusError as e:
                status_code = e.status_code
                error_message = str(e)

                # Handle token limit exceeded (400 error)
                if status_code == 400 and "ContextWindowExceeded" in error_message:
                    logger.warning(f"Token limit exceeded, splitting {len(texts)} texts")
                    return self._embed_with_split(texts)

                # Handle server errors (500, 502, 503, 504) - retry indefinitely
                if status_code >= 500:
                    retry_count += 1
                    delay = min(RETRY_DELAY * (1 + (retry_count - 1) * 0.5), MAX_RETRY_DELAY)
                    elapsed = int(time.time() - start_time)

                    # Log status periodically
                    if retry_count % STATUS_LOG_INTERVAL == 1:
                        logger.warning(
                            f"Server error {status_code} - waiting for recovery. "
                            f"Retry #{retry_count}, elapsed: {elapsed}s. "
                            f"Next retry in {int(delay)}s. (Ctrl+C to cancel)"
                        )
                        print(
                            f"\r[Server unavailable] Retry #{retry_count}, "
                            f"waiting {int(delay)}s... (elapsed: {elapsed}s, Ctrl+C to cancel)",
                            end="", flush=True
                        )

                    time.sleep(delay)
                    continue

                # Other client errors - don't retry
                raise

            except APIError as e:
                # Generic API error - retry with backoff
                retry_count += 1
                delay = min(RETRY_DELAY * (1 + (retry_count - 1) * 0.5), MAX_RETRY_DELAY)
                elapsed = int(time.time() - start_time)

                if retry_count % STATUS_LOG_INTERVAL == 1:
                    logger.warning(
                        f"API error - retrying. Retry #{retry_count}, elapsed: {elapsed}s. "
                        f"Error: {str(e)[:100]}"
                    )

                time.sleep(delay)
                continue

            except KeyboardInterrupt:
                logger.info("Embedding interrupted by user")
                raise

    def _embed_with_split(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts by splitting those that exceed token limits.

        For each text, if embedding fails due to token limit,
        split it in half and embed each half separately,
        then average the embeddings.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []

        for text in texts:
            embedding = self._embed_single_with_split(text)
            all_embeddings.append(embedding)

        return all_embeddings

    def _embed_single_with_split(self, text: str, depth: int = 0) -> List[float]:
        """
        Embed a single text, splitting if it exceeds token limit.

        Args:
            text: Text to embed.
            depth: Recursion depth (to prevent infinite splitting).

        Returns:
            Embedding vector.
        """
        max_depth = 4  # Max splits: original -> 2 -> 4 -> 8 -> 16 pieces

        if depth > max_depth:
            # Too many splits - truncate aggressively
            truncated = text[:500]
            logger.warning(f"Max split depth reached, truncating to {len(truncated)} chars")
            text = truncated

        retry_count = 0
        start_time = time.time()

        while True:
            try:
                response = self._client.embeddings.create(
                    model=self.config.semantic.embedding_model,
                    input=[text]
                )
                return response.data[0].embedding

            except APIStatusError as e:
                if e.status_code == 400 and "ContextWindowExceeded" in str(e):
                    if depth >= max_depth:
                        # Last resort: truncate
                        truncated = text[:500]
                        logger.warning(f"Truncating text to {len(truncated)} chars after max splits")
                        response = self._client.embeddings.create(
                            model=self.config.semantic.embedding_model,
                            input=[truncated]
                        )
                        return response.data[0].embedding

                    # Split text in half at a good boundary
                    mid = len(text) // 2
                    split_point = self._find_split_point(text, mid)

                    first_half = text[:split_point].strip()
                    second_half = text[split_point:].strip()

                    logger.debug(f"Splitting text at depth {depth}: {len(text)} -> {len(first_half)} + {len(second_half)}")

                    # Recursively embed each half
                    emb1 = self._embed_single_with_split(first_half, depth + 1)
                    emb2 = self._embed_single_with_split(second_half, depth + 1)

                    # Average the embeddings
                    return [(a + b) / 2 for a, b in zip(emb1, emb2)]

                # Server errors - retry indefinitely
                if e.status_code >= 500:
                    retry_count += 1
                    delay = min(RETRY_DELAY * (1 + (retry_count - 1) * 0.5), MAX_RETRY_DELAY)

                    if retry_count % STATUS_LOG_INTERVAL == 1:
                        elapsed = int(time.time() - start_time)
                        logger.warning(f"Server error {e.status_code}, retry #{retry_count}, elapsed: {elapsed}s")

                    time.sleep(delay)
                    continue

                # Other errors - re-raise
                raise

            except APIError as e:
                retry_count += 1
                delay = min(RETRY_DELAY * (1 + (retry_count - 1) * 0.5), MAX_RETRY_DELAY)
                time.sleep(delay)
                continue

            except KeyboardInterrupt:
                logger.info("Embedding interrupted by user")
                raise

    def _find_split_point(self, text: str, mid: int) -> int:
        """
        Find a good split point near the middle of the text.

        Prefers paragraph > sentence > word boundaries.

        Args:
            text: Text to split.
            mid: Target middle position.

        Returns:
            Best split position.
        """
        search_range = min(200, mid // 2)
        search_start = max(0, mid - search_range)
        search_end = min(len(text), mid + search_range)
        search_region = text[search_start:search_end]

        # Look for paragraph break
        para_pos = search_region.rfind('\n\n')
        if para_pos != -1:
            return search_start + para_pos + 2

        # Look for sentence end
        for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            pos = search_region.rfind(sep)
            if pos != -1:
                return search_start + pos + len(sep)

        # Look for any newline
        newline_pos = search_region.rfind('\n')
        if newline_pos != -1:
            return search_start + newline_pos + 1

        # Look for space
        space_pos = search_region.rfind(' ')
        if space_pos != -1:
            return search_start + space_pos + 1

        # Fall back to exact middle
        return mid

    def get_model_info(self) -> Dict:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model": self.config.semantic.embedding_model,
            "dimensions": self.config.semantic.embedding_dimensions,
            "endpoint": self.config.semantic.endpoint,
            "initialized": self._initialized
        }


def get_embedding_service() -> EmbeddingService:
    """
    Get the singleton EmbeddingService instance.

    Returns:
        Global EmbeddingService instance.
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


if __name__ == "__main__":
    service = get_embedding_service()

    print("Embedding Service Info:")
    info = service.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(f"\nResilience settings:")
    print(f"  Server error retry: unlimited (waits for recovery)")
    print(f"  Base retry delay: {RETRY_DELAY}s")
    print(f"  Max retry delay: {MAX_RETRY_DELAY}s")
    print(f"  Auto-split on token limit: enabled (up to 4 levels)")

    print("\nNote: Actual embedding requires valid API credentials in config.")
