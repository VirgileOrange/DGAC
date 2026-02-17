"""
Semantic indexer for generating and storing document embeddings.

Orchestrates the semantic indexing pipeline: chunking pages,
generating embeddings, and storing in the vector index.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from ..core import get_config, get_logger
from ..database import get_connection, init_schema, init_vector_index
from ..database.vector_repository import VectorRepository
from ..extraction.semantic_chunker import SemanticChunker
from ..search.embedding_service import get_embedding_service

logger = get_logger(__name__)


@dataclass
class SemanticIndexingStats:
    """Statistics from semantic indexing run."""
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    errors: List[str] = field(default_factory=list)


class SemanticIndexer:
    """
    Orchestrates the semantic indexing pipeline.

    Handles chunking, embedding generation, and vector storage
    for document pages.
    """

    def __init__(
        self,
        progress_callback: Callable[[int, int, str], None] = None,
        enabled: Optional[bool] = None
    ):
        """
        Initialize the semantic indexer.

        Args:
            progress_callback: Optional callback(current, total, message)
                              for progress updates.
            enabled: Override config semantic.enabled setting.
                    If None, uses config value.
        """
        self.config = get_config()
        self.progress_callback = progress_callback

        if enabled is None:
            self.enabled = self.config.semantic.enabled
        else:
            self.enabled = enabled

        self.chunker = SemanticChunker()
        self.embedding_service = get_embedding_service()
        self.vector_repo = VectorRepository()

        self.batch_size = self.config.semantic.embedding_batch_size

    def ensure_schema(self) -> None:
        """Ensure semantic search tables exist."""
        init_schema()
        init_vector_index()

    def index_document(
        self,
        doc_id: int,
        pages: List[Tuple[int, str]],
        filename: str
    ) -> int:
        """
        Index a single document for semantic search.

        Args:
            doc_id: Document ID from the database.
            pages: List of (page_num, text) tuples.
            filename: Document filename for logging.

        Returns:
            Number of chunks indexed.
        """
        if not self.enabled:
            return 0

        chunks = self.chunker.chunk_document(pages, doc_id)

        if not chunks:
            logger.debug(f"No chunks created for document {filename}")
            return 0

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.embed_passages(texts)

        self.vector_repo.store_chunks_batch(chunks, embeddings)

        logger.debug(f"Indexed {len(chunks)} chunks for {filename}")
        return len(chunks)

    def index_documents_batch(
        self,
        documents: List[dict]
    ) -> SemanticIndexingStats:
        """
        Index multiple documents for semantic search.

        Args:
            documents: List of dicts with keys:
                      - doc_id: Document database ID
                      - pages: List of (page_num, text) tuples
                      - filename: Document filename

        Returns:
            SemanticIndexingStats with indexing results.
        """
        stats = SemanticIndexingStats()

        if not self.enabled:
            logger.info("Semantic indexing is disabled")
            return stats

        self.ensure_schema()

        total = len(documents)

        for i, doc in enumerate(documents):
            doc_id = doc["doc_id"]
            pages = doc["pages"]
            filename = doc["filename"]

            if self.progress_callback:
                self.progress_callback(i + 1, total, f"Indexing {filename}")

            try:
                existing_chunks = self.vector_repo.get_document_chunk_count(doc_id)
                if existing_chunks > 0:
                    stats.documents_skipped += 1
                    continue

                chunks_count = self.index_document(doc_id, pages, filename)

                if chunks_count > 0:
                    stats.documents_processed += 1
                    stats.chunks_created += chunks_count
                    stats.embeddings_generated += chunks_count

            except Exception as e:
                stats.documents_failed += 1
                error_msg = f"{filename}: {str(e)}"
                stats.errors.append(error_msg)
                logger.error(f"Failed to index {filename}: {e}")

        logger.info(
            f"Semantic indexing complete: {stats.documents_processed} documents, "
            f"{stats.chunks_created} chunks, {stats.documents_failed} failures"
        )

        return stats

    def reindex_all(self) -> SemanticIndexingStats:
        """
        Reindex all documents in the database.

        Fetches all documents from FTS5 index and creates
        semantic embeddings for each.

        Returns:
            SemanticIndexingStats with indexing results.
        """
        stats = SemanticIndexingStats()

        if not self.enabled:
            logger.info("Semantic indexing is disabled")
            return stats

        self.ensure_schema()

        with get_connection() as conn:
            doc_rows = conn.execute("""
                SELECT DISTINCT filepath, filename
                FROM documents
                ORDER BY filepath
            """).fetchall()

        total_docs = len(doc_rows)
        logger.info(f"Starting semantic reindex of {total_docs} documents")

        for i, doc_row in enumerate(doc_rows):
            filepath = doc_row["filepath"]
            filename = doc_row["filename"]

            if self.progress_callback:
                self.progress_callback(i + 1, total_docs, f"Reindexing {filename}")

            try:
                with get_connection() as conn:
                    page_rows = conn.execute("""
                        SELECT id, page_num, content
                        FROM documents
                        WHERE filepath = ?
                        ORDER BY page_num
                    """, (filepath,)).fetchall()

                if not page_rows:
                    continue

                doc_id = page_rows[0]["id"]
                pages = [(row["page_num"], row["content"]) for row in page_rows]

                self.vector_repo.delete_document_chunks(doc_id)

                chunks_count = self.index_document(doc_id, pages, filename)

                if chunks_count > 0:
                    stats.documents_processed += 1
                    stats.chunks_created += chunks_count
                    stats.embeddings_generated += chunks_count

            except Exception as e:
                stats.documents_failed += 1
                error_msg = f"{filename}: {str(e)}"
                stats.errors.append(error_msg)
                logger.error(f"Failed to reindex {filename}: {e}")

        logger.info(
            f"Semantic reindex complete: {stats.documents_processed} documents, "
            f"{stats.chunks_created} chunks"
        )

        return stats

    def delete_document(self, doc_id: int) -> int:
        """
        Remove a document from the semantic index.

        Args:
            doc_id: Document ID to remove.

        Returns:
            Number of chunks deleted.
        """
        return self.vector_repo.delete_document_chunks(doc_id)

    def get_stats(self) -> dict:
        """
        Get semantic index statistics.

        Returns:
            Dictionary with index statistics.
        """
        chunk_count = self.vector_repo.get_chunk_count()
        model_info = self.embedding_service.get_model_info()

        return {
            "total_chunks": chunk_count,
            "embedding_model": model_info["model"],
            "embedding_dimensions": model_info["dimensions"],
            "semantic_enabled": self.config.semantic.enabled
        }


def semantic_progress_printer(current: int, total: int, message: str) -> None:
    """Simple progress callback that prints to console."""
    percent = (current / total) * 100 if total > 0 else 0
    print(f"\r[{percent:5.1f}%] {current}/{total} - {message[:50]:<50}", end="", flush=True)


if __name__ == "__main__":
    import sys

    indexer = SemanticIndexer(progress_callback=semantic_progress_printer)

    print("Semantic Indexer Stats:")
    stats = indexer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if "--reindex" in sys.argv:
        print("\nStarting full reindex...")
        print("-" * 60)

        result = indexer.reindex_all()

        print("\n" + "-" * 60)
        print("Reindex Summary:")
        print(f"  Documents processed: {result.documents_processed}")
        print(f"  Documents skipped:   {result.documents_skipped}")
        print(f"  Documents failed:    {result.documents_failed}")
        print(f"  Chunks created:      {result.chunks_created}")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:
                print(f"  - {error}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more")
