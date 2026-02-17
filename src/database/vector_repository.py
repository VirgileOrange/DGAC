"""
Vector repository for storing and querying embeddings.

Provides storage and k-nearest-neighbor search using sqlite-vec extension.
"""

import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..core import get_config, get_logger
from ..extraction.semantic_chunker import SemanticChunk
from .connection import get_connection, get_cursor
from .schema import _load_vec_extension, is_vec_extension_available

logger = get_logger(__name__)


@dataclass
class VectorSearchResult:
    """
    Result from a vector similarity search.

    Attributes:
        chunk_id: Unique chunk identifier.
        document_id: Parent document ID.
        page_num: Source page number.
        position: Position within page.
        content: Text content of the chunk.
        similarity: Cosine similarity score (higher is better).
    """
    chunk_id: str
    document_id: int
    page_num: int
    position: int
    content: str
    similarity: float


class VectorRepository:
    """
    Repository for vector embedding storage and retrieval.

    Uses sqlite-vec for efficient k-nearest-neighbor search.
    """

    def __init__(self):
        """Initialize the vector repository."""
        self.config = get_config()
        self.dimensions = self.config.semantic.embedding_dimensions

    def _ensure_vec_extension(self, conn) -> bool:
        """
        Load sqlite-vec extension into this connection.

        Note: Each connection needs the extension loaded separately.
        The global _vec_extension_available tracks package availability,
        but sqlite_vec.load() must be called per connection.

        Args:
            conn: SQLite connection.

        Returns:
            True if extension is available.
        """
        # Always try to load - each connection needs the extension loaded
        return _load_vec_extension(conn)

    def store_chunk(
        self,
        chunk: SemanticChunk,
        embedding: np.ndarray
    ) -> None:
        """
        Store a single chunk with its embedding.

        Args:
            chunk: SemanticChunk object with metadata.
            embedding: numpy array of embedding vector.
        """
        embedding_blob = self._array_to_blob(embedding)

        with get_cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO chunks_metadata
                (chunk_id, document_id, page_num, position, content, char_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.document_id,
                chunk.page_num,
                chunk.position,
                chunk.content,
                chunk.char_count
            ))

            cur.execute("""
                INSERT OR REPLACE INTO chunks_vec
                (chunk_id, embedding)
                VALUES (?, ?)
            """, (chunk.chunk_id, embedding_blob))

        with get_connection() as conn:
            if self._ensure_vec_extension(conn):
                conn.execute(
                    "DELETE FROM chunks_vec_idx WHERE chunk_id = ?",
                    (chunk.chunk_id,)
                )
                conn.execute("""
                    INSERT INTO chunks_vec_idx (chunk_id, embedding)
                    VALUES (?, ?)
                """, (chunk.chunk_id, embedding_blob))
                conn.commit()

    def store_chunks_batch(
        self,
        chunks: List[SemanticChunk],
        embeddings: np.ndarray
    ) -> int:
        """
        Store multiple chunks with their embeddings.

        Args:
            chunks: List of SemanticChunk objects.
            embeddings: numpy array of shape (n, dimensions).

        Returns:
            Number of chunks stored.
        """
        if not chunks or len(embeddings) == 0:
            return 0

        metadata_rows = []
        vec_rows = []

        for chunk, embedding in zip(chunks, embeddings):
            embedding_blob = self._array_to_blob(embedding)

            metadata_rows.append((
                chunk.chunk_id,
                chunk.document_id,
                chunk.page_num,
                chunk.position,
                chunk.content,
                chunk.char_count
            ))

            vec_rows.append((chunk.chunk_id, embedding_blob))

        with get_cursor() as cur:
            cur.executemany("""
                INSERT OR REPLACE INTO chunks_metadata
                (chunk_id, document_id, page_num, position, content, char_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, metadata_rows)

            cur.executemany("""
                INSERT OR REPLACE INTO chunks_vec
                (chunk_id, embedding)
                VALUES (?, ?)
            """, vec_rows)

        with get_connection() as conn:
            if self._ensure_vec_extension(conn):
                for chunk_id, embedding_blob in vec_rows:
                    conn.execute(
                        "DELETE FROM chunks_vec_idx WHERE chunk_id = ?",
                        (chunk_id,)
                    )
                    conn.execute("""
                        INSERT INTO chunks_vec_idx (chunk_id, embedding)
                        VALUES (?, ?)
                    """, (chunk_id, embedding_blob))
                conn.commit()

        logger.debug(f"Stored {len(chunks)} chunks with embeddings")
        return len(chunks)

    def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10
    ) -> List[VectorSearchResult]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query vector as numpy array.
            limit: Maximum number of results.

        Returns:
            List of VectorSearchResult ordered by similarity (descending).
        """
        query_blob = self._array_to_blob(query_embedding)

        with get_connection() as conn:
            if not self._ensure_vec_extension(conn):
                logger.error("sqlite-vec not available for search")
                return []

            rows = conn.execute("""
                SELECT
                    v.chunk_id,
                    v.distance,
                    m.document_id,
                    m.page_num,
                    m.position,
                    m.content
                FROM chunks_vec_idx v
                JOIN chunks_metadata m ON v.chunk_id = m.chunk_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
            """, (query_blob, limit)).fetchall()

        results = []
        for row in rows:
            distance = row["distance"]
            similarity = 1.0 / (1.0 + distance)

            results.append(VectorSearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                page_num=row["page_num"],
                position=row["position"],
                content=row["content"],
                similarity=similarity
            ))

        return results

    def delete_document_chunks(self, document_id: int) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID to delete chunks for.

        Returns:
            Number of chunks deleted.
        """
        with get_connection() as conn:
            chunk_ids = conn.execute(
                "SELECT chunk_id FROM chunks_metadata WHERE document_id = ?",
                (document_id,)
            ).fetchall()

            if not chunk_ids:
                return 0

            chunk_id_list = [row["chunk_id"] for row in chunk_ids]

        with get_cursor() as cur:
            placeholders = ",".join("?" * len(chunk_id_list))

            cur.execute(
                f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})",
                chunk_id_list
            )

            cur.execute(
                f"DELETE FROM chunks_metadata WHERE chunk_id IN ({placeholders})",
                chunk_id_list
            )

        with get_connection() as conn:
            if self._ensure_vec_extension(conn):
                placeholders = ",".join("?" * len(chunk_id_list))
                conn.execute(
                    f"DELETE FROM chunks_vec_idx WHERE chunk_id IN ({placeholders})",
                    chunk_id_list
                )
                conn.commit()

        logger.debug(f"Deleted {len(chunk_id_list)} chunks for document {document_id}")
        return len(chunk_id_list)

    def get_chunk_count(self) -> int:
        """
        Get total number of indexed chunks.

        Returns:
            Total chunk count.
        """
        with get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM chunks_metadata").fetchone()
            return row["count"] if row else 0

    def get_document_chunk_count(self, document_id: int) -> int:
        """
        Get chunk count for a specific document.

        Args:
            document_id: Document ID to count chunks for.

        Returns:
            Number of chunks for the document.
        """
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM chunks_metadata WHERE document_id = ?",
                (document_id,)
            ).fetchone()
            return row["count"] if row else 0

    def get_chunk_by_id(self, chunk_id: str) -> Optional[VectorSearchResult]:
        """
        Retrieve a chunk by its ID.

        Args:
            chunk_id: Unique chunk identifier.

        Returns:
            VectorSearchResult or None if not found.
        """
        with get_connection() as conn:
            row = conn.execute("""
                SELECT chunk_id, document_id, page_num, position, content
                FROM chunks_metadata
                WHERE chunk_id = ?
            """, (chunk_id,)).fetchone()

            if not row:
                return None

            return VectorSearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                page_num=row["page_num"],
                position=row["position"],
                content=row["content"],
                similarity=1.0
            )

    def _array_to_blob(self, arr: np.ndarray) -> bytes:
        """
        Convert numpy array to bytes for SQLite storage.

        Args:
            arr: numpy array of floats.

        Returns:
            Packed bytes representation.
        """
        return struct.pack(f"{len(arr)}f", *arr.astype(np.float32))

    def _blob_to_array(self, blob: bytes) -> np.ndarray:
        """
        Convert bytes back to numpy array.

        Args:
            blob: Packed bytes from database.

        Returns:
            numpy array of floats.
        """
        count = len(blob) // 4
        return np.array(struct.unpack(f"{count}f", blob))


if __name__ == "__main__":
    from .schema import init_schema, init_vector_index

    print("Initializing schema...")
    init_schema()

    print("\nInitializing vector index...")
    success = init_vector_index()
    if success:
        print("Vector index created.")
    else:
        print("Vector index already exists or extension unavailable.")

    print("\nVector repository ready.")

    repo = VectorRepository()
    print(f"Total chunks: {repo.get_chunk_count()}")
