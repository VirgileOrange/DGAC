"""
Document repository for CRUD operations on the documents table.

Provides a clean interface for inserting, querying, and managing
indexed PDF documents.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from ..core import get_logger
from .connection import get_connection, get_cursor

logger = get_logger(__name__)


@dataclass
class Document:
    """Represents a single indexed document page."""
    id: int
    filepath: str
    filename: str
    page_num: int
    content: str
    relative_path: str
    file_hash: str
    indexed_at: datetime


class DocumentRepository:
    """
    Repository for document CRUD operations.

    Provides methods for inserting, querying, and deleting
    indexed document pages.
    """

    def insert(
        self,
        filepath: Union[str, Path],
        filename: str,
        page_num: int,
        content: str,
        relative_path: str = None,
        file_hash: str = None
    ) -> Optional[int]:
        """
        Insert a single document page.

        Uses INSERT OR IGNORE to skip duplicates.

        Args:
            filepath: Absolute path to the PDF file.
            filename: PDF filename.
            page_num: Page number (1-indexed).
            content: Extracted text content.
            relative_path: Path relative to data directory.
            file_hash: MD5 hash for change detection.

        Returns:
            Inserted row ID or None if duplicate.
        """
        with get_cursor() as cur:
            cur.execute("""
                INSERT OR IGNORE INTO documents
                (filepath, filename, page_num, content, relative_path, file_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (str(filepath), filename, page_num, content, relative_path, file_hash))

            return cur.lastrowid if cur.rowcount > 0 else None

    def insert_batch(self, documents: List[tuple]) -> int:
        """
        Insert multiple documents in a single transaction.

        Args:
            documents: List of tuples matching insert() parameters:
                       (filepath, filename, page_num, content, relative_path, file_hash)

        Returns:
            Number of rows inserted.
        """
        if not documents:
            return 0

        with get_cursor() as cur:
            cur.executemany("""
                INSERT OR IGNORE INTO documents
                (filepath, filename, page_num, content, relative_path, file_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, documents)

            return cur.rowcount

    def exists(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a file is already indexed.

        Args:
            filepath: Path to check.

        Returns:
            True if any pages exist for this file.
        """
        with get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE filepath = ? LIMIT 1",
                (str(filepath),)
            ).fetchone()
            return row is not None

    def get_by_id(self, doc_id: int) -> Optional[Document]:
        """
        Fetch a document by its ID.

        Args:
            doc_id: Document row ID.

        Returns:
            Document object or None.
        """
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?",
                (doc_id,)
            ).fetchone()

            if row:
                return self._row_to_document(row)
            return None

    def get_by_filepath(self, filepath: Union[str, Path]) -> List[Document]:
        """
        Fetch all pages for a file.

        Args:
            filepath: Path to the PDF file.

        Returns:
            List of Document objects ordered by page number.
        """
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE filepath = ? ORDER BY page_num",
                (str(filepath),)
            ).fetchall()

            return [self._row_to_document(row) for row in rows]

    def delete_by_filepath(self, filepath: Union[str, Path]) -> int:
        """
        Delete all pages for a file.

        Args:
            filepath: Path to the PDF file.

        Returns:
            Number of rows deleted.
        """
        with get_cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE filepath = ?",
                (str(filepath),)
            )
            deleted = cur.rowcount

        if deleted > 0:
            logger.debug(f"Deleted {deleted} pages for: {filepath}")

        return deleted

    def count(self) -> int:
        """
        Get total document page count.

        Returns:
            Total number of indexed pages.
        """
        with get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM documents").fetchone()
            return row["count"]

    def count_files(self) -> int:
        """
        Get total unique file count.

        Returns:
            Number of unique PDF files indexed.
        """
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT filepath) as count FROM documents"
            ).fetchone()
            return row["count"]

    def get_indexed_filepaths(self) -> set:
        """
        Get set of all indexed file paths.

        Returns:
            Set of filepath strings.
        """
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT filepath FROM documents"
            ).fetchall()
            return {row["filepath"] for row in rows}

    @staticmethod
    def _row_to_document(row) -> Document:
        """Convert a database row to a Document object."""
        return Document(
            id=row["id"],
            filepath=row["filepath"],
            filename=row["filename"],
            page_num=row["page_num"],
            content=row["content"],
            relative_path=row["relative_path"],
            file_hash=row["file_hash"],
            indexed_at=row["indexed_at"]
        )


if __name__ == "__main__":
    from .schema import init_schema, reset_schema

    init_schema()

    repo = DocumentRepository()

    print("Testing DocumentRepository...")

    doc_id = repo.insert(
        filepath="/test/sample.pdf",
        filename="sample.pdf",
        page_num=1,
        content="This is test content for page one.",
        relative_path="test/sample.pdf",
        file_hash="abc123"
    )
    print(f"Inserted document with ID: {doc_id}")

    batch = [
        ("/test/sample.pdf", "sample.pdf", 2, "Page two content.", "test/sample.pdf", "abc123"),
        ("/test/sample.pdf", "sample.pdf", 3, "Page three content.", "test/sample.pdf", "abc123"),
        ("/test/other.pdf", "other.pdf", 1, "Different document.", "test/other.pdf", "def456"),
    ]
    inserted = repo.insert_batch(batch)
    print(f"Batch inserted: {inserted} rows")

    print(f"File exists: {repo.exists('/test/sample.pdf')}")
    print(f"Total pages: {repo.count()}")
    print(f"Total files: {repo.count_files()}")

    doc = repo.get_by_id(1)
    if doc:
        print(f"Retrieved: {doc.filename} page {doc.page_num}")

    pages = repo.get_by_filepath("/test/sample.pdf")
    print(f"Pages for sample.pdf: {len(pages)}")
