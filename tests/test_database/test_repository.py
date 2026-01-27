"""
Tests for the document repository.

Tests CRUD operations for document storage.

SAFETY NOTE: All tests use the `configured_db` fixture which:
- Creates a temporary directory via tempfile.mkdtemp()
- Configures the database singleton to use a temp path
- Cleans up all temp files after the test
- Never touches real data directories
"""

import pytest
from pathlib import Path

from src.database.schema import init_schema
from src.database.repository import DocumentRepository


@pytest.fixture
def repository(configured_db) -> DocumentRepository:
    """Create a repository with initialized schema in temp database."""
    init_schema()
    return DocumentRepository()


class TestDocumentRepository:
    """Tests for DocumentRepository class."""

    def test_insert_document(self, repository: DocumentRepository):
        """Test inserting a single document page into temp DB."""
        doc_id = repository.insert(
            filepath="/path/to/doc.pdf",
            filename="doc.pdf",
            relative_path="doc.pdf",
            file_hash="abc123",
            page_num=1,
            content="This is the document content."
        )

        assert doc_id is not None
        assert doc_id > 0

    def test_insert_multiple_pages(self, repository: DocumentRepository):
        """Test inserting multiple pages from same document."""
        ids = []
        for page in range(1, 4):
            doc_id = repository.insert(
                filepath="/path/to/doc.pdf",
                filename="doc.pdf",
                relative_path="doc.pdf",
                file_hash="abc123",
                page_num=page,
                content=f"Content for page {page}"
            )
            ids.append(doc_id)

        # All IDs should be unique
        assert len([i for i in ids if i is not None]) == 3

    def test_exists_by_filepath(self, repository: DocumentRepository):
        """Test checking if document exists by filepath."""
        # Initially should not exist
        assert not repository.exists("/path/to/nonexistent.pdf")

        # Insert document
        repository.insert(
            filepath="/path/to/doc.pdf",
            filename="doc.pdf",
            relative_path="doc.pdf",
            file_hash="unique_hash",
            page_num=1,
            content="Content"
        )

        # Now should exist
        assert repository.exists("/path/to/doc.pdf")

    def test_get_by_id(self, repository: DocumentRepository):
        """Test retrieving document by ID."""
        doc_id = repository.insert(
            filepath="/path/to/doc.pdf",
            filename="doc.pdf",
            relative_path="doc.pdf",
            file_hash="abc123",
            page_num=1,
            content="Test content"
        )

        doc = repository.get_by_id(doc_id)

        assert doc is not None
        assert doc.filename == "doc.pdf"
        assert doc.content == "Test content"

    def test_get_by_id_nonexistent(self, repository: DocumentRepository):
        """Test that getting nonexistent ID returns None."""
        doc = repository.get_by_id(99999)

        assert doc is None

    def test_delete_by_filepath(self, repository: DocumentRepository):
        """Test deleting documents by filepath from temp DB.

        SAFETY: Only deletes from temp test database.
        """
        # Insert multiple pages
        for page in range(1, 3):
            repository.insert(
                filepath="/path/to/delete.pdf",
                filename="delete.pdf",
                relative_path="delete.pdf",
                file_hash="to_delete",
                page_num=page,
                content=f"Page {page}"
            )

        # Also insert different file
        repository.insert(
            filepath="/path/to/keep.pdf",
            filename="keep.pdf",
            relative_path="keep.pdf",
            file_hash="to_keep",
            page_num=1,
            content="Keep this"
        )

        # Delete first file from temp DB
        deleted = repository.delete_by_filepath("/path/to/delete.pdf")

        assert deleted == 2  # Two pages deleted

        # Second file should still exist
        assert repository.exists("/path/to/keep.pdf")
        assert not repository.exists("/path/to/delete.pdf")


class TestRepositoryBulkOperations:
    """Tests for bulk operations."""

    def test_insert_batch(self, repository: DocumentRepository):
        """Test bulk insertion into temp DB."""
        # insert_batch expects list of tuples:
        # (filepath, filename, page_num, content, relative_path, file_hash)
        documents = [
            (f"/path/doc{i}.pdf", f"doc{i}.pdf", 1, f"Content {i}", f"doc{i}.pdf", f"hash{i}")
            for i in range(5)
        ]

        count = repository.insert_batch(documents)

        assert count == 5

    def test_get_indexed_filepaths(self, repository: DocumentRepository):
        """Test getting all unique filepaths from temp DB."""
        # Insert documents from different files
        for i in range(3):
            repository.insert(
                filepath=f"/path/doc{i}.pdf",
                filename=f"doc{i}.pdf",
                relative_path=f"doc{i}.pdf",
                file_hash=f"hash{i}",
                page_num=1,
                content=f"Content {i}"
            )

        filepaths = repository.get_indexed_filepaths()

        assert len(filepaths) == 3
        assert "/path/doc0.pdf" in filepaths

    def test_count_methods(self, repository: DocumentRepository):
        """Test count and count_files methods."""
        # Insert 2 files with 3 total pages
        repository.insert(
            filepath="/path/file1.pdf",
            filename="file1.pdf",
            relative_path="file1.pdf",
            file_hash="hash1",
            page_num=1,
            content="Page 1 of file 1"
        )
        repository.insert(
            filepath="/path/file1.pdf",
            filename="file1.pdf",
            relative_path="file1.pdf",
            file_hash="hash1",
            page_num=2,
            content="Page 2 of file 1"
        )
        repository.insert(
            filepath="/path/file2.pdf",
            filename="file2.pdf",
            relative_path="file2.pdf",
            file_hash="hash2",
            page_num=1,
            content="Page 1 of file 2"
        )

        assert repository.count() == 3
        assert repository.count_files() == 2

    def test_get_by_filepath(self, repository: DocumentRepository):
        """Test getting all pages for a file."""
        # Insert multiple pages
        for page in range(1, 4):
            repository.insert(
                filepath="/path/multi.pdf",
                filename="multi.pdf",
                relative_path="multi.pdf",
                file_hash="multihash",
                page_num=page,
                content=f"Page {page} content"
            )

        pages = repository.get_by_filepath("/path/multi.pdf")

        assert len(pages) == 3
        assert pages[0].page_num == 1
        assert pages[1].page_num == 2
        assert pages[2].page_num == 3
