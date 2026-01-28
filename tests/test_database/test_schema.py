"""
Tests for database schema management.

Tests schema initialization, reset, and statistics queries.

SAFETY NOTE: All tests use the `configured_db` fixture which:
- Creates a temporary directory via tempfile.mkdtemp()
- Configures the database singleton to use a temp path
- Cleans up all temp files after the test
- Never touches real data directories
"""

from src.database.schema import init_schema, reset_schema, get_statistics
from src.database.connection import get_connection, get_cursor


class TestInitSchema:
    """Tests for schema initialization."""

    def test_init_creates_documents_table(self, configured_db):
        """Test that init_schema creates documents table.

        Uses temp database via configured_db fixture.
        """
        init_schema()

        with get_connection() as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
            ).fetchone()

            assert result is not None
            assert result["name"] == "documents"

    def test_init_creates_fts_table(self, configured_db):
        """Test that init_schema creates FTS5 virtual table.

        Uses temp database via configured_db fixture.
        """
        init_schema()

        with get_connection() as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'"
            ).fetchone()

            assert result is not None

    def test_init_is_idempotent(self, configured_db):
        """Test that init_schema can be called multiple times safely.

        Uses temp database via configured_db fixture.
        """
        # Call twice - both operate on temp DB only
        init_schema()
        init_schema()

        # Should still work
        with get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
            assert result["cnt"] == 0


class TestResetSchema:
    """Tests for schema reset.

    SAFETY: reset_schema() drops and recreates tables.
    This only affects the temporary test database, never production data.
    """

    def test_reset_clears_data(self, configured_db):
        """Test that reset_schema removes all data from temp DB."""
        init_schema()

        # Insert some data into TEMP database
        with get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO documents (filepath, filename, relative_path, file_hash,
                                       page_num, content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("/test.pdf", "test.pdf", "test.pdf", "abc123", 1, "Test content"))

        # Reset temp database
        reset_schema()

        # Data should be gone from temp DB
        with get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
            assert result["cnt"] == 0

    def test_reset_recreates_tables(self, configured_db):
        """Test that reset recreates table structure in temp DB."""
        init_schema()
        reset_schema()

        # Tables should still exist and be usable in temp DB
        with get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO documents (filepath, filename, relative_path, file_hash,
                                       page_num, content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("/new.pdf", "new.pdf", "new.pdf", "def456", 1, "New content"))


class TestGetStatistics:
    """Tests for statistics retrieval (read-only operations)."""

    def test_statistics_empty_database(self, configured_db):
        """Test statistics on empty temp database."""
        init_schema()

        stats = get_statistics()

        assert stats["total_files"] == 0
        assert stats["total_pages"] == 0

    def test_statistics_with_data(self, configured_db):
        """Test statistics with data in temp database."""
        init_schema()

        # Insert test data into TEMP database - 2 files, 3 pages
        with get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO documents (filepath, filename, relative_path, file_hash,
                                       page_num, content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("/file1.pdf", "file1.pdf", "file1.pdf", "hash1", 1, "Content A"))

            cursor.execute("""
                INSERT INTO documents (filepath, filename, relative_path, file_hash,
                                       page_num, content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("/file1.pdf", "file1.pdf", "file1.pdf", "hash1", 2, "Content B"))

            cursor.execute("""
                INSERT INTO documents (filepath, filename, relative_path, file_hash,
                                       page_num, content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("/file2.pdf", "file2.pdf", "file2.pdf", "hash2", 1, "Content C"))

        stats = get_statistics()

        assert stats["total_pages"] == 3
        assert stats["total_files"] == 2
