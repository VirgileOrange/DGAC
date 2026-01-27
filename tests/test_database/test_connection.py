"""
Tests for database connection management.

Tests connection creation, context managers, and SQLite configuration.
"""

import sqlite3
from pathlib import Path

from src.database.connection import DatabaseManager


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_manager_creation(self, temp_database: Path):
        """Test creating a database manager."""
        manager = DatabaseManager(temp_database)

        assert manager.db_path == temp_database

    def test_manager_creates_parent_directory(self, temp_dir: Path):
        """Test that manager creates parent directories."""
        db_path = temp_dir / "subdir" / "nested" / "test.db"

        _manager = DatabaseManager(db_path)  # noqa: F841

        assert db_path.parent.exists()

    def test_connection_context_manager(self, temp_database: Path):
        """Test connection context manager."""
        manager = DatabaseManager(temp_database)

        with manager.connection() as conn:
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)

            # Connection should be usable
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

    def test_cursor_context_manager(self, temp_database: Path):
        """Test cursor context manager."""
        manager = DatabaseManager(temp_database)

        with manager.cursor() as cursor:
            assert cursor is not None
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            cursor.execute("INSERT INTO test (id) VALUES (1)")

        # Verify data was committed
        with manager.connection() as conn:
            result = conn.execute("SELECT id FROM test").fetchone()
            assert result[0] == 1

    def test_cursor_rollback_on_error(self, temp_database: Path):
        """Test that errors cause rollback."""
        manager = DatabaseManager(temp_database)

        # First, create table
        with manager.cursor() as cursor:
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")

        # Try to insert with error
        try:
            with manager.cursor() as cursor:
                cursor.execute("INSERT INTO test (id) VALUES (1)")
                cursor.execute("INSERT INTO test (id) VALUES (1)")  # Duplicate
        except sqlite3.IntegrityError:
            pass

        # Table should exist but be empty (rollback)
        with manager.connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] == 0

    def test_row_factory_enabled(self, temp_database: Path):
        """Test that row factory is enabled for dict-like access."""
        manager = DatabaseManager(temp_database)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice')")
            conn.commit()

            row = conn.execute("SELECT * FROM test").fetchone()

            # Should be accessible by column name
            assert row["id"] == 1
            assert row["name"] == "Alice"

    def test_wal_mode_enabled(self, temp_database: Path):
        """Test that WAL journal mode is enabled."""
        manager = DatabaseManager(temp_database)

        with manager.connection() as conn:
            result = conn.execute("PRAGMA journal_mode").fetchone()
            # WAL mode should be enabled
            assert result[0].lower() == "wal"


class TestDatabaseManagerExecute:
    """Tests for execute methods."""

    def test_execute_query(self, temp_database: Path):
        """Test simple query execution."""
        manager = DatabaseManager(temp_database)

        with manager.cursor() as cursor:
            cursor.execute("CREATE TABLE test (id INTEGER, value TEXT)")

        manager.execute("INSERT INTO test VALUES (1, 'test')", ())

        with manager.connection() as conn:
            result = conn.execute("SELECT * FROM test").fetchone()
            assert result["value"] == "test"

    def test_executemany(self, temp_database: Path):
        """Test batch execution."""
        manager = DatabaseManager(temp_database)

        with manager.cursor() as cursor:
            cursor.execute("CREATE TABLE test (id INTEGER, value TEXT)")

        data = [(1, "a"), (2, "b"), (3, "c")]
        count = manager.executemany("INSERT INTO test VALUES (?, ?)", data)

        with manager.connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] == 3
