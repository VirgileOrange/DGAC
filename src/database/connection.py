"""
SQLite connection management for the PDF Search Engine.

Provides context managers for safe connection handling with
WAL mode for better concurrency and automatic directory creation.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from ..core import get_config, get_logger, DatabaseError

logger = get_logger(__name__)


class DatabaseManager:
    """
    Manages SQLite database connections with proper lifecycle handling.

    Enables WAL mode for concurrent reads during indexing and provides
    context managers for safe resource cleanup.
    """

    def __init__(self, db_path: Path = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file. Defaults to config value.
        """
        config = get_config()
        self.db_path = Path(db_path or config.paths.database_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection: Optional[sqlite3.Connection] = None

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimal settings."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )

            conn.row_factory = sqlite3.Row

            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")

            return conn

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to database: {e}")

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.

        Yields:
            SQLite connection with Row factory enabled.
        """
        conn = self._create_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def cursor(self, commit: bool = True) -> Generator[sqlite3.Cursor, None, None]:
        """
        Context manager for database cursors with automatic commit/rollback.

        Args:
            commit: Whether to commit on successful exit.

        Yields:
            SQLite cursor for query execution.
        """
        conn = self._create_connection()
        cursor = conn.cursor()

        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Execute a single query and return cursor.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            Cursor with query results.
        """
        with self.cursor() as cur:
            cur.execute(query, params)
            return cur

    def executemany(self, query: str, params_list: list) -> int:
        """
        Execute a query with multiple parameter sets.

        Args:
            query: SQL query string.
            params_list: List of parameter tuples.

        Returns:
            Number of rows affected.
        """
        with self.cursor() as cur:
            cur.executemany(query, params_list)
            return cur.rowcount


_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the singleton DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Get a database connection via context manager.

    Yields:
        SQLite connection.
    """
    with get_db_manager().connection() as conn:
        yield conn


@contextmanager
def get_cursor(commit: bool = True) -> Generator[sqlite3.Cursor, None, None]:
    """
    Get a database cursor via context manager.

    Args:
        commit: Whether to auto-commit on exit.

    Yields:
        SQLite cursor.
    """
    with get_db_manager().cursor(commit=commit) as cur:
        yield cur


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test.db"

        manager = DatabaseManager(test_db)

        with manager.cursor() as cur:
            cur.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
            cur.execute("INSERT INTO test (name) VALUES (?)", ("Alice",))
            cur.execute("INSERT INTO test (name) VALUES (?)", ("Bob",))

        with manager.connection() as conn:
            rows = conn.execute("SELECT * FROM test").fetchall()
            print("Rows in test table:")
            for row in rows:
                print(f"  id={row['id']}, name={row['name']}")

        print(f"\nDatabase created at: {test_db}")
        print(f"Database size: {test_db.stat().st_size} bytes")
