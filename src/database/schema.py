"""
Database schema definitions for the PDF Search Engine.

Defines the documents table, FTS5 virtual table for full-text search,
and synchronization triggers.
"""

import sqlite3

from ..core import get_config, get_logger, DatabaseError
from .connection import get_cursor, get_connection

logger = get_logger(__name__)


DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL,
    filename TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    content TEXT,
    relative_path TEXT,
    file_hash TEXT,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(filepath, page_num)
)
"""

DOCUMENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath)",
    "CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)",
    "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)"
]


def _get_fts_table_sql() -> str:
    """Generate FTS5 table creation SQL with configured tokenizer."""
    config = get_config()
    tokenizer = config.search.tokenizer

    return f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        filename,
        content,
        content='documents',
        content_rowid='id',
        tokenize='{tokenizer}'
    )
    """


FTS_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
        INSERT INTO documents_fts(rowid, filename, content)
        VALUES (new.id, new.filename, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, filename, content)
        VALUES ('delete', old.id, old.filename, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, filename, content)
        VALUES ('delete', old.id, old.filename, old.content);
        INSERT INTO documents_fts(rowid, filename, content)
        VALUES (new.id, new.filename, new.content);
    END
    """
]


def init_schema() -> None:
    """
    Initialize database schema if not exists.

    Creates documents table, FTS5 virtual table, indexes, and triggers.
    """
    logger.info("Initializing database schema")

    with get_cursor() as cur:
        cur.execute(DOCUMENTS_TABLE)

        for index_sql in DOCUMENTS_INDEXES:
            cur.execute(index_sql)

        try:
            cur.execute(_get_fts_table_sql())
        except sqlite3.OperationalError as e:
            if "already exists" not in str(e):
                raise DatabaseError(f"Failed to create FTS table: {e}")

        for trigger_sql in FTS_TRIGGERS:
            try:
                cur.execute(trigger_sql)
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    raise DatabaseError(f"Failed to create trigger: {e}")

    logger.info("Schema initialization complete")


def reset_schema() -> None:
    """
    Drop and recreate all tables.

    Warning: This deletes all indexed data.
    """
    logger.warning("Resetting database schema - all data will be deleted")

    with get_cursor() as cur:
        cur.execute("DROP TRIGGER IF EXISTS documents_ai")
        cur.execute("DROP TRIGGER IF EXISTS documents_ad")
        cur.execute("DROP TRIGGER IF EXISTS documents_au")
        cur.execute("DROP TABLE IF EXISTS documents_fts")
        cur.execute("DROP TABLE IF EXISTS documents")

    init_schema()

    logger.info("Schema reset complete")


def get_statistics() -> dict:
    """
    Get database statistics for dashboard display.

    Returns:
        Dictionary with document counts and size info.
    """
    with get_connection() as conn:
        stats = {}

        row = conn.execute("SELECT COUNT(*) as count FROM documents").fetchone()
        stats["total_pages"] = row["count"]

        row = conn.execute(
            "SELECT COUNT(DISTINCT filepath) as count FROM documents"
        ).fetchone()
        stats["total_files"] = row["count"]

        row = conn.execute(
            "SELECT SUM(LENGTH(content)) as total FROM documents"
        ).fetchone()
        total_bytes = row["total"] or 0
        stats["total_content_mb"] = round(total_bytes / (1024 * 1024), 2)

        row = conn.execute(
            "SELECT MIN(indexed_at) as oldest, MAX(indexed_at) as newest FROM documents"
        ).fetchone()
        stats["oldest_index"] = row["oldest"]
        stats["newest_index"] = row["newest"]

    return stats


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test.db"

        from ..core.config_loader import _config_instance
        original_config = _config_instance

        try:
            init_schema()
            print("Schema initialized successfully")

            with get_cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (filepath, filename, page_num, content, relative_path, file_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ("/test/doc.pdf", "doc.pdf", 1, "Test content for searching", "test/doc.pdf", "abc123"))

            stats = get_statistics()
            print(f"\nStatistics: {stats}")

            with get_connection() as conn:
                results = conn.execute(
                    "SELECT * FROM documents_fts WHERE documents_fts MATCH ?",
                    ("searching",)
                ).fetchall()
                print(f"\nFTS search for 'searching': {len(results)} result(s)")

        except Exception as e:
            print(f"Error: {e}")
