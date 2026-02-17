"""
Database schema definitions for the PDF Search Engine.

Defines the documents table, FTS5 virtual table for full-text search,
synchronization triggers, and semantic search tables.
"""

import sqlite3

from ..core import get_config, get_logger, DatabaseError
from .connection import get_cursor, get_connection

logger = get_logger(__name__)

_vec_extension_available = None


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    """
    Load sqlite-vec extension into the connection.

    Args:
        conn: SQLite connection to load extension into.

    Returns:
        True if extension loaded successfully, False otherwise.
    """
    global _vec_extension_available

    # If package is known to be unavailable, skip
    if _vec_extension_available is False:
        return False

    try:
        import sqlite_vec
    except ImportError:
        logger.warning("sqlite-vec package not installed. Semantic search disabled.")
        _vec_extension_available = False
        return False

    # Package is available
    _vec_extension_available = True

    try:
        # Enable extension loading (required by SQLite security model)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)  # Disable after loading for security
        return True
    except Exception as e:
        # Connection-specific error - don't disable globally
        logger.warning(f"Failed to load sqlite-vec extension on connection: {e}")
        return False


def is_vec_extension_available() -> bool:
    """Check if sqlite-vec extension is available."""
    global _vec_extension_available

    if _vec_extension_available is not None:
        return _vec_extension_available

    with get_connection() as conn:
        return _load_vec_extension(conn)


def reset_vec_extension_cache() -> None:
    """Reset the vec extension availability cache.

    Call this when switching database connections to ensure
    fresh extension loading attempt.
    """
    global _vec_extension_available
    _vec_extension_available = None


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

CHUNKS_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS chunks_metadata (
    chunk_id TEXT PRIMARY KEY,
    document_id INTEGER NOT NULL,
    page_num INTEGER NOT NULL,
    position INTEGER NOT NULL,
    content TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
)
"""

CHUNKS_VEC_TABLE = """
CREATE TABLE IF NOT EXISTS chunks_vec (
    chunk_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks_metadata(chunk_id) ON DELETE CASCADE
)
"""

CHUNKS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks_metadata(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_page_num ON chunks_metadata(page_num)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks_metadata(document_id, page_num)"
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

    Creates documents table, FTS5 virtual table, indexes, triggers,
    and semantic search tables.
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

        cur.execute(CHUNKS_METADATA_TABLE)
        cur.execute(CHUNKS_VEC_TABLE)

        for index_sql in CHUNKS_INDEXES:
            cur.execute(index_sql)

    logger.info("Schema initialization complete")


def init_vector_index(dimensions: int = None) -> bool:
    """
    Initialize the sqlite-vec virtual table for vector search.

    Loads the sqlite-vec extension and creates the vector index table.

    Args:
        dimensions: Embedding dimensions (default from config).

    Returns:
        True if index was created, False if already exists or extension unavailable.
    """
    if dimensions is None:
        config = get_config()
        dimensions = config.semantic.embedding_dimensions

    with get_connection() as conn:
        if not _load_vec_extension(conn):
            logger.warning("Cannot create vector index: sqlite-vec extension not available")
            return False

        try:
            conn.execute("SELECT 1 FROM chunks_vec_idx LIMIT 1")
            logger.debug("Vector index already exists")
            return False
        except Exception:
            pass

        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec_idx USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{dimensions}]
            )
        """)
        conn.commit()
        logger.info(f"Created vector index with {dimensions} dimensions")

    return True


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
        cur.execute("DROP TABLE IF EXISTS chunks_vec_idx")
        cur.execute("DROP TABLE IF EXISTS chunks_vec")
        cur.execute("DROP TABLE IF EXISTS chunks_metadata")
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
