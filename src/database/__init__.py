"""
Database module for SQLite persistence with FTS5 full-text search.

Provides connection management, schema definitions, and CRUD operations
for the document index.
"""

from .connection import get_connection, get_cursor, DatabaseManager
from .schema import init_schema, reset_schema, get_statistics
from .repository import DocumentRepository

__all__ = [
    "get_connection",
    "get_cursor",
    "DatabaseManager",
    "init_schema",
    "reset_schema",
    "get_statistics",
    "DocumentRepository"
]
