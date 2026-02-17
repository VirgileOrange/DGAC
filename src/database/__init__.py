"""
Database module for SQLite persistence with FTS5 full-text search.

Provides connection management, schema definitions, and CRUD operations
for the document index. Includes vector storage for semantic search.
"""

from .connection import get_connection, get_cursor, DatabaseManager
from .schema import init_schema, reset_schema, get_statistics, init_vector_index, is_vec_extension_available, reset_vec_extension_cache
from .repository import DocumentRepository
from .vector_repository import VectorRepository, VectorSearchResult

__all__ = [
    "get_connection",
    "get_cursor",
    "DatabaseManager",
    "init_schema",
    "reset_schema",
    "get_statistics",
    "init_vector_index",
    "is_vec_extension_available",
    "reset_vec_extension_cache",
    "DocumentRepository",
    "VectorRepository",
    "VectorSearchResult"
]
