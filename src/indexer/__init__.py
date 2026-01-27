"""
Indexer module for orchestrating the PDF indexing pipeline.

Coordinates file scanning, text extraction, and database storage
to build the full-text search index.
"""

from .index_builder import IndexBuilder

__all__ = ["IndexBuilder"]
