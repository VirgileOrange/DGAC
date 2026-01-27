"""
Utility module providing shared helper functions.

Contains file operations and text processing utilities used across
the application. Depends only on the core module.
"""

from .file_utils import (
    get_file_hash,
    get_file_size_mb,
    get_relative_path,
    ensure_directory
)
from .text_utils import (
    clean_text,
    truncate_text,
    extract_keywords
)

__all__ = [
    "get_file_hash",
    "get_file_size_mb",
    "get_relative_path",
    "ensure_directory",
    "clean_text",
    "truncate_text",
    "extract_keywords"
]
