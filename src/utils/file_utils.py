"""
File utility functions for the PDF Search Engine.

Provides common file operations: hashing for change detection,
size calculations, path manipulation, and directory management.
"""

import hashlib
from pathlib import Path
from typing import Union


def get_file_hash(filepath: Union[str, Path], chunk_size: int = 8192) -> str:
    """
    Compute MD5 hash of the first chunk of a file for fast change detection.

    Args:
        filepath: Path to the file.
        chunk_size: Number of bytes to read (default 8KB).

    Returns:
        Hexadecimal MD5 hash string.
    """
    filepath = Path(filepath)
    hasher = hashlib.md5()

    with open(filepath, "rb") as f:
        chunk = f.read(chunk_size)
        hasher.update(chunk)

    return hasher.hexdigest()


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """
    Get file size in megabytes.

    Args:
        filepath: Path to the file.

    Returns:
        File size in MB, rounded to 2 decimal places.
    """
    filepath = Path(filepath)
    size_bytes = filepath.stat().st_size
    return round(size_bytes / (1024 * 1024), 2)


def get_relative_path(filepath: Union[str, Path], base: Union[str, Path]) -> str:
    """
    Compute relative path from base directory.

    Args:
        filepath: Absolute path to the file.
        base: Base directory to compute relative path from.

    Returns:
        Relative path as string, or absolute path if not relative to base.
    """
    filepath = Path(filepath).resolve()
    base = Path(base).resolve()

    try:
        return str(filepath.relative_to(base))
    except ValueError:
        return str(filepath)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Create directory tree if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        Path object of the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"Test content for hashing")
        temp_path = Path(f.name)

    print(f"Test file: {temp_path}")
    print(f"Hash: {get_file_hash(temp_path)}")
    print(f"Size: {get_file_size_mb(temp_path)} MB")
    print(f"Relative path: {get_relative_path(temp_path, temp_path.parent.parent)}")

    test_dir = temp_path.parent / "test_subdir"
    ensure_directory(test_dir)
    print(f"Created directory: {test_dir.exists()}")

    temp_path.unlink()
    test_dir.rmdir()
