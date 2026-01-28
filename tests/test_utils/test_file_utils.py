"""
Tests for file utility functions.

Tests hashing, size calculation, path manipulation, and directory creation.
All tests use temporary files/directories for safety.
"""

from pathlib import Path

from src.utils.file_utils import (
    get_file_hash,
    get_file_size_mb,
    get_relative_path,
    ensure_directory
)


class TestGetFileHash:
    """Tests for get_file_hash function."""

    def test_hash_returns_hex_string(self, temp_dir: Path):
        """Test that hash returns a valid hexadecimal string."""
        test_file = temp_dir / "test.txt"
        test_file.write_bytes(b"Hello World")

        hash_value = get_file_hash(test_file)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 32  # MD5 hex length
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_same_content_same_hash(self, temp_dir: Path):
        """Test that identical content produces identical hash."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"

        content = b"Identical content"
        file1.write_bytes(content)
        file2.write_bytes(content)

        assert get_file_hash(file1) == get_file_hash(file2)

    def test_different_content_different_hash(self, temp_dir: Path):
        """Test that different content produces different hash."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"

        file1.write_bytes(b"Content A")
        file2.write_bytes(b"Content B")

        assert get_file_hash(file1) != get_file_hash(file2)

    def test_hash_with_string_path(self, temp_dir: Path):
        """Test that hash works with string paths."""
        test_file = temp_dir / "test.txt"
        test_file.write_bytes(b"Test")

        hash_value = get_file_hash(str(test_file))

        assert isinstance(hash_value, str)

    def test_hash_binary_file(self, sample_pdf: Path):
        """Test hashing a binary file (PDF)."""
        hash_value = get_file_hash(sample_pdf)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 32


class TestGetFileSizeMb:
    """Tests for get_file_size_mb function."""

    def test_small_file_size(self, temp_dir: Path):
        """Test size calculation for small file."""
        test_file = temp_dir / "small.txt"
        test_file.write_bytes(b"x" * 1024)  # 1 KB

        size = get_file_size_mb(test_file)

        assert size == 0.0  # Less than 0.01 MB

    def test_larger_file_size(self, temp_dir: Path):
        """Test size calculation for larger file."""
        test_file = temp_dir / "larger.txt"
        test_file.write_bytes(b"x" * (1024 * 1024))  # 1 MB

        size = get_file_size_mb(test_file)

        assert size == 1.0

    def test_size_with_string_path(self, temp_dir: Path):
        """Test size works with string paths."""
        test_file = temp_dir / "test.txt"
        test_file.write_bytes(b"Test content")

        size = get_file_size_mb(str(test_file))

        assert isinstance(size, float)


class TestGetRelativePath:
    """Tests for get_relative_path function."""

    def test_relative_path_within_base(self, temp_dir: Path):
        """Test relative path for file within base directory."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        test_file = subdir / "file.txt"
        test_file.touch()

        rel_path = get_relative_path(test_file, temp_dir)

        assert rel_path == "subdir/file.txt" or rel_path == "subdir\\file.txt"

    def test_relative_path_outside_base(self, temp_dir: Path):
        """Test that path outside base returns absolute path."""
        other_dir = temp_dir.parent / "other"

        rel_path = get_relative_path(other_dir, temp_dir)

        # Should return absolute path since not relative to base
        assert Path(rel_path).is_absolute() or ".." in rel_path

    def test_relative_path_with_strings(self, temp_dir: Path):
        """Test that function works with string arguments."""
        test_file = temp_dir / "test.txt"
        test_file.touch()

        rel_path = get_relative_path(str(test_file), str(temp_dir))

        assert rel_path == "test.txt"


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_creates_single_directory(self, temp_dir: Path):
        """Test creating a single new directory."""
        new_dir = temp_dir / "new_folder"
        assert not new_dir.exists()

        result = ensure_directory(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_creates_nested_directories(self, temp_dir: Path):
        """Test creating nested directory structure."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        ensure_directory(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_existing_directory_no_error(self, temp_dir: Path):
        """Test that existing directory doesn't raise error."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        assert existing_dir.exists()

        # Should not raise
        result = ensure_directory(existing_dir)

        assert result == existing_dir

    def test_with_string_path(self, temp_dir: Path):
        """Test that function works with string paths."""
        new_dir = temp_dir / "string_dir"

        result = ensure_directory(str(new_dir))

        assert isinstance(result, Path)
        assert new_dir.exists()
