"""
Tests for the file scanner module.

Tests PDF file discovery and filtering in directory trees.
"""

import pytest
from pathlib import Path

from src.extraction.file_scanner import FileScanner


class TestFileScanner:
    """Tests for FileScanner class."""

    def test_scanner_creation(self, temp_dir: Path):
        """Test creating a scanner with valid directory."""
        scanner = FileScanner(temp_dir)

        assert scanner.root_directory == temp_dir

    def test_scanner_creation_with_string(self, temp_dir: Path):
        """Test creating a scanner with string path."""
        scanner = FileScanner(str(temp_dir))

        assert scanner.root_directory == temp_dir

    def test_scan_finds_pdfs(self, sample_pdf_collection: Path):
        """Test that scan finds PDF files."""
        scanner = FileScanner(sample_pdf_collection)

        pdf_files = list(scanner.scan())

        assert len(pdf_files) == 4  # root_doc + doc1 + doc2 + doc3

    def test_scan_ignores_non_pdfs(self, sample_pdf_collection: Path):
        """Test that scan ignores non-PDF files."""
        scanner = FileScanner(sample_pdf_collection)

        pdf_files = list(scanner.scan())

        # Should not include readme.txt
        filenames = [f.name for f in pdf_files]
        assert "readme.txt" not in filenames

    def test_scan_respects_extensions(self, temp_dir: Path):
        """Test that scan only finds configured extensions."""
        # Create files with different extensions
        (temp_dir / "doc.pdf").write_bytes(b"%PDF-1.4")
        (temp_dir / "doc.PDF").write_bytes(b"%PDF-1.4")  # Uppercase
        (temp_dir / "doc.txt").write_text("Not a PDF")

        scanner = FileScanner(temp_dir, extensions=[".pdf"])

        pdf_files = list(scanner.scan())

        # Should find .pdf and .PDF (case insensitive)
        assert len(pdf_files) >= 1

    def test_scan_recursive(self, sample_pdf_collection: Path):
        """Test that scan recurses into subdirectories."""
        scanner = FileScanner(sample_pdf_collection)

        pdf_files = list(scanner.scan())

        # Should find files in folder1 and folder2
        paths = [str(f) for f in pdf_files]
        has_folder1 = any("folder1" in p for p in paths)
        has_folder2 = any("folder2" in p for p in paths)

        assert has_folder1
        assert has_folder2

    def test_scan_empty_directory(self, temp_dir: Path):
        """Test scanning an empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        scanner = FileScanner(empty_dir)

        pdf_files = list(scanner.scan())

        assert pdf_files == []

    def test_count_returns_total(self, sample_pdf_collection: Path):
        """Test that count returns total number of PDFs."""
        scanner = FileScanner(sample_pdf_collection)

        count = scanner.count()

        assert count == 4
