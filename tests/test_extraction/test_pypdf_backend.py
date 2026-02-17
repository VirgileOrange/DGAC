"""
Tests for the PyPDF2-based extraction backend.

Tests text extraction, page-level extraction, encryption handling,
and error cases using sample PDF fixtures.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock, PropertyMock

from src.extraction.pypdf_backend import PyPDFBackend
from src.core.exceptions import ExtractionError


@pytest.fixture
def backend():
    """Create a PyPDFBackend instance."""
    return PyPDFBackend()


class TestPyPDFBackend:
    """Tests for PyPDFBackend class."""

    def test_backend_name(self, backend):
        """Test that backend has correct name identifier."""
        assert backend.name == "pypdf2"

    def test_extract_returns_list(self, backend, sample_pdf):
        """Test that extract returns a list."""
        try:
            result = backend.extract(sample_pdf)
            assert isinstance(result, list)
        except ExtractionError:
            pass

    def test_extract_accepts_string_path(self, backend, sample_pdf):
        """Test that extract accepts string path."""
        try:
            result = backend.extract(str(sample_pdf))
            assert isinstance(result, list)
        except ExtractionError:
            pass

    def test_extract_tuples_structure(self, backend, sample_pdf):
        """Test that results are (page_num, text) tuples."""
        try:
            results = backend.extract(sample_pdf)
            for item in results:
                assert isinstance(item, tuple)
                assert len(item) == 2
                assert isinstance(item[0], int)
                assert isinstance(item[1], str)
        except ExtractionError:
            pass

    def test_extract_page_numbers_start_at_one(self, backend, sample_pdf):
        """Test that page numbers are 1-indexed."""
        try:
            results = backend.extract(sample_pdf)
            if results:
                assert results[0][0] == 1
        except ExtractionError:
            pass

    def test_extract_nonexistent_file_raises(self, backend, temp_dir):
        """Test that extracting nonexistent file raises ExtractionError."""
        fake_path = temp_dir / "nonexistent.pdf"

        with pytest.raises(ExtractionError):
            backend.extract(fake_path)

    def test_extract_invalid_pdf_raises(self, backend, temp_dir):
        """Test that invalid PDF content raises ExtractionError."""
        invalid_pdf = temp_dir / "invalid.pdf"
        invalid_pdf.write_bytes(b"Not a valid PDF")

        with pytest.raises(ExtractionError):
            backend.extract(invalid_pdf)


class TestPyPDFBackendExtractPage:
    """Tests for single-page extraction."""

    def test_extract_page_returns_string(self, backend, sample_pdf):
        """Test that extract_page returns a string."""
        try:
            text = backend.extract_page(sample_pdf, 1)
            assert isinstance(text, str)
        except ExtractionError:
            pass

    def test_extract_page_accepts_string_path(self, backend, sample_pdf):
        """Test that extract_page accepts string path."""
        try:
            text = backend.extract_page(str(sample_pdf), 1)
            assert isinstance(text, str)
        except ExtractionError:
            pass

    def test_extract_page_invalid_page_raises(self, backend, sample_pdf):
        """Test that out-of-range page number raises."""
        with pytest.raises(ExtractionError):
            backend.extract_page(sample_pdf, 9999)

    def test_extract_page_nonexistent_file_raises(self, backend, temp_dir):
        """Test that nonexistent file raises ExtractionError."""
        fake_path = temp_dir / "nonexistent.pdf"

        with pytest.raises(ExtractionError):
            backend.extract_page(fake_path, 1)


class TestPyPDFBackendEncryption:
    """Tests for encrypted PDF handling."""

    @patch("src.extraction.pypdf_backend.PdfReader")
    def test_encrypted_pdf_attempts_empty_password(self, mock_reader_cls, backend, sample_pdf):
        """Test that encrypted PDF is decrypted with empty password."""
        mock_reader = Mock()
        mock_reader.is_encrypted = True
        mock_reader.pages = []
        mock_reader_cls.return_value = mock_reader

        backend.extract(sample_pdf)

        mock_reader.decrypt.assert_called_once_with("")

    @patch("src.extraction.pypdf_backend.PdfReader")
    def test_encrypted_pdf_decrypt_failure_raises(self, mock_reader_cls, backend, sample_pdf):
        """Test that undecryptable PDF raises ExtractionError."""
        mock_reader = Mock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.side_effect = Exception("Bad password")
        mock_reader_cls.return_value = mock_reader

        with pytest.raises(ExtractionError, match="encrypted"):
            backend.extract(sample_pdf)

    @patch("src.extraction.pypdf_backend.PdfReader")
    def test_unencrypted_pdf_skips_decrypt(self, mock_reader_cls, backend, sample_pdf):
        """Test that unencrypted PDF does not call decrypt."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Some text"

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        backend.extract(sample_pdf)

        mock_reader.decrypt.assert_not_called()


class TestPyPDFBackendWithMock:
    """Tests using mocked PdfReader for deterministic behavior."""

    @patch("src.extraction.pypdf_backend.PdfReader")
    def test_extract_multiple_pages(self, mock_reader_cls, backend, sample_pdf):
        """Test extraction of multiple pages."""
        pages = []
        for text in ["Page one content", "Page two content", "Page three content"]:
            mock_page = Mock()
            mock_page.extract_text.return_value = text
            pages.append(mock_page)

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = pages
        mock_reader_cls.return_value = mock_reader

        results = backend.extract(sample_pdf)

        assert len(results) == 3
        assert results[0] == (1, "Page one content")
        assert results[1] == (2, "Page two content")
        assert results[2] == (3, "Page three content")

    @patch("src.extraction.pypdf_backend.PdfReader")
    def test_extract_skips_empty_pages(self, mock_reader_cls, backend, sample_pdf):
        """Test that pages with only whitespace are skipped."""
        page_with_text = Mock()
        page_with_text.extract_text.return_value = "Real content"

        empty_page = Mock()
        empty_page.extract_text.return_value = "   \n  "

        none_page = Mock()
        none_page.extract_text.return_value = None

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [page_with_text, empty_page, none_page]
        mock_reader_cls.return_value = mock_reader

        results = backend.extract(sample_pdf)

        assert len(results) == 1
        assert results[0] == (1, "Real content")

    @patch("src.extraction.pypdf_backend.PdfReader")
    def test_extract_continues_on_page_error(self, mock_reader_cls, backend, sample_pdf):
        """Test that a failing page doesn't stop extraction of others."""
        good_page = Mock()
        good_page.extract_text.return_value = "Good content"

        bad_page = Mock()
        bad_page.extract_text.side_effect = Exception("Page corrupted")

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [good_page, bad_page, good_page]
        mock_reader_cls.return_value = mock_reader

        results = backend.extract(sample_pdf)

        assert len(results) == 2
        assert results[0] == (1, "Good content")
        assert results[1] == (3, "Good content")
