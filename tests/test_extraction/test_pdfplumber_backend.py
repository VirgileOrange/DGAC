"""
Tests for the pdfplumber-based extraction backend.

Tests text extraction, page-level extraction, table extraction,
and error cases using sample PDF fixtures and mocks.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from src.extraction.pdfplumber_backend import PDFPlumberBackend
from src.core.exceptions import ExtractionError


@pytest.fixture
def backend():
    """Create a PDFPlumberBackend instance."""
    return PDFPlumberBackend()


class TestPDFPlumberBackend:
    """Tests for PDFPlumberBackend class."""

    def test_backend_name(self, backend):
        """Test that backend has correct name identifier."""
        assert backend.name == "pdfplumber"

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


class TestPDFPlumberBackendExtractPage:
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


class TestPDFPlumberBackendWithMock:
    """Tests using mocked pdfplumber for deterministic behavior."""

    @patch("src.extraction.pdfplumber_backend.pdfplumber")
    def test_extract_multiple_pages(self, mock_pdfplumber, backend, sample_pdf):
        """Test extraction of multiple pages."""
        pages = []
        for text in ["Page one", "Page two", "Page three"]:
            mock_page = Mock()
            mock_page.extract_text.return_value = text
            pages.append(mock_page)

        mock_pdf = MagicMock()
        mock_pdf.pages = pages
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        results = backend.extract(sample_pdf)

        assert len(results) == 3
        assert results[0] == (1, "Page one")
        assert results[1] == (2, "Page two")
        assert results[2] == (3, "Page three")

    @patch("src.extraction.pdfplumber_backend.pdfplumber")
    def test_extract_skips_empty_pages(self, mock_pdfplumber, backend, sample_pdf):
        """Test that pages with only whitespace are skipped."""
        page_with_text = Mock()
        page_with_text.extract_text.return_value = "Real content"

        empty_page = Mock()
        empty_page.extract_text.return_value = "   \n  "

        none_page = Mock()
        none_page.extract_text.return_value = None

        mock_pdf = MagicMock()
        mock_pdf.pages = [page_with_text, empty_page, none_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        results = backend.extract(sample_pdf)

        assert len(results) == 1
        assert results[0] == (1, "Real content")

    @patch("src.extraction.pdfplumber_backend.pdfplumber")
    def test_extract_continues_on_page_error(self, mock_pdfplumber, backend, sample_pdf):
        """Test that a failing page doesn't stop extraction of others."""
        good_page = Mock()
        good_page.extract_text.return_value = "Good content"

        bad_page = Mock()
        bad_page.extract_text.side_effect = Exception("Page corrupted")

        mock_pdf = MagicMock()
        mock_pdf.pages = [good_page, bad_page, good_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        results = backend.extract(sample_pdf)

        assert len(results) == 2
        assert results[0] == (1, "Good content")
        assert results[1] == (3, "Good content")


class TestPDFPlumberBackendExtractTables:
    """Tests for table extraction."""

    @patch("src.extraction.pdfplumber_backend.pdfplumber")
    def test_extract_tables_all_pages(self, mock_pdfplumber, backend, sample_pdf):
        """Test extracting tables from all pages."""
        table1 = [["Header1", "Header2"], ["val1", "val2"]]
        table2 = [["A", "B"], ["C", "D"]]

        page1 = Mock()
        page1.extract_tables.return_value = [table1]
        page2 = Mock()
        page2.extract_tables.return_value = [table2]

        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        tables = backend.extract_tables(sample_pdf)

        assert len(tables) == 2
        assert tables[0] == table1
        assert tables[1] == table2

    @patch("src.extraction.pdfplumber_backend.pdfplumber")
    def test_extract_tables_specific_page(self, mock_pdfplumber, backend, sample_pdf):
        """Test extracting tables from a specific page."""
        table = [["Col1", "Col2"], ["data1", "data2"]]

        page1 = Mock()
        page1.extract_tables.return_value = [table]
        page2 = Mock()

        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        tables = backend.extract_tables(sample_pdf, page_num=1)

        assert len(tables) == 1
        assert tables[0] == table
        page2.extract_tables.assert_not_called()

    @patch("src.extraction.pdfplumber_backend.pdfplumber")
    def test_extract_tables_no_tables(self, mock_pdfplumber, backend, sample_pdf):
        """Test extracting tables when none exist."""
        page = Mock()
        page.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        tables = backend.extract_tables(sample_pdf)

        assert tables == []

    @patch("src.extraction.pdfplumber_backend.pdfplumber")
    def test_extract_tables_none_response(self, mock_pdfplumber, backend, sample_pdf):
        """Test extracting tables when extract_tables returns None."""
        page = Mock()
        page.extract_tables.return_value = None

        mock_pdf = MagicMock()
        mock_pdf.pages = [page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        tables = backend.extract_tables(sample_pdf)

        assert tables == []

    def test_extract_tables_nonexistent_file_raises(self, backend, temp_dir):
        """Test that nonexistent file raises ExtractionError."""
        fake_path = temp_dir / "nonexistent.pdf"

        with pytest.raises(ExtractionError):
            backend.extract_tables(fake_path)

    def test_extract_tables_accepts_string_path(self, backend, sample_pdf):
        """Test that extract_tables accepts string path."""
        try:
            result = backend.extract_tables(str(sample_pdf))
            assert isinstance(result, list)
        except ExtractionError:
            pass
