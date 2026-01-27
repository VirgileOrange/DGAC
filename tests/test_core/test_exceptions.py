"""
Tests for custom exception classes.

Tests exception creation, message formatting, and details handling.
"""

import pytest

from src.core.exceptions import (
    PDFSearchError,
    ConfigurationError,
    ExtractionError,
    DatabaseError,
    SearchError
)


class TestPDFSearchError:
    """Tests for base PDFSearchError."""

    def test_basic_creation(self):
        """Test creating exception with just a message."""
        error = PDFSearchError("Something went wrong")

        assert error.message == "Something went wrong"
        assert error.details == {}
        assert str(error) == "Something went wrong"

    def test_creation_with_details(self):
        """Test creating exception with details dict."""
        error = PDFSearchError(
            "File error",
            {"filename": "test.pdf", "size": 1024}
        )

        assert error.message == "File error"
        assert error.details["filename"] == "test.pdf"
        assert error.details["size"] == 1024

    def test_str_includes_message(self):
        """Test that string representation includes message."""
        error = PDFSearchError("Error", {"key": "value"})

        error_str = str(error)
        assert "Error" in error_str


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_is_subclass_of_base(self):
        """Test that ConfigurationError inherits from PDFSearchError."""
        error = ConfigurationError("Config missing")

        assert isinstance(error, PDFSearchError)

    def test_can_be_caught_as_base(self):
        """Test that ConfigurationError can be caught as PDFSearchError."""
        with pytest.raises(PDFSearchError):
            raise ConfigurationError("Test error")


class TestExtractionError:
    """Tests for ExtractionError."""

    def test_with_filepath(self):
        """Test ExtractionError with filepath parameter."""
        error = ExtractionError(
            "Failed to extract text",
            filepath="/path/to/file.pdf"
        )

        assert error.filepath == "/path/to/file.pdf"
        assert error.message == "Failed to extract text"

    def test_with_filepath_and_details(self):
        """Test ExtractionError with filepath and details."""
        error = ExtractionError(
            "Failed to extract",
            filepath="/test.pdf",
            details={"backend": "pypdf2"}
        )

        assert error.filepath == "/test.pdf"
        assert error.details["backend"] == "pypdf2"


class TestDatabaseError:
    """Tests for DatabaseError."""

    def test_database_error_message(self):
        """Test DatabaseError with database operation details."""
        error = DatabaseError(
            "Connection failed",
            {"database": "test.db", "operation": "connect"}
        )

        assert "Connection failed" in error.message
        assert error.details["operation"] == "connect"


class TestSearchError:
    """Tests for SearchError."""

    def test_search_error_with_query(self):
        """Test SearchError with query parameter."""
        error = SearchError(
            "Invalid query syntax",
            query="test AND OR"
        )

        assert error.query == "test AND OR"
        assert error.message == "Invalid query syntax"

    def test_search_error_with_details(self):
        """Test SearchError with query and details."""
        error = SearchError(
            "Parse error",
            query="bad query",
            details={"position": 5}
        )

        assert error.query == "bad query"
        assert error.details["position"] == 5
