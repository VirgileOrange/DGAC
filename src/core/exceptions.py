"""
Custom exception hierarchy for the PDF Search Engine.

Provides specific exception types for different failure modes:
configuration errors, extraction failures, database issues, and search problems.
"""


class PDFSearchError(Exception):
    """Base exception for all PDF Search Engine errors."""

    def __init__(self, message: str, details: dict = None):
        """
        Initialize the exception.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(PDFSearchError):
    """Raised when configuration is invalid or missing."""
    pass


class ExtractionError(PDFSearchError):
    """Raised when PDF text extraction fails."""

    def __init__(self, message: str, filepath: str = None, details: dict = None):
        """
        Initialize extraction error.

        Args:
            message: Error description.
            filepath: Path to the problematic PDF file.
            details: Additional context.
        """
        super().__init__(message, details)
        self.filepath = filepath


class DatabaseError(PDFSearchError):
    """Raised when SQLite operations fail."""
    pass


class SearchError(PDFSearchError):
    """Raised when search query execution fails."""

    def __init__(self, message: str, query: str = None, details: dict = None):
        """
        Initialize search error.

        Args:
            message: Error description.
            query: The problematic search query.
            details: Additional context.
        """
        super().__init__(message, details)
        self.query = query


if __name__ == "__main__":
    try:
        raise ConfigurationError("Config file not found", {"path": "/config/config.json"})
    except PDFSearchError as e:
        print(f"Caught: {e.__class__.__name__}: {e.message}")
        print(f"Details: {e.details}")

    try:
        raise ExtractionError("Failed to extract text", filepath="/docs/test.pdf")
    except ExtractionError as e:
        print(f"Extraction failed for: {e.filepath}")
