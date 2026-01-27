"""
Unified PDF extraction interface with automatic fallback.

Wraps multiple extraction backends and attempts fallback when
the primary backend fails or returns empty results.
"""

from pathlib import Path
from typing import List, Tuple, Union

from ..core import get_config, get_logger, ExtractionError
from .pypdf_backend import PyPDFBackend
from .pdfplumber_backend import PDFPlumberBackend

logger = get_logger(__name__)


BACKENDS = {
    "pypdf2": PyPDFBackend,
    "pdfplumber": PDFPlumberBackend
}


class PDFExtractor:
    """
    Unified PDF extraction with automatic backend fallback.

    Tries the primary backend first, falls back to secondary
    if extraction fails or produces empty results.
    """

    def __init__(
        self,
        primary_backend: str = None,
        fallback_backend: str = None
    ):
        """
        Initialize the extractor with configured backends.

        Args:
            primary_backend: Name of primary backend ("pypdf2" or "pdfplumber").
            fallback_backend: Name of fallback backend.
        """
        config = get_config()

        primary_name = primary_backend or config.extraction.primary_backend
        fallback_name = fallback_backend or config.extraction.fallback_backend

        if primary_name not in BACKENDS:
            raise ExtractionError(f"Unknown backend: {primary_name}")

        self.primary = BACKENDS[primary_name]()
        self.fallback = BACKENDS.get(fallback_name, lambda: None)()

        logger.debug(
            f"Initialized extractor: primary={primary_name}, fallback={fallback_name}"
        )

    def extract(self, filepath: Union[str, Path]) -> List[Tuple[int, str]]:
        """
        Extract text from a PDF using available backends.

        Tries primary backend first, falls back if needed.

        Args:
            filepath: Path to the PDF file.

        Returns:
            List of (page_number, text) tuples.

        Raises:
            ExtractionError: If all backends fail.
        """
        filepath = Path(filepath)
        primary_error = None

        try:
            results = self.primary.extract(filepath)

            if results:
                return results

            logger.debug(f"Primary backend returned empty results: {filepath.name}")

        except ExtractionError as e:
            primary_error = e
            logger.debug(f"Primary backend failed: {e.message}")

        if self.fallback:
            try:
                logger.debug(f"Trying fallback backend for: {filepath.name}")
                results = self.fallback.extract(filepath)

                if results:
                    return results

            except ExtractionError as e:
                logger.debug(f"Fallback backend also failed: {e.message}")

        if primary_error:
            raise primary_error

        raise ExtractionError(
            "All backends returned empty results",
            filepath=str(filepath)
        )

    def extract_page(self, filepath: Union[str, Path], page_num: int) -> str:
        """
        Extract text from a specific page.

        Args:
            filepath: Path to the PDF file.
            page_num: Page number (1-indexed).

        Returns:
            Extracted text.
        """
        filepath = Path(filepath)

        try:
            return self.primary.extract_page(filepath, page_num)
        except ExtractionError:
            if self.fallback:
                return self.fallback.extract_page(filepath, page_num)
            raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <pdf_file>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    extractor = PDFExtractor()

    try:
        pages = extractor.extract(pdf_path)
        print(f"Extracted {len(pages)} pages from {pdf_path.name}")

        total_chars = sum(len(text) for _, text in pages)
        print(f"Total characters: {total_chars:,}")

        if pages:
            page_num, text = pages[0]
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"\n=== Preview (Page {page_num}) ===")
            print(preview)

    except ExtractionError as e:
        print(f"Extraction failed: {e.message}")
