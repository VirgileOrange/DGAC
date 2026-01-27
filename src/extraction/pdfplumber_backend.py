"""
pdfplumber-based text extraction backend.

Better handling of complex layouts, tables, and multi-column documents.
Slower than PyPDF2 but more accurate for difficult PDFs.
"""

from pathlib import Path
from typing import List, Tuple, Union

import pdfplumber

from ..core import get_logger, ExtractionError

logger = get_logger(__name__)


class PDFPlumberBackend:
    """
    PDF text extraction using pdfplumber library.

    Provides more accurate extraction for complex layouts
    at the cost of slower processing.
    """

    name = "pdfplumber"

    def extract(self, filepath: Union[str, Path]) -> List[Tuple[int, str]]:
        """
        Extract text from all pages of a PDF.

        Args:
            filepath: Path to the PDF file.

        Returns:
            List of (page_number, text) tuples. Page numbers are 1-indexed.

        Raises:
            ExtractionError: If extraction fails completely.
        """
        filepath = Path(filepath)
        results = []

        try:
            with pdfplumber.open(filepath) as pdf:
                total_pages = len(pdf.pages)
                logger.debug(f"Processing {total_pages} pages: {filepath.name}")

                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text() or ""

                        if text.strip():
                            results.append((page_num, text))
                        else:
                            logger.debug(f"Empty page {page_num} in {filepath.name}")

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract page {page_num} from {filepath.name}: {e}"
                        )

        except Exception as e:
            raise ExtractionError(
                f"pdfplumber extraction failed: {e}",
                filepath=str(filepath)
            )

        return results

    def extract_page(self, filepath: Union[str, Path], page_num: int) -> str:
        """
        Extract text from a specific page.

        Args:
            filepath: Path to the PDF file.
            page_num: Page number (1-indexed).

        Returns:
            Extracted text from the page.
        """
        filepath = Path(filepath)

        try:
            with pdfplumber.open(filepath) as pdf:
                # Convert 1-indexed to 0-indexed
                page = pdf.pages[page_num - 1]
                return page.extract_text() or ""

        except Exception as e:
            raise ExtractionError(
                f"Failed to extract page {page_num}: {e}",
                filepath=str(filepath)
            )

    def extract_tables(self, filepath: Union[str, Path], page_num: int = None) -> List[List[List[str]]]:
        """
        Extract tables from PDF pages.

        Args:
            filepath: Path to the PDF file.
            page_num: Specific page (1-indexed) or None for all pages.

        Returns:
            List of tables, where each table is a list of rows.
        """
        filepath = Path(filepath)
        all_tables = []

        try:
            with pdfplumber.open(filepath) as pdf:
                pages_to_process = (
                    [pdf.pages[page_num - 1]] if page_num
                    else pdf.pages
                )

                for page in pages_to_process:
                    tables = page.extract_tables() or []
                    all_tables.extend(tables)

        except Exception as e:
            raise ExtractionError(
                f"Failed to extract tables: {e}",
                filepath=str(filepath)
            )

        return all_tables


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdfplumber_backend.py <pdf_file>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    backend = PDFPlumberBackend()

    try:
        pages = backend.extract(pdf_path)
        print(f"Extracted {len(pages)} pages from {pdf_path.name}")
        print("-" * 50)

        for page_num, text in pages[:2]:
            preview = text[:500] + "..." if len(text) > 500 else text
            print(f"\n=== Page {page_num} ===")
            print(preview)

        print("\n=== Tables ===")
        tables = backend.extract_tables(pdf_path)
        print(f"Found {len(tables)} tables")

    except ExtractionError as e:
        print(f"Extraction error: {e.message}")
