"""
PyPDF2-based text extraction backend.

Fast extraction suitable for most standard PDF files.
Handles encryption detection and empty password decryption.
"""

from pathlib import Path
from typing import List, Tuple, Union

from pypdf import PdfReader

from ..core import get_logger, ExtractionError

logger = get_logger(__name__)


class PyPDFBackend:
    """
    PDF text extraction using PyPDF2 library.

    Provides fast extraction for standard PDFs with basic
    encryption handling.
    """

    name = "pypdf2"

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
            reader = PdfReader(filepath)

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise ExtractionError(
                        "PDF is encrypted and cannot be decrypted",
                        filepath=str(filepath)
                    )

            total_pages = len(reader.pages)
            logger.debug(f"Processing {total_pages} pages: {filepath.name}")

            for page_num, page in enumerate(reader.pages, start=1):
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

        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(
                f"PyPDF extraction failed: {e}",
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
            reader = PdfReader(filepath)

            if reader.is_encrypted:
                reader.decrypt("")

            # Convert 1-indexed to 0-indexed
            page = reader.pages[page_num - 1]
            return page.extract_text() or ""

        except Exception as e:
            raise ExtractionError(
                f"Failed to extract page {page_num}: {e}",
                filepath=str(filepath)
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pypdf_backend.py <pdf_file>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    backend = PyPDFBackend()

    try:
        pages = backend.extract(pdf_path)
        print(f"Extracted {len(pages)} pages from {pdf_path.name}")
        print("-" * 50)

        for page_num, text in pages[:2]:
            preview = text[:500] + "..." if len(text) > 500 else text
            print(f"\n=== Page {page_num} ===")
            print(preview)

    except ExtractionError as e:
        print(f"Extraction error: {e.message}")
