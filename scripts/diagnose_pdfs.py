"""
Diagnostic script to analyze PDFs that fail text extraction.

Checks PDF structure, fonts, text layers, and compares extraction backends.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Tuple
from dataclasses import dataclass
import PyPDF2
import pdfplumber

from src.core import get_config


@dataclass
class PDFDiagnostic:
    """Diagnostic results for a single PDF."""
    filepath: Path
    num_pages: int
    has_text_pypdf: bool
    has_text_pdfplumber: bool
    pypdf_char_count: int
    pdfplumber_char_count: int
    has_images: bool
    has_fonts: bool
    font_names: List[str]
    is_encrypted: bool
    error: str = ""


def diagnose_pdf(filepath: Path) -> PDFDiagnostic:
    """
    Run comprehensive diagnostics on a single PDF.

    Args:
        filepath: Path to the PDF file.

    Returns:
        PDFDiagnostic with analysis results.
    """
    result = PDFDiagnostic(
        filepath=filepath,
        num_pages=0,
        has_text_pypdf=False,
        has_text_pdfplumber=False,
        pypdf_char_count=0,
        pdfplumber_char_count=0,
        has_images=False,
        has_fonts=False,
        font_names=[],
        is_encrypted=False
    )

    # PyPDF2 analysis
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            result.is_encrypted = reader.is_encrypted
            result.num_pages = len(reader.pages)

            total_text = ""
            for page in reader.pages:
                text = page.extract_text() or ""
                total_text += text

            result.pypdf_char_count = len(total_text.strip())
            result.has_text_pypdf = result.pypdf_char_count > 10

    except Exception as e:
        result.error += f"PyPDF2 error: {e}; "

    # pdfplumber analysis
    try:
        with pdfplumber.open(filepath) as pdf:
            total_text = ""
            all_fonts = set()

            for page in pdf.pages:
                text = page.extract_text() or ""
                total_text += text

                # Check for images
                if page.images:
                    result.has_images = True

                # Check fonts
                if hasattr(page, 'chars') and page.chars:
                    for char in page.chars[:100]:
                        if 'fontname' in char:
                            all_fonts.add(char['fontname'])

            result.pdfplumber_char_count = len(total_text.strip())
            result.has_text_pdfplumber = result.pdfplumber_char_count > 10
            result.font_names = list(all_fonts)[:10]
            result.has_fonts = len(all_fonts) > 0

    except Exception as e:
        result.error += f"pdfplumber error: {e}; "

    return result


def scan_directory(directory: Path, limit: int = None) -> List[PDFDiagnostic]:
    """
    Scan a directory and diagnose all PDFs.

    Args:
        directory: Directory to scan.
        limit: Optional limit on number of files to process.

    Returns:
        List of diagnostic results.
    """
    results = []
    pdf_files = list(directory.rglob("*.pdf"))

    if limit:
        pdf_files = pdf_files[:limit]

    total = len(pdf_files)
    print(f"Analyzing {total} PDF files...\n")

    for i, filepath in enumerate(pdf_files, 1):
        print(f"\r[{i}/{total}] Analyzing: {filepath.name[:50]:<50}", end="", flush=True)
        diag = diagnose_pdf(filepath)
        results.append(diag)

    print("\n")
    return results


def print_report(results: List[PDFDiagnostic]) -> None:
    """Print summary report of diagnostics."""
    total = len(results)
    no_text_both = [r for r in results if not r.has_text_pypdf and not r.has_text_pdfplumber]
    no_text_pypdf_only = [r for r in results if not r.has_text_pypdf and r.has_text_pdfplumber]
    no_text_pdfplumber_only = [r for r in results if r.has_text_pypdf and not r.has_text_pdfplumber]
    has_images_no_text = [r for r in no_text_both if r.has_images]
    encrypted = [r for r in results if r.is_encrypted]
    has_errors = [r for r in results if r.error]

    print("=" * 70)
    print("PDF DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"\nTotal PDFs analyzed: {total}")
    print(f"  - Extracted by both backends:    {total - len(no_text_both) - len(no_text_pypdf_only) - len(no_text_pdfplumber_only)}")
    print(f"  - No text (both backends):       {len(no_text_both)}")
    print(f"  - PyPDF2 failed, pdfplumber OK:  {len(no_text_pypdf_only)}")
    print(f"  - pdfplumber failed, PyPDF2 OK:  {len(no_text_pdfplumber_only)}")
    print(f"  - Encrypted:                     {len(encrypted)}")
    print(f"  - Has images but no text:        {len(has_images_no_text)}")
    print(f"  - Processing errors:             {len(has_errors)}")

    if no_text_both:
        print("\n" + "-" * 70)
        print("FILES WITH NO EXTRACTABLE TEXT (showing first 20):")
        print("-" * 70)
        for r in no_text_both[:20]:
            status = []
            if r.has_images:
                status.append("has_images")
            if r.has_fonts:
                status.append(f"fonts:{','.join(r.font_names[:3])}")
            if r.is_encrypted:
                status.append("encrypted")
            if r.error:
                status.append(f"error")

            status_str = f" [{', '.join(status)}]" if status else ""
            print(f"  - {r.filepath.name}{status_str}")

    if no_text_pypdf_only:
        print("\n" + "-" * 70)
        print("FILES WHERE ONLY PDFPLUMBER WORKS (showing first 10):")
        print("-" * 70)
        for r in no_text_pypdf_only[:10]:
            print(f"  - {r.filepath.name} (pdfplumber: {r.pdfplumber_char_count} chars)")

    if has_errors:
        print("\n" + "-" * 70)
        print("FILES WITH PROCESSING ERRORS (showing first 10):")
        print("-" * 70)
        for r in has_errors[:10]:
            print(f"  - {r.filepath.name}: {r.error[:80]}")


def analyze_single_file(filepath: Path) -> None:
    """Detailed analysis of a single PDF file."""
    print(f"Detailed analysis of: {filepath.name}")
    print("=" * 70)

    diag = diagnose_pdf(filepath)

    print(f"Pages:              {diag.num_pages}")
    print(f"Encrypted:          {diag.is_encrypted}")
    print(f"Has images:         {diag.has_images}")
    print(f"Has fonts:          {diag.has_fonts}")
    print(f"Font names:         {', '.join(diag.font_names) if diag.font_names else 'None detected'}")
    print(f"PyPDF2 chars:       {diag.pypdf_char_count}")
    print(f"pdfplumber chars:   {diag.pdfplumber_char_count}")

    if diag.error:
        print(f"Errors:             {diag.error}")

    # Show sample text
    print("\n" + "-" * 70)
    print("SAMPLE TEXT (first 500 chars from each backend):")
    print("-" * 70)

    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if reader.pages:
                text = reader.pages[0].extract_text() or ""
                print(f"\nPyPDF2:\n{text[:500] if text else '(empty)'}")
    except Exception as e:
        print(f"\nPyPDF2: Error - {e}")

    try:
        with pdfplumber.open(filepath) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""
                print(f"\npdfplumber:\n{text[:500] if text else '(empty)'}")
    except Exception as e:
        print(f"\npdfplumber: Error - {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose PDF extraction issues")
    parser.add_argument("--file", type=str, help="Analyze a single file")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of files to scan")
    parser.add_argument("--all", action="store_true", help="Scan all files (no limit)")

    args = parser.parse_args()

    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)
        analyze_single_file(filepath)
    else:
        config = get_config()
        data_dir = config.paths.data_directory

        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            sys.exit(1)

        limit = None if args.all else args.limit
        results = scan_directory(data_dir, limit=limit)
        print_report(results)


if __name__ == "__main__":
    main()
