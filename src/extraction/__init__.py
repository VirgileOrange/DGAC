"""
PDF extraction module for the PDF Search Engine.

Provides file discovery and text extraction with multiple backends
(PyPDF2 and pdfplumber) with automatic fallback support.
"""

from .file_scanner import FileScanner
from .pypdf_backend import PyPDFBackend
from .pdfplumber_backend import PDFPlumberBackend
from .extractor import PDFExtractor

__all__ = [
    "FileScanner",
    "PyPDFBackend",
    "PDFPlumberBackend",
    "PDFExtractor"
]
