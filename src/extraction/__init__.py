"""
PDF extraction module for the PDF Search Engine.

Provides file discovery and text extraction with multiple backends
(PyPDF2 and pdfplumber) with automatic fallback support.
Includes semantic chunking for embedding preparation.
"""

from .file_scanner import FileScanner
from .pypdf_backend import PyPDFBackend
from .pdfplumber_backend import PDFPlumberBackend
from .extractor import PDFExtractor
from .semantic_chunker import SemanticChunker, SemanticChunk

__all__ = [
    "FileScanner",
    "PyPDFBackend",
    "PDFPlumberBackend",
    "PDFExtractor",
    "SemanticChunker",
    "SemanticChunk"
]
