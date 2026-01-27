"""
Pytest fixtures and configuration for the test suite.

Provides temporary directories, sample PDFs, and mock configurations
to ensure tests are isolated and safe.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator

import sys
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files.

    Yields:
        Path to temporary directory, cleaned up after test.
    """
    tmp = tempfile.mkdtemp(prefix="pdf_search_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_config(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Create a temporary config.json for testing.

    Args:
        temp_dir: Temporary directory fixture.

    Yields:
        Path to temporary config file.
    """
    config_dir = temp_dir / "config"
    config_dir.mkdir()

    data_dir = temp_dir / "data"
    data_dir.mkdir()

    output_dir = temp_dir / "output"
    output_dir.mkdir()

    logs_dir = output_dir / "logs"
    logs_dir.mkdir()

    config_data = {
        "paths": {
            "data_directory": str(data_dir),
            "database_path": str(output_dir / "test.db"),
            "logs_directory": str(logs_dir)
        },
        "assets": {
            "css_path": "assets/style.css",
            "logo_path": "assets/logo.png"
        },
        "extraction": {
            "primary_backend": "pypdf2",
            "fallback_backend": "pdfplumber",
            "max_file_size_mb": 100,
            "supported_extensions": [".pdf"]
        },
        "indexing": {
            "batch_size": 10,
            "commit_frequency": 10,
            "skip_existing": True,
            "log_progress_every": 5
        },
        "search": {
            "default_limit": 20,
            "max_limit": 100,
            "snippet_length": 100,
            "bm25_weights": {
                "filename": 1.0,
                "content": 10.0
            },
            "tokenizer": "unicode61"
        },
        "gui": {
            "page_title": "Test PDF Search",
            "results_per_page": 10,
            "enable_pdf_preview": True
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(levelname)s - %(message)s",
            "max_file_size_mb": 1,
            "backup_count": 1
        }
    }

    config_path = config_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    yield config_path


@pytest.fixture
def sample_pdf_content() -> bytes:
    """
    Create minimal valid PDF content for testing.

    Returns:
        Bytes representing a minimal PDF with text.
    """
    # Minimal PDF with "Hello World" text
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF"""
    return pdf_content


@pytest.fixture
def sample_pdf(temp_dir: Path, sample_pdf_content: bytes) -> Path:
    """
    Create a sample PDF file for testing.

    Args:
        temp_dir: Temporary directory fixture.
        sample_pdf_content: PDF content fixture.

    Returns:
        Path to the created PDF file.
    """
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(sample_pdf_content)
    return pdf_path


@pytest.fixture
def sample_pdf_collection(temp_dir: Path, sample_pdf_content: bytes) -> Path:
    """
    Create multiple sample PDF files in a directory structure.

    Args:
        temp_dir: Temporary directory fixture.
        sample_pdf_content: PDF content fixture.

    Returns:
        Path to the data directory containing PDFs.
    """
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Create subdirectories with PDFs
    subdir1 = data_dir / "folder1"
    subdir1.mkdir()

    subdir2 = data_dir / "folder2"
    subdir2.mkdir()

    # Create PDF files
    (data_dir / "root_doc.pdf").write_bytes(sample_pdf_content)
    (subdir1 / "doc1.pdf").write_bytes(sample_pdf_content)
    (subdir1 / "doc2.pdf").write_bytes(sample_pdf_content)
    (subdir2 / "doc3.pdf").write_bytes(sample_pdf_content)

    # Create a non-PDF file (should be ignored)
    (data_dir / "readme.txt").write_text("Not a PDF")

    return data_dir


@pytest.fixture
def temp_database(temp_dir: Path) -> Path:
    """
    Create path for a temporary database.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Path where test database should be created.
    """
    return temp_dir / "test.db"


@pytest.fixture
def reset_config_singleton():
    """
    Reset the config singleton between tests.

    This ensures each test gets a fresh config instance.
    """
    from src.core import config_loader
    config_loader._config_instance = None
    yield
    config_loader._config_instance = None


@pytest.fixture
def reset_logger_singleton():
    """
    Reset the logger initialization flag between tests.
    """
    from src.core import logger
    logger._logger_initialized = False
    yield
    logger._logger_initialized = False


@pytest.fixture
def reset_db_singleton():
    """
    Reset the database manager singleton between tests.
    """
    from src.database import connection
    connection._db_manager = None
    yield
    connection._db_manager = None


@pytest.fixture
def configured_db(temp_config, reset_config_singleton, reset_db_singleton):
    """
    Set up a fully configured database using temp config.

    This fixture initializes config with temp paths and resets
    both config and db singletons, ready for schema operations.
    """
    from src.core.config_loader import get_config
    get_config(temp_config)
    yield
    # Cleanup happens via reset fixtures
