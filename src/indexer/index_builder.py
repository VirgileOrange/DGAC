"""
Main indexing pipeline for the PDF Search Engine.

Orchestrates the complete indexing workflow: scanning files,
extracting text, and storing in the database with progress tracking.
Includes optional semantic indexing for vector search.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from ..core import get_config, get_logger, ExtractionError
from ..database import init_schema, reset_schema, DocumentRepository, get_connection
from ..extraction import FileScanner, PDFExtractor
from ..utils import get_file_hash, get_relative_path, clean_text

logger = get_logger(__name__)


@dataclass
class IndexingStats:
    """Statistics from an indexing run."""
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    pages_indexed: int = 0
    chunks_indexed: int = 0
    semantic_errors: int = 0
    errors: List[str] = field(default_factory=list)


class IndexBuilder:
    """
    Orchestrates the PDF indexing pipeline.

    Handles file discovery, text extraction, and database storage
    with batching for performance and progress callbacks for UI.
    Includes optional semantic indexing for vector search.
    """

    def __init__(
        self,
        reset: bool = False,
        progress_callback: Callable[[int, int, str], None] = None,
        semantic_enabled: Optional[bool] = None
    ):
        """
        Initialize the index builder.

        Args:
            reset: If True, drop and recreate the database schema.
            progress_callback: Optional callback(current, total, filename)
                              called during indexing for progress updates.
            semantic_enabled: Override config semantic.enabled setting.
                             If None, uses config value.
        """
        self.config = get_config()
        self.reset = reset
        self.progress_callback = progress_callback

        self.scanner = FileScanner()
        self.extractor = PDFExtractor()
        self.repository = DocumentRepository()

        self.batch_size = self.config.indexing.batch_size
        self.skip_existing = self.config.indexing.skip_existing
        self.log_every = self.config.indexing.log_progress_every

        if semantic_enabled is None:
            self.semantic_enabled = self.config.semantic.enabled
        else:
            self.semantic_enabled = semantic_enabled

        self.semantic_indexer = None
        self._pending_semantic: List[Dict] = []

    def build(self) -> IndexingStats:
        """
        Run the complete indexing pipeline.

        Returns:
            IndexingStats with counts and any errors encountered.
        """
        stats = IndexingStats()

        logger.info("Starting indexing pipeline")

        if self.reset:
            reset_schema()
        else:
            init_schema()

        if self.semantic_enabled:
            self._init_semantic_indexer()

        indexed_paths = self._get_indexed_paths() if self.skip_existing else set()

        pdf_files = list(self.scanner.scan())
        stats.files_scanned = len(pdf_files)

        logger.info(f"Found {stats.files_scanned} PDF files to process")

        batch: List[tuple] = []

        for i, filepath in enumerate(pdf_files):
            filepath_str = str(filepath)

            if filepath_str in indexed_paths:
                stats.files_skipped += 1
                continue

            if self.progress_callback:
                self.progress_callback(i + 1, stats.files_scanned, filepath.name)

            try:
                pages_added, pages_data = self._process_file_with_pages(filepath, batch)

                if pages_added > 0:
                    stats.files_indexed += 1
                    stats.pages_indexed += pages_added

                    if self.semantic_enabled and pages_data:
                        self._pending_semantic.append({
                            "filepath": filepath,
                            "filename": filepath.name,
                            "pages": pages_data
                        })

            except ExtractionError as e:
                stats.files_failed += 1
                error_msg = f"{filepath.name}: {e.message}"
                stats.errors.append(error_msg)
                logger.warning(f"Failed to extract: {error_msg}")

            except Exception as e:
                stats.files_failed += 1
                error_msg = f"{filepath.name}: {str(e)}"
                stats.errors.append(error_msg)
                logger.error(f"Unexpected error: {error_msg}")

            if len(batch) >= self.batch_size:
                self._commit_batch(batch)
                batch.clear()

            if (i + 1) % self.log_every == 0:
                logger.info(
                    f"Progress: {i + 1}/{stats.files_scanned} files "
                    f"({stats.files_indexed} indexed, {stats.files_failed} failed)"
                )

        if batch:
            self._commit_batch(batch)

        logger.info(
            f"FTS5 indexing complete: {stats.files_indexed} files indexed, "
            f"{stats.pages_indexed} pages, {stats.files_failed} failures"
        )

        if self.semantic_enabled and self._pending_semantic:
            self._run_semantic_indexing(stats)

        return stats

    def _init_semantic_indexer(self) -> None:
        """Initialize the semantic indexer lazily."""
        if self.semantic_indexer is None:
            from .semantic_indexer import SemanticIndexer
            self.semantic_indexer = SemanticIndexer(
                progress_callback=self.progress_callback,
                enabled=self.semantic_enabled
            )
            self.semantic_indexer.ensure_schema()
            logger.info("Semantic indexer initialized")

    def _run_semantic_indexing(self, stats: IndexingStats) -> None:
        """Run semantic indexing for all pending documents."""
        logger.info(f"Starting semantic indexing for {len(self._pending_semantic)} documents")

        for doc_data in self._pending_semantic:
            filepath = doc_data["filepath"]
            filename = doc_data["filename"]
            pages = doc_data["pages"]

            try:
                doc_id = self._get_doc_id_for_filepath(str(filepath))
                if doc_id is None:
                    logger.warning(f"No doc_id found for {filename}, skipping semantic")
                    continue

                chunks_count = self.semantic_indexer.index_document(
                    doc_id=doc_id,
                    pages=pages,
                    filename=filename
                )
                stats.chunks_indexed += chunks_count

            except Exception as e:
                stats.semantic_errors += 1
                logger.error(f"Semantic indexing failed for {filename}: {e}")

        self._pending_semantic.clear()

        logger.info(
            f"Semantic indexing complete: {stats.chunks_indexed} chunks, "
            f"{stats.semantic_errors} errors"
        )

    def _get_doc_id_for_filepath(self, filepath: str) -> Optional[int]:
        """Get the document ID for a filepath from the database."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM documents WHERE filepath = ? LIMIT 1",
                (filepath,)
            ).fetchone()
            return row["id"] if row else None

    def _get_indexed_paths(self) -> Set[str]:
        """Get set of already indexed file paths."""
        return self.repository.get_indexed_filepaths()

    def _process_file(self, filepath: Path, batch: List[tuple]) -> int:
        """
        Extract and prepare a single PDF for indexing.

        Args:
            filepath: Path to the PDF file.
            batch: List to append document tuples to.

        Returns:
            Number of pages extracted.
        """
        pages_added, _ = self._process_file_with_pages(filepath, batch)
        return pages_added

    def _process_file_with_pages(
        self,
        filepath: Path,
        batch: List[tuple]
    ) -> Tuple[int, List[Tuple[int, str]]]:
        """
        Extract and prepare a single PDF for indexing.

        Args:
            filepath: Path to the PDF file.
            batch: List to append document tuples to.

        Returns:
            Tuple of (pages_count, list of (page_num, cleaned_content) tuples).
        """
        pages = self.extractor.extract(filepath)

        if not pages:
            return 0, []

        file_hash = get_file_hash(filepath)
        relative_path = get_relative_path(filepath, self.config.paths.data_directory)
        filename = filepath.name

        pages_data: List[Tuple[int, str]] = []

        for page_num, content in pages:
            cleaned_content = clean_text(content)

            if not cleaned_content:
                continue

            batch.append((
                str(filepath),
                filename,
                page_num,
                cleaned_content,
                relative_path,
                file_hash
            ))

            pages_data.append((page_num, cleaned_content))

        return len(pages), pages_data

    def _commit_batch(self, batch: List[tuple]) -> int:
        """
        Commit a batch of documents to the database.

        Args:
            batch: List of document tuples.

        Returns:
            Number of rows inserted.
        """
        if not batch:
            return 0

        inserted = self.repository.insert_batch(batch)
        logger.debug(f"Committed batch: {inserted} rows")
        return inserted

    def index_single(self, filepath: Path) -> int:
        """
        Index a single PDF file.

        Useful for incremental updates or testing.

        Args:
            filepath: Path to the PDF file.

        Returns:
            Number of pages indexed.
        """
        init_schema()

        batch: List[tuple] = []
        pages_added = self._process_file(filepath, batch)

        if batch:
            self._commit_batch(batch)

        return pages_added

    def reindex_file(self, filepath: Path) -> int:
        """
        Re-index a file, replacing existing entries.

        Args:
            filepath: Path to the PDF file.

        Returns:
            Number of pages indexed.
        """
        self.repository.delete_by_filepath(str(filepath))
        return self.index_single(filepath)


def progress_printer(current: int, total: int, filename: str) -> None:
    """Simple progress callback that prints to console."""
    percent = (current / total) * 100 if total > 0 else 0
    print(f"\r[{percent:5.1f}%] {current}/{total} - {filename[:50]:<50}", end="", flush=True)


if __name__ == "__main__":
    import sys

    reset_flag = "--reset" in sys.argv

    builder = IndexBuilder(reset=reset_flag, progress_callback=progress_printer)

    print("Starting indexing...")
    print("-" * 60)

    stats = builder.build()

    print("\n" + "-" * 60)
    print("Indexing Summary:")
    print(f"  Files scanned:  {stats.files_scanned}")
    print(f"  Files indexed:  {stats.files_indexed}")
    print(f"  Files skipped:  {stats.files_skipped}")
    print(f"  Files failed:   {stats.files_failed}")
    print(f"  Pages indexed:  {stats.pages_indexed}")

    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for error in stats.errors[:10]:
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")
