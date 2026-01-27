"""
File scanner for recursive PDF discovery.

Handles deeply nested directory structures (13-14 levels) with memory-efficient
iteration and configurable filtering by extension and file size.
"""

from pathlib import Path
from typing import Iterator, List, Union

from ..core import get_config, get_logger
from ..utils import get_file_size_mb

logger = get_logger(__name__)


class FileScanner:
    """
    Recursively discovers PDF files in a directory tree.

    Uses generator-based iteration for memory efficiency when
    processing large collections.
    """

    def __init__(
        self,
        root_directory: Union[str, Path] = None,
        extensions: List[str] = None,
        max_file_size_mb: float = None
    ):
        """
        Initialize the file scanner.

        Args:
            root_directory: Directory to scan. Defaults to config value.
            extensions: List of file extensions to include (e.g., [".pdf"]).
            max_file_size_mb: Skip files larger than this size.
        """
        config = get_config()

        self.root_directory = Path(root_directory or config.paths.data_directory)
        self.extensions = extensions or config.extraction.supported_extensions
        self.max_file_size_mb = max_file_size_mb or config.extraction.max_file_size_mb

        self.extensions = [ext.lower() for ext in self.extensions]

    def scan(self) -> Iterator[Path]:
        """
        Scan directory and yield matching file paths.

        Yields:
            Path objects for each matching file.

        Logs:
            Progress every 1000 files discovered.
        """
        if not self.root_directory.exists():
            logger.error(f"Root directory does not exist: {self.root_directory}")
            return

        logger.info(f"Scanning directory: {self.root_directory}")

        file_count = 0
        skipped_size = 0
        skipped_ext = 0

        for filepath in self.root_directory.rglob("*"):
            if not filepath.is_file():
                continue

            if filepath.suffix.lower() not in self.extensions:
                skipped_ext += 1
                continue

            try:
                size_mb = get_file_size_mb(filepath)
                if size_mb > self.max_file_size_mb:
                    logger.debug(f"Skipping large file ({size_mb}MB): {filepath.name}")
                    skipped_size += 1
                    continue
            except OSError as e:
                logger.warning(f"Cannot access file {filepath}: {e}")
                continue

            file_count += 1

            if file_count % 1000 == 0:
                logger.info(f"Discovered {file_count} files...")

            yield filepath

        logger.info(
            f"Scan complete: {file_count} files found, "
            f"{skipped_size} skipped (too large), "
            f"{skipped_ext} skipped (wrong extension)"
        )

    def count(self) -> int:
        """
        Count total matching files without loading all paths.

        Returns:
            Number of matching files.
        """
        return sum(1 for _ in self.scan())

    def list_all(self) -> List[Path]:
        """
        Get all matching files as a list.

        Warning: May consume significant memory for large collections.

        Returns:
            List of all matching file paths.
        """
        return list(self.scan())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_dir = Path(sys.argv[1])
    else:
        test_dir = Path(".")

    scanner = FileScanner(
        root_directory=test_dir,
        extensions=[".pdf"],
        max_file_size_mb=100
    )

    print(f"Scanning: {test_dir}")
    print("-" * 50)

    for i, filepath in enumerate(scanner.scan()):
        print(f"  {filepath.name}")
        if i >= 9:
            print("  ... (showing first 10 only)")
            break
