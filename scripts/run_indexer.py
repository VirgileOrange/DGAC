"""
CLI script to run the PDF indexing pipeline.

Usage:
    python scripts/run_indexer.py           # Incremental index (FTS5 + semantic)
    python scripts/run_indexer.py --reset   # Full rebuild
    python scripts/run_indexer.py --no-semantic  # Skip semantic indexing
    python scripts/run_indexer.py --config path/to/config.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import get_config, get_logger, ConfigurationError
from src.core.config_loader import reload_config
from src.indexer import IndexBuilder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Index PDF files for full-text and semantic search"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing index and rebuild from scratch"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config.json file"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic indexing (embeddings)"
    )

    return parser.parse_args()


def progress_callback(current: int, total: int, filename: str) -> None:
    """Print progress to console."""
    percent = (current / total) * 100 if total > 0 else 0
    bar_width = 30
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_width - filled)

    print(f"\r[{bar}] {percent:5.1f}% ({current}/{total}) {filename[:40]:<40}", end="", flush=True)


def main():
    """Main entry point for the indexer CLI."""
    args = parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        reload_config(config_path)

    try:
        config = get_config()
    except ConfigurationError as e:
        print(f"Configuration error: {e.message}")
        sys.exit(1)

    logger = get_logger(__name__)

    semantic_enabled = config.semantic.enabled and not args.no_semantic

    print("=" * 60)
    print("PDF Search Engine - Indexer")
    print("=" * 60)
    print(f"Data directory:    {config.paths.data_directory}")
    print(f"Database path:     {config.paths.database_path}")
    print(f"Reset mode:        {args.reset}")
    print(f"Semantic indexing: {'enabled' if semantic_enabled else 'disabled'}")
    print("=" * 60)

    if args.reset:
        response = input("This will DELETE all existing index data. Continue? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    callback = None if args.quiet else progress_callback

    builder = IndexBuilder(
        reset=args.reset,
        progress_callback=callback,
        semantic_enabled=semantic_enabled
    )

    print("\nStarting indexing...\n")

    stats = builder.build()

    if not args.quiet:
        print("\n")

    print("=" * 60)
    print("Indexing Complete")
    print("=" * 60)
    print(f"Files scanned:     {stats.files_scanned:,}")
    print(f"Files indexed:     {stats.files_indexed:,}")
    print(f"Files skipped:     {stats.files_skipped:,}")
    print(f"Files failed:      {stats.files_failed:,}")
    print(f"Pages indexed:     {stats.pages_indexed:,}")

    if semantic_enabled:
        print("-" * 60)
        print(f"Chunks indexed:    {stats.chunks_indexed:,}")
        print(f"Semantic errors:   {stats.semantic_errors:,}")

    print("=" * 60)

    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for error in stats.errors[:20]:
            print(f"  - {error}")
        if len(stats.errors) > 20:
            print(f"  ... and {len(stats.errors) - 20} more errors")

    if stats.files_failed > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
