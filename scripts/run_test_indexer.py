"""
Test script for the PDF indexing pipeline.

Creates a temporary test database, indexes 10 random documents,
demonstrates all search features, then cleans up.

Usage:
    python scripts/run_test_indexer.py              # Test with semantic if available
    python scripts/run_test_indexer.py --no-semantic  # Test without semantic
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import get_config, get_logger, ConfigurationError
from src.core.config_loader import reload_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the PDF indexing pipeline with a temporary database"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config.json file"
    )

    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic indexing (embeddings)"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of documents to index (default: 10)"
    )

    return parser.parse_args()


def run_test(config, semantic_enabled: bool, doc_count: int) -> None:
    """Run indexer test with temporary database."""
    import src.database.connection as db_conn
    import src.database.schema as db_schema
    from src.database import (
        init_schema, reset_schema, get_statistics,
        init_vector_index, is_vec_extension_available,
        reset_vec_extension_cache
    )
    from src.extraction import FileScanner
    from src.search import HybridEngine, SearchMode

    test_db_path = config.paths.database_path.parent / "test_indexer.db"

    # Check if sqlite-vec package is available (not database-specific)
    vec_available = False
    try:
        import sqlite_vec
        vec_available = True
    except ImportError:
        pass

    if semantic_enabled and not vec_available:
        print("Warning: sqlite-vec package not installed. Semantic search disabled.")
        print("         Install with: pip install sqlite-vec")
        semantic_enabled = False

    print("=" * 60)
    print("PDF Search Engine - TEST MODE")
    print("=" * 60)
    print(f"Data directory:    {config.paths.data_directory}")
    print(f"Test database:     {test_db_path}")
    print(f"Documents to test: {doc_count}")
    print(f"Semantic indexing: {'enabled' if semantic_enabled else 'disabled'}")
    print(f"sqlite-vec:        {'available' if vec_available else 'NOT AVAILABLE'}")
    print("=" * 60)
    print("\nThis will create a temporary test database.")
    print("The production database will NOT be modified.\n")

    scanner = FileScanner()
    all_pdfs = list(scanner.scan())

    if not all_pdfs:
        print("No PDF files found in data directory.")
        return

    test_count = min(doc_count, len(all_pdfs))
    test_files = random.sample(all_pdfs, test_count)

    print(f"Found {len(all_pdfs)} PDF files. Selecting {test_count} random files for test.\n")
    print("Selected files:")
    for i, f in enumerate(test_files, 1):
        print(f"  {i}. {f.name}")
    print()

    original_db_manager = db_conn._db_manager
    db_conn._db_manager = db_conn.DatabaseManager(test_db_path)

    # Reset the vec extension cache to ensure fresh loading attempt with new database
    reset_vec_extension_cache()

    try:
        reset_schema()

        if semantic_enabled:
            print("Initializing vector index...")
            try:
                vec_result = init_vector_index()
                print(f"  init_vector_index returned: {vec_result}")
                if not vec_result:
                    # Check if it already exists (that's ok) vs failed
                    with db_conn.get_connection() as conn:
                        try:
                            conn.execute("SELECT 1 FROM chunks_vec_idx LIMIT 1")
                            print("  Vector index already exists - OK")
                        except Exception:
                            print("Warning: Failed to create vector index. Semantic search disabled.")
                            semantic_enabled = False
            except Exception as e:
                print(f"Warning: Error creating vector index: {e}")
                import traceback
                traceback.print_exc()
                semantic_enabled = False

        from src.database import DocumentRepository
        from src.extraction import PDFExtractor
        from src.utils import get_file_hash, get_relative_path, clean_text

        extractor = PDFExtractor()
        repository = DocumentRepository()

        print("Indexing test documents...\n")

        total_pages = 0
        indexed_files = 0
        failed_files = 0
        pages_data_all = []

        for i, filepath in enumerate(test_files, 1):
            print(f"[{i}/{test_count}] {filepath.name}...", end=" ", flush=True)

            try:
                pages = extractor.extract(filepath)

                if not pages:
                    print("(no content)")
                    continue

                file_hash = get_file_hash(filepath)
                relative_path = get_relative_path(filepath, config.paths.data_directory)
                filename = filepath.name

                batch = []
                doc_pages = []

                for page_num, content in pages:
                    cleaned = clean_text(content)
                    if cleaned:
                        batch.append((
                            str(filepath),
                            filename,
                            page_num,
                            cleaned,
                            relative_path,
                            file_hash
                        ))
                        doc_pages.append((page_num, cleaned))

                if batch:
                    repository.insert_batch(batch)
                    total_pages += len(batch)
                    indexed_files += 1
                    pages_data_all.append({
                        "filepath": filepath,
                        "filename": filename,
                        "pages": doc_pages
                    })
                    print(f"OK ({len(batch)} pages)")
                else:
                    print("(empty)")

            except Exception as e:
                failed_files += 1
                print(f"FAILED: {e}")

        print("\n" + "=" * 60)
        print("Test Indexing Complete (FTS5)")
        print("=" * 60)
        print(f"Files indexed:     {indexed_files}")
        print(f"Files failed:      {failed_files}")
        print(f"Pages indexed:     {total_pages}")

        if semantic_enabled and pages_data_all:
            print("\n" + "-" * 60)
            print("Running semantic indexing...")
            print("-" * 60)

            from src.indexer.semantic_indexer import SemanticIndexer
            from src.database import get_connection

            # Show semantic config being used
            print(f"  Endpoint: {config.semantic.endpoint}")
            print(f"  Model: {config.semantic.embedding_model}")
            print(f"  Dimensions: {config.semantic.embedding_dimensions}")
            print()

            try:
                semantic_indexer = SemanticIndexer(enabled=True)
                print(f"  SemanticIndexer created (enabled={semantic_indexer.enabled})")
            except Exception as e:
                print(f"  ERROR creating SemanticIndexer: {e}")
                import traceback
                traceback.print_exc()
                semantic_indexer = None

            if semantic_indexer:
                try:
                    semantic_indexer.ensure_schema()
                    print("  Schema ensured successfully")
                except Exception as e:
                    print(f"  ERROR ensuring schema: {e}")
                    import traceback
                    traceback.print_exc()

                total_chunks = 0

                for doc_data in pages_data_all:
                    filepath = doc_data["filepath"]
                    filename = doc_data["filename"]
                    pages = doc_data["pages"]

                    with get_connection() as conn:
                        row = conn.execute(
                            "SELECT id FROM documents WHERE filepath = ? LIMIT 1",
                            (str(filepath),)
                        ).fetchone()
                        if not row:
                            print(f"  {filename}: document not found in DB")
                            continue
                        doc_id = row["id"]

                    try:
                        print(f"  Indexing {filename} (doc_id={doc_id}, pages={len(pages)})...", end=" ", flush=True)
                        chunks = semantic_indexer.index_document(doc_id, pages, filename)
                        total_chunks += chunks
                        print(f"{chunks} chunks")
                    except Exception as e:
                        print(f"FAILED")
                        print(f"    Error: {e}")
                        import traceback
                        traceback.print_exc()

                print(f"\nTotal chunks indexed: {total_chunks}")

        print("\n" + "=" * 60)
        print("Database Statistics")
        print("=" * 60)
        stats = get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("Search Demo")
        print("=" * 60)

        engine = HybridEngine()

        with db_conn.get_connection() as conn:
            sample_row = conn.execute(
                "SELECT content FROM documents WHERE LENGTH(content) > 100 LIMIT 1"
            ).fetchone()

        if sample_row:
            sample_text = sample_row["content"]
            words = [w for w in sample_text.split() if len(w) > 4][:20]
            if words:
                test_query = " ".join(random.sample(words, min(2, len(words))))
            else:
                test_query = "document"
        else:
            test_query = "document"

        print(f"\nTest query: \"{test_query}\"\n")

        print("-" * 40)
        print("LEXICAL SEARCH (BM25)")
        print("-" * 40)
        try:
            results, search_stats = engine.search(test_query, mode=SearchMode.LEXICAL, limit=5)
            print(f"Results: {search_stats.total_results} | Time: {search_stats.execution_time_ms:.1f}ms")
            for i, r in enumerate(results[:3], 1):
                snippet = r.snippet[:100].replace('\n', ' ')
                print(f"  {i}. {r.filename} (p.{r.page_num}) - Score: {r.score:.2f}")
                print(f"     {snippet}...")
        except Exception as e:
            print(f"  Error: {e}")

        if semantic_enabled:
            print("\n" + "-" * 40)
            print("SEMANTIC SEARCH")
            print("-" * 40)
            try:
                results, search_stats = engine.search(test_query, mode=SearchMode.SEMANTIC, limit=5)
                print(f"Results: {search_stats.total_results} | Time: {search_stats.execution_time_ms:.1f}ms")
                for i, r in enumerate(results[:3], 1):
                    snippet = r.snippet[:100].replace('\n', ' ')
                    sim = f" | Sim: {r.similarity:.2%}" if r.similarity else ""
                    print(f"  {i}. {r.filename} (p.{r.page_num}) - Score: {r.score:.4f}{sim}")
                    print(f"     {snippet}...")
            except Exception as e:
                print(f"  Error: {e}")

            print("\n" + "-" * 40)
            print("HYBRID SEARCH (RRF)")
            print("-" * 40)
            try:
                results, search_stats = engine.search(test_query, mode=SearchMode.HYBRID, limit=5)
                print(f"Results: {search_stats.total_results} | Time: {search_stats.execution_time_ms:.1f}ms")
                print(f"Lexical: {search_stats.lexical_results} | Semantic: {search_stats.semantic_results} | Overlap: {search_stats.overlap_count}")
                for i, r in enumerate(results[:3], 1):
                    snippet = r.snippet[:100].replace('\n', ' ')
                    source = {"lexical": "[L]", "semantic": "[S]", "both": "[L+S]"}.get(r.source, "")
                    print(f"  {i}. {source} {r.filename} (p.{r.page_num}) - Score: {r.score:.4f}")
                    print(f"     {snippet}...")
            except Exception as e:
                print(f"  Error: {e}")

        print("\n" + "=" * 60)

    finally:
        db_conn._db_manager = original_db_manager

        print("\nTest complete.")
        input("Press Enter to delete test database and exit...")

        if test_db_path.exists():
            test_db_path.unlink()
            print(f"Deleted: {test_db_path}")

        wal_path = test_db_path.parent / (test_db_path.name + "-wal")
        shm_path = test_db_path.parent / (test_db_path.name + "-shm")

        if wal_path.exists():
            wal_path.unlink()
        if shm_path.exists():
            shm_path.unlink()

        print("Test cleanup complete.")


def main():
    """Main entry point for the test indexer."""
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

    semantic_enabled = config.semantic.enabled and not args.no_semantic

    run_test(config, semantic_enabled, args.count)


if __name__ == "__main__":
    main()
