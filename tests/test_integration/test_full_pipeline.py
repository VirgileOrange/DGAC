"""
Integration tests for the full indexing and search pipeline.

Tests the complete flow from PDF files to searchable index.

SAFETY NOTE: All tests use the `configured_db` fixture which:
- Creates a temporary directory via tempfile.mkdtemp()
- Configures the database singleton to use a temp path
- Cleans up all temp files after the test
- Never touches real data directories
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.database.schema import init_schema, get_statistics
from src.database.repository import DocumentRepository
from src.search.bm25_engine import BM25Engine
from src.search.models import SearchQuery


class TestFullPipeline:
    """
    Integration tests for the complete index-and-search pipeline.

    These tests verify that:
    1. Documents can be indexed into the database
    2. The FTS5 index is properly created
    3. Search returns relevant results
    4. Statistics are accurate
    """

    @pytest.fixture
    def integrated_system(self, configured_db):
        """
        Set up an integrated system with indexed documents in temp DB.

        Creates a database with pre-indexed test documents
        for search testing.
        """
        init_schema()
        repo = DocumentRepository()

        # Index a collection of test documents
        test_documents = [
            {
                "filepath": "/docs/aviation/reglementation.pdf",
                "filename": "reglementation.pdf",
                "relative_path": "aviation/reglementation.pdf",
                "file_hash": "hash_aviation_1",
                "pages": [
                    (1, "Règlement européen sur l'aviation civile. Les normes de sécurité aérienne sont définies par l'EASA."),
                    (2, "Contrôle du trafic aérien et gestion de l'espace aérien français."),
                ]
            },
            {
                "filepath": "/docs/maritime/navigation.pdf",
                "filename": "navigation.pdf",
                "relative_path": "maritime/navigation.pdf",
                "file_hash": "hash_maritime_1",
                "pages": [
                    (1, "Règles de navigation maritime et sécurité en mer."),
                    (2, "Protocoles de communication entre navires."),
                ]
            },
            {
                "filepath": "/docs/transport/infrastructure.pdf",
                "filename": "infrastructure.pdf",
                "relative_path": "transport/infrastructure.pdf",
                "file_hash": "hash_transport_1",
                "pages": [
                    (1, "Infrastructure de transport civil et développement durable."),
                ]
            },
            {
                "filepath": "/docs/securite/procedures.pdf",
                "filename": "procedures.pdf",
                "relative_path": "securite/procedures.pdf",
                "file_hash": "hash_securite_1",
                "pages": [
                    (1, "Procédures de sécurité pour l'aviation et le transport."),
                    (2, "Gestion des incidents et accidents aériens."),
                    (3, "Formation du personnel de sécurité aérienne."),
                ]
            },
        ]

        for doc in test_documents:
            for page_num, content in doc["pages"]:
                repo.insert(
                    filepath=doc["filepath"],
                    filename=doc["filename"],
                    relative_path=doc["relative_path"],
                    file_hash=doc["file_hash"],
                    page_num=page_num,
                    content=content
                )

        return {
            "repository": repo,
            "engine": BM25Engine()
        }

    def test_statistics_after_indexing(self, integrated_system):
        """Test that statistics reflect indexed documents."""
        stats = get_statistics()

        # 4 unique files
        assert stats["total_files"] == 4
        # 8 total pages (2+2+1+3)
        assert stats["total_pages"] == 8

    def test_search_finds_relevant_documents(self, integrated_system):
        """Test that search returns relevant documents."""
        engine = integrated_system["engine"]

        query = SearchQuery(text="aviation")
        results, stats = engine.search(query)

        # Should find aviation-related documents
        assert stats.total_results > 0

        # Results should be from relevant files
        relevant_files = {"reglementation.pdf", "procedures.pdf"}
        result_files = {r.filename for r in results}
        assert result_files & relevant_files  # Intersection should not be empty

    def test_search_with_french_terms(self, integrated_system):
        """Test search with French accented characters."""
        engine = integrated_system["engine"]

        query = SearchQuery(text="sécurité")
        results, stats = engine.search(query)

        # Should find documents mentioning sécurité
        assert stats.total_results > 0

    def test_search_excludes_irrelevant(self, integrated_system):
        """Test that search excludes irrelevant documents."""
        engine = integrated_system["engine"]

        query = SearchQuery(text="maritime")
        results, stats = engine.search(query, include_content=True)

        # Should only find maritime document
        for result in results:
            assert "navigation" in result.filename or "maritime" in (result.content or "").lower()

    def test_advanced_or_search(self, integrated_system):
        """Test OR search combines results."""
        engine = integrated_system["engine"]

        query = SearchQuery(text="aviation OR maritime", advanced=True)
        results, stats = engine.search(query)

        # Should find both aviation and maritime documents
        filenames = {r.filename for r in results}
        assert len(filenames) >= 2

    def test_advanced_not_search(self, integrated_system):
        """Test NOT search excludes terms."""
        engine = integrated_system["engine"]

        # Search for sécurité but not maritime
        query = SearchQuery(text="sécurité NOT maritime", advanced=True)
        results, stats = engine.search(query)

        # Results should not include maritime content
        for result in results:
            # Maritime document should not be in results
            assert result.filename != "navigation.pdf"

    def test_phrase_search(self, integrated_system):
        """Test exact phrase matching."""
        engine = integrated_system["engine"]

        query = SearchQuery(text='"aviation civile"', advanced=True)
        results, stats = engine.search(query, include_content=True)

        # Should find the exact phrase
        if results:
            found_phrase = any("aviation civile" in (r.content or "").lower() for r in results)
            assert found_phrase

    def test_search_pagination(self, integrated_system):
        """Test search with pagination."""
        engine = integrated_system["engine"]

        # First page
        query1 = SearchQuery(text="sécurité", limit=2, offset=0)
        results1, stats1 = engine.search(query1)

        # Second page
        query2 = SearchQuery(text="sécurité", limit=2, offset=2)
        results2, stats2 = engine.search(query2)

        # Total should be same
        assert stats1.total_results == stats2.total_results

        # Results should be different (if enough results)
        if len(results1) > 0 and len(results2) > 0:
            ids1 = {r.id for r in results1}
            ids2 = {r.id for r in results2}
            assert ids1 != ids2  # Different pages

    def test_search_includes_content(self, integrated_system):
        """Test that content can be included in results."""
        engine = integrated_system["engine"]

        query = SearchQuery(text="aviation")
        results, stats = engine.search(query, include_content=True)

        for result in results:
            assert result.content is not None
            assert len(result.content) > 0

    def test_empty_search_returns_empty(self, integrated_system):
        """Test that empty search returns empty results."""
        engine = integrated_system["engine"]

        query = SearchQuery(text="xyznonexistentterm123")
        results, stats = engine.search(query)

        assert len(results) == 0
        assert stats.total_results == 0


class TestDatabaseIntegrity:
    """Tests for database integrity and consistency."""

    def test_duplicate_prevention(self, configured_db):
        """Test that duplicate documents are handled in temp DB."""
        init_schema()
        repo = DocumentRepository()

        # Insert document
        repo.insert(
            filepath="/test.pdf",
            filename="test.pdf",
            relative_path="test.pdf",
            file_hash="unique_hash",
            page_num=1,
            content="Content"
        )

        # Check exists by filepath
        assert repo.exists("/test.pdf")

        # Insert again with same filepath but different page
        # This should create a new record (different page)
        repo.insert(
            filepath="/test.pdf",
            filename="test.pdf",
            relative_path="test.pdf",
            file_hash="unique_hash",
            page_num=2,
            content="Different content"
        )

        # Both should be searchable
        engine = BM25Engine()
        query = SearchQuery(text="content")
        results, stats = engine.search(query)

        assert stats.total_results >= 1

    def test_statistics_update_on_changes(self, configured_db):
        """Test that statistics update when data changes in temp DB."""
        init_schema()
        repo = DocumentRepository()

        # Initial stats
        stats_before = get_statistics()
        assert stats_before["total_files"] == 0

        # Add document
        repo.insert(
            filepath="/new.pdf",
            filename="new.pdf",
            relative_path="new.pdf",
            file_hash="new_hash",
            page_num=1,
            content="New content"
        )

        # Stats should update
        stats_after = get_statistics()
        assert stats_after["total_files"] == 1
        assert stats_after["total_pages"] == 1


class TestSearchQuality:
    """Tests for search result quality."""

    @pytest.fixture
    def ranked_documents(self, configured_db):
        """Create documents with varying relevance in temp DB."""
        init_schema()
        repo = DocumentRepository()

        # Document with high relevance (multiple mentions)
        repo.insert(
            filepath="/high.pdf",
            filename="high.pdf",
            relative_path="high.pdf",
            file_hash="high_hash",
            page_num=1,
            content="Aviation aviation aviation. Safety in aviation is critical. Aviation regulations."
        )

        # Document with low relevance (one mention)
        repo.insert(
            filepath="/low.pdf",
            filename="low.pdf",
            relative_path="low.pdf",
            file_hash="low_hash",
            page_num=1,
            content="Maritime transport and shipping. Brief mention of aviation."
        )

        # Document with no relevance
        repo.insert(
            filepath="/none.pdf",
            filename="none.pdf",
            relative_path="none.pdf",
            file_hash="none_hash",
            page_num=1,
            content="Completely unrelated content about cooking recipes."
        )

        return BM25Engine()

    def test_relevance_ranking(self, ranked_documents):
        """Test that more relevant documents rank higher."""
        engine = ranked_documents

        query = SearchQuery(text="aviation")
        results, stats = engine.search(query)

        # Should have results
        assert len(results) >= 2

        # High relevance document should rank first
        assert results[0].filename == "high.pdf"

    def test_irrelevant_excluded(self, ranked_documents):
        """Test that irrelevant documents are not returned."""
        engine = ranked_documents

        query = SearchQuery(text="aviation")
        results, stats = engine.search(query)

        filenames = [r.filename for r in results]
        assert "none.pdf" not in filenames
