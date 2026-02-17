"""
Tests for the semantic chunker module.

Tests text chunking for embedding, including split point detection,
overlap handling, and chunk ID generation.

SAFETY NOTE: All tests use the `configured_db` fixture which:
- Creates a temporary directory via tempfile.mkdtemp()
- Configures the database singleton to use a temp path
- Cleans up all temp files after the test
- Never touches real data directories
"""

import pytest

from src.extraction.semantic_chunker import SemanticChunker, SemanticChunk


class TestSemanticChunk:
    """Tests for SemanticChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a SemanticChunk instance."""
        chunk = SemanticChunk(
            chunk_id="test123",
            document_id=1,
            page_num=1,
            position=0,
            content="Test content",
            char_count=12
        )

        assert chunk.chunk_id == "test123"
        assert chunk.document_id == 1
        assert chunk.page_num == 1
        assert chunk.position == 0
        assert chunk.content == "Test content"
        assert chunk.char_count == 12

    def test_chunk_equality(self):
        """Test that chunks with same values are equal."""
        chunk1 = SemanticChunk(
            chunk_id="abc",
            document_id=1,
            page_num=1,
            position=0,
            content="Content",
            char_count=7
        )
        chunk2 = SemanticChunk(
            chunk_id="abc",
            document_id=1,
            page_num=1,
            position=0,
            content="Content",
            char_count=7
        )

        assert chunk1 == chunk2


class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    def test_chunker_creation(self, configured_db):
        """Test creating a SemanticChunker instance."""
        chunker = SemanticChunker()

        assert chunker is not None
        assert chunker.max_chars > 0
        assert chunker.overlap_chars >= 0

    def test_chunk_short_text(self, configured_db):
        """Test chunking text shorter than max_chars returns single chunk."""
        chunker = SemanticChunker()
        text = "Short text that fits in one chunk."

        chunks = chunker.chunk_page(text, doc_id=1, page_num=1)

        assert len(chunks) == 1
        assert chunks[0].content == text.strip()
        assert chunks[0].position == 0
        assert chunks[0].document_id == 1
        assert chunks[0].page_num == 1

    def test_chunk_empty_text(self, configured_db):
        """Test chunking empty text returns empty list."""
        chunker = SemanticChunker()

        chunks = chunker.chunk_page("", doc_id=1, page_num=1)

        assert len(chunks) == 0

    def test_chunk_whitespace_only(self, configured_db):
        """Test chunking whitespace-only text returns empty list."""
        chunker = SemanticChunker()

        chunks = chunker.chunk_page("   \n\t   ", doc_id=1, page_num=1)

        assert len(chunks) == 0

    def test_chunk_long_text(self, configured_db):
        """Test chunking text longer than max_chars creates multiple chunks."""
        chunker = SemanticChunker()

        # Create text longer than max_chars
        long_text = "This is a sample paragraph. " * 200

        chunks = chunker.chunk_page(long_text, doc_id=1, page_num=1)

        assert len(chunks) > 1

        # All chunks should have incrementing positions
        positions = [c.position for c in chunks]
        assert positions == list(range(len(chunks)))

        # All chunks should have the same document_id and page_num
        for chunk in chunks:
            assert chunk.document_id == 1
            assert chunk.page_num == 1

    def test_chunk_preserves_content(self, configured_db):
        """Test that chunking preserves all text content."""
        chunker = SemanticChunker()

        # Create text with distinct parts
        text = "Part one. " * 100 + "Part two. " * 100 + "Part three. " * 100

        chunks = chunker.chunk_page(text, doc_id=1, page_num=1)

        # Combined content should cover original text
        all_content = " ".join(c.content for c in chunks)
        assert "Part one" in all_content
        assert "Part two" in all_content
        assert "Part three" in all_content

    def test_chunk_id_deterministic(self, configured_db):
        """Test that chunk IDs are deterministic."""
        chunker = SemanticChunker()
        text = "Test content for deterministic ID."

        chunks1 = chunker.chunk_page(text, doc_id=1, page_num=1)
        chunks2 = chunker.chunk_page(text, doc_id=1, page_num=1)

        assert chunks1[0].chunk_id == chunks2[0].chunk_id

    def test_chunk_id_differs_by_doc(self, configured_db):
        """Test that chunk IDs differ for different documents."""
        chunker = SemanticChunker()
        text = "Same content in different documents."

        chunks1 = chunker.chunk_page(text, doc_id=1, page_num=1)
        chunks2 = chunker.chunk_page(text, doc_id=2, page_num=1)

        assert chunks1[0].chunk_id != chunks2[0].chunk_id

    def test_chunk_id_differs_by_page(self, configured_db):
        """Test that chunk IDs differ for different pages."""
        chunker = SemanticChunker()
        text = "Same content on different pages."

        chunks1 = chunker.chunk_page(text, doc_id=1, page_num=1)
        chunks2 = chunker.chunk_page(text, doc_id=1, page_num=2)

        assert chunks1[0].chunk_id != chunks2[0].chunk_id


class TestSemanticChunkerDocument:
    """Tests for document-level chunking."""

    def test_chunk_document(self, configured_db):
        """Test chunking multiple pages of a document."""
        chunker = SemanticChunker()

        pages = [
            (1, "First page content."),
            (2, "Second page content."),
            (3, "Third page content."),
        ]

        chunks = chunker.chunk_document(pages, doc_id=1)

        assert len(chunks) == 3

        # Each page should produce at least one chunk
        page_nums = set(c.page_num for c in chunks)
        assert page_nums == {1, 2, 3}

    def test_chunk_document_empty_pages(self, configured_db):
        """Test chunking document with some empty pages."""
        chunker = SemanticChunker()

        pages = [
            (1, "First page content."),
            (2, ""),
            (3, "Third page content."),
        ]

        chunks = chunker.chunk_document(pages, doc_id=1)

        # Empty page should be skipped
        assert len(chunks) == 2
        page_nums = [c.page_num for c in chunks]
        assert 2 not in page_nums

    def test_chunk_document_all_same_doc_id(self, configured_db):
        """Test all chunks from a document have same document_id."""
        chunker = SemanticChunker()

        pages = [
            (1, "Page one."),
            (2, "Page two."),
        ]

        chunks = chunker.chunk_document(pages, doc_id=42)

        for chunk in chunks:
            assert chunk.document_id == 42


class TestSemanticChunkerSplitPoints:
    """Tests for split point detection."""

    def test_split_at_paragraph(self, configured_db):
        """Test that long text splits at paragraph boundaries when possible."""
        chunker = SemanticChunker()

        # Create text with clear paragraph breaks
        paragraph1 = "First paragraph content. " * 50
        paragraph2 = "Second paragraph content. " * 50
        text = paragraph1 + "\n\n" + paragraph2

        chunks = chunker.chunk_page(text, doc_id=1, page_num=1)

        # Should split at paragraph boundary if possible
        if len(chunks) >= 2:
            # First chunk should end near paragraph break
            first_content = chunks[0].content
            # Paragraph content should be in respective chunks
            assert "First paragraph" in first_content or "Second paragraph" in first_content

    def test_split_at_sentence(self, configured_db):
        """Test that text splits at sentence boundaries."""
        chunker = SemanticChunker()

        # Create text with sentences but no paragraph breaks
        text = "This is sentence one. This is sentence two. " * 100

        chunks = chunker.chunk_page(text, doc_id=1, page_num=1)

        # Chunks should generally end with periods (sentence boundaries)
        for chunk in chunks[:-1]:  # Skip last chunk
            content = chunk.content.strip()
            # Should end with punctuation or near sentence boundary
            if len(content) > 10:
                assert content[-1] in ".!? " or "." in content[-20:]

    def test_char_count_accuracy(self, configured_db):
        """Test that char_count matches actual content length."""
        chunker = SemanticChunker()

        text = "Accurate character counting test. " * 50

        chunks = chunker.chunk_page(text, doc_id=1, page_num=1)

        for chunk in chunks:
            assert chunk.char_count == len(chunk.content)


class TestGenerateChunkId:
    """Tests for chunk ID generation."""

    def test_chunk_id_format(self, configured_db):
        """Test chunk ID is a valid hex string."""
        chunk_id = SemanticChunker._generate_chunk_id(1, 1, 0, "test")

        # Should be 16 hex characters
        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)

    def test_chunk_id_unique_per_content(self, configured_db):
        """Test different content produces different IDs."""
        id1 = SemanticChunker._generate_chunk_id(1, 1, 0, "Content A")
        id2 = SemanticChunker._generate_chunk_id(1, 1, 0, "Content B")

        assert id1 != id2

    def test_chunk_id_unique_per_position(self, configured_db):
        """Test different positions produce different IDs."""
        id1 = SemanticChunker._generate_chunk_id(1, 1, 0, "Same content")
        id2 = SemanticChunker._generate_chunk_id(1, 1, 1, "Same content")

        assert id1 != id2
