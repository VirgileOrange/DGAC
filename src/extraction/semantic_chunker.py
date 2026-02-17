"""
Semantic chunker for preparing text for embedding.

Splits page text into chunks that fit within the embedding model's token limit,
using intelligent split points and overlap for context continuity.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import List, Tuple

from ..core import get_config, get_logger

logger = get_logger(__name__)


@dataclass
class SemanticChunk:
    """
    Represents a text chunk prepared for embedding.

    Attributes:
        chunk_id: Deterministic hash-based identifier.
        document_id: Reference to parent document.
        page_num: Source page number.
        position: Position within page (0 for most pages).
        content: The text content.
        char_count: Length of content.
    """
    chunk_id: str
    document_id: int
    page_num: int
    position: int
    content: str
    char_count: int


class SemanticChunker:
    """
    Prepares page text for embedding by chunking within token limits.

    Handles intelligent splitting at paragraph, sentence, or word boundaries
    with configurable overlap for context continuity.
    """

    def __init__(self):
        """Initialize the chunker with configuration."""
        self.config = get_config()
        self.max_chars = self.config.semantic.max_chunk_chars
        self.overlap_chars = self.config.semantic.chunk_overlap_chars

    def chunk_page(
        self,
        text: str,
        doc_id: int,
        page_num: int
    ) -> List[SemanticChunk]:
        """
        Split page text into chunks suitable for embedding.

        Args:
            text: Page text content.
            doc_id: Document ID from database.
            page_num: Page number (1-indexed).

        Returns:
            List of SemanticChunk objects.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        if len(text) <= self.max_chars:
            chunk_id = self._generate_chunk_id(doc_id, page_num, 0, text)
            return [SemanticChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                page_num=page_num,
                position=0,
                content=text,
                char_count=len(text)
            )]

        chunks = []
        position = 0
        start = 0

        while start < len(text):
            end = min(start + self.max_chars, len(text))

            if end < len(text):
                end = self._find_split_point(text, start, end)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = self._generate_chunk_id(doc_id, page_num, position, chunk_text)
                chunks.append(SemanticChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    page_num=page_num,
                    position=position,
                    content=chunk_text,
                    char_count=len(chunk_text)
                ))
                position += 1

            start = max(start + 1, end - self.overlap_chars)

            if start >= len(text):
                break

        return chunks

    def chunk_document(
        self,
        pages: List[Tuple[int, str]],
        doc_id: int
    ) -> List[SemanticChunk]:
        """
        Chunk all pages of a document.

        Args:
            pages: List of (page_num, text) tuples.
            doc_id: Document ID from database.

        Returns:
            List of SemanticChunk objects for all pages.
        """
        all_chunks = []

        for page_num, text in pages:
            page_chunks = self.chunk_page(text, doc_id, page_num)
            all_chunks.extend(page_chunks)

        logger.debug(f"Document {doc_id}: {len(pages)} pages -> {len(all_chunks)} chunks")
        return all_chunks

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """
        Find the best split point near end position.

        Prefers paragraph > sentence > word boundaries.

        Args:
            text: Full text being split.
            start: Start position of current chunk.
            end: Maximum end position.

        Returns:
            Best split position.
        """
        search_start = max(start, end - 300)
        search_region = text[search_start:end]

        para_match = None
        for match in re.finditer(r'\n\s*\n', search_region):
            para_match = match

        if para_match:
            return search_start + para_match.end()

        sentence_match = None
        for match in re.finditer(r'[.!?]\s+', search_region):
            sentence_match = match

        if sentence_match:
            return search_start + sentence_match.end()

        word_match = None
        for match in re.finditer(r'\s+', search_region):
            word_match = match

        if word_match:
            return search_start + word_match.start()

        return end

    @staticmethod
    def _generate_chunk_id(doc_id: int, page_num: int, position: int, content: str) -> str:
        """
        Generate a deterministic chunk ID.

        Args:
            doc_id: Document ID.
            page_num: Page number.
            position: Position within page.
            content: Chunk text content.

        Returns:
            SHA256 hash truncated to 16 characters.
        """
        unique_string = f"{doc_id}:{page_num}:{position}:{content[:100]}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]


if __name__ == "__main__":
    chunker = SemanticChunker()

    sample_text = """
    This is a sample paragraph about aviation safety regulations.
    It contains multiple sentences that discuss various topics.

    This is another paragraph that continues the discussion.
    The regulations are important for maintaining flight safety.

    A third paragraph provides additional context and details
    about the implementation of these safety measures.
    """ * 20

    chunks = chunker.chunk_page(sample_text, doc_id=1, page_num=1)

    print(f"Text length: {len(sample_text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print()

    for chunk in chunks:
        print(f"Chunk {chunk.position}: {chunk.char_count} chars")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Preview: {chunk.content[:80]}...")
        print()
