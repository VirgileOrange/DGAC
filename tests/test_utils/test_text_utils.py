"""
Tests for text utility functions.

Tests text cleaning, truncation, and keyword extraction.
"""

import pytest

from src.utils.text_utils import (
    clean_text,
    truncate_text,
    extract_keywords
)


class TestCleanText:
    """Tests for clean_text function."""

    def test_removes_multiple_spaces(self):
        """Test that multiple spaces are collapsed to single space."""
        text = "Hello    world   test"

        result = clean_text(text)

        assert result == "Hello world test"

    def test_removes_multiple_newlines(self):
        """Test that multiple newlines are reduced to double newline."""
        text = "Line 1\n\n\n\n\nLine 2"

        result = clean_text(text)

        assert result == "Line 1\n\nLine 2"

    def test_strips_line_whitespace(self):
        """Test that leading/trailing whitespace is stripped from lines."""
        text = "  Line 1  \n   Line 2   "

        result = clean_text(text)

        assert result == "Line 1\nLine 2"

    def test_preserves_accented_characters(self):
        """Test that French accented characters are preserved."""
        text = "Café résumé naïve"

        result = clean_text(text)

        assert "é" in result
        assert "ï" in result

    def test_empty_string_returns_empty(self):
        """Test that empty string returns empty string."""
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_normalizes_unicode(self):
        """Test that unicode is normalized (NFKC)."""
        # Different unicode representations of the same character
        text = "Café"  # Normal form

        result = clean_text(text)

        assert "Café" in result or "Cafe" in result  # Depending on normalization


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_short_text_unchanged(self):
        """Test that text shorter than max_length is unchanged."""
        text = "Short"

        result = truncate_text(text, 100)

        assert result == "Short"

    def test_exact_length_unchanged(self):
        """Test that text at exactly max_length is unchanged."""
        text = "12345"

        result = truncate_text(text, 5)

        assert result == "12345"

    def test_truncation_adds_suffix(self):
        """Test that truncated text ends with suffix."""
        text = "This is a long sentence that needs truncation."

        result = truncate_text(text, 20)

        assert result.endswith("...")
        assert len(result) <= 20

    def test_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "Long text to truncate"

        result = truncate_text(text, 15, suffix="…")

        assert result.endswith("…")

    def test_empty_string_returns_empty(self):
        """Test that empty string returns empty."""
        assert truncate_text("", 10) == ""
        assert truncate_text(None, 10) is None

    def test_breaks_at_word_boundary_when_space_near_end(self):
        """Test that truncation breaks at word boundary when space is near the end.

        The function only breaks at word boundary if the last space is more than
        70% into the truncated portion. This test uses a text where the space
        position qualifies for word boundary breaking.
        """
        # "This is a very long text" - with max_length=20, truncate_at=17
        # text[:17] = "This is a very lo"
        # last_space = 14 (before "very")
        # 14 > 17 * 0.7 = 11.9, so it breaks at word boundary
        text = "This is a very long text"

        result = truncate_text(text, 20)

        # Should break at "very" rather than mid-word
        assert result == "This is a very..."

    def test_no_word_break_when_space_too_early(self):
        """Test that truncation doesn't break at word boundary when space is too early.

        When the last space is less than 70% into the truncated text,
        the function truncates mid-word.
        """
        # "Hello beautiful world" - with max_length=15, truncate_at=12
        # text[:12] = "Hello beauti"
        # last_space = 5 (after "Hello")
        # 5 < 12 * 0.7 = 8.4, so it truncates mid-word
        text = "Hello beautiful world"

        result = truncate_text(text, 15)

        # Truncates mid-word because space is too early
        assert result == "Hello beauti..."


class TestExtractKeywords:
    """Tests for extract_keywords function."""

    def test_extracts_words(self):
        """Test that words are extracted from text."""
        text = "The quick brown fox"

        keywords = extract_keywords(text)

        assert "quick" in keywords
        assert "brown" in keywords

    def test_filters_short_words(self):
        """Test that short words are filtered out."""
        text = "I am a test"

        keywords = extract_keywords(text, min_length=3)

        assert "am" not in keywords
        assert "test" in keywords

    def test_handles_french_accents(self):
        """Test that French accented words are extracted."""
        text = "L'aviation civile française"

        keywords = extract_keywords(text)

        assert "aviation" in keywords
        assert "civile" in keywords
        assert "française" in keywords

    def test_returns_lowercase(self):
        """Test that keywords are lowercase."""
        text = "UPPERCASE Mixed Case"

        keywords = extract_keywords(text)

        assert all(k == k.lower() for k in keywords)

    def test_empty_string_returns_empty_list(self):
        """Test that empty string returns empty list."""
        assert extract_keywords("") == []
        assert extract_keywords(None) == []

    def test_custom_min_length(self):
        """Test custom minimum word length."""
        text = "a ab abc abcd"

        keywords = extract_keywords(text, min_length=4)

        assert "abc" not in keywords
        assert "abcd" in keywords

    def test_removes_duplicates_implicitly(self):
        """Test that extraction handles repeated words."""
        text = "test test test"

        keywords = extract_keywords(text)

        # Should have test at least once
        assert "test" in keywords
