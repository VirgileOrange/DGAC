"""
Tests for the FTS5 query parser.

Tests query sanitization, operator handling, and term extraction.
"""

import pytest

from src.search.query_parser import QueryParser


class TestQueryParserBasicMode:
    """Tests for basic (simple) query parsing."""

    def test_simple_words(self):
        """Test parsing simple words."""
        parser = QueryParser()

        result = parser.parse("aviation civile")

        assert result == "aviation civile"

    def test_removes_special_characters(self):
        """Test that FTS5 special characters are removed."""
        parser = QueryParser()

        # Characters like *, -, +, :, ^, ', " should be removed
        result = parser.parse('test*query "phrase" term+')

        assert "*" not in result
        assert '"' not in result
        assert "+" not in result

    def test_normalizes_whitespace(self):
        """Test that multiple spaces are normalized."""
        parser = QueryParser()

        result = parser.parse("   multiple    spaces   here   ")

        assert "  " not in result
        assert result == "multiple spaces here"

    def test_empty_string_returns_empty(self):
        """Test that empty input returns empty string."""
        parser = QueryParser()

        assert parser.parse("") == ""
        assert parser.parse("   ") == ""

    def test_preserves_accented_characters(self):
        """Test that French accents are preserved."""
        parser = QueryParser()

        result = parser.parse("règlement sécurité")

        assert "è" in result
        assert "é" in result


class TestQueryParserAdvancedMode:
    """Tests for advanced query parsing with operators."""

    def test_preserves_or_operator(self):
        """Test that OR operator is preserved."""
        parser = QueryParser()

        result = parser.parse_advanced("aviation OR maritime")

        assert "OR" in result

    def test_preserves_and_operator(self):
        """Test that AND operator is preserved."""
        parser = QueryParser()

        result = parser.parse_advanced("aviation AND civile")

        assert "AND" in result

    def test_preserves_not_operator(self):
        """Test that NOT operator is preserved."""
        parser = QueryParser()

        result = parser.parse_advanced("aviation NOT militaire")

        assert "NOT" in result

    def test_preserves_quoted_phrases(self):
        """Test that quoted phrases are preserved."""
        parser = QueryParser()

        result = parser.parse_advanced('"aviation civile"')

        assert '"aviation civile"' in result

    def test_preserves_prefix_wildcard(self):
        """Test that prefix wildcards are preserved."""
        parser = QueryParser()

        result = parser.parse_advanced("aéro*")

        assert "aéro*" in result or "aero*" in result

    def test_case_insensitive_operators(self):
        """Test that operators are case-insensitive."""
        parser = QueryParser()

        result = parser.parse_advanced("word1 or word2 not word3")

        assert "OR" in result
        assert "NOT" in result

    def test_complex_query(self):
        """Test complex query with multiple operators."""
        parser = QueryParser()

        result = parser.parse_advanced('"sécurité aérienne" OR règlement NOT obsolète')

        assert '"sécurité aérienne"' in result
        assert "OR" in result
        assert "NOT" in result


class TestQueryParserExtractTerms:
    """Tests for term extraction."""

    def test_extracts_simple_terms(self):
        """Test extracting terms from simple query."""
        parser = QueryParser()

        terms = parser.extract_terms("aviation civile")

        assert "aviation" in terms
        assert "civile" in terms

    def test_removes_operators(self):
        """Test that operators are not included as terms."""
        parser = QueryParser()

        terms = parser.extract_terms("aviation OR civile NOT militaire")

        assert "or" not in terms
        assert "and" not in terms
        assert "not" not in terms

    def test_handles_quoted_phrases(self):
        """Test extracting terms from quoted phrases."""
        parser = QueryParser()

        terms = parser.extract_terms('"aviation civile"')

        assert "aviation" in terms
        assert "civile" in terms

    def test_removes_wildcards(self):
        """Test that wildcards are stripped from terms."""
        parser = QueryParser()

        terms = parser.extract_terms("aéro* test")

        # Should have "aéro" or "aero", not "aéro*"
        assert not any("*" in term for term in terms)

    def test_returns_lowercase(self):
        """Test that terms are lowercase."""
        parser = QueryParser()

        terms = parser.extract_terms("Aviation CIVILE Test")

        assert all(term == term.lower() for term in terms)

    def test_deduplicates_terms(self):
        """Test that duplicate terms are removed."""
        parser = QueryParser()

        terms = parser.extract_terms("test test test")

        assert terms.count("test") == 1
