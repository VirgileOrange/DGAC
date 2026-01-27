"""
Query parser for FTS5 full-text search.

Sanitizes user input and transforms queries into valid FTS5 syntax.
Supports both basic (implicit AND) and advanced query modes.
"""

import re
from typing import List

from ..core import get_logger

logger = get_logger(__name__)


# Characters with special meaning in FTS5 that need escaping
FTS5_SPECIAL_CHARS = set('"\'*-+():^')


class QueryParser:
    """
    Parses and sanitizes search queries for FTS5.

    Provides both basic mode (simple word matching) and advanced mode
    (preserving operators like OR, NOT, and quoted phrases).
    """

    def parse(self, query: str) -> str:
        """
        Parse query in basic mode.

        Removes all FTS5 special characters and returns space-separated
        terms for implicit AND matching.

        Args:
            query: Raw user input.

        Returns:
            Sanitized query string safe for FTS5 MATCH.
        """
        if not query or not query.strip():
            return ""

        cleaned = "".join(
            char if char not in FTS5_SPECIAL_CHARS else " "
            for char in query
        )

        terms = cleaned.split()
        terms = [term.strip() for term in terms if term.strip()]

        return " ".join(terms)

    def parse_advanced(self, query: str) -> str:
        """
        Parse query in advanced mode, preserving FTS5 operators.

        Preserves:
        - OR, NOT, AND operators
        - Quoted phrases "like this"
        - Prefix wildcards term*

        Args:
            query: Raw user input with FTS5 syntax.

        Returns:
            Sanitized query with valid operators preserved.
        """
        if not query or not query.strip():
            return ""

        # Extract and protect quoted phrases
        phrases = []
        protected_query = query

        for match in re.finditer(r'"([^"]*)"', query):
            placeholder = f"__PHRASE_{len(phrases)}__"
            phrases.append(match.group(0))
            protected_query = protected_query.replace(match.group(0), placeholder, 1)

        tokens = protected_query.split()
        result_tokens = []

        for token in tokens:
            upper = token.upper()

            if upper in ("OR", "AND", "NOT"):
                result_tokens.append(upper)
                continue

            if token.startswith("__PHRASE_") and token.endswith("__"):
                idx = int(token[9:-2])
                result_tokens.append(phrases[idx])
                continue

            if token.endswith("*"):
                clean_prefix = self._clean_term(token[:-1])
                if clean_prefix:
                    result_tokens.append(clean_prefix + "*")
                continue

            clean_token = self._clean_term(token)
            if clean_token:
                result_tokens.append(clean_token)

        return " ".join(result_tokens)

    def _clean_term(self, term: str) -> str:
        """Remove special characters from a single term."""
        return "".join(
            char for char in term
            if char not in FTS5_SPECIAL_CHARS
        ).strip()

    def extract_terms(self, query: str) -> List[str]:
        """
        Extract individual search terms from a query.

        Useful for highlighting matches in results.

        Args:
            query: Parsed or raw query string.

        Returns:
            List of individual terms.
        """
        # Remove operators and quotes
        cleaned = re.sub(r'\b(OR|AND|NOT)\b', ' ', query, flags=re.IGNORECASE)
        cleaned = cleaned.replace('"', ' ')
        cleaned = cleaned.replace('*', '')

        terms = [t.strip().lower() for t in cleaned.split() if t.strip()]

        return list(set(terms))


if __name__ == "__main__":
    parser = QueryParser()

    print("=== Basic Mode ===")
    test_queries = [
        "aviation civile",
        "règlement (EU) 2024/123",
        "test*query",
        '"exact phrase"',
        "   multiple   spaces   "
    ]

    for q in test_queries:
        result = parser.parse(q)
        print(f"  '{q}' -> '{result}'")

    print("\n=== Advanced Mode ===")
    advanced_queries = [
        "aviation OR maritime",
        "règlement NOT obsolète",
        '"sécurité aérienne"',
        "aéro*",
        "aviation AND civile NOT militaire"
    ]

    for q in advanced_queries:
        result = parser.parse_advanced(q)
        print(f"  '{q}' -> '{result}'")

    print("\n=== Term Extraction ===")
    query = '"aviation civile" OR règlement NOT test*'
    terms = parser.extract_terms(query)
    print(f"  '{query}' -> {terms}")
