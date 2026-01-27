"""
Text utility functions for the PDF Search Engine.

Provides text cleaning, truncation, and basic tokenization
for processing extracted PDF content.
"""

import re
import unicodedata
from typing import List


def clean_text(text: str) -> str:
    """
    Normalize and clean extracted text.

    Removes control characters, normalizes whitespace, and handles
    common PDF extraction artifacts.

    Args:
        text: Raw text from PDF extraction.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters except newlines and tabs
    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith("C")
        or char in "\n\t"
    )

    # Replace multiple spaces/tabs with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: String to append when truncated.

    Returns:
        Truncated text or original if within limit.
    """
    if not text or len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)
    if truncate_at <= 0:
        return suffix[:max_length]

    # Try to break at word boundary
    truncated = text[:truncate_at]
    last_space = truncated.rfind(" ")

    if last_space > truncate_at * 0.7:
        truncated = truncated[:last_space]

    return truncated + suffix


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text via simple tokenization.

    Splits on non-alphanumeric characters and filters short words.

    Args:
        text: Text to extract keywords from.
        min_length: Minimum word length to include.

    Returns:
        List of lowercase keywords.
    """
    if not text:
        return []

    words = re.findall(r"\b[a-zA-ZÀ-ÿ0-9]+\b", text.lower())

    return [word for word in words if len(word) >= min_length]


if __name__ == "__main__":
    sample_text = """
    Ceci est un    texte   avec des   espaces multiples.



    Et plusieurs lignes vides.

    Caractères spéciaux: é à ç ü
    """

    print("=== clean_text ===")
    cleaned = clean_text(sample_text)
    print(repr(cleaned))

    print("\n=== truncate_text ===")
    long_text = "Ceci est une phrase assez longue qui sera tronquée."
    print(f"Original: {long_text}")
    print(f"Truncated (30): {truncate_text(long_text, 30)}")
    print(f"Truncated (20): {truncate_text(long_text, 20)}")

    print("\n=== extract_keywords ===")
    keywords = extract_keywords("L'aviation civile française et les règlements européens 2024")
    print(f"Keywords: {keywords}")
