"""
Core module providing foundational components.

This module contains the configuration loader, centralized logging setup,
and custom exception hierarchy. It has no internal dependencies.
"""

from .config_loader import get_config, Config, SemanticConfig, HybridConfig
from .logger import get_logger
from .exceptions import (
    PDFSearchError,
    ConfigurationError,
    ExtractionError,
    DatabaseError,
    SearchError
)

__all__ = [
    "get_config",
    "Config",
    "SemanticConfig",
    "HybridConfig",
    "get_logger",
    "PDFSearchError",
    "ConfigurationError",
    "ExtractionError",
    "DatabaseError",
    "SearchError"
]
