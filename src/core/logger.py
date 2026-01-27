"""
Centralized logging setup for the PDF Search Engine.

Provides console and rotating file output with configuration from config.json.
Uses a guard to prevent multiple initialization.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_logger_initialized = False


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    logs_directory: Path = None,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> None:
    """
    Initialize the root logger with console and optional file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Format string for log messages.
        logs_directory: Directory for log files. If None, file logging disabled.
        max_file_size_mb: Maximum size of each log file in MB.
        backup_count: Number of backup files to keep.
    """
    global _logger_initialized

    if _logger_initialized:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if logs_directory:
        logs_directory = Path(logs_directory)
        logs_directory.mkdir(parents=True, exist_ok=True)

        log_file = logs_directory / "pdf_search.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _logger_initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.

    Automatically initializes logging from config on first call.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    global _logger_initialized

    if not _logger_initialized:
        try:
            from .config_loader import get_config
            config = get_config()
            setup_logging(
                log_level=config.logging.level,
                log_format=config.logging.format,
                logs_directory=config.paths.logs_directory,
                max_file_size_mb=config.logging.max_file_size_mb,
                backup_count=config.logging.backup_count
            )
        except Exception:
            setup_logging()

    return logging.getLogger(name)


if __name__ == "__main__":
    setup_logging(log_level="DEBUG")

    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    other_logger = get_logger("other_module")
    other_logger.info("Message from another logger")
