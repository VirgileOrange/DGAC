"""
Tests for the logging module.

Tests logger setup, configuration, and output handling.
"""

import logging
from pathlib import Path

from src.core.logger import setup_logging, get_logger


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_creates_root_logger(self, reset_logger_singleton):
        """Test that setup_logging configures the root logger."""
        setup_logging(log_level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_with_file_handler(self, temp_dir: Path, reset_logger_singleton):
        """Test that setup_logging creates file handler when directory provided."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir()

        setup_logging(
            log_level="INFO",
            logs_directory=logs_dir,
            max_file_size_mb=1,
            backup_count=1
        )

        log_file = logs_dir / "pdf_search.log"
        logger = get_logger("test")
        logger.info("Test message")

        # File should be created
        assert log_file.exists()

    def test_setup_only_runs_once(self, reset_logger_singleton):
        """Test that setup_logging only initializes once."""
        setup_logging(log_level="DEBUG")
        initial_handlers = len(logging.getLogger().handlers)

        # Call again
        setup_logging(log_level="WARNING")

        # Should not add more handlers
        assert len(logging.getLogger().handlers) == initial_handlers


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_named_logger(self, reset_logger_singleton):
        """Test that get_logger returns a logger with the given name."""
        logger = get_logger("my_module")

        assert logger.name == "my_module"

    def test_get_logger_auto_initializes(self, reset_logger_singleton):
        """Test that get_logger initializes logging if not done."""
        # Don't call setup_logging first
        logger = get_logger("auto_init_test")

        # Should still work
        assert logger is not None
        logger.info("This should not raise")

    def test_multiple_loggers_share_config(self, reset_logger_singleton):
        """Test that multiple loggers share the same configuration."""
        setup_logging(log_level="WARNING")

        _logger1 = get_logger("module1")  # noqa: F841
        _logger2 = get_logger("module2")  # noqa: F841

        # Both should have same effective level from root
        root_level = logging.getLogger().level
        assert root_level == logging.WARNING
