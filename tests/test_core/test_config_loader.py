"""
Tests for the configuration loader module.

Tests config loading, parsing, path resolution, and error handling.
"""

import json
import pytest
from pathlib import Path

from src.core.config_loader import (
    Config,
    PathsConfig,
    get_config,
    reload_config,
)
from src.core.exceptions import ConfigurationError


class TestPathsConfig:
    """Tests for PathsConfig dataclass."""

    def test_paths_config_creation(self, temp_dir: Path):
        """Test creating PathsConfig with valid paths."""
        config = PathsConfig(
            data_directory=temp_dir / "data",
            database_path=temp_dir / "db.sqlite",
            logs_directory=temp_dir / "logs"
        )

        assert config.data_directory == temp_dir / "data"
        assert config.database_path == temp_dir / "db.sqlite"
        assert config.logs_directory == temp_dir / "logs"


class TestConfigFromFile:
    """Tests for loading config from file."""

    def test_load_valid_config(self, temp_config: Path, reset_config_singleton):
        """Test loading a valid configuration file."""
        config = Config.from_file(temp_config)

        assert config is not None
        assert config.extraction.primary_backend == "pypdf2"
        assert config.search.tokenizer == "unicode61"
        assert config.gui.page_title == "Test PDF Search"

    def test_load_missing_config_raises_error(self, temp_dir: Path):
        """Test that loading non-existent config raises ConfigurationError."""
        fake_path = temp_dir / "nonexistent" / "config.json"

        with pytest.raises(ConfigurationError) as exc_info:
            Config.from_file(fake_path)

        assert "not found" in str(exc_info.value.message).lower()

    def test_load_invalid_json_raises_error(self, temp_dir: Path):
        """Test that invalid JSON raises ConfigurationError."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        config_path = config_dir / "config.json"
        config_path.write_text("{ invalid json }")

        with pytest.raises(ConfigurationError) as exc_info:
            Config.from_file(config_path)

        assert "invalid json" in str(exc_info.value.message).lower()

    def test_config_resolves_relative_paths(self, temp_config: Path, reset_config_singleton):
        """Test that relative paths are resolved to absolute paths."""
        config = Config.from_file(temp_config)

        # All paths should be absolute
        assert config.paths.data_directory.is_absolute()
        assert config.paths.database_path.is_absolute()
        assert config.paths.logs_directory.is_absolute()

    def test_config_default_values(self, temp_dir: Path, reset_config_singleton):
        """Test that missing config values get defaults."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        config_path = config_dir / "config.json"

        # Minimal config
        minimal_config = {"paths": {}, "search": {}}
        config_path.write_text(json.dumps(minimal_config))

        config = Config.from_file(config_path)

        # Check defaults are applied
        assert config.extraction.primary_backend == "pypdf2"
        assert config.search.default_limit == 50
        assert config.indexing.batch_size == 100


class TestGetConfig:
    """Tests for the get_config singleton function."""

    def test_get_config_returns_same_instance(self, temp_config: Path, reset_config_singleton):
        """Test that get_config returns singleton instance."""
        config1 = get_config(temp_config)
        config2 = get_config()

        assert config1 is config2

    def test_reload_config_creates_new_instance(self, temp_config: Path, reset_config_singleton):
        """Test that reload_config creates a fresh instance."""
        _config1 = get_config(temp_config)  # noqa: F841

        # Modify config file
        with open(temp_config, "r") as f:
            data = json.load(f)
        data["gui"]["page_title"] = "Modified Title"
        with open(temp_config, "w") as f:
            json.dump(data, f)

        config2 = reload_config(temp_config)

        assert config2.gui.page_title == "Modified Title"


class TestAssetsConfig:
    """Tests for AssetsConfig."""

    def test_assets_config_paths(self, temp_config: Path, reset_config_singleton):
        """Test that assets paths are properly loaded."""
        config = Config.from_file(temp_config)

        assert config.assets.css_path is not None
        assert config.assets.logo_path is not None
