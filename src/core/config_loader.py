"""
Configuration loader for the PDF Search Engine.

Loads settings from config.json and provides typed access via dataclasses.
Supports singleton pattern for global access and runtime reload capability.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .exceptions import ConfigurationError


@dataclass
class PathsConfig:
    """Configuration for file system paths."""
    data_directory: Path
    database_path: Path
    logs_directory: Path


@dataclass
class AssetsConfig:
    """Configuration for UI assets paths."""
    css_path: Path
    logo_path: Path


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction settings."""
    primary_backend: str
    fallback_backend: str
    max_file_size_mb: int
    supported_extensions: List[str]


@dataclass
class IndexingConfig:
    """Configuration for indexing behavior."""
    batch_size: int
    commit_frequency: int
    skip_existing: bool
    log_progress_every: int


@dataclass
class BM25Weights:
    """BM25 ranking weights for search fields."""
    filename: float
    content: float


@dataclass
class SearchConfig:
    """Configuration for search functionality."""
    default_limit: int
    max_limit: int
    snippet_length: int
    bm25_weights: BM25Weights
    tokenizer: str


@dataclass
class GUIConfig:
    """Configuration for Streamlit web interface."""
    page_title: str
    results_per_page: int
    enable_pdf_preview: bool


@dataclass
class LoggingConfig:
    """Configuration for logging behavior."""
    level: str
    format: str
    max_file_size_mb: int
    backup_count: int


@dataclass
class SemanticConfig:
    """Configuration for semantic search and embedding."""
    enabled: bool
    endpoint: str
    api_key: str
    model: str
    embedding_model: str
    embedding_dimensions: int
    max_chunk_chars: int
    chunk_overlap_chars: int
    embedding_batch_size: int


@dataclass
class HybridConfig:
    """Configuration for hybrid search combining lexical and semantic."""
    default_mode: str
    rrf_k: int
    default_lexical_weight: float
    default_semantic_weight: float


@dataclass
class Config:
    """
    Main configuration container holding all config sections.

    Provides singleton access via get_config() function.
    """
    paths: PathsConfig
    assets: AssetsConfig
    extraction: ExtractionConfig
    indexing: IndexingConfig
    search: SearchConfig
    gui: GUIConfig
    logging: LoggingConfig
    semantic: SemanticConfig
    hybrid: HybridConfig
    project_root: Path = field(default_factory=Path)

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the config.json file.

        Returns:
            Populated Config instance.

        Raises:
            ConfigurationError: If file is missing or invalid.
        """
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                {"path": str(config_path)}
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in config file: {e}",
                {"path": str(config_path)}
            )

        project_root = config_path.parent.parent

        return cls._parse_config(data, project_root)

    @classmethod
    def _parse_config(cls, data: dict, project_root: Path) -> "Config":
        """Parse raw config dict into typed Config object."""
        paths_data = data.get("paths", {})
        paths = PathsConfig(
            data_directory=cls._resolve_path(paths_data.get("data_directory", "data"), project_root),
            database_path=cls._resolve_path(paths_data.get("database_path", "output/documents.db"), project_root),
            logs_directory=cls._resolve_path(paths_data.get("logs_directory", "output/logs"), project_root)
        )

        assets_data = data.get("assets", {})
        assets = AssetsConfig(
            css_path=cls._resolve_path(assets_data.get("css_path", "assets/style_orange.css"), project_root),
            logo_path=cls._resolve_path(assets_data.get("logo_path", "data/logo/logo_orange.png"), project_root)
        )

        ext_data = data.get("extraction", {})
        extraction = ExtractionConfig(
            primary_backend=ext_data.get("primary_backend", "pypdf2"),
            fallback_backend=ext_data.get("fallback_backend", "pdfplumber"),
            max_file_size_mb=ext_data.get("max_file_size_mb", 500),
            supported_extensions=ext_data.get("supported_extensions", [".pdf"])
        )

        idx_data = data.get("indexing", {})
        indexing = IndexingConfig(
            batch_size=idx_data.get("batch_size", 100),
            commit_frequency=idx_data.get("commit_frequency", 100),
            skip_existing=idx_data.get("skip_existing", True),
            log_progress_every=idx_data.get("log_progress_every", 100)
        )

        search_data = data.get("search", {})
        bm25_data = search_data.get("bm25_weights", {})
        search = SearchConfig(
            default_limit=search_data.get("default_limit", 50),
            max_limit=search_data.get("max_limit", 500),
            snippet_length=search_data.get("snippet_length", 150),
            bm25_weights=BM25Weights(
                filename=bm25_data.get("filename", 1.0),
                content=bm25_data.get("content", 10.0)
            ),
            tokenizer=search_data.get("tokenizer", "unicode61")
        )

        gui_data = data.get("gui", {})
        gui = GUIConfig(
            page_title=gui_data.get("page_title", "PDF Search Engine"),
            results_per_page=gui_data.get("results_per_page", 20),
            enable_pdf_preview=gui_data.get("enable_pdf_preview", True)
        )

        log_data = data.get("logging", {})
        logging_cfg = LoggingConfig(
            level=log_data.get("level", "INFO"),
            format=log_data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            max_file_size_mb=log_data.get("max_file_size_mb", 10),
            backup_count=log_data.get("backup_count", 5)
        )

        semantic_data = data.get("semantic", {})
        semantic = SemanticConfig(
            enabled=semantic_data.get("enabled", True),
            endpoint=semantic_data.get("endpoint", ""),
            api_key=semantic_data.get("api_key", ""),
            model=semantic_data.get("model", "alfred-4.2"),
            embedding_model=semantic_data.get("embedding_model", "multilingual-e5-large"),
            embedding_dimensions=semantic_data.get("embedding_dimensions", 1024),
            max_chunk_chars=semantic_data.get("max_chunk_chars", 1800),
            chunk_overlap_chars=semantic_data.get("chunk_overlap_chars", 200),
            embedding_batch_size=semantic_data.get("embedding_batch_size", 32)
        )

        hybrid_data = data.get("hybrid", {})
        hybrid = HybridConfig(
            default_mode=hybrid_data.get("default_mode", "hybrid"),
            rrf_k=hybrid_data.get("rrf_k", 60),
            default_lexical_weight=hybrid_data.get("default_lexical_weight", 1.0),
            default_semantic_weight=hybrid_data.get("default_semantic_weight", 1.0)
        )

        return cls(
            paths=paths,
            assets=assets,
            extraction=extraction,
            indexing=indexing,
            search=search,
            gui=gui,
            logging=logging_cfg,
            semantic=semantic,
            hybrid=hybrid,
            project_root=project_root
        )

    @staticmethod
    def _resolve_path(path_str: str, project_root: Path) -> Path:
        """Resolve a path string, making relative paths absolute."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return project_root / path


_config_instance: Optional[Config] = None


def get_config(config_path: Path = None) -> Config:
    """
    Get the singleton Config instance.

    Args:
        config_path: Optional path to config file. If not provided,
                    searches upward from current directory.

    Returns:
        The global Config instance.

    Raises:
        ConfigurationError: If config cannot be loaded.
    """
    global _config_instance

    if _config_instance is None or config_path is not None:
        if config_path is None:
            config_path = _find_config_file()
        _config_instance = Config.from_file(config_path)

    return _config_instance


def _find_config_file() -> Path:
    """Search upward from current directory to find config/config.json."""
    current = Path.cwd()

    for _ in range(10):
        config_path = current / "config" / "config.json"
        if config_path.exists():
            return config_path

        parent = current.parent
        if parent == current:
            break
        current = parent

    raise ConfigurationError(
        "Could not find config/config.json in current directory or parents"
    )


def reload_config(config_path: Path = None) -> Config:
    """
    Force reload of configuration.

    Args:
        config_path: Optional path to config file.

    Returns:
        Fresh Config instance.
    """
    global _config_instance
    _config_instance = None
    return get_config(config_path)


if __name__ == "__main__":
    try:
        config = get_config()
        print(f"Project root: {config.project_root}")
        print(f"Data directory: {config.paths.data_directory}")
        print(f"Database path: {config.paths.database_path}")
        print(f"Primary backend: {config.extraction.primary_backend}")
        print(f"Tokenizer: {config.search.tokenizer}")
    except ConfigurationError as e:
        print(f"Config error: {e.message}")
