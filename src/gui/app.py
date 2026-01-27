"""
Main Streamlit application for the PDF Search Engine.

Entry point that assembles all components into the complete
web interface with search, results, and PDF viewing.

Note: This file is run directly by Streamlit, so it needs to
set up the Python path before importing other modules.
"""

import sys
import base64
from pathlib import Path

# Add project root to path for imports when run directly by Streamlit
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st  # noqa: E402

from src.core import get_config, get_logger  # noqa: E402
from src.database import init_schema  # noqa: E402
from src.search import BM25Engine, SearchQuery  # noqa: E402

from src.gui.state import init_state, get_state, set_state, get_pagination_state  # noqa: E402
from src.gui.components import (  # noqa: E402
    render_sidebar,
    render_search_bar,
    render_results,
)
from src.gui.components.search_bar import render_search_header, render_no_results  # noqa: E402
from src.gui.components.results_list import render_pagination  # noqa: E402
from src.gui.components.pdf_viewer import render_pdf_from_state  # noqa: E402

logger = get_logger(__name__)


def load_css(css_path: Path) -> None:
    """
    Load and inject custom CSS into Streamlit.

    Args:
        css_path: Path to the CSS file.
    """
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_banner(logo_path: Path, title: str) -> None:
    """
    Render the main header banner with logo and title.

    Args:
        logo_path: Path to the logo image.
        title: Title text to display.
    """
    logo_html = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode("utf-8")
        logo_html = f'<img src="data:image/png;base64,{logo_data}" class="main-header-logo" alt="Logo">'

    banner_html = f"""
    <div class="main-header">
        {logo_html}
        <div class="main-header-content">
            <h1>{title}</h1>
            <p>Moteur de recherche dans vos documents PDF</p>
        </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    config = get_config()

    st.set_page_config(
        page_title=config.gui.page_title,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css(config.assets.css_path)

    init_state()

    init_schema()

    options = render_sidebar()

    render_banner(config.assets.logo_path, config.gui.page_title)

    query_text, submitted = render_search_bar()

    if submitted and query_text.strip():
        _execute_search(query_text, options)

    _render_results_section()

    render_pdf_from_state()


def _execute_search(query_text: str, options: dict) -> None:
    """
    Execute search and store results in state.

    Args:
        query_text: The search query string.
        options: Search options from sidebar.
    """
    engine = BM25Engine()
    pagination = get_pagination_state()

    query = SearchQuery(
        text=query_text,
        limit=options["results_per_page"],
        offset=pagination["offset"],
        advanced=options["advanced_search"]
    )

    with st.spinner("Recherche en cours..."):
        try:
            results, stats = engine.search(query, include_content=True)

            set_state("search_results", results)
            set_state("search_stats", stats)

            logger.info(f"Search '{query_text}': {stats.total_results} results")

        except Exception as e:
            st.error(f"Erreur lors de la recherche : {e}")
            logger.error(f"Search error: {e}")


def _render_results_section() -> None:
    """Render the search results section."""
    results = get_state("search_results", [])
    stats = get_state("search_stats")

    if not stats:
        _render_welcome()
        return

    render_search_header(stats)

    if not results:
        render_no_results(stats.query)
        return

    st.divider()

    render_results(results)

    st.divider()

    render_pagination(
        stats.total_results,
        get_state("results_per_page", 20)
    )


def _render_welcome() -> None:
    """Render welcome message when no search has been performed."""
    st.markdown("""
    ### Bienvenue sur le moteur de recherche PDF

    Utilisez la barre de recherche ci-dessus pour trouver des documents dans votre collection PDF.

    **Fonctionnalités :**
    - Recherche plein texte avec classement BM25
    - Visualisation des PDF directement dans le navigateur
    - Navigation vers des pages spécifiques
    - Opérateurs de recherche avancée

    Saisissez une requête pour commencer.
    """)


if __name__ == "__main__":
    main()
