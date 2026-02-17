"""
Search bar component for the PDF Search Engine.

Provides the main search input and submit functionality.
"""

import streamlit as st
from typing import Tuple

from ..state import get_state, set_state, clear_search_state


def render_search_bar() -> Tuple[str, bool]:
    """
    Render the search input bar.

    Returns:
        Tuple of (query_text, was_submitted).
    """
    col1, col2 = st.columns([5, 1])

    with col1:
        query = st.text_input(
            "Rechercher",
            value=get_state("search_query", ""),
            placeholder="Saisissez vos termes de recherche...",
            key="search_input",
            label_visibility="collapsed"
        )

    with col2:
        submitted = st.button(
            "Rechercher",
            type="primary",
            use_container_width=True
        )

    previous_query = get_state("search_query", "")
    query_changed = query != previous_query and query.strip() != ""

    if query_changed:
        clear_search_state()
        set_state("search_query", query)

    return query, submitted or query_changed


MODE_LABELS = {
    "lexical": "Lexical",
    "semantic": "Semantique",
    "hybrid": "Hybride"
}


def render_search_header(stats) -> None:
    """
    Render search results header with stats.

    Args:
        stats: SearchStats or HybridSearchStats object with result information.
    """
    if not stats:
        return

    mode_label = ""
    if hasattr(stats, "mode"):
        mode_label = f" ({MODE_LABELS.get(stats.mode, stats.mode)})"

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"**{stats.total_results:,}** resultats trouves{mode_label}")

    with col2:
        st.caption(f"Requete : \"{stats.query}\"")

    with col3:
        st.caption(f"{stats.execution_time_ms:.0f} ms")

    if hasattr(stats, "overlap_count") and stats.overlap_count > 0:
        st.caption(
            f"Sources : {stats.lexical_results} lexical, "
            f"{stats.semantic_results} semantique, "
            f"{stats.overlap_count} en commun"
        )


def render_no_results(query: str) -> None:
    """Display no results message with suggestions."""
    st.info(f"Aucun resultat trouve pour \"{query}\"")

    with st.expander("Suggestions"):
        st.markdown("""
        - Verifiez l'orthographe
        - Essayez avec moins de mots-cles ou des mots differents
        - Essayez un autre mode de recherche (lexical, semantique, hybride)
        - Essayez sans les accents
        - Utilisez la recherche avancee avec OR pour des alternatives
        """)
