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


def render_search_header(stats) -> None:
    """
    Render search results header with stats.

    Args:
        stats: SearchStats object with result information.
    """
    if not stats:
        return

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"**{stats.total_results:,}** résultats trouvés")

    with col2:
        st.caption(f"Requête : \"{stats.query}\"")

    with col3:
        st.caption(f"{stats.execution_time_ms:.0f} ms")


def render_no_results(query: str) -> None:
    """Display no results message with suggestions."""
    st.info(f"Aucun résultat trouvé pour \"{query}\"")

    with st.expander("Suggestions"):
        st.markdown("""
        - Vérifiez l'orthographe
        - Essayez avec moins de mots-clés ou des mots différents
        - Essayez sans les accents
        - Utilisez la recherche avancée avec OR pour des alternatives
        """)
