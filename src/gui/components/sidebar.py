"""
Sidebar component for the PDF Search Engine.

Displays database statistics, search options, and help text.
"""

import streamlit as st
from typing import Dict

from ...database import get_statistics
from ..state import get_state, set_state


def render_sidebar() -> Dict:
    """
    Render the sidebar with stats and options.

    Returns:
        Dictionary of selected options.
    """
    with st.sidebar:
        st.title("Recherche PDF")

        st.subheader("Statistiques")
        _render_statistics()

        st.divider()

        st.subheader("Options de recherche")
        options = _render_options()

        st.divider()

        _render_help()

    return options


def _render_statistics() -> None:
    """Display database statistics."""
    try:
        stats = get_statistics()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fichiers", f"{stats['total_files']:,}")

        with col2:
            st.metric("Pages", f"{stats['total_pages']:,}")

        st.caption(f"Taille de l'index : {stats['total_content_mb']:.1f} Mo")

        if stats['newest_index']:
            st.caption(f"Dernière mise à jour : {stats['newest_index'][:16]}")

    except Exception as e:
        st.warning(f"Impossible de charger les statistiques : {e}")


def _render_options() -> Dict:
    """Render search option controls."""
    results_per_page = st.slider(
        "Résultats par page",
        min_value=10,
        max_value=100,
        value=get_state("results_per_page", 20),
        step=10,
        key="results_slider"
    )
    set_state("results_per_page", results_per_page)

    advanced_search = st.checkbox(
        "Recherche avancée",
        value=get_state("advanced_search", False),
        key="advanced_checkbox",
        help="Activer les opérateurs OR, NOT et la recherche de phrases exactes"
    )
    set_state("advanced_search", advanced_search)

    return {
        "results_per_page": results_per_page,
        "advanced_search": advanced_search
    }


def _render_help() -> None:
    """Display search help text."""
    with st.expander("Aide à la recherche"):
        st.markdown("""
        **Recherche simple :**
        - Tapez des mots pour trouver les documents les contenant tous
        - La recherche ignore les accents

        **Recherche avancée :**
        - `mot1 OR mot2` - Correspond à l'un ou l'autre terme
        - `mot1 NOT mot2` - Exclut un terme
        - `"phrase exacte"` - Correspond à la phrase exacte
        - `préfixe*` - Correspond aux mots commençant par le préfixe

        **Exemples :**
        - `aviation civile`
        - `règlement OR directive`
        - `sécurité NOT maritime`
        - `"contrôle aérien"`
        """)
