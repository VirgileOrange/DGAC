"""
Sidebar component for the PDF Search Engine.

Displays database statistics, search options, and help text.
"""

import streamlit as st
from typing import Dict

from ...database import get_statistics
from ..state import get_state, set_state


SEARCH_MODES = {
    "hybrid": "Hybride (recommandé)",
    "lexical": "Lexical (BM25)",
    "semantic": "Sémantique"
}


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

        st.subheader("Mode de recherche")
        search_mode = _render_search_mode()

        st.divider()

        st.subheader("Options")
        options = _render_options(search_mode)

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


def _render_search_mode() -> str:
    """Render search mode selector."""
    current_mode = get_state("search_mode", "hybrid")

    mode_options = list(SEARCH_MODES.keys())
    mode_labels = list(SEARCH_MODES.values())

    current_index = mode_options.index(current_mode) if current_mode in mode_options else 0

    selected_label = st.radio(
        "Sélectionnez le mode",
        options=mode_labels,
        index=current_index,
        key="search_mode_radio",
        help="Hybride combine les deux méthodes pour de meilleurs résultats"
    )

    selected_mode = mode_options[mode_labels.index(selected_label)]
    set_state("search_mode", selected_mode)

    if selected_mode == "lexical":
        st.caption("Recherche par mots-clés exacts (BM25)")
    elif selected_mode == "semantic":
        st.caption("Recherche par sens et synonymes")
    else:
        st.caption("Combine lexical + sémantique")

    return selected_mode


def _render_options(search_mode: str) -> Dict:
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

    advanced_search = False
    if search_mode in ("lexical", "hybrid"):
        advanced_search = st.checkbox(
            "Recherche avancée",
            value=get_state("advanced_search", False),
            key="advanced_checkbox",
            help="Activer les opérateurs OR, NOT et la recherche de phrases exactes"
        )
        set_state("advanced_search", advanced_search)

    lexical_weight = get_state("lexical_weight", 1.0)
    semantic_weight = get_state("semantic_weight", 1.0)

    if search_mode == "hybrid":
        with st.expander("Pondération hybride", expanded=False):
            lexical_weight = st.slider(
                "Poids lexical",
                min_value=0.0,
                max_value=2.0,
                value=get_state("lexical_weight", 1.0),
                step=0.1,
                key="lexical_weight_slider"
            )
            set_state("lexical_weight", lexical_weight)

            semantic_weight = st.slider(
                "Poids sémantique",
                min_value=0.0,
                max_value=2.0,
                value=get_state("semantic_weight", 1.0),
                step=0.1,
                key="semantic_weight_slider"
            )
            set_state("semantic_weight", semantic_weight)

    return {
        "results_per_page": results_per_page,
        "advanced_search": advanced_search,
        "search_mode": search_mode,
        "lexical_weight": lexical_weight,
        "semantic_weight": semantic_weight
    }


def _render_help() -> None:
    """Display search help text."""
    with st.expander("Aide à la recherche"):
        st.markdown("""
        **Modes de recherche :**
        - **Hybride** : Combine les deux methodes pour les meilleurs resultats
        - **Lexical** : Recherche exacte par mots-cles (BM25)
        - **Semantique** : Recherche par sens, gere les synonymes

        **Recherche simple :**
        - Tapez des mots pour trouver les documents les contenant
        - La recherche ignore les accents

        **Recherche avancee (mode lexical/hybride) :**
        - `mot1 OR mot2` - Correspond a l'un ou l'autre terme
        - `mot1 NOT mot2` - Exclut un terme
        - `"phrase exacte"` - Correspond a la phrase exacte
        - `prefixe*` - Correspond aux mots commencant par le prefixe

        **Exemples :**
        - `aviation civile`
        - `reglement OR directive`
        - `securite NOT maritime`
        - `"controle aerien"`

        **Indicateurs de source :**
        - [L] Trouve par recherche lexicale
        - [S] Trouve par recherche semantique
        - [L+S] Trouve par les deux methodes
        """)
