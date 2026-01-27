"""
Results list component for displaying search results.

Renders search results with snippets, actions, and pagination.
"""

import streamlit as st
from pathlib import Path
from typing import List

from ...search import SearchResult
from ..state import get_state, set_state


def render_results(results: List[SearchResult]) -> None:
    """
    Render the list of search results.

    Args:
        results: List of SearchResult objects to display.
    """
    if not results:
        return

    for result in results:
        _render_result_card(result)


def _render_result_card(result: SearchResult) -> None:
    """Render a single result card with expander."""
    header = f"**{result.filename}** - Page {result.page_num}"

    with st.expander(header, expanded=False):
        st.caption(f"Chemin : {result.relative_path}")
        st.caption(f"Score : {result.display_score:.2f}")

        st.markdown("---")

        snippet_html = result.snippet.replace("<mark>", "**").replace("</mark>", "**")
        st.markdown(f"...{snippet_html}...")

        st.markdown("---")

        _render_actions(result)


def _render_actions(result: SearchResult) -> None:
    """Render action buttons for a result."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Voir le PDF", key=f"pdf_btn_{result.id}", use_container_width=True):
            current = get_state("show_pdf", {})
            current[result.id] = not current.get(result.id, False)
            set_state("show_pdf", current)

    with col2:
        if st.button("Texte complet", key=f"text_btn_{result.id}", use_container_width=True):
            current = get_state("show_content", {})
            current[result.id] = not current.get(result.id, False)
            set_state("show_content", current)

    with col3:
        filepath = Path(result.filepath)
        if filepath.exists():
            folder = filepath.parent
            st.markdown(
                f'<a href="file:///{folder}" target="_blank">'
                f'<button style="width:100%">Ouvrir le dossier</button></a>',
                unsafe_allow_html=True
            )

    show_pdf = get_state("show_pdf", {})
    if show_pdf.get(result.id, False):
        set_state("selected_doc_id", result.id)
        set_state("selected_doc_path", result.filepath)
        set_state("selected_doc_page", result.page_num)

    show_content = get_state("show_content", {})
    if show_content.get(result.id, False):
        _render_full_content(result)


def _render_full_content(result: SearchResult) -> None:
    """Render full page content in a text area."""
    if result.content:
        st.text_area(
            f"Contenu complet - Page {result.page_num}",
            value=result.content,
            height=300,
            key=f"content_area_{result.id}"
        )
    else:
        st.info("Contenu non chargé. Relancez la recherche avec l'option contenu.")


def render_pagination(total_results: int, results_per_page: int) -> None:
    """
    Render pagination controls.

    Args:
        total_results: Total number of matching results.
        results_per_page: Number of results per page.
    """
    if total_results <= results_per_page:
        return

    total_pages = (total_results + results_per_page - 1) // results_per_page
    current_page = get_state("current_page", 1)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("Début", disabled=current_page <= 1):
            set_state("current_page", 1)
            st.rerun()

    with col2:
        if st.button("Préc.", disabled=current_page <= 1):
            set_state("current_page", current_page - 1)
            st.rerun()

    with col3:
        st.markdown(
            f"<div style='text-align:center'>Page {current_page} sur {total_pages}</div>",
            unsafe_allow_html=True
        )

    with col4:
        if st.button("Suiv.", disabled=current_page >= total_pages):
            set_state("current_page", current_page + 1)
            st.rerun()

    with col5:
        if st.button("Fin", disabled=current_page >= total_pages):
            set_state("current_page", total_pages)
            st.rerun()
