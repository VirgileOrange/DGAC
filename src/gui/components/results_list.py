"""
Results list component for displaying search results.

Renders search results with snippets, actions, and pagination.
Supports both SearchResult and HybridSearchResult objects.
"""

import streamlit as st
from pathlib import Path
from typing import List, Union

from ...search import SearchResult, HybridSearchResult
from ..state import get_state, set_state


SOURCE_LABELS = {
    "lexical": "[L]",
    "semantic": "[S]",
    "both": "[L+S]"
}


def render_results(results: List[Union[SearchResult, HybridSearchResult]]) -> None:
    """
    Render the list of search results.

    Args:
        results: List of SearchResult or HybridSearchResult objects to display.
    """
    if not results:
        return

    for idx, result in enumerate(results):
        _render_result_card(result, idx)


def _render_result_card(result: Union[SearchResult, HybridSearchResult], idx: int) -> None:
    """Render a single result card with expander."""
    source_label = ""
    if hasattr(result, "source"):
        source_label = f" {SOURCE_LABELS.get(result.source, '')}"

    header = f"**{result.filename}** - Page {result.page_num}{source_label}"

    with st.expander(header, expanded=False):
        st.caption(f"Chemin : {result.relative_path}")

        score_text = _format_score(result)
        st.caption(score_text)

        st.markdown("---")

        snippet_html = result.snippet.replace("<mark>", "**").replace("</mark>", "**")
        st.markdown(f"...{snippet_html}...")

        st.markdown("---")

        _render_actions(result, idx)


def _format_score(result: Union[SearchResult, HybridSearchResult]) -> str:
    """Format score display based on result type."""
    if hasattr(result, "source"):
        parts = [f"Score : {result.score:.4f}"]

        if hasattr(result, "similarity") and result.similarity is not None:
            parts.append(f"Similarite : {result.similarity:.2%}")

        if hasattr(result, "lexical_rank") and result.lexical_rank is not None:
            parts.append(f"Rang lexical : {result.lexical_rank}")

        if hasattr(result, "semantic_rank") and result.semantic_rank is not None:
            parts.append(f"Rang semantique : {result.semantic_rank}")

        return " | ".join(parts)

    return f"Score : {result.display_score:.2f}"


def _render_actions(result: Union[SearchResult, HybridSearchResult], idx: int) -> None:
    """Render action buttons for a result."""
    result_id = _get_result_id(result, idx)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Voir le PDF", key=f"pdf_btn_{result_id}", use_container_width=True):
            current = get_state("show_pdf", {})
            current[result_id] = not current.get(result_id, False)
            set_state("show_pdf", current)

    with col2:
        if st.button("Texte complet", key=f"text_btn_{result_id}", use_container_width=True):
            current = get_state("show_content", {})
            current[result_id] = not current.get(result_id, False)
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
    if show_pdf.get(result_id, False):
        set_state("selected_doc_id", result_id)
        set_state("selected_doc_path", result.filepath)
        set_state("selected_doc_page", result.page_num)
        stats = get_state("search_stats")
        if stats:
            set_state("selected_doc_search_query", stats.query)
            set_state("selected_doc_search_mode", stats.mode)
        source = getattr(result, "source", "lexical")
        if source in ("semantic", "both"):
            set_state("selected_doc_chunk_content", result.snippet)
        else:
            set_state("selected_doc_chunk_content", None)

    show_content = get_state("show_content", {})
    if show_content.get(result_id, False):
        _render_full_content(result, result_id)


def _get_result_id(result: Union[SearchResult, HybridSearchResult], idx: int) -> str:
    """Get unique identifier for a result."""
    if hasattr(result, "id"):
        return f"{result.id}_{idx}"
    return f"{result.document_id}_{result.page_num}_{idx}"


def _render_full_content(
    result: Union[SearchResult, HybridSearchResult],
    result_id: str
) -> None:
    """Render full page content in a text area."""
    content = getattr(result, "content", None) or getattr(result, "snippet", "")

    if content:
        st.text_area(
            f"Contenu - Page {result.page_num}",
            value=content,
            height=300,
            key=f"content_area_{result_id}"
        )
    else:
        st.info("Contenu non disponible pour ce resultat.")


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
