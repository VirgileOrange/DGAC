"""
PDF viewer component for embedded PDF display with highlighting.

Renders PDFs inline with optional highlighting:
- Light red for semantic chunks
- Yellow for lexical keywords
Download provides the original unmodified PDF.
"""

import base64
import io
import re
from pathlib import Path
from typing import List, Optional

import streamlit as st

from ..state import get_state, set_state

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


CHUNK_HIGHLIGHT_COLOR = (1, 0.7, 0.7)
KEYWORD_HIGHLIGHT_COLOR = (1, 1, 0.6)


def render_pdf_viewer(
    filepath: str,
    page_num: int = 1,
    height: int = 600
) -> None:
    """
    Render an embedded PDF viewer with optional highlighting.

    Args:
        filepath: Path to the PDF file.
        page_num: Page number to display (1-indexed).
        height: Viewer height in pixels.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        st.error(f"Fichier introuvable : {filepath}")
        return

    st.subheader(f"Visualiseur PDF : {filepath.name}")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("Fermer", key="close_pdf"):
            set_state("show_pdf", {})
            set_state("selected_doc_id", None)
            st.rerun()

    try:
        original_pdf_data = filepath.read_bytes()

        search_query = get_state("selected_doc_search_query")
        search_mode = get_state("selected_doc_search_mode")
        chunk_content = get_state("selected_doc_chunk_content")

        if PYMUPDF_AVAILABLE and (search_query or chunk_content):
            highlighted_data = _create_highlighted_pdf(
                original_pdf_data,
                page_num,
                search_query,
                search_mode,
                chunk_content
            )
            display_data = highlighted_data if highlighted_data else original_pdf_data
        else:
            display_data = original_pdf_data

        base64_pdf = base64.b64encode(display_data).decode("utf-8")

        pdf_display = f"""
            <iframe
                src="data:application/pdf;base64,{base64_pdf}#page={page_num}"
                width="100%"
                height="{height}px"
                type="application/pdf"
                style="border: 1px solid #ccc; border-radius: 4px;"
            >
                <p>Votre navigateur ne supporte pas l'affichage des PDF.
                <a href="data:application/pdf;base64,{base64_pdf}" download="{filepath.name}">
                Telechargez le PDF</a> a la place.</p>
            </iframe>
        """

        st.markdown(pdf_display, unsafe_allow_html=True)

        _render_download_button(filepath, original_pdf_data)

    except Exception as e:
        st.error(f"Impossible de charger le PDF : {e}")
        _render_fallback(filepath)


def _create_highlighted_pdf(
    pdf_data: bytes,
    page_num: int,
    search_query: Optional[str],
    search_mode: Optional[str],
    chunk_content: Optional[str]
) -> Optional[bytes]:
    """
    Create a PDF with highlights on the specified page.

    Args:
        pdf_data: Original PDF file bytes.
        page_num: Page number to highlight (1-indexed).
        search_query: Search query for keyword highlighting.
        search_mode: Search mode (lexical/semantic/hybrid).
        chunk_content: Chunk text for semantic highlighting.

    Returns:
        Highlighted PDF bytes, or None if highlighting failed.
    """
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        page_idx = page_num - 1

        if page_idx < 0 or page_idx >= len(doc):
            doc.close()
            return None

        page = doc[page_idx]

        if chunk_content and search_mode in ("semantic", "hybrid"):
            _highlight_chunk(page, chunk_content, CHUNK_HIGHLIGHT_COLOR)

        if search_query and search_mode in ("lexical", "hybrid"):
            keywords = _extract_keywords(search_query)
            _highlight_keywords(page, keywords, KEYWORD_HIGHLIGHT_COLOR)

        output = io.BytesIO()
        doc.save(output)
        doc.close()
        output.seek(0)
        return output.read()

    except Exception:
        return None


def _highlight_chunk(page, chunk_content: str, color: tuple) -> None:
    """
    Highlight a semantic chunk on the page.

    Args:
        page: PyMuPDF page object.
        chunk_content: Text content of the chunk to highlight.
        color: RGB color tuple for highlighting.
    """
    clean_chunk = re.sub(r'<[^>]+>', '', chunk_content).strip()
    if len(clean_chunk) < 20:
        return

    search_text = clean_chunk[:200]
    text_instances = page.search_for(search_text)

    if not text_instances:
        words = clean_chunk.split()
        if len(words) >= 5:
            search_text = ' '.join(words[:5])
            text_instances = page.search_for(search_text)

    for inst in text_instances:
        annot = page.add_highlight_annot(inst)
        annot.set_colors(stroke=color)
        annot.update()


def _highlight_keywords(page, keywords: List[str], color: tuple) -> None:
    """
    Highlight keywords on the page.

    Args:
        page: PyMuPDF page object.
        keywords: List of keywords to highlight.
        color: RGB color tuple for highlighting.
    """
    for keyword in keywords:
        if len(keyword) < 2:
            continue
        text_instances = page.search_for(keyword)
        for inst in text_instances:
            annot = page.add_highlight_annot(inst)
            annot.set_colors(stroke=color)
            annot.update()


def _extract_keywords(query: str) -> List[str]:
    """
    Extract search keywords from a query.

    Args:
        query: Raw search query string.

    Returns:
        List of individual keywords.
    """
    cleaned = re.sub(r'\b(OR|AND|NOT)\b', ' ', query, flags=re.IGNORECASE)
    cleaned = cleaned.replace('"', ' ')
    cleaned = cleaned.replace('*', '')
    cleaned = re.sub(r'[^\w\s\-àâäéèêëïîôùûüç]', ' ', cleaned, flags=re.IGNORECASE)
    keywords = [k.strip() for k in cleaned.split() if k.strip() and len(k.strip()) >= 2]
    return list(set(keywords))


def _render_download_button(filepath: Path, pdf_data: bytes) -> None:
    """
    Render a download button for the original PDF.

    Args:
        filepath: Path to the PDF file.
        pdf_data: Original PDF bytes (without highlights).
    """
    st.download_button(
        label="Telecharger le PDF",
        data=pdf_data,
        file_name=filepath.name,
        mime="application/pdf",
        key=f"download_{filepath.name}"
    )


def _render_fallback(filepath: Path) -> None:
    """
    Render fallback options when PDF cannot be displayed.

    Args:
        filepath: Path to the PDF file.
    """
    st.warning("Impossible d'afficher le PDF dans le navigateur.")

    col1, col2 = st.columns(2)

    with col1:
        if filepath.exists():
            with open(filepath, "rb") as f:
                st.download_button(
                    label="Telecharger le PDF",
                    data=f.read(),
                    file_name=filepath.name,
                    mime="application/pdf"
                )

    with col2:
        st.markdown(
            f'<a href="file:///{filepath}" target="_blank">'
            f'<button>Ouvrir avec le lecteur systeme</button></a>',
            unsafe_allow_html=True
        )


def render_pdf_from_state() -> None:
    """
    Render PDF viewer based on current session state.

    Checks for selected document and renders viewer if active.
    """
    selected_id = get_state("selected_doc_id")
    show_pdf = get_state("show_pdf", {})

    if not selected_id or not show_pdf.get(selected_id, False):
        return

    filepath = get_state("selected_doc_path")
    page_num = get_state("selected_doc_page", 1)

    if filepath:
        st.divider()
        render_pdf_viewer(filepath, page_num)
