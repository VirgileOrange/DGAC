"""
PDF viewer component for embedded PDF display.

Renders PDFs inline using base64-encoded iframe or provides
download fallback for unsupported browsers.
"""

import base64
from pathlib import Path
from typing import Optional

import streamlit as st

from ..state import get_state, set_state


def render_pdf_viewer(
    filepath: str,
    page_num: int = 1,
    height: int = 600
) -> None:
    """
    Render an embedded PDF viewer.

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
        pdf_data = filepath.read_bytes()
        base64_pdf = base64.b64encode(pdf_data).decode("utf-8")

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
                Téléchargez le PDF</a> à la place.</p>
            </iframe>
        """

        st.markdown(pdf_display, unsafe_allow_html=True)

        _render_download_button(filepath, pdf_data)

    except Exception as e:
        st.error(f"Impossible de charger le PDF : {e}")
        _render_fallback(filepath)


def _render_download_button(filepath: Path, pdf_data: bytes) -> None:
    """Render a download button for the PDF."""
    st.download_button(
        label="Télécharger le PDF",
        data=pdf_data,
        file_name=filepath.name,
        mime="application/pdf",
        key=f"download_{filepath.name}"
    )


def _render_fallback(filepath: Path) -> None:
    """Render fallback options when PDF cannot be displayed."""
    st.warning("Impossible d'afficher le PDF dans le navigateur.")

    col1, col2 = st.columns(2)

    with col1:
        if filepath.exists():
            with open(filepath, "rb") as f:
                st.download_button(
                    label="Télécharger le PDF",
                    data=f.read(),
                    file_name=filepath.name,
                    mime="application/pdf"
                )

    with col2:
        st.markdown(
            f'<a href="file:///{filepath}" target="_blank">'
            f'<button>Ouvrir avec le lecteur système</button></a>',
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
