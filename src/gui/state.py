"""
Streamlit session state management.

Provides helpers for initializing, reading, and updating
session state values used across the application.
"""

import streamlit as st
from typing import Any, Dict, List, Optional


DEFAULT_STATE = {
    "search_query": "",
    "search_results": [],
    "search_stats": None,
    "selected_doc_id": None,
    "show_pdf": {},
    "show_content": {},
    "results_per_page": 20,
    "current_page": 1,
    "advanced_search": False,
}


def init_state() -> None:
    """
    Initialize session state with default values.

    Only sets values that don't already exist, preserving
    state across reruns.
    """
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_state(key: str, default: Any = None) -> Any:
    """
    Get a value from session state.

    Args:
        key: State key to retrieve.
        default: Default value if key doesn't exist.

    Returns:
        The stored value or default.
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """
    Set a value in session state.

    Args:
        key: State key to set.
        value: Value to store.
    """
    st.session_state[key] = value


def toggle_state(key: str) -> bool:
    """
    Toggle a boolean state value.

    Args:
        key: State key to toggle.

    Returns:
        The new value after toggling.
    """
    current = st.session_state.get(key, False)
    st.session_state[key] = not current
    return st.session_state[key]


def update_state(updates: Dict[str, Any]) -> None:
    """
    Update multiple state values at once.

    Args:
        updates: Dictionary of key-value pairs to update.
    """
    for key, value in updates.items():
        st.session_state[key] = value


def clear_search_state() -> None:
    """Reset search-related state to defaults."""
    set_state("search_results", [])
    set_state("search_stats", None)
    set_state("current_page", 1)
    set_state("show_pdf", {})
    set_state("show_content", {})


def get_pagination_state() -> Dict[str, int]:
    """
    Get pagination-related state.

    Returns:
        Dictionary with current_page, results_per_page, and offset.
    """
    current_page = get_state("current_page", 1)
    results_per_page = get_state("results_per_page", 20)
    offset = (current_page - 1) * results_per_page

    return {
        "current_page": current_page,
        "results_per_page": results_per_page,
        "offset": offset
    }
