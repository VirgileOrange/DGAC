"""
GUI module providing the Streamlit web interface.

Contains the main application, session state management,
and reusable UI components for the PDF search engine.
"""

from .state import init_state, get_state, set_state

__all__ = [
    "init_state",
    "get_state",
    "set_state"
]
