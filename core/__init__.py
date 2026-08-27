"""
Core infrastructure for LookThePerson.

This package holds everything that is not vision-specific: configuration,
the event bus, signal filtering, geometry helpers, performance metrics,
theming and the shared application state.

Nothing here imports OpenCV or MediaPipe, so it can be imported and tested
without a camera or model downloads.
"""

from core.config import Config, load_config, save_config
from core.events import Event, EventBus, Events
from core.state import AppState, FrameContext
from core.theme import Theme, get_theme, theme_names

__all__ = [
    "Config",
    "load_config",
    "save_config",
    "Event",
    "EventBus",
    "Events",
    "AppState",
    "FrameContext",
    "Theme",
    "get_theme",
    "theme_names",
]
