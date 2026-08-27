"""
Mode system for LookThePerson.

A mode owns a complete behaviour — its processing, drawing, HUD lines and key
bindings — and the main loop simply delegates to whichever one is active. This
is what makes the app extensible: a new mode is a new class, never a new branch
in the frame loop.

Use :func:`build_mode_manager` to get a manager with every built-in mode
registered.
"""

from typing import List, Optional

from core.state import AppState
from modes.base import Mode, ModeCategory, ModeManager
from modes.creative_modes import creative_modes
from modes.detection_modes import detection_modes
from modes.fitness_modes import fitness_modes
from modes.interaction_modes import interaction_modes
from modes.utility_modes import utility_modes
from modes.wellness_modes import wellness_modes

__all__ = [
    "Mode",
    "ModeCategory",
    "ModeManager",
    "all_modes",
    "build_mode_manager",
    "mode_keys",
    "DEFAULT_MODE",
]

DEFAULT_MODE = "full"


def all_modes() -> List[Mode]:
    """
    Fresh instances of every built-in mode, grouped by category.

    Order matters: it drives the mode picker and the ``[`` / ``]`` cycling.
    """
    return [
        *detection_modes(),
        *fitness_modes(),
        *wellness_modes(),
        *creative_modes(),
        *interaction_modes(),
        *utility_modes(),
    ]


def mode_keys() -> List[str]:
    """Every built-in mode key — used by ``--list-modes`` and validation."""
    return [mode.key for mode in all_modes()]


def build_mode_manager(state: AppState, initial: Optional[str] = None) -> ModeManager:
    """
    Create a :class:`ModeManager` with every mode registered and one active.

    Falls back to :data:`DEFAULT_MODE` when *initial* names a mode that does
    not exist, so a stale config value cannot leave the app with no mode.
    """
    manager = ModeManager(state)
    manager.register_all(all_modes())

    wanted = initial or state.config.mode or DEFAULT_MODE
    if wanted not in manager:
        if initial:
            print(f"[modes] Modo desconocido '{wanted}', usando '{DEFAULT_MODE}'", flush=True)
        wanted = DEFAULT_MODE
    manager.switch(wanted)
    return manager
