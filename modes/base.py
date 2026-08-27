"""
Mode framework for LookThePerson.

A *mode* is a self-contained behaviour with its own processing, drawing, HUD
and key bindings. The main loop knows nothing about any specific mode — it
prepares a :class:`~core.state.FrameContext` and hands it to whichever mode is
active. Adding a mode therefore never means editing the main loop.

Lifecycle per frame:

1. :meth:`Mode.process` — analysis and reactions (no drawing).
2. the pipeline draws the standard overlays the mode asked for via ``toggles``
3. :meth:`Mode.draw` — the mode's own overlay, on top.
4. :meth:`Mode.hud_lines` — text the HUD appends to its panel.

Modes are also asked what they need through :attr:`Mode.requires`, so heavy
models stay off unless some active mode actually uses them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from core.state import AppState, FrameContext

__all__ = ["Mode", "ModeCategory", "ModeManager"]


class ModeCategory:
    """Grouping used by the mode picker and the help screen."""

    DETECTION = "deteccion"
    FITNESS = "fitness"
    WELLNESS = "bienestar"
    CREATIVE = "creativo"
    INTERACTION = "interaccion"
    UTILITY = "utilidad"

    ALL = (DETECTION, FITNESS, WELLNESS, CREATIVE, INTERACTION, UTILITY)


class Mode:
    """
    Base class for every mode.

    Subclasses override the hooks they care about; every hook has a working
    default, so a minimal mode is just a class with a ``key`` and ``label``.
    """

    #: Stable identifier used in config and by ``--mode``.
    key: str = "base"
    #: Human-readable name shown in the HUD.
    label: str = "Base"
    #: One-line explanation for the help screen.
    description: str = ""
    #: Grouping for the picker.
    category: str = ModeCategory.UTILITY
    #: Models this mode needs: pose, hands, face_mesh, face_detect, object.
    requires: Tuple[str, ...] = ("pose",)
    #: Toggle values applied when the mode is entered.
    toggles: Dict[str, bool] = {}
    #: Extra keys the mode handles, as ``{"p": "description"}`` for the help panel.
    keys: Dict[str, str] = {}
    #: Set by a mode that draws its own complete operator display.
    #:
    #: The shared HUD then suppresses its own status line and key hints instead
    #: of painting them over the mode's, which otherwise collide in the corners
    #: both want to use.
    owns_overlay: bool = False

    def __init__(self) -> None:
        self._entered_at: float = 0.0
        self._frames: int = 0

    # -- Lifecycle ----------------------------------------------------------

    def on_enter(self, state: AppState) -> None:
        """
        Called once when the mode becomes active.

        The base implementation applies :attr:`toggles`; override and call
        ``super().on_enter(state)`` to keep that behaviour.
        """
        self._entered_at = state.uptime
        self._frames = 0
        if self.toggles:
            state.apply_toggles(self.toggles)

    def on_exit(self, state: AppState) -> None:
        """Called once when leaving the mode. Release anything held here."""

    def reset(self, state: AppState) -> None:
        """Clear accumulated mode state without leaving the mode."""

    # -- Per-frame hooks ----------------------------------------------------

    def process(self, ctx: FrameContext, state: AppState) -> None:
        """Analysis and side effects for this frame. Do not draw here."""

    def draw(self, ctx: FrameContext, state: AppState) -> None:
        """Draw the mode's overlay onto ``ctx.frame``."""

    def hud_lines(self, ctx: FrameContext, state: AppState) -> List[str]:
        """Extra text lines for the HUD panel."""
        return []

    def status_text(self, ctx: FrameContext, state: AppState) -> Optional[str]:
        """
        Override the main status line for this frame.

        Returning ``None`` leaves the pipeline's own status text in place.
        """
        return None

    def on_key(self, key: int, state: AppState) -> bool:
        """
        Handle a mode-specific keypress.

        Return True when the key was consumed, so global bindings do not also
        fire for it.
        """
        return False

    # -- Introspection ------------------------------------------------------

    def needs(self, model: str) -> bool:
        """Whether this mode requires the named model."""
        return model in self.requires

    @property
    def frames_active(self) -> int:
        return self._frames

    def tick(self) -> None:
        """Called by the manager once per frame the mode is active."""
        self._frames += 1

    def help_lines(self) -> List[str]:
        """Description plus this mode's own key bindings, for the help panel."""
        lines = [f"{self.label}: {self.description}"] if self.description else [self.label]
        lines.extend(f"  [{k}] {v}" for k, v in self.keys.items())
        return lines

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Mode {self.key}>"


class ModeManager:
    """
    Registry and switcher for modes.

    Holds one instance of each registered mode so state (rep counts, drawings,
    calibration) survives switching away and back.
    """

    def __init__(self, state: AppState):
        self.state = state
        self._modes: Dict[str, Mode] = {}
        self._order: List[str] = []
        self._current: Optional[Mode] = None
        self._history: List[str] = []

    # -- Registration -------------------------------------------------------

    def register(self, mode: Mode) -> Mode:
        """Add a mode. Re-registering a key replaces the previous instance."""
        if mode.key not in self._modes:
            self._order.append(mode.key)
        self._modes[mode.key] = mode
        return mode

    def register_all(self, modes: Sequence[Mode]) -> None:
        for mode in modes:
            self.register(mode)

    # -- Access -------------------------------------------------------------

    @property
    def current(self) -> Optional[Mode]:
        return self._current

    @property
    def current_key(self) -> str:
        return self._current.key if self._current else ""

    def get(self, key: str) -> Optional[Mode]:
        return self._modes.get(key)

    def keys(self) -> List[str]:
        """Every registered mode key, in registration order."""
        return list(self._order)

    def all_modes(self) -> List[Mode]:
        return [self._modes[key] for key in self._order]

    def by_category(self, category: str) -> List[Mode]:
        return [m for m in self.all_modes() if m.category == category]

    def categories(self) -> Dict[str, List[Mode]]:
        """Modes grouped by category, in category order."""
        grouped: Dict[str, List[Mode]] = {}
        for mode in self.all_modes():
            grouped.setdefault(mode.category, []).append(mode)
        return grouped

    def __len__(self) -> int:
        return len(self._modes)

    def __contains__(self, key: str) -> bool:
        return key in self._modes

    # -- Switching ----------------------------------------------------------

    def switch(self, key: str) -> bool:
        """
        Activate a mode by key.

        Returns False for an unknown key, leaving the current mode running —
        a bad config value must not take the app down.
        """
        mode = self._modes.get(key)
        if mode is None:
            self.state.notify(f"Modo desconocido: {key}", "warn")
            return False
        if self._current is mode:
            return True

        if self._current is not None:
            self._current.on_exit(self.state)
            self._history.append(self._current.key)

        self._current = mode
        self.state.set_mode(key)
        mode.on_enter(self.state)
        self.state.bus.emit("mode.changed", mode=key, label=mode.label)
        self.state.notify(f"Modo: {mode.label}", "info")
        return True

    def next_mode(self, category: Optional[str] = None) -> bool:
        """Switch to the next mode, optionally staying within a category."""
        return self._step(1, category)

    def previous_mode(self, category: Optional[str] = None) -> bool:
        """Switch to the previous mode."""
        return self._step(-1, category)

    def _step(self, delta: int, category: Optional[str]) -> bool:
        pool = [m.key for m in (self.by_category(category) if category else self.all_modes())]
        if not pool:
            return False
        try:
            index = pool.index(self.current_key)
        except ValueError:
            index = -1 if delta > 0 else 0
        return self.switch(pool[(index + delta) % len(pool)])

    def go_back(self) -> bool:
        """Return to the previously active mode."""
        if not self._history:
            return False
        return self.switch(self._history.pop())

    # -- Per-frame dispatch -------------------------------------------------

    def process(self, ctx: FrameContext) -> None:
        if self._current:
            self._current.tick()
            self._current.process(ctx, self.state)

    def draw(self, ctx: FrameContext) -> None:
        if self._current:
            self._current.draw(ctx, self.state)

    def hud_lines(self, ctx: FrameContext) -> List[str]:
        return self._current.hud_lines(ctx, self.state) if self._current else []

    def status_text(self, ctx: FrameContext) -> Optional[str]:
        return self._current.status_text(ctx, self.state) if self._current else None

    def handle_key(self, key: int) -> bool:
        return self._current.on_key(key, self.state) if self._current else False

    @property
    def owns_overlay(self) -> bool:
        """Whether the active mode draws its own complete interface."""
        return bool(self._current.owns_overlay) if self._current else False

    def key_help(self) -> List[Tuple[str, str]]:
        """The active mode's own key bindings, as ``(key, description)`` rows."""
        if not self._current:
            return []
        return [(key, description) for key, description in self._current.keys.items()]

    # -- Model requirements -------------------------------------------------

    def required_models(self) -> Tuple[str, ...]:
        """Models the active mode needs — used to skip unnecessary inference."""
        return self._current.requires if self._current else ("pose",)

    def requires(self, model: str) -> bool:
        return self._current.needs(model) if self._current else model == "pose"

    def shutdown(self) -> None:
        """Exit the active mode cleanly at application shutdown."""
        if self._current:
            self._current.on_exit(self.state)
            self._current = None
