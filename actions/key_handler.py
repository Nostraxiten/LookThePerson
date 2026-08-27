"""
Keyboard handling for LookThePerson.

Maps keys to toggles and one-shot actions while the camera keeps running.
Extends the original handler with key groups, conflict detection, rebinding
and a help listing that stays in sync with what is actually registered.

The original API — ``register_toggle``, ``register_oneshot``, ``process_key``,
``is_active``, ``set_active``, ``get_status_lines`` — is unchanged.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

__all__ = ["KeyAction", "KeyHandler", "key_label", "SPECIAL_KEYS"]

#: Non-printable keys OpenCV reports, with the labels we show for them.
SPECIAL_KEYS: Dict[int, str] = {
    27: "ESC",
    9: "TAB",
    13: "ENTER",
    32: "SPACE",
    8: "BACK",
    81: "LEFT",
    82: "UP",
    83: "RIGHT",
    84: "DOWN",
}


def key_label(key_code: int) -> str:
    """Display label for a key code."""
    if key_code in SPECIAL_KEYS:
        return SPECIAL_KEYS[key_code]
    if 32 <= key_code < 127:
        return chr(key_code).upper()
    return f"0x{key_code:02X}"


class KeyAction:
    """A single key binding."""

    __slots__ = ("name", "key_label", "is_toggle", "active", "description", "group", "hidden")

    def __init__(
        self,
        name: str,
        key_label: str,
        description: str,
        is_toggle: bool = True,
        default_active: bool = False,
        group: str = "general",
        hidden: bool = False,
    ):
        self.name = name
        self.key_label = key_label
        self.description = description
        self.is_toggle = is_toggle
        self.active = default_active
        self.group = group
        self.hidden = hidden

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        kind = "toggle" if self.is_toggle else "oneshot"
        return f"<KeyAction {self.name} [{self.key_label}] {kind}>"


class KeyHandler:
    """
    Registry of key bindings, dispatched once per frame.

    Call :meth:`process_key` with whatever ``cv2.waitKey`` returned; it
    returns ``(action_name, state)`` when a binding matched, else ``None``.
    """

    def __init__(self):
        self._actions: Dict[int, KeyAction] = {}
        self._by_name: Dict[str, int] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._one_shot_callbacks: Dict[str, Callable] = {}
        self._conflicts: List[Tuple[int, str, str]] = []
        self._press_counts: Dict[str, int] = {}

    # -- Registration -------------------------------------------------------

    def register_toggle(
        self,
        key_code: int,
        name: str,
        description: str,
        default_active: bool = False,
        callback: Optional[Callable] = None,
        group: str = "general",
        hidden: bool = False,
    ) -> KeyAction:
        """Register a toggle (press flips it on or off)."""
        action = KeyAction(
            name, key_label(key_code), description,
            is_toggle=True, default_active=default_active, group=group, hidden=hidden,
        )
        self._add(key_code, action)
        if callback:
            self._callbacks[name] = callback
        return action

    def register_oneshot(
        self,
        key_code: int,
        name: str,
        description: str,
        callback: Optional[Callable] = None,
        group: str = "general",
        hidden: bool = False,
    ) -> KeyAction:
        """Register a one-shot action (press fires it once)."""
        action = KeyAction(
            name, key_label(key_code), description,
            is_toggle=False, group=group, hidden=hidden,
        )
        self._add(key_code, action)
        if callback:
            self._one_shot_callbacks[name] = callback
        return action

    def _add(self, key_code: int, action: KeyAction) -> None:
        existing = self._actions.get(key_code)
        if existing is not None and existing.name != action.name:
            # Recorded rather than raised: a later registration deliberately
            # wins, but the clash is worth surfacing in the debug view.
            self._conflicts.append((key_code, existing.name, action.name))
            self._by_name.pop(existing.name, None)
        self._actions[key_code] = action
        self._by_name[action.name] = key_code

    def rebind(self, name: str, new_key: int) -> bool:
        """Move an existing action to a different key."""
        old_key = self._by_name.get(name)
        if old_key is None:
            return False
        action = self._actions.pop(old_key)
        action.key_label = key_label(new_key)
        self._add(new_key, action)
        return True

    # -- Dispatch -----------------------------------------------------------

    def process_key(self, key_code: int):
        """
        Handle a keypress.

        Returns ``(action_name, new_state)`` for a match, else ``None``.
        Toggles report their new boolean; one-shots always report ``True``.
        """
        action = self._actions.get(key_code)
        if action is None:
            return None

        self._press_counts[action.name] = self._press_counts.get(action.name, 0) + 1

        if action.is_toggle:
            action.active = not action.active
            callback = self._callbacks.get(action.name)
            if callback:
                callback(action, action.active)
            return action.name, action.active

        callback = self._one_shot_callbacks.get(action.name)
        if callback:
            callback(action)
        return action.name, True

    def handles(self, key_code: int) -> bool:
        """Whether any binding claims this key."""
        return key_code in self._actions

    # -- State --------------------------------------------------------------

    def is_active(self, name: str) -> bool:
        """Whether a named toggle is on."""
        key_code = self._by_name.get(name)
        if key_code is None:
            return False
        return self._actions[key_code].active

    def set_active(self, name: str, active: bool) -> None:
        """Set a toggle's state without firing its callback."""
        key_code = self._by_name.get(name)
        if key_code is not None:
            self._actions[key_code].active = bool(active)

    def sync_from(self, toggles: Dict[str, bool]) -> None:
        """
        Copy state in from an external source.

        The application state owns the truth about toggles; this keeps the
        help panel's ON/OFF indicators honest after a mode preset changes
        things behind the key handler's back.
        """
        for name, active in toggles.items():
            self.set_active(name, active)

    # -- Introspection ------------------------------------------------------

    def get_all_actions(self) -> List[Tuple[int, KeyAction]]:
        """All bindings, sorted by key code."""
        return sorted(self._actions.items(), key=lambda item: item[0])

    def get_status_lines(self) -> List[Tuple[str, str, Optional[bool], str]]:
        """``(key_label, name, active_or_None, description)`` per visible binding."""
        return [
            (action.key_label, action.name,
             action.active if action.is_toggle else None, action.description)
            for _code, action in self.get_all_actions()
            if not action.hidden
        ]

    def help_rows(self) -> List[Tuple[str, str, Optional[bool]]]:
        """``(key, description, active)`` rows for the HUD help panel."""
        return [
            (action.key_label, action.description,
             action.active if action.is_toggle else None)
            for _code, action in self.get_all_actions()
            if not action.hidden
        ]

    def groups(self) -> Dict[str, List[KeyAction]]:
        """Bindings grouped by their declared group."""
        grouped: Dict[str, List[KeyAction]] = {}
        for _code, action in self.get_all_actions():
            grouped.setdefault(action.group, []).append(action)
        return grouped

    @property
    def conflicts(self) -> List[Tuple[int, str, str]]:
        """Keys that were registered twice, as ``(code, replaced, winner)``."""
        return list(self._conflicts)

    def press_counts(self) -> Dict[str, int]:
        return dict(self._press_counts)

    def __len__(self) -> int:
        return len(self._actions)
