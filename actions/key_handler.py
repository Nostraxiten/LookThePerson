"""
Real-time keyboard action handler for LookThePerson.
Maps keys to toggleable features that run while the camera stays active.
"""


class KeyAction:
    """Represents a single keyboard-togglable action."""

    __slots__ = ("name", "key_label", "is_toggle", "active", "description")

    def __init__(self, name, key_label, description, is_toggle=True, default_active=False):
        self.name = name
        self.key_label = key_label
        self.description = description
        self.is_toggle = is_toggle
        self.active = default_active


class KeyHandler:
    """
    Central registry of key bindings.
    Call `process_key(key_code)` each frame to dispatch actions.
    """

    def __init__(self):
        self._actions = {}  # key_code (int) -> KeyAction
        self._callbacks = {}  # action_name -> callable(action, active)
        self._one_shot_callbacks = {}  # action_name -> callable(action)

    def register_toggle(self, key_code, name, description, default_active=False, callback=None):
        """Register a toggle action (press to flip on/off)."""
        key_label = chr(key_code).upper() if 32 <= key_code < 127 else f"0x{key_code:02X}"
        action = KeyAction(name, key_label, description, is_toggle=True, default_active=default_active)
        self._actions[key_code] = action
        if callback:
            self._callbacks[name] = callback
        return action

    def register_oneshot(self, key_code, name, description, callback=None):
        """Register a one-shot action (press to fire once)."""
        key_label = chr(key_code).upper() if 32 <= key_code < 127 else f"0x{key_code:02X}"
        action = KeyAction(name, key_label, description, is_toggle=False, default_active=False)
        self._actions[key_code] = action
        if callback:
            self._one_shot_callbacks[name] = callback
        return action

    def process_key(self, key_code):
        """
        Process a keypress. Returns (action_name, new_state) or None.
        For toggles, new_state is the boolean active state.
        For one-shots, new_state is True.
        """
        action = self._actions.get(key_code)
        if action is None:
            return None

        if action.is_toggle:
            action.active = not action.active
            cb = self._callbacks.get(action.name)
            if cb:
                cb(action, action.active)
            return action.name, action.active
        else:
            cb = self._one_shot_callbacks.get(action.name)
            if cb:
                cb(action)
            return action.name, True

    def is_active(self, name):
        """Check if a named toggle action is currently active."""
        for action in self._actions.values():
            if action.name == name:
                return action.active
        return False

    def set_active(self, name, active):
        """Programmatically set a toggle's state."""
        for action in self._actions.values():
            if action.name == name:
                action.active = active
                return

    def get_all_actions(self):
        """Return list of (key_code, KeyAction) sorted by key."""
        return sorted(self._actions.items(), key=lambda item: item[0])

    def get_status_lines(self):
        """Return list of (key_label, name, active, description) for HUD display."""
        lines = []
        for _code, action in self.get_all_actions():
            lines.append((
                action.key_label,
                action.name,
                action.active if action.is_toggle else None,
                action.description,
            ))
        return lines
