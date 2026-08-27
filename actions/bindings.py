"""
Gesture-to-action bindings for LookThePerson.

Decouples "what the body did" from "what the app should do". Gestures are
detected in ``gestures``, actions are registered here, and the mapping between
them lives in configuration — so a user can rebind a gesture without touching
any code.

Every binding is gated by a permission and a cooldown: nothing that reaches
outside the application can fire without an explicit allow, and nothing can
fire faster than its cooldown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.events import Events
from core.filters import Cooldown

__all__ = ["Action", "ActionRegistry", "GestureBindings", "DEFAULT_BINDINGS"]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """
    Something the app can be asked to do.

    Attributes:
        name: stable identifier used in bindings.
        label: human-readable description.
        handler: callable invoked as ``handler(state, **payload)``.
        permission: config flag in ``gestures`` that must be true to run, or
            empty for actions that stay inside the app.
        cooldown: minimum seconds between firings.
    """

    name: str
    label: str
    handler: Callable[..., Any]
    permission: str = ""
    cooldown: float = 1.0
    category: str = "general"


class ActionRegistry:
    """
    Holds every registered action and enforces permissions and cooldowns.

    Modes and the main pipeline register their actions here; the binding layer
    only ever refers to them by name.
    """

    def __init__(self):
        self._actions: Dict[str, Action] = {}
        self._cooldowns: Dict[str, Cooldown] = {}
        self._fire_counts: Dict[str, int] = {}
        self._blocked: List[Tuple[str, str]] = []

    def register(
        self,
        name: str,
        label: str,
        handler: Callable[..., Any],
        permission: str = "",
        cooldown: float = 1.0,
        category: str = "general",
    ) -> Action:
        """Add an action, replacing any existing one with the same name."""
        action = Action(name, label, handler, permission, cooldown, category)
        self._actions[name] = action
        self._cooldowns[name] = Cooldown(cooldown)
        return action

    def unregister(self, name: str) -> bool:
        self._cooldowns.pop(name, None)
        return self._actions.pop(name, None) is not None

    def get(self, name: str) -> Optional[Action]:
        return self._actions.get(name)

    def names(self) -> List[str]:
        return sorted(self._actions)

    def by_category(self) -> Dict[str, List[Action]]:
        grouped: Dict[str, List[Action]] = {}
        for action in self._actions.values():
            grouped.setdefault(action.category, []).append(action)
        return grouped

    def __contains__(self, name: str) -> bool:
        return name in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    # -- Execution ----------------------------------------------------------

    def can_run(self, name: str, state: Any, now: float) -> Tuple[bool, str]:
        """
        Whether an action may run right now.

        Returns ``(allowed, reason)`` — the reason explains a refusal, which
        is what the HUD shows when a gesture visibly does nothing.
        """
        action = self._actions.get(name)
        if action is None:
            return False, "accion desconocida"

        if action.permission:
            allowed = getattr(state.config.gestures, action.permission, False)
            if not allowed:
                return False, f"permiso '{action.permission}' desactivado"

        if not self._cooldowns[name].ready(now):
            return False, "en enfriamiento"
        return True, ""

    def run(self, name: str, state: Any, now: float, **payload: Any) -> bool:
        """
        Execute an action if it is allowed.

        Exceptions from handlers are caught and reported: a failing action must
        never take the frame loop down with it.
        """
        allowed, reason = self.can_run(name, state, now)
        if not allowed:
            self._blocked.append((name, reason))
            if len(self._blocked) > 50:
                self._blocked.pop(0)
            return False

        action = self._actions[name]
        self._cooldowns[name].trigger(now)
        self._fire_counts[name] = self._fire_counts.get(name, 0) + 1

        try:
            action.handler(state, **payload)
        except Exception as exc:
            state.notify(f"Accion '{name}' fallo", "danger")
            print(f"[actions] '{name}' lanzo una excepcion: {exc}", flush=True)
            return False

        state.bus.emit(Events.ACTION_TRIGGERED, name=name, label=action.label)
        return True

    # -- Introspection ------------------------------------------------------

    def fire_counts(self) -> Dict[str, int]:
        return dict(self._fire_counts)

    def recent_blocks(self, limit: int = 5) -> List[Tuple[str, str]]:
        """Recently refused actions, for diagnosing 'why did nothing happen'."""
        return self._blocked[-limit:]


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------

#: Gestures mapped to action names out of the box. Config can override any of
#: these, and a binding pointing at an unknown action is simply ignored.
DEFAULT_BINDINGS: Dict[str, str] = {
    "clap": "change_color",
    "t_pose": "open_calculator",
    "arms_closed": "close_calculator",
    "both_hands_raised": "open_browser",
    "head_touch": "screenshot",
    "hands_on_hips": "toggle_help",
    "peace": "screenshot",
    "thumbs_up": "start_recording",
    "thumbs_down": "stop_recording",
    "rock": "next_theme",
    "ok": "next_mode",
    "spock": "toggle_grid",
}


@dataclass
class GestureBindings:
    """
    Maps gesture names to action names.

    Built from :data:`DEFAULT_BINDINGS` and then overlaid with whatever the
    user configured, so a partial config only changes what it mentions.
    """

    registry: ActionRegistry
    bindings: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BINDINGS))
    enabled: bool = True

    @classmethod
    def from_config(cls, registry: ActionRegistry, config: Any) -> "GestureBindings":
        """Build bindings from a :class:`~core.config.GestureConfig`."""
        merged = dict(DEFAULT_BINDINGS)
        merged.update(getattr(config, "bindings", None) or {})
        return cls(registry=registry, bindings=merged, enabled=getattr(config, "enabled", True))

    # -- Mapping ------------------------------------------------------------

    def bind(self, gesture: str, action: str) -> None:
        """Point a gesture at an action."""
        self.bindings[gesture] = action

    def unbind(self, gesture: str) -> bool:
        return self.bindings.pop(gesture, None) is not None

    def action_for(self, gesture: str) -> Optional[str]:
        return self.bindings.get(gesture)

    def gestures_for(self, action: str) -> List[str]:
        """Every gesture currently bound to an action."""
        return [g for g, a in self.bindings.items() if a == action]

    def describe(self) -> List[str]:
        """Readable ``gesture -> action`` listing for the help panel."""
        lines = []
        for gesture in sorted(self.bindings):
            action_name = self.bindings[gesture]
            action = self.registry.get(action_name)
            label = action.label if action else f"{action_name} (no registrada)"
            lines.append(f"{gesture} -> {label}")
        return lines

    # -- Dispatch -----------------------------------------------------------

    def fire(self, gesture: str, state: Any, now: float, **payload: Any) -> bool:
        """Run whatever is bound to *gesture*, if anything."""
        if not self.enabled:
            return False
        action_name = self.bindings.get(gesture)
        if not action_name:
            return False
        return self.registry.run(action_name, state, now, gesture=gesture, **payload)

    def dispatch(self, gestures: Dict[str, Any], state: Any, now: float) -> List[str]:
        """
        Fire every active gesture in a gesture dict.

        Values may be booleans or strings (``"left"`` / ``"right"``); a string
        also fires the side-specific binding, e.g. ``pointing_left``.
        """
        fired: List[str] = []
        if not self.enabled:
            return fired

        for gesture, value in gestures.items():
            if not value:
                continue
            if isinstance(value, str):
                if self.fire(f"{gesture}_{value}", state, now):
                    fired.append(f"{gesture}_{value}")
                    continue
            if self.fire(gesture, state, now):
                fired.append(gesture)
        return fired
