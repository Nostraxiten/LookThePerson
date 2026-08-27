"""
Event bus for LookThePerson.

Detection code publishes facts ("a clap happened", "the mode changed") without
knowing who reacts to them. Modes, the HUD, the action layer and the session
logger all subscribe independently, which is what makes it possible to add new
behaviour without touching the main loop.

The bus is synchronous and single-threaded by design — handlers run inside the
frame loop, so they must stay fast. A handler that raises is reported once and
then muted for that event to protect the loop.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

__all__ = ["Event", "EventBus", "Events"]


class Events:
    """Canonical event names. Using these constants avoids typo-only bugs."""

    FRAME_START = "frame.start"
    FRAME_END = "frame.end"

    POSE_DETECTED = "pose.detected"
    POSE_LOST = "pose.lost"
    HAND_DETECTED = "hand.detected"
    HAND_LOST = "hand.lost"
    FACE_DETECTED = "face.detected"
    FACE_LOST = "face.lost"
    OBJECT_DETECTED = "object.detected"

    GESTURE = "gesture"                 # any gesture, payload has "name"
    BODY_GESTURE = "gesture.body"
    HAND_GESTURE = "gesture.hand"
    FACE_GESTURE = "gesture.face"

    REP_COMPLETED = "fitness.rep"
    SET_COMPLETED = "fitness.set"
    POSTURE_ALERT = "posture.alert"
    BLINK = "face.blink"
    DROWSINESS_ALERT = "face.drowsy"
    PRESENCE_CHANGED = "presence.changed"
    MOTION_ALERT = "security.motion"

    MODE_CHANGED = "mode.changed"
    TOGGLE_CHANGED = "toggle.changed"
    ACTION_TRIGGERED = "action.triggered"
    RECORDING_STARTED = "recording.started"
    RECORDING_STOPPED = "recording.stopped"
    SCREENSHOT_TAKEN = "recording.screenshot"

    NOTIFY = "ui.notify"
    ERROR = "app.error"
    SHUTDOWN = "app.shutdown"


@dataclass
class Event:
    """A single published event."""

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Event({self.name!r}, {self.payload!r})"


Handler = Callable[[Event], None]


class EventBus:
    """
    Synchronous publish/subscribe hub with wildcard support.

    Subscribing to ``"gesture.*"`` receives every gesture sub-event, and
    ``"*"`` receives everything — useful for the session logger and the
    on-screen event timeline.
    """

    def __init__(self, history_size: int = 200):
        self._handlers: Dict[str, List[Handler]] = defaultdict(list)
        self._history: Deque[Event] = deque(maxlen=max(1, history_size))
        self._muted: set = set()
        self._failed: set = set()
        self._emit_count: Dict[str, int] = defaultdict(int)

    # -- Subscription -------------------------------------------------------

    def subscribe(self, name: str, handler: Handler) -> Callable[[], None]:
        """
        Register *handler* for events matching *name*.

        Returns a callable that unsubscribes, so callers can clean up without
        holding on to both the name and the function.
        """
        self._handlers[name].append(handler)

        def unsubscribe() -> None:
            self.unsubscribe(name, handler)

        return unsubscribe

    def subscribe_many(self, names: Iterable[str], handler: Handler) -> Callable[[], None]:
        """Subscribe one handler to several event names at once."""
        removers = [self.subscribe(name, handler) for name in names]

        def unsubscribe() -> None:
            for remove in removers:
                remove()

        return unsubscribe

    def unsubscribe(self, name: str, handler: Handler) -> bool:
        handlers = self._handlers.get(name)
        if not handlers or handler not in handlers:
            return False
        handlers.remove(handler)
        if not handlers:
            self._handlers.pop(name, None)
        return True

    def clear(self, name: Optional[str] = None) -> None:
        """Drop handlers for *name*, or every handler when *name* is None."""
        if name is None:
            self._handlers.clear()
        else:
            self._handlers.pop(name, None)

    # -- Publication --------------------------------------------------------

    def emit(self, name: str, /, **payload: Any) -> Event:
        """
        Publish an event and dispatch it to matching handlers.

        *name* is positional-only so that a payload may itself carry a ``name``
        key — which several events (toggles, gestures, actions) do.
        """
        return self.publish(Event(name, payload))

    def publish(self, event: Event) -> Event:
        self._history.append(event)
        self._emit_count[event.name] += 1

        if event.name in self._muted:
            return event

        for pattern in self._matching_patterns(event.name):
            for handler in list(self._handlers.get(pattern, ())):
                key = (pattern, id(handler))
                if key in self._failed:
                    continue
                try:
                    handler(event)
                except Exception as exc:  # keep the frame loop alive
                    self._failed.add(key)
                    print(
                        f"[events] Handler de '{pattern}' fallo y fue desactivado: {exc}",
                        flush=True,
                    )
        return event

    def _matching_patterns(self, name: str) -> List[str]:
        """Exact name, every ancestor wildcard, and the global wildcard."""
        patterns = [name, "*"]
        parts = name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            patterns.append(".".join(parts[:i]) + ".*")
        return patterns

    # -- Introspection ------------------------------------------------------

    def mute(self, name: str) -> None:
        """Stop dispatching an event name (it is still recorded in history)."""
        self._muted.add(name)

    def unmute(self, name: str) -> None:
        self._muted.discard(name)

    def history(self, name: Optional[str] = None, limit: int = 20) -> List[Event]:
        """Most recent events, newest last, optionally filtered by name."""
        events = [e for e in self._history if name is None or e.name == name]
        return events[-limit:]

    def counts(self) -> Dict[str, int]:
        """How many times each event has been emitted this session."""
        return dict(self._emit_count)

    def handler_count(self, name: Optional[str] = None) -> int:
        if name is not None:
            return len(self._handlers.get(name, ()))
        return sum(len(v) for v in self._handlers.values())

    def reset_failures(self) -> None:
        """Re-enable handlers that were muted after raising."""
        self._failed.clear()
