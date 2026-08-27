"""
Signal filters and temporal debouncing for LookThePerson.

Raw landmark streams are noisy and raw booleans flicker. Everything in this
module exists to turn jittery per-frame values into stable ones:

* :class:`OneEuroFilter` — adaptive low-pass, the standard for hand/pose data.
* :class:`ExponentialFilter` / :class:`MedianFilter` — cheap smoothing.
* :class:`Hysteresis` — a Schmitt trigger for threshold crossings.
* :class:`Debouncer` — requires a boolean to hold before it counts.
* :class:`EdgeDetector` — rising/falling edges from a boolean stream.
* :class:`Cooldown` — rate-limits repeated triggers.
* :class:`VelocityTracker` — derivative of a positional signal.

Pure Python: no numpy, so these are safe to unit-test anywhere.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ExponentialFilter",
    "MedianFilter",
    "OneEuroFilter",
    "PointFilter",
    "Hysteresis",
    "Debouncer",
    "EdgeDetector",
    "Cooldown",
    "VelocityTracker",
    "RingBuffer",
    "moving_average",
]


# ---------------------------------------------------------------------------
# Simple smoothing
# ---------------------------------------------------------------------------

class ExponentialFilter:
    """
    Exponential moving average.

    *alpha* is the weight of the newest sample: 1.0 passes the signal through
    untouched, values near 0 smooth heavily.
    """

    __slots__ = ("alpha", "_value")

    def __init__(self, alpha: float = 0.35, initial: Optional[float] = None):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value = initial

    @property
    def value(self) -> Optional[float]:
        return self._value

    def update(self, sample: float) -> float:
        if self._value is None:
            self._value = float(sample)
        else:
            self._value += self.alpha * (float(sample) - self._value)
        return self._value

    def reset(self) -> None:
        self._value = None


class MedianFilter:
    """Rolling median — removes isolated spikes without rounding real edges."""

    __slots__ = ("_window", "_samples")

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._samples: Deque[float] = deque(maxlen=window)

    @property
    def value(self) -> Optional[float]:
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def update(self, sample: float) -> float:
        self._samples.append(float(sample))
        return self.value  # type: ignore[return-value]

    def reset(self) -> None:
        self._samples.clear()


class OneEuroFilter:
    """
    The 1€ filter (Casiez et al., 2012).

    Adaptive low-pass: heavy smoothing when the signal is slow (kills jitter),
    light smoothing when it moves fast (kills lag). Tune with *min_cutoff* for
    baseline steadiness and *beta* for how aggressively it opens up on motion.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x_prev", "_dx_prev", "_t_prev")

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, sample: float, timestamp: float) -> float:
        """Filter *sample* observed at *timestamp* (seconds, monotonic)."""
        sample = float(sample)
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = sample
            self._t_prev = timestamp
            self._dx_prev = 0.0
            return sample

        dt = timestamp - self._t_prev
        if dt <= 0.0:
            return self._x_prev
        self._t_prev = timestamp

        # Filtered derivative drives the adaptive cutoff.
        dx = (sample - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx_prev += a_d * (dx - self._dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(self._dx_prev)
        a = self._alpha(cutoff, dt)
        self._x_prev += a * (sample - self._x_prev)
        return self._x_prev

    @property
    def value(self) -> Optional[float]:
        return self._x_prev

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class PointFilter:
    """A 1€ filter applied independently to the X, Y and Z axes of a point."""

    __slots__ = ("_fx", "_fy", "_fz")

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007):
        self._fx = OneEuroFilter(min_cutoff, beta)
        self._fy = OneEuroFilter(min_cutoff, beta)
        self._fz = OneEuroFilter(min_cutoff, beta)

    def update(self, x: float, y: float, z: float, timestamp: float) -> Tuple[float, float, float]:
        return (
            self._fx.update(x, timestamp),
            self._fy.update(y, timestamp),
            self._fz.update(z, timestamp),
        )

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self._fz.reset()


# ---------------------------------------------------------------------------
# Boolean conditioning
# ---------------------------------------------------------------------------

class Hysteresis:
    """
    Schmitt trigger: turns on above *high* and off below *low*.

    Keeping the two thresholds apart stops a value hovering near a single
    threshold from rapidly toggling the output.
    """

    __slots__ = ("low", "high", "_state")

    def __init__(self, low: float, high: float, initial: bool = False):
        if low > high:
            low, high = high, low
        self.low = low
        self.high = high
        self._state = initial

    @property
    def state(self) -> bool:
        return self._state

    def update(self, value: float) -> bool:
        if self._state:
            if value < self.low:
                self._state = False
        elif value > self.high:
            self._state = True
        return self._state

    def reset(self, state: bool = False) -> None:
        self._state = state


class Debouncer:
    """
    Requires a boolean to hold steady before the change is accepted.

    *rise_seconds* is how long a True must persist to turn the output on, and
    *fall_seconds* how long a False must persist to turn it back off.
    """

    __slots__ = ("rise_seconds", "fall_seconds", "_state", "_pending_since", "_pending_value")

    def __init__(self, rise_seconds: float = 0.2, fall_seconds: float = 0.2, initial: bool = False):
        self.rise_seconds = max(0.0, rise_seconds)
        self.fall_seconds = max(0.0, fall_seconds)
        self._state = initial
        self._pending_since: Optional[float] = None
        self._pending_value: Optional[bool] = None

    @property
    def state(self) -> bool:
        return self._state

    def update(self, value: bool, now: float) -> bool:
        value = bool(value)
        if value == self._state:
            self._pending_since = None
            self._pending_value = None
            return self._state

        if self._pending_value != value:
            self._pending_value = value
            self._pending_since = now
            return self._state

        needed = self.rise_seconds if value else self.fall_seconds
        if self._pending_since is not None and now - self._pending_since >= needed:
            self._state = value
            self._pending_since = None
            self._pending_value = None
        return self._state

    def reset(self, state: bool = False) -> None:
        self._state = state
        self._pending_since = None
        self._pending_value = None


class EdgeDetector:
    """Reports rising and falling transitions of a boolean stream."""

    __slots__ = ("_state",)

    def __init__(self, initial: bool = False):
        self._state = initial

    @property
    def state(self) -> bool:
        return self._state

    def update(self, value: bool) -> str:
        """Return ``"rising"``, ``"falling"`` or ``""`` when nothing changed."""
        value = bool(value)
        if value == self._state:
            return ""
        self._state = value
        return "rising" if value else "falling"

    def rising(self, value: bool) -> bool:
        return self.update(value) == "rising"


class Cooldown:
    """Allows an action at most once per *seconds*."""

    __slots__ = ("seconds", "_last")

    def __init__(self, seconds: float):
        self.seconds = max(0.0, seconds)
        self._last = float("-inf")

    def ready(self, now: float) -> bool:
        return now - self._last >= self.seconds

    def trigger(self, now: float) -> bool:
        """Consume the cooldown; True when the action is allowed to run."""
        if not self.ready(now):
            return False
        self._last = now
        return True

    def remaining(self, now: float) -> float:
        return max(0.0, self.seconds - (now - self._last))

    def reset(self) -> None:
        self._last = float("-inf")


# ---------------------------------------------------------------------------
# Derivatives and buffers
# ---------------------------------------------------------------------------

class VelocityTracker:
    """
    Estimates speed from positional samples over a short time window.

    Uses the oldest and newest sample in the window rather than consecutive
    frames, which is far less sensitive to single-frame jitter.
    """

    __slots__ = ("_window_seconds", "_samples")

    def __init__(self, window_seconds: float = 0.25):
        self._window_seconds = max(1e-3, window_seconds)
        self._samples: Deque[Tuple[float, float, float]] = deque()

    def update(self, x: float, y: float, now: float) -> Tuple[float, float, float]:
        """
        Add a sample and return ``(vx, vy, speed)`` in units per second.
        """
        self._samples.append((now, float(x), float(y)))
        cutoff = now - self._window_seconds
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if len(self._samples) < 2:
            return 0.0, 0.0, 0.0

        t0, x0, y0 = self._samples[0]
        t1, x1, y1 = self._samples[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0, 0.0, 0.0
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        return vx, vy, math.hypot(vx, vy)

    def reset(self) -> None:
        self._samples.clear()


class RingBuffer:
    """Fixed-capacity sample history with basic statistics."""

    __slots__ = ("_data",)

    def __init__(self, capacity: int = 120, values: Optional[Iterable[float]] = None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._data: Deque[float] = deque(values or (), maxlen=capacity)

    def append(self, value: float) -> None:
        self._data.append(float(value))

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.append(value)

    def values(self) -> List[float]:
        return list(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    @property
    def capacity(self) -> int:
        return self._data.maxlen or 0

    @property
    def is_full(self) -> bool:
        return len(self._data) == self._data.maxlen

    def mean(self) -> float:
        return sum(self._data) / len(self._data) if self._data else 0.0

    def minimum(self) -> float:
        return min(self._data) if self._data else 0.0

    def maximum(self) -> float:
        return max(self._data) if self._data else 0.0

    def std_dev(self) -> float:
        n = len(self._data)
        if n < 2:
            return 0.0
        mean = self.mean()
        variance = sum((v - mean) ** 2 for v in self._data) / (n - 1)
        return math.sqrt(variance)

    def percentile(self, pct: float) -> float:
        """Linear-interpolated percentile, *pct* in 0..100."""
        if not self._data:
            return 0.0
        ordered = sorted(self._data)
        if len(ordered) == 1:
            return ordered[0]
        pos = (max(0.0, min(100.0, pct)) / 100.0) * (len(ordered) - 1)
        low = int(math.floor(pos))
        high = min(low + 1, len(ordered) - 1)
        frac = pos - low
        return ordered[low] * (1.0 - frac) + ordered[high] * frac

    def clear(self) -> None:
        self._data.clear()


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Simple trailing moving average; the first samples use a shorter window."""
    if window < 1:
        raise ValueError("window must be >= 1")
    out: List[float] = []
    running: Deque[float] = deque(maxlen=window)
    for value in values:
        running.append(float(value))
        out.append(sum(running) / len(running))
    return out
