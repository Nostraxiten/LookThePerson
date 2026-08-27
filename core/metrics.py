"""
Performance instrumentation for LookThePerson.

Answers three questions while the app runs:

* How fast is the loop? (:class:`FPSTracker` — instant, smoothed, 1% low)
* Where does the time go? (:class:`StageProfiler` — per-stage timings)
* Is anything degrading? (:class:`PerformanceMonitor` — health verdict and
  automatic quality suggestions)

All timings are in milliseconds unless stated otherwise.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from core.filters import RingBuffer

__all__ = ["FPSTracker", "StageProfiler", "StageStats", "PerformanceMonitor"]


# ---------------------------------------------------------------------------
# Frame rate
# ---------------------------------------------------------------------------

class FPSTracker:
    """
    Frame-rate statistics with an exponentially smoothed headline number.

    ``smoothing`` is the weight kept from the previous estimate: higher means
    a steadier but slower-reacting readout.
    """

    def __init__(self, smoothing: float = 0.9, history: int = 240):
        self.smoothing = min(max(smoothing, 0.0), 0.999)
        self._frame_times = RingBuffer(history)
        self._smoothed = 0.0
        self._instant = 0.0
        self._last_time: Optional[float] = None
        self._frame_count = 0
        self._start_time = time.monotonic()

    def tick(self, now: Optional[float] = None) -> float:
        """Record a frame boundary and return the smoothed FPS."""
        now = time.monotonic() if now is None else now
        if self._last_time is not None:
            dt = max(now - self._last_time, 1e-6)
            self._instant = 1.0 / dt
            self._frame_times.append(dt * 1000.0)
            if self._smoothed == 0.0:
                self._smoothed = self._instant
            else:
                self._smoothed = (
                    self._smoothed * self.smoothing + self._instant * (1.0 - self.smoothing)
                )
        self._last_time = now
        self._frame_count += 1
        return self._smoothed

    @property
    def fps(self) -> float:
        """Smoothed frames per second."""
        return self._smoothed

    @property
    def instant_fps(self) -> float:
        return self._instant

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def elapsed(self) -> float:
        """Seconds since the tracker was created."""
        return time.monotonic() - self._start_time

    @property
    def average_fps(self) -> float:
        """Mean FPS across the whole session."""
        elapsed = self.elapsed
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    @property
    def frame_time_ms(self) -> float:
        """Mean milliseconds per frame over the recent history."""
        return self._frame_times.mean()

    @property
    def one_percent_low(self) -> float:
        """
        The 1% low FPS — the average of the worst frames.

        A large gap between this and the headline FPS means stutter, even
        when the average looks healthy.
        """
        if len(self._frame_times) < 10:
            return self._smoothed
        worst_ms = self._frame_times.percentile(99)
        return 1000.0 / worst_ms if worst_ms > 0 else 0.0

    @property
    def jitter_ms(self) -> float:
        """Standard deviation of frame times — how uneven the pacing is."""
        return self._frame_times.std_dev()

    def history_ms(self) -> List[float]:
        return self._frame_times.values()

    def fps_history(self) -> List[float]:
        """Recent per-frame FPS values, for the on-screen graph."""
        return [1000.0 / ms if ms > 0 else 0.0 for ms in self._frame_times]

    def reset(self) -> None:
        self._frame_times.clear()
        self._smoothed = 0.0
        self._instant = 0.0
        self._last_time = None
        self._frame_count = 0
        self._start_time = time.monotonic()


# ---------------------------------------------------------------------------
# Stage profiling
# ---------------------------------------------------------------------------

@dataclass
class StageStats:
    """Aggregated timings for one named pipeline stage."""

    name: str
    calls: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0
    max_ms: float = 0.0
    samples: RingBuffer = field(default_factory=lambda: RingBuffer(120))

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    @property
    def recent_ms(self) -> float:
        """Mean over the recent window — reacts faster than the lifetime mean."""
        return self.samples.mean()


class StageProfiler:
    """
    Times named stages of the frame pipeline.

    ```python
    with profiler.stage("pose"):
        result = pose_model.detect(image, ts)
    ```

    Profiling can be switched off entirely, in which case the context manager
    becomes a near-free no-op.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._stats: Dict[str, StageStats] = {}
        self._frame_started: Optional[float] = None
        self._frame_total = RingBuffer(120)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block of code as *name*."""
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0)

    def record(self, name: str, elapsed_ms: float) -> None:
        """Record a timing directly, for callers that time things themselves."""
        stats = self._stats.get(name)
        if stats is None:
            stats = StageStats(name)
            self._stats[name] = stats
        stats.calls += 1
        stats.total_ms += elapsed_ms
        stats.last_ms = elapsed_ms
        stats.max_ms = max(stats.max_ms, elapsed_ms)
        stats.samples.append(elapsed_ms)

    def begin_frame(self) -> None:
        if self.enabled:
            self._frame_started = time.perf_counter()

    def end_frame(self) -> float:
        """Close the frame and return its total duration in milliseconds."""
        if not self.enabled or self._frame_started is None:
            return 0.0
        elapsed = (time.perf_counter() - self._frame_started) * 1000.0
        self._frame_total.append(elapsed)
        self._frame_started = None
        return elapsed

    @property
    def frame_ms(self) -> float:
        return self._frame_total.mean()

    def get(self, name: str) -> Optional[StageStats]:
        return self._stats.get(name)

    def all_stats(self) -> List[StageStats]:
        """Stages ordered by recent cost, most expensive first."""
        return sorted(self._stats.values(), key=lambda s: s.recent_ms, reverse=True)

    def top(self, count: int = 5) -> List[StageStats]:
        return self.all_stats()[:count]

    def breakdown(self) -> List[Tuple[str, float, float]]:
        """``(name, recent_ms, share_of_frame_percent)`` per stage."""
        total = sum(s.recent_ms for s in self._stats.values()) or 1.0
        return [
            (s.name, s.recent_ms, s.recent_ms / total * 100.0)
            for s in self.all_stats()
        ]

    def report_lines(self, count: int = 6) -> List[str]:
        """Human-readable summary for the debug overlay or the console."""
        lines = [f"frame: {self.frame_ms:5.1f} ms"]
        for name, ms, share in self.breakdown()[:count]:
            lines.append(f"{name:<14} {ms:6.2f} ms  {share:4.1f}%")
        return lines

    def reset(self) -> None:
        self._stats.clear()
        self._frame_total.clear()
        self._frame_started = None


# ---------------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Watches the frame rate and suggests quality changes when it drops.

    The app can act on :meth:`suggestion` to degrade gracefully — raising the
    object-detection stride, dropping the face mesh — instead of stuttering.
    """

    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

    def __init__(self, target_fps: float = 25.0, grace_seconds: float = 3.0):
        self.target_fps = max(1.0, target_fps)
        self.grace_seconds = grace_seconds
        self._started = time.monotonic()
        self._poor_since: Optional[float] = None
        self._suggestions: List[str] = []

    def update(self, fps: float, now: Optional[float] = None) -> str:
        """Feed the current FPS and get back a health verdict."""
        now = time.monotonic() if now is None else now
        ratio = fps / self.target_fps

        if ratio >= 0.9:
            verdict = self.GOOD
        elif ratio >= 0.6:
            verdict = self.FAIR
        else:
            verdict = self.POOR

        if verdict == self.POOR:
            self._poor_since = self._poor_since or now
        else:
            self._poor_since = None
        return verdict

    def is_struggling(self, now: Optional[float] = None) -> bool:
        """True once the frame rate has been poor for longer than the grace period."""
        if self._poor_since is None:
            return False
        now = time.monotonic() if now is None else now
        if now - self._started < self.grace_seconds:
            return False  # ignore the warm-up while models initialise
        return now - self._poor_since >= self.grace_seconds

    def suggestion(self, active_features: Dict[str, bool]) -> Optional[str]:
        """
        Name the most expensive feature currently enabled, to shed load.

        Returns ``None`` when nothing obvious is left to disable.
        """
        for feature in ("object_detect", "face_mesh", "segmentation", "face_detect"):
            if active_features.get(feature):
                return feature
        return None

    def reset(self) -> None:
        self._poor_since = None
        self._started = time.monotonic()
