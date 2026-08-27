"""Tests for the core infrastructure: geometry, filters, config, events."""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import pytest

from core.config import Config, deep_merge, load_config, save_config
from core.events import Event, EventBus, Events
from core.filters import (
    Cooldown,
    Debouncer,
    EdgeDetector,
    ExponentialFilter,
    Hysteresis,
    MedianFilter,
    OneEuroFilter,
    RingBuffer,
    VelocityTracker,
    moving_average,
)
from core.geometry import (
    Point,
    angle_between,
    as_point,
    box_iou,
    centroid,
    clamp,
    distance,
    inverse_lerp,
    joint_angle,
    normalize,
    midpoint,
    polygon_area,
    remap,
    signed_angle,
)
from core.metrics import FPSTracker, StageProfiler
from core.state import AppState
from core.theme import get_theme, next_theme, theme_names


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_distance(self):
        assert distance(Point(0, 0), Point(3, 4)) == pytest.approx(5.0)

    def test_as_point_defaults_none_visibility_without_changing_zero(self):
        missing = SimpleNamespace(x=0.0, y=0.0, visibility=None)
        hidden = SimpleNamespace(x=1.0, y=1.0, visibility=0.0)

        assert as_point(missing).visibility == 1.0
        assert midpoint(missing, hidden).visibility == 0.0

    def test_joint_angle_straight_limb_is_180(self):
        angle = joint_angle(Point(0, 0), Point(1, 0), Point(2, 0))
        assert angle == pytest.approx(180.0)

    def test_joint_angle_right_angle(self):
        angle = joint_angle(Point(0, 0), Point(1, 0), Point(1, 1))
        assert angle == pytest.approx(90.0)

    def test_joint_angle_folded_limb_is_zero(self):
        angle = joint_angle(Point(0, 0), Point(1, 0), Point(0, 0))
        assert angle == pytest.approx(0.0)

    def test_angle_between_handles_zero_vectors(self):
        assert angle_between((0, 0), (1, 0)) == 0.0

    def test_signed_angle_direction(self):
        assert signed_angle((1, 0), (0, 1)) == pytest.approx(90.0)
        assert signed_angle((1, 0), (0, -1)) == pytest.approx(-90.0)

    def test_normalize_zero_vector(self):
        assert normalize((0.0, 0.0)) == (0.0, 0.0)

    def test_centroid_of_empty_is_none(self):
        assert centroid([]) is None

    def test_centroid(self):
        result = centroid([Point(0, 0), Point(2, 0), Point(1, 3)])
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(1.0)

    def test_clamp_handles_reversed_bounds(self):
        assert clamp(5, 10, 0) == 5
        assert clamp(-5, 0, 10) == 0

    def test_inverse_lerp_equal_bounds(self):
        assert inverse_lerp(1.0, 1.0, 1.0) == 0.0

    def test_remap_clamps_to_output_range(self):
        assert remap(20, 0, 10, 0, 100) == pytest.approx(100.0)

    def test_box_iou_identical_boxes(self):
        assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == pytest.approx(1.0)

    def test_box_iou_disjoint_boxes(self):
        assert box_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0

    def test_polygon_area_square(self):
        square = [Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)]
        assert polygon_area(square) == pytest.approx(4.0)

    def test_polygon_area_degenerate(self):
        assert polygon_area([Point(0, 0), Point(1, 1)]) == 0.0


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFilters:
    def test_exponential_filter_converges(self):
        f = ExponentialFilter(alpha=0.5)
        f.update(0.0)
        for _ in range(30):
            f.update(10.0)
        assert f.value == pytest.approx(10.0, abs=0.01)

    def test_exponential_filter_rejects_bad_alpha(self):
        with pytest.raises(ValueError):
            ExponentialFilter(alpha=0.0)

    def test_median_filter_removes_spike(self):
        f = MedianFilter(window=5)
        for value in (1.0, 1.0, 99.0, 1.0, 1.0):
            f.update(value)
        assert f.value == pytest.approx(1.0)

    def test_one_euro_smooths_but_tracks(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        f.update(0.0, 0.0)
        # A step input should be damped, not followed instantly.
        first = f.update(10.0, 0.033)
        assert 0.0 < first < 10.0
        for i in range(2, 200):
            value = f.update(10.0, i * 0.033)
        assert value == pytest.approx(10.0, abs=0.1)

    def test_one_euro_ignores_non_advancing_time(self):
        f = OneEuroFilter()
        f.update(1.0, 1.0)
        assert f.update(5.0, 1.0) == pytest.approx(1.0)

    def test_hysteresis_needs_full_swing(self):
        h = Hysteresis(low=0.3, high=0.7)
        assert h.update(0.5) is False      # below the high threshold
        assert h.update(0.8) is True
        assert h.update(0.5) is True       # still above the low threshold
        assert h.update(0.2) is False

    def test_debouncer_requires_hold(self):
        d = Debouncer(rise_seconds=0.2, fall_seconds=0.2)
        assert d.update(True, 0.0) is False
        assert d.update(True, 0.1) is False
        assert d.update(True, 0.25) is True

    def test_debouncer_resets_on_flicker(self):
        d = Debouncer(rise_seconds=0.2)
        d.update(True, 0.0)
        d.update(False, 0.1)
        d.update(True, 0.15)
        assert d.update(True, 0.3) is False   # the timer restarted at 0.15
        assert d.update(True, 0.36) is True

    def test_edge_detector(self):
        e = EdgeDetector()
        assert e.update(True) == "rising"
        assert e.update(True) == ""
        assert e.update(False) == "falling"

    def test_cooldown_rate_limits(self):
        c = Cooldown(1.0)
        assert c.trigger(0.0) is True
        assert c.trigger(0.5) is False
        assert c.trigger(1.1) is True

    def test_velocity_tracker(self):
        v = VelocityTracker(window_seconds=1.0)
        v.update(0.0, 0.0, 0.0)
        _vx, _vy, speed = v.update(1.0, 0.0, 1.0)
        assert speed == pytest.approx(1.0, abs=0.01)

    def test_ring_buffer_statistics(self):
        buffer = RingBuffer(capacity=5, values=[1, 2, 3, 4, 5])
        assert buffer.mean() == pytest.approx(3.0)
        assert buffer.minimum() == 1
        assert buffer.maximum() == 5
        assert buffer.percentile(50) == pytest.approx(3.0)
        assert buffer.is_full

    def test_ring_buffer_evicts_oldest(self):
        buffer = RingBuffer(capacity=2)
        buffer.extend([1, 2, 3])
        assert buffer.values() == [2, 3]

    def test_moving_average(self):
        assert moving_average([1, 2, 3, 4], 2) == [1.0, 1.5, 2.5, 3.5]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.camera.width == 1280
        assert config.mode == "full"

    def test_from_dict_rebuilds_nested_sections(self):
        config = Config.from_dict({"camera": {"width": 640}, "detection": {"max_hands": 4}})
        assert config.camera.width == 640
        assert config.camera.height == 720          # default preserved
        assert config.detection.max_hands == 4

    def test_from_dict_ignores_unknown_keys(self):
        config = Config.from_dict({"camera": {"width": 800}, "nonexistent": 1})
        assert config.camera.width == 800

    def test_round_trip(self):
        config = Config.from_dict({"camera": {"fps": 60}})
        assert Config.from_dict(config.to_dict()).camera.fps == 60

    def test_dotted_get_and_set(self):
        config = Config()
        assert config.get("camera.width") == 1280
        assert config.set("camera.width", 999) is True
        assert config.get("camera.width") == 999
        assert config.get("nope.nope", "fallback") == "fallback"
        assert config.set("nope.nope", 1) is False

    def test_deep_merge_skips_none(self):
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": None, "c": 9}})
        assert merged == {"a": {"b": 1, "c": 9}}

    def test_deep_merge_does_not_mutate_input(self):
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}

    def test_save_and_load_profile(self, tmp_path, monkeypatch):
        path = tmp_path / "config.json"
        config = Config.from_dict({"camera": {"width": 640}, "mode": "reps"})
        save_config(config, str(path), profile="gym")

        document = json.loads(path.read_text())
        assert "profiles" in document
        assert document["profiles"]["gym"]["mode"] == "reps"

        # A profile overlays the base document when loaded.
        monkeypatch.setattr("core.config.USER_CONFIG_PATH", str(path))
        monkeypatch.chdir(tmp_path)
        loaded = load_config(profile="gym")
        assert loaded.mode == "reps"
        assert loaded.camera.width == 640

    def test_load_config_survives_malformed_file(self, tmp_path, monkeypatch):
        path = tmp_path / "config.json"
        path.write_text("{ this is not json")
        monkeypatch.setattr("core.config.USER_CONFIG_PATH", str(path))
        monkeypatch.chdir(tmp_path)
        assert load_config().camera.width == 1280

    def test_cli_overrides_win(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("core.config.USER_CONFIG_PATH", str(tmp_path / "none.json"))
        config = load_config({"camera": {"width": 320}})
        assert config.camera.width == 320


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEventBus:
    def test_emit_and_receive(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", received.append)
        bus.emit("test", value=42)
        assert received[0].get("value") == 42

    def test_payload_may_contain_name_key(self):
        # ``emit``'s first parameter is positional-only precisely for this.
        bus = EventBus()
        received = []
        bus.subscribe("gesture", received.append)
        bus.emit("gesture", name="clap")
        assert received[0].get("name") == "clap"

    def test_wildcard_subscription(self):
        bus = EventBus()
        received = []
        bus.subscribe("gesture.*", received.append)
        bus.emit("gesture.body", name="t_pose")
        bus.emit("gesture.hand", name="peace")
        bus.emit("other.thing")
        assert len(received) == 2

    def test_global_wildcard(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", received.append)
        bus.emit("anything")
        assert len(received) == 1

    def test_unsubscribe_via_returned_callable(self):
        bus = EventBus()
        received = []
        unsubscribe = bus.subscribe("test", received.append)
        bus.emit("test")
        unsubscribe()
        bus.emit("test")
        assert len(received) == 1

    def test_failing_handler_is_isolated_and_muted(self):
        bus = EventBus()
        good = []

        def broken(_event):
            raise RuntimeError("boom")

        bus.subscribe("test", broken)
        bus.subscribe("test", good.append)
        bus.emit("test")
        bus.emit("test")
        assert len(good) == 2          # the good handler kept running

    def test_history_and_counts(self):
        bus = EventBus()
        bus.emit("a")
        bus.emit("a")
        bus.emit("b")
        assert bus.counts() == {"a": 2, "b": 1}
        assert len(bus.history("a")) == 2


# ---------------------------------------------------------------------------
# State, metrics, theme
# ---------------------------------------------------------------------------

class TestAppState:
    def test_toggle_emits_only_on_change(self):
        state = AppState(Config())
        events = []
        state.bus.subscribe(Events.TOGGLE_CHANGED, events.append)

        state.set_toggle("grid", True)
        state.set_toggle("grid", True)     # no change, no event
        assert len(events) == 1
        assert state.is_active("grid")

    def test_toggle_flips(self):
        state = AppState(Config())
        first = state.toggle("grid")
        assert state.toggle("grid") is not first

    def test_counters(self):
        state = AppState(Config())
        state.increment("reps")
        state.increment("reps", 2)
        assert state.counter("reps") == 3

    def test_set_theme_updates_body_color(self):
        state = AppState(Config())
        state.set_theme("matrix")
        assert state.theme.name == "matrix"
        assert state.body_color == state.theme.skeleton

    def test_unknown_theme_falls_back(self):
        assert get_theme("does-not-exist").name == "cyber"

    def test_next_theme_wraps(self):
        names = theme_names()
        assert next_theme(names[-1]) == names[0]

    def test_summary_shape(self):
        summary = AppState(Config()).summary()
        assert {"uptime_seconds", "frames", "mode", "counters"} <= set(summary)


class TestMetrics:
    def test_fps_tracker(self):
        tracker = FPSTracker(smoothing=0.0)
        tracker.tick(0.0)
        tracker.tick(0.1)         # 10 fps
        assert tracker.fps == pytest.approx(10.0, abs=0.1)
        assert tracker.frame_count == 2

    def test_profiler_records_stages(self):
        profiler = StageProfiler()
        with profiler.stage("work"):
            sum(range(1000))
        stats = profiler.get("work")
        assert stats is not None and stats.calls == 1

    def test_disabled_profiler_records_nothing(self):
        profiler = StageProfiler(enabled=False)
        with profiler.stage("work"):
            pass
        assert profiler.get("work") is None

    def test_breakdown_percentages_sum_to_100(self):
        profiler = StageProfiler()
        profiler.record("a", 10.0)
        profiler.record("b", 30.0)
        total = sum(share for _name, _ms, share in profiler.breakdown())
        assert total == pytest.approx(100.0)
