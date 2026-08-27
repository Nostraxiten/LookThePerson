"""
Tests for the mode system, gestures and the action layer.

The mode tests are deliberately exhaustive rather than sampled: every
registered mode is driven through a full frame cycle, because a mode that
raises on its first frame is the failure users would hit immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from actions.bindings import ActionRegistry, GestureBindings
from actions.key_handler import KeyHandler
from analytics.angles import compute_joint_angles
from core.config import Config
from core.geometry import Point
from core.state import AppState, FrameContext
from gestures.body_gestures import detect_every_gesture
from gestures.hand_gestures import (
    classify_hand,
    detect_all_hand_gestures,
    extended_fingers,
    hand_openness,
    hand_orientation,
    is_pinching,
    HandGestureTracker,
)
from modes import all_modes, build_mode_manager, mode_keys
from modes.base import Mode, ModeCategory, ModeManager
from tests.conftest import make_face, make_hand, make_pose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeMask:
    """Stands in for a MediaPipe segmentation mask."""

    def __init__(self, height: int = 180, width: int = 320):
        self._data = np.zeros((height, width), dtype=np.float32)
        self._data[40:140, 100:220] = 1.0

    def numpy_view(self):
        return self._data


class FakePoseResult:
    def __init__(self, landmarks):
        self.pose_landmarks = [landmarks]
        self.segmentation_masks = [FakeMask()]


class FakeDetection:
    class bounding_box:
        origin_x, origin_y, width, height = 120, 60, 90, 90

    categories = [type("C", (), {"category_name": "person", "score": 0.91})()]


class FakeDetectionResult:
    detections = [FakeDetection()]


def make_context(frame_index: int = 0, knee: float = 175.0) -> FrameContext:
    """A fully populated frame context, as the real pipeline would build."""
    pose = make_pose(knee)
    return FrameContext(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        width=640, height=480,
        timestamp_ms=frame_index * 33,
        now=frame_index * 0.033,
        delta=0.033,
        frame_index=frame_index,
        pose_result=FakePoseResult(pose),
        face_detect_result=FakeDetectionResult(),
        object_result=FakeDetectionResult(),
        pose_landmarks=[pose],
        hand_landmarks=[make_hand()],
        face_landmarks=[make_face()],
        handedness=["Right"],
        body_center=(320, 240),
        body_gestures=detect_every_gesture(pose, 0.53),
        hand_info=detect_all_hand_gestures([make_hand()]),
        face_info={
            "raw": {"smile": True}, "triggered": [], "gaze": (0.1, -0.05),
            "head_pose": {"yaw": 4.0, "pitch": -2.0, "roll": 1.0},
        },
        angles=compute_joint_angles(pose),
        motion={"energy": 0.03, "left_hand_speed": 0.2, "right_hand_speed": 0.4},
    )


@pytest.fixture
def state() -> AppState:
    return AppState(Config())


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

class TestModeRegistry:
    def test_many_modes_are_registered(self):
        assert len(all_modes()) >= 40

    def test_mode_keys_are_unique(self):
        keys = mode_keys()
        assert len(keys) == len(set(keys))

    def test_every_mode_declares_its_metadata(self):
        for mode in all_modes():
            assert mode.key and mode.label
            assert mode.category in ModeCategory.ALL
            assert mode.requires, f"{mode.key} declara requires vacio"

    def test_every_category_has_modes(self):
        categories = {mode.category for mode in all_modes()}
        assert categories == set(ModeCategory.ALL)

    def test_build_manager_activates_requested_mode(self, state):
        manager = build_mode_manager(state, "reps")
        assert manager.current_key == "reps"

    def test_unknown_mode_falls_back_to_default(self, state):
        manager = build_mode_manager(state, "does-not-exist")
        assert manager.current_key == "full"

    def test_switch_applies_mode_toggles(self, state):
        manager = build_mode_manager(state, "full")
        manager.switch("minimal")
        assert state.is_active("skeleton") is False

    def test_switch_emits_event(self, state):
        manager = build_mode_manager(state, "full")
        events = []
        state.bus.subscribe("mode.changed", events.append)
        manager.switch("pose")
        assert events and events[0].get("mode") == "pose"

    def test_next_and_previous_wrap(self, state):
        manager = build_mode_manager(state, "full")
        first = manager.current_key
        manager.next_mode()
        assert manager.current_key != first
        manager.previous_mode()
        assert manager.current_key == first

    def test_go_back_returns_to_previous(self, state):
        manager = build_mode_manager(state, "full")
        manager.switch("pose")
        manager.go_back()
        assert manager.current_key == "full"

    def test_category_cycling_stays_in_category(self, state):
        manager = build_mode_manager(state, "reps")
        manager.next_mode(category=ModeCategory.FITNESS)
        assert manager.current.category == ModeCategory.FITNESS

    def test_mode_instances_persist_across_switches(self, state):
        manager = build_mode_manager(state, "reps")
        reps = manager.get("reps")
        manager.switch("pose")
        manager.switch("reps")
        assert manager.get("reps") is reps


class TestModeExecution:
    """Every mode must survive a full frame cycle and its own key bindings."""

    @pytest.mark.parametrize("mode_key", mode_keys())
    def test_mode_runs_a_frame(self, mode_key, state):
        manager = build_mode_manager(state, mode_key)
        mode = manager.current

        for index in range(4):
            ctx = make_context(index, knee=175.0 if index % 2 else 85.0)
            manager.process(ctx)
            manager.draw(ctx)
            lines = manager.hud_lines(ctx)
            assert isinstance(lines, list)
            assert all(isinstance(line, str) for line in lines)
            manager.status_text(ctx)

        for key in mode.keys:
            manager.handle_key(32 if key == "space" else ord(key[0]))

        mode.reset(state)
        manager.shutdown()

    @pytest.mark.parametrize("mode_key", mode_keys())
    def test_mode_survives_empty_frame(self, mode_key, state):
        """No person, no hands, no face — the most common real-world frame."""
        manager = build_mode_manager(state, mode_key)
        empty = FrameContext(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            width=640, height=480, timestamp_ms=0, now=0.0, delta=0.033,
        )
        manager.process(empty)
        manager.draw(empty)
        manager.hud_lines(empty)


# ---------------------------------------------------------------------------
# Gestures
# ---------------------------------------------------------------------------

class TestHandGestures:
    def test_open_palm(self):
        assert classify_hand(make_hand(fingers=5)) == "open_palm"

    def test_fist(self):
        assert classify_hand(make_hand(fingers=0)) == "fist"

    def test_peace_sign(self):
        # Index and middle up, thumb folded in — make_hand extends the thumb
        # first, so it has to be tucked back explicitly.
        peace = make_hand(fingers=3)
        folded = make_hand(fingers=0)
        for index in (1, 2, 3, 4):
            peace[index] = folded[index]
        assert classify_hand(peace) == "peace"

    def test_finger_counting(self):
        assert sum(extended_fingers(make_hand(fingers=5))) == 5
        assert sum(extended_fingers(make_hand(fingers=0))) == 0

    def test_openness_ordering(self):
        assert hand_openness(make_hand(fingers=5)) > hand_openness(make_hand(fingers=0))

    def test_orientation_is_up_for_default_hand(self):
        assert hand_orientation(make_hand()) == "arriba"

    def test_pinch_detection(self):
        hand = make_hand(fingers=2)
        hand[4] = Point(hand[8].x, hand[8].y)      # thumb tip onto index tip
        assert is_pinching(hand)

    def test_short_hand_is_unclassified(self):
        assert classify_hand([Point(0, 0)] * 5) is None

    def test_summary_shape(self):
        info = detect_all_hand_gestures([make_hand(fingers=5), make_hand(fingers=5)])
        assert info["both_hands_open"] is True
        assert len(info["finger_counts"]) == 2

    def test_empty_hand_list(self):
        info = detect_all_hand_gestures([])
        assert info["gestures"] == []
        assert info["calculator_input"] is None

    def test_tracker_requires_stability(self):
        tracker = HandGestureTracker(stable_seconds=0.3, cooldown_seconds=1.0)
        info = detect_all_hand_gestures([make_hand(fingers=5)])
        assert tracker.update(info, 0.0) == []
        assert tracker.update(info, 0.1) == []
        assert "open_palm" in tracker.update(info, 0.5)

    def test_tracker_enforces_cooldown(self):
        tracker = HandGestureTracker(stable_seconds=0.1, cooldown_seconds=5.0)
        info = detect_all_hand_gestures([make_hand(fingers=5)])
        tracker.update(info, 0.0)
        tracker.update(info, 0.2)
        empty = detect_all_hand_gestures([])
        tracker.update(empty, 0.5)
        assert tracker.update(info, 1.0) == []      # still inside the cooldown


class TestBodyGestures:
    def test_neutral_stance_fires_nothing(self):
        active = {k: v for k, v in detect_every_gesture(make_pose(), 0.53).items() if v}
        assert active == {}

    def test_raised_hand(self):
        pose = make_pose(LEFT_WRIST=Point(0.40, 0.05))
        assert detect_every_gesture(pose, 0.53)["one_hand_raised"] == "left"

    def test_pointing_outward_only(self):
        pointing = make_pose(RIGHT_WRIST=Point(0.85, 0.28))
        assert detect_every_gesture(pointing, 0.53)["pointing"] == "right"

        crossed = make_pose(
            LEFT_WRIST=Point(0.57, 0.40), RIGHT_WRIST=Point(0.43, 0.40),
        )
        gestures = detect_every_gesture(crossed, 0.53)
        assert gestures["pointing"] is None
        assert gestures["arms_crossed"] is True

    def test_jump_uses_baseline(self):
        jumping = make_pose(LEFT_HIP=Point(0.45, 0.44), RIGHT_HIP=Point(0.55, 0.44))
        assert detect_every_gesture(jumping, 0.53)["jumping"] is True
        assert detect_every_gesture(jumping, 0.44)["jumping"] is False

    def test_leaning(self):
        leaning = make_pose(
            LEFT_SHOULDER=Point(0.53, 0.26), RIGHT_SHOULDER=Point(0.69, 0.26),
        )
        assert detect_every_gesture(leaning, 0.53)["leaning"] == "right"

    def test_hands_on_hips_needs_flared_elbows(self):
        hips = make_pose(
            LEFT_WRIST=Point(0.44, 0.54), RIGHT_WRIST=Point(0.56, 0.54),
            LEFT_ELBOW=Point(0.32, 0.42), RIGHT_ELBOW=Point(0.68, 0.42),
        )
        assert detect_every_gesture(hips, 0.53)["hands_on_hips"] is True


# ---------------------------------------------------------------------------
# Actions and keys
# ---------------------------------------------------------------------------

class TestActions:
    def test_action_runs(self, state):
        registry = ActionRegistry()
        calls = []
        registry.register("test", "Test", lambda s, **_: calls.append(1), cooldown=0.0)
        assert registry.run("test", state, 0.0) is True
        assert calls == [1]

    def test_unknown_action_is_refused(self, state):
        registry = ActionRegistry()
        assert registry.run("nope", state, 0.0) is False

    def test_permission_blocks_action(self, state):
        registry = ActionRegistry()
        registry.register(
            "mouse", "Mouse", lambda s, **_: None,
            permission="allow_mouse_control", cooldown=0.0,
        )
        allowed, reason = registry.can_run("mouse", state, 0.0)
        assert allowed is False and "permiso" in reason

        state.config.gestures.allow_mouse_control = True
        assert registry.can_run("mouse", state, 0.0)[0] is True

    def test_cooldown_blocks_repeat(self, state):
        registry = ActionRegistry()
        registry.register("test", "Test", lambda s, **_: None, cooldown=5.0)
        assert registry.run("test", state, 0.0) is True
        assert registry.run("test", state, 1.0) is False
        assert registry.run("test", state, 6.0) is True

    def test_failing_handler_does_not_propagate(self, state):
        registry = ActionRegistry()

        def broken(_state, **_kwargs):
            raise RuntimeError("boom")

        registry.register("broken", "Broken", broken, cooldown=0.0)
        assert registry.run("broken", state, 0.0) is False

    def test_bindings_dispatch_active_gestures(self, state):
        registry = ActionRegistry()
        fired = []
        registry.register("change_color", "Color", lambda s, **_: fired.append(1), cooldown=0.0)
        bindings = GestureBindings(registry=registry)
        bindings.dispatch({"clap": True, "squat": False}, state, 0.0)
        assert fired == [1]

    def test_bindings_handle_side_specific_gestures(self, state):
        registry = ActionRegistry()
        fired = []
        registry.register("go_left", "Left", lambda s, **_: fired.append(1), cooldown=0.0)
        bindings = GestureBindings(registry=registry, bindings={"pointing_left": "go_left"})
        bindings.dispatch({"pointing": "left"}, state, 0.0)
        assert fired == [1]

    def test_disabled_bindings_do_nothing(self, state):
        registry = ActionRegistry()
        fired = []
        registry.register("change_color", "Color", lambda s, **_: fired.append(1), cooldown=0.0)
        bindings = GestureBindings(registry=registry, enabled=False)
        bindings.dispatch({"clap": True}, state, 0.0)
        assert fired == []

    def test_bindings_from_config_merge_overrides(self, state):
        registry = ActionRegistry()
        state.config.gestures.bindings = {"clap": "next_theme"}
        bindings = GestureBindings.from_config(registry, state.config.gestures)
        assert bindings.action_for("clap") == "next_theme"
        assert bindings.action_for("t_pose") == "open_calculator"   # default kept


class TestKeyHandler:
    def test_toggle_flips(self):
        handler = KeyHandler()
        handler.register_toggle(ord("g"), "grid", "Grid")
        assert handler.process_key(ord("g")) == ("grid", True)
        assert handler.process_key(ord("g")) == ("grid", False)

    def test_oneshot_always_reports_true(self):
        handler = KeyHandler()
        handler.register_oneshot(ord("s"), "shot", "Screenshot")
        assert handler.process_key(ord("s")) == ("shot", True)

    def test_unregistered_key_returns_none(self):
        assert KeyHandler().process_key(ord("z")) is None

    def test_conflicts_are_recorded_not_raised(self):
        handler = KeyHandler()
        handler.register_toggle(ord("g"), "grid", "Grid")
        handler.register_toggle(ord("g"), "ghost", "Ghost")
        assert handler.conflicts
        assert handler.process_key(ord("g"))[0] == "ghost"    # last wins

    def test_rebind(self):
        handler = KeyHandler()
        handler.register_toggle(ord("g"), "grid", "Grid")
        assert handler.rebind("grid", ord("j")) is True
        assert handler.process_key(ord("g")) is None
        assert handler.process_key(ord("j"))[0] == "grid"

    def test_sync_from_external_state(self):
        handler = KeyHandler()
        handler.register_toggle(ord("g"), "grid", "Grid", default_active=False)
        handler.sync_from({"grid": True})
        assert handler.is_active("grid") is True

    def test_special_key_labels(self):
        handler = KeyHandler()
        handler.register_oneshot(9, "picker", "Modes")
        assert handler.help_rows()[0][0] == "TAB"
