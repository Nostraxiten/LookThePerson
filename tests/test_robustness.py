"""
Robustness tests: every mode against hostile input.

The rest of the suite checks that modes compute the right thing from a
well-formed frame. This file checks the opposite — that nothing takes the frame
loop down when the input is wrong, because a crash in ``draw`` ends the session
and loses the recording.

Each scenario below stands for a failure seen in real footage:

* ``nan_landmarks`` — MediaPipe emits NaN for a landmark it cannot solve, and
  the One-Euro smoother then carries that NaN forward for the whole session.
* ``offscreen`` — a limb leaves the shot and the tracker extrapolates far
  outside 0..1, which OpenCV cannot accept as a coordinate.
* ``truncated_pose`` — a partial landmark list, which every fixed-index
  detector would index straight past the end of.
* ``collapsed`` — every landmark at one point, so every scale and ratio divides
  by zero.
* ``tiny`` / ``skinny`` — frame geometry that makes a kernel or a downscale
  degenerate.
* ``non_contiguous`` — a mode handed on a cropped or transposed view; OpenCV
  refuses to draw into one.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Config
from core.geometry import (
    PIXEL_LIMIT,
    Point,
    is_finite_point,
    safe_int,
    to_pixels,
    to_pixels_clamped,
)
from core.state import AppState, FrameContext
from core.theme import theme_names
from fx.filters import FILTERS, apply_filter
from gestures.body_gestures import detect_every_gesture
from modes import build_mode_manager, mode_keys
from tests.conftest import make_face, make_hand, make_pose

MODE_KEYS = mode_keys()


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeMask:
    def __init__(self, height: int = 180, width: int = 320):
        self._data = np.zeros((height, width), dtype=np.float32)
        self._data[height // 4:height * 3 // 4, width // 4:width * 3 // 4] = 1.0

    def numpy_view(self):
        return self._data


class FakePoseResult:
    def __init__(self, landmarks, masks=True, mask_size=(180, 320)):
        self.pose_landmarks = [landmarks]
        self.segmentation_masks = [FakeMask(*mask_size)] if masks else None


class FakeDetection:
    def __init__(self, x=120, y=60, w=90, h=90, name="person", score=0.91):
        self.bounding_box = type(
            "BB", (), {"origin_x": x, "origin_y": y, "width": w, "height": h},
        )()
        self.categories = [type("C", (), {"category_name": name, "score": score})()]


class FakeDetectionResult:
    def __init__(self, detections=None):
        self.detections = detections if detections is not None else [FakeDetection()]


def context(frame, width, height, index=0, pose=None, hands=None, faces=None,
            masks=True, mask_size=(180, 320), **extra) -> FrameContext:
    """Assemble a frame context the way the pipeline would."""
    fields = dict(
        frame=frame, width=width, height=height, timestamp_ms=index * 33,
        now=index * 0.033, delta=0.033, frame_index=index,
    )
    if pose is not None:
        fields.update(
            pose_result=FakePoseResult(pose, masks=masks, mask_size=mask_size),
            pose_landmarks=[pose],
            body_center=(width // 2, height // 2),
            body_gestures=detect_every_gesture(pose, 0.53),
            motion={"energy": 0.03, "left_hand_speed": 0.2, "right_hand_speed": 0.4},
        )
    if hands is not None:
        fields.update(hand_landmarks=hands, handedness=["Right", "Left"][:len(hands)])
    if faces is not None:
        fields.update(
            face_landmarks=faces,
            face_info={"raw": {}, "triggered": [], "gaze": (0.1, -0.05),
                       "head_pose": {"yaw": 4.0, "pitch": -2.0, "roll": 1.0}},
        )
    fields.update(extra)
    return FrameContext(**fields)


# ---------------------------------------------------------------------------
# Hostile scenarios
# ---------------------------------------------------------------------------

def _full(index):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return context(frame, 640, 480, index, pose=make_pose(175.0 if index % 2 else 85.0),
                   hands=[make_hand()], faces=[make_face()],
                   face_detect_result=FakeDetectionResult(),
                   object_result=FakeDetectionResult())


def _empty(index):
    return context(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480, index)


def _nan_landmarks(index):
    pose = list(make_pose())
    pose[11] = Point(float("nan"), float("nan"))
    pose[12] = Point(float("inf"), float("-inf"))
    hand = list(make_hand())
    hand[8] = Point(float("nan"), 0.5)
    face = list(make_face())
    face[13] = Point(float("nan"), float("nan"))
    return context(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480, index,
                   pose=pose, hands=[hand], faces=[face])


def _offscreen(index):
    pose = list(make_pose())
    pose[15] = Point(-40.0, -30.0)
    pose[16] = Point(55.0, 45.0)
    pose[0] = Point(-12.0, 19.0)
    return context(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480, index,
                   pose=pose, hands=[make_hand(x=-20.0, y=30.0)], faces=[make_face()],
                   face_detect_result=FakeDetectionResult(),
                   object_result=FakeDetectionResult())


def _truncated_pose(index):
    return context(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480, index,
                   pose=make_pose()[:25])


def _collapsed(index):
    return context(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480, index,
                   pose=[Point(0.5, 0.5)] * 33, hands=[[Point(0.5, 0.5)] * 21],
                   faces=[[Point(0.5, 0.5)] * 478],
                   face_detect_result=FakeDetectionResult(),
                   object_result=FakeDetectionResult())


def _tiny(index):
    return context(np.zeros((2, 2, 3), dtype=np.uint8), 2, 2, index,
                   pose=make_pose(), hands=[make_hand()], faces=[make_face()],
                   mask_size=(2, 2), face_detect_result=FakeDetectionResult(),
                   object_result=FakeDetectionResult())


def _skinny(index):
    return context(np.zeros((480, 3, 3), dtype=np.uint8), 3, 480, index,
                   pose=make_pose(), hands=[make_hand()], mask_size=(480, 3))


def _no_mask(index):
    return context(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480, index,
                   pose=make_pose(), masks=False, hands=[make_hand()],
                   faces=[make_face()])


def _mask_mismatch(index):
    """Striding can leave a mask from a frame of a different size."""
    return context(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480, index,
                   pose=make_pose(), mask_size=(97, 131))


def _bad_boxes(index):
    boxes = [FakeDetection(-50, -40, 200, 200), FakeDetection(600, 460, 300, 300),
             FakeDetection(10, 10, 0, 0), FakeDetection(10, 10, -30, -30)]
    return context(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480, index,
                   pose=make_pose(), faces=[make_face()],
                   face_detect_result=FakeDetectionResult(boxes),
                   object_result=FakeDetectionResult(boxes))


def _crowd(index):
    poses = [make_pose() for _ in range(5)]
    ctx = context(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480, index,
                  pose=poses[0], hands=[make_hand(), make_hand()],
                  faces=[make_face(), make_face()],
                  face_detect_result=FakeDetectionResult(
                      [FakeDetection(i * 40, i * 30) for i in range(6)]),
                  object_result=FakeDetectionResult(
                      [FakeDetection(i * 40, i * 30, name="chair") for i in range(8)]))
    ctx.pose_landmarks = poses
    ctx.pose_result.pose_landmarks = poses
    ctx.pose_result.segmentation_masks = [FakeMask() for _ in poses]
    return ctx


def _non_contiguous(index):
    """A cropped/strided view, as a mode that zooms would produce."""
    backing = np.zeros((960, 1280, 3), dtype=np.uint8)
    view = backing[::2, ::2]
    return context(view, view.shape[1], view.shape[0], index,
                   pose=make_pose(), hands=[make_hand()], faces=[make_face()])


def _zero_delta(index):
    ctx = _full(index)
    ctx.delta = 0.0
    ctx.now = 0.0
    return ctx


SCENARIOS = {
    "full": _full,
    "empty": _empty,
    "nan_landmarks": _nan_landmarks,
    "offscreen": _offscreen,
    "truncated_pose": _truncated_pose,
    "collapsed": _collapsed,
    "tiny": _tiny,
    "skinny": _skinny,
    "no_mask": _no_mask,
    "mask_mismatch": _mask_mismatch,
    "bad_boxes": _bad_boxes,
    "crowd": _crowd,
    "non_contiguous": _non_contiguous,
    "zero_delta": _zero_delta,
}


def run_frames(manager, factory, frames: int = 3) -> None:
    """Drive a mode through the same stages, and guards, as the frame loop."""
    for index in range(frames):
        ctx = factory(index)
        manager.process(ctx)
        # The pipeline normalises the frame between stages; mirroring that here
        # is what keeps this a test of the modes rather than of numpy views.
        ctx.ensure_drawable()
        manager.draw(ctx)
        lines = manager.hud_lines(ctx)
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)
        manager.status_text(ctx)


@pytest.fixture
def state() -> AppState:
    return AppState(Config())


# ---------------------------------------------------------------------------
# Safe pixel conversion
# ---------------------------------------------------------------------------

class TestSafeConversion:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_safe_int_absorbs_non_finite_values(self, value):
        assert isinstance(safe_int(value), int)

    def test_safe_int_uses_the_fallback_for_nan(self):
        assert safe_int(float("nan"), fallback=7) == 7

    def test_safe_int_clamps_instead_of_overflowing(self):
        assert safe_int(1e30) == PIXEL_LIMIT
        assert safe_int(-1e30) == -PIXEL_LIMIT
        assert safe_int(float("inf")) == PIXEL_LIMIT

    def test_safe_int_passes_ordinary_numbers_through(self):
        assert safe_int(12.7) == 12
        assert safe_int(-3.2) == -3

    def test_safe_int_survives_junk(self):
        assert safe_int("no soy un numero", fallback=5) == 5
        assert safe_int(None, fallback=5) == 5

    def test_to_pixels_never_raises_on_nan(self):
        x, y = to_pixels(Point(float("nan"), float("inf")), 640, 480)
        assert isinstance(x, int) and isinstance(y, int)

    def test_to_pixels_is_exact_for_normal_input(self):
        assert to_pixels(Point(0.5, 0.25), 640, 480) == (320, 120)

    def test_to_pixels_keeps_offscreen_points_offscreen(self):
        """Drawing must not pin a limb to the edge; OpenCV clips for free."""
        x, _y = to_pixels(Point(-2.0, 0.5), 640, 480)
        assert x < 0

    def test_to_pixels_clamped_stays_inside_the_frame(self):
        assert to_pixels_clamped(Point(-2.0, 5.0), 640, 480) == (0, 479)
        assert to_pixels_clamped(Point(0.5, 0.5), 640, 480) == (320, 240)

    def test_to_pixels_clamped_handles_a_degenerate_frame(self):
        assert to_pixels_clamped(Point(0.5, 0.5), 0, 0) == (0, 0)

    @pytest.mark.parametrize("point,expected", [
        (Point(0.5, 0.5), True),
        (Point(float("nan"), 0.5), False),
        (Point(0.5, float("inf")), False),
        (Point(-5.0, 9.0), True),          # off-screen is still usable
    ])
    def test_is_finite_point(self, point, expected):
        assert is_finite_point(point) is expected

    def test_is_finite_point_rejects_junk(self):
        assert is_finite_point(None) is False
        assert is_finite_point("hola") is False


class TestFrameContextHelpers:
    def test_px_absorbs_non_finite_landmarks(self):
        ctx = _empty(0)
        assert ctx.px(Point(float("nan"), float("nan"))) == (0, 0)

    def test_landmark_points_reports_broken_landmarks_as_invisible(self):
        ctx = _empty(0)
        pose = list(make_pose())
        pose[5] = Point(float("nan"), float("nan"))
        points = ctx.landmark_points(pose)
        assert len(points) == len(pose)      # indices stay aligned
        assert points[5][2] is False

    def test_landmark_points_accepts_an_empty_list(self):
        assert _empty(0).landmark_points([]) == []

    def test_ensure_drawable_makes_a_view_contiguous(self):
        ctx = _non_contiguous(0)
        assert not ctx.frame.flags.c_contiguous
        ctx.ensure_drawable()
        assert ctx.frame.flags.c_contiguous
        assert ctx.frame.flags.writeable

    def test_ensure_drawable_leaves_a_normal_frame_alone(self):
        ctx = _empty(0)
        before = ctx.frame
        ctx.ensure_drawable()
        assert ctx.frame is before

    def test_ensure_drawable_tolerates_a_missing_frame(self):
        ctx = _empty(0)
        ctx.frame = None
        assert ctx.ensure_drawable() is None


# ---------------------------------------------------------------------------
# Gestures
# ---------------------------------------------------------------------------

class TestGestureRobustness:
    @pytest.mark.parametrize("landmarks", [
        None, [], make_pose()[:10], make_pose()[:32], [Point(0.5, 0.5)] * 5,
    ])
    def test_a_short_landmark_list_yields_an_inactive_result(self, landmarks):
        gestures = detect_every_gesture(landmarks, 0.53)
        assert isinstance(gestures, dict)
        assert not any(gestures.values())

    def test_non_finite_landmarks_do_not_raise(self):
        pose = list(make_pose())
        pose[0] = Point(float("nan"), float("nan"))
        pose[15] = Point(float("inf"), float("-inf"))
        assert isinstance(detect_every_gesture(pose, 0.53), dict)

    def test_a_collapsed_pose_does_not_raise(self):
        assert isinstance(detect_every_gesture([Point(0.5, 0.5)] * 33, 0.53), dict)

    def test_the_inactive_result_has_the_same_keys_as_a_real_one(self):
        """A caller reading gestures['squat'] must not hit a KeyError."""
        real = detect_every_gesture(make_pose(), 0.53)
        empty = detect_every_gesture(None, 0.53)
        assert set(empty) == set(real)


# ---------------------------------------------------------------------------
# Every mode against every scenario
# ---------------------------------------------------------------------------

class TestModesSurviveHostileFrames:
    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    @pytest.mark.parametrize("mode_key", MODE_KEYS)
    def test_mode_survives(self, mode_key, scenario, state):
        manager = build_mode_manager(state, mode_key)
        run_frames(manager, SCENARIOS[scenario])


class TestModesSurviveInput:
    @pytest.mark.parametrize("mode_key", MODE_KEYS)
    def test_every_printable_key_is_safe(self, mode_key, state):
        """A mode must never raise on a key, whether or not it wants it."""
        manager = build_mode_manager(state, mode_key)
        for code in list(range(32, 127)) + [9, 13, 27, 0, 255, -1]:
            manager.handle_key(code)
        # Keys may have changed internal state; the mode must still render.
        run_frames(manager, SCENARIOS["full"], frames=2)

    @pytest.mark.parametrize("mode_key", MODE_KEYS)
    def test_reset_is_safe_at_any_time(self, mode_key, state):
        manager = build_mode_manager(state, mode_key)
        mode = manager.current
        mode.reset(state)                       # before any frame
        run_frames(manager, SCENARIOS["full"], frames=2)
        mode.reset(state)                       # after frames
        run_frames(manager, SCENARIOS["empty"], frames=2)

    def test_switching_between_every_pair_of_modes_is_safe(self, state):
        """Mode state must not leak into whichever mode is entered next."""
        manager = build_mode_manager(state, "full")
        for index, key in enumerate(MODE_KEYS):
            manager.switch(key)
            run_frames(manager, SCENARIOS["full" if index % 2 else "empty"], frames=1)

    @pytest.mark.parametrize("theme", theme_names())
    def test_every_theme_renders(self, theme, state):
        state.set_theme(theme)
        for key in MODE_KEYS:
            manager = build_mode_manager(state, key)
            run_frames(manager, SCENARIOS["full"], frames=1)

    def test_an_unknown_theme_falls_back_rather_than_raising(self, state):
        state.set_theme("no-existe")
        manager = build_mode_manager(state, "full")
        run_frames(manager, SCENARIOS["full"], frames=1)


class TestToggleCombinations:
    TOGGLES = ("skeleton", "segmentation", "face_mesh", "face_detect",
               "object_detect", "bounding_boxes", "grid", "night_mode",
               "telemetry", "landmark_ids", "trails", "heatmap", "debug")

    @pytest.mark.parametrize("value", [True, False])
    def test_all_toggles_on_or_off_together(self, value, state):
        """The two extremes catch what a random sample usually misses."""
        for key in MODE_KEYS:
            manager = build_mode_manager(state, key)
            state.apply_toggles({name: value for name in self.TOGGLES})
            run_frames(manager, SCENARIOS["full"], frames=1)

    @pytest.mark.parametrize("toggle", TOGGLES)
    def test_each_toggle_alone_against_a_broken_frame(self, toggle, state):
        for key in ("full", "pose", "security", "privacy", "debug", "matrix"):
            manager = build_mode_manager(state, key)
            state.apply_toggles({name: (name == toggle) for name in self.TOGGLES})
            run_frames(manager, SCENARIOS["nan_landmarks"], frames=1)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFilterRobustness:
    SIZES = [(1, 1), (2, 2), (3, 3), (1, 640), (640, 1), (17, 31), (61, 81)]

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("name", sorted(FILTERS))
    def test_filter_preserves_shape_at_any_size(self, name, size):
        height, width = size
        frame = np.full((height, width, 3), 90, dtype=np.uint8)
        out = apply_filter(frame, name)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("name", sorted(FILTERS))
    def test_filter_handles_flat_extremes(self, name):
        for level in (0, 255):
            frame = np.full((32, 48, 3), level, dtype=np.uint8)
            assert apply_filter(frame, name).shape == frame.shape

    def test_an_unknown_filter_returns_the_frame_untouched(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        assert apply_filter(frame, "no-existe") is frame

    def test_a_small_frame_does_not_degrade_the_cartoon_filter(self, capsys):
        """apply_filter rescues a broken filter by printing — nothing should."""
        for size in [(2, 2), (3, 5), (8, 8)]:
            apply_filter(np.full((*size, 3), 120, dtype=np.uint8), "cartoon")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

class TestHudRobustness:
    """The HUD draws over whatever the mode produced, so it sees the same mess."""

    def _hud(self):
        from ui.hud import HUD
        return HUD()

    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    def test_hud_draws_over_every_scenario(self, scenario, state):
        hud = self._hud()
        manager = build_mode_manager(state, "full")
        ctx = SCENARIOS[scenario](0)
        manager.process(ctx)
        ctx.ensure_drawable()
        manager.draw(ctx)
        hud.draw(ctx, state, mode_lines=manager.hud_lines(ctx))

    def test_hud_draws_every_panel_at_once(self, state):
        hud = self._hud()
        hud.show_picker = True
        state.apply_toggles({"telemetry": True, "fps_graph": True,
                             "debug": True, "help": True})
        manager = build_mode_manager(state, "full")
        ctx = SCENARIOS["full"](0)
        for _ in range(3):
            state.fps.tick(_ * 0.033)
        hud.draw(
            ctx, state,
            mode_lines=manager.hud_lines(ctx),
            key_bindings=[("H", "Ayuda", True), ("Q", "Salir", None)],
            mode_keys=[("a", "Armar"), ("space", "Iniciar")],
        )

    def test_help_panel_lists_the_active_modes_own_keys(self, state):
        """A mode key invisible in the app is a key the operator never finds."""
        hud = self._hud()
        state.set_toggle("help", True)
        manager = build_mode_manager(state, "security")
        rows = manager.key_help()
        assert rows, "el modo security deberia exponer sus teclas"
        assert all(len(row) == 2 for row in rows)

        ctx = SCENARIOS["full"](0)
        hud.draw(ctx, state, key_bindings=[("Q", "Salir", None)], mode_keys=rows)

    def test_key_help_is_empty_without_a_mode(self, state):
        from modes.base import ModeManager
        assert ModeManager(state).key_help() == []
        assert ModeManager(state).owns_overlay is False

    def test_overlay_owner_suppresses_the_minimal_hud(self, state):
        """
        With telemetry off the HUD normally still paints a status line and FPS
        in the corners the security OSD uses. It must stay out of the way.
        """
        state.set_toggle("telemetry", False)
        state.status_text = "TEXTO QUE NO DEBE APARECER"

        plain = SCENARIOS["empty"](0)
        self._hud().draw(plain, state, overlay_owner=False)

        owned = SCENARIOS["empty"](0)
        self._hud().draw(owned, state, overlay_owner=True)

        assert plain.frame.any(), "el HUD minimo deberia haber dibujado algo"
        assert not owned.frame.any(), "un modo con overlay propio no debe recibir chrome"

    def test_security_mode_declares_it_owns_the_overlay(self, state):
        manager = build_mode_manager(state, "security")
        assert manager.owns_overlay is True

    def test_ordinary_modes_do_not_own_the_overlay(self, state):
        for key in ("full", "pose", "reps", "matrix"):
            assert build_mode_manager(state, key).owns_overlay is False


# ---------------------------------------------------------------------------
# Session logging
# ---------------------------------------------------------------------------

class TestSessionLoggingSurvivesEventPayloads:
    """
    An event payload is arbitrary, and the bus disables a handler that raises.

    ``EventBus.emit`` takes its name positional-only precisely so a payload may
    carry ``name`` — modes do exactly that (``emit(ACTION_TRIGGERED,
    name="screenshot")``). The recorder has to be just as tolerant, or the
    first photo booth shot silently ends session logging for the whole run.
    """

    def _recorder(self):
        from analytics.session import SessionRecorder
        return SessionRecorder(enabled=True)

    def test_a_payload_carrying_name_is_recorded(self):
        recorder = self._recorder()
        recorder.record_event("action.triggered", name="screenshot")
        assert len(recorder) == 1

    @pytest.mark.parametrize("key", ["name", "kind", "label", "value", "timestamp"])
    def test_a_payload_may_reuse_any_reserved_key(self, key):
        recorder = self._recorder()
        recorder.record_event("evento", **{key: "carga"})
        assert len(recorder) == 1

    def test_reserved_payload_keys_do_not_overwrite_the_record(self):
        recorder = self._recorder()
        recorder.record_event("el_evento_real", label="impostor", value=999)
        payload = recorder.records[0].to_dict()
        assert payload["label"] == "el_evento_real"
        assert payload["value"] == 1.0
        # The payload's own values are kept, just renamed out of the way.
        assert payload["data_label"] == "impostor"
        assert payload["data_value"] == 999

    def test_the_bus_keeps_the_handler_alive_through_action_events(self):
        """The end-to-end shape of the bug: emit, and logging must continue."""
        from analytics.session import SessionRecorder
        from core.events import EventBus, Events

        bus = EventBus()
        recorder = SessionRecorder(enabled=True)

        def log_event(event):
            recorder.record_event(event.name, **{
                k: v for k, v in event.payload.items()
                if isinstance(v, (str, int, float, bool))
            })

        bus.subscribe(Events.ACTION_TRIGGERED, log_event)
        for _ in range(3):
            bus.emit(Events.ACTION_TRIGGERED, name="screenshot")

        assert len(recorder) == 3, "el handler fue desactivado tras el primer evento"

    def test_export_survives_reserved_payload_keys(self, tmp_path):
        recorder = self._recorder()
        recorder.output_dir = str(tmp_path)
        recorder.record_event("action.triggered", name="screenshot")
        recorder.record_gesture("clap")
        for fmt in ("json", "csv", "jsonl"):
            assert recorder.export(fmt) is not None
