"""
Tests for the surveillance mode and the two subsystems it is built on.

Three things are worth asserting about a security camera, and none of them are
"it does not crash":

* an identity survives the subject walking out of frame and back in,
* a subject *outside* the armed zone does not raise an alarm,
* and the night-vision metering does not flap between day and night at dusk.

Everything here runs on synthetic skeletons and flat frames — no camera, no
model download.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from analytics.identity import (
    MIN_SHARED_RATIOS,
    BodySignature,
    Detection,
    PersonTracker,
    build_signature,
    describe_build,
    pose_box,
    stature_span,
)
from core.config import Config
from core.geometry import Point
from core.state import AppState, FrameContext
from fx.nightvision import (
    NIGHT_MODES,
    NightVisionProcessor,
    next_night_mode,
    scene_luminance,
)
from modes import build_mode_manager
from modes.security import ZONES, SecurityEvent, SecurityMode, Zone
from tests.conftest import make_face, make_pose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def person(build: float = 1.0, limb: float = 1.0, x: float = 0.5, y: float = 0.5):
    """
    A skeleton whose proportions are driven by *build* and *limb*.

    Two calls with different values must produce signatures the tracker can
    tell apart; two calls with the same values but different *x* / *y* must
    produce the same signature, since it is meant to be position-invariant.
    """
    points = list(make_pose(178.0))
    half = 0.08 * build

    points[11] = Point(0.5 - half, 0.26)                      # shoulders
    points[12] = Point(0.5 + half, 0.26)
    points[13] = Point(0.5 - half - 0.02, 0.26 + 0.14 * limb)  # elbows
    points[14] = Point(0.5 + half + 0.02, 0.26 + 0.14 * limb)
    points[15] = Point(0.5 - half - 0.03, 0.26 + 0.28 * limb)  # wrists
    points[16] = Point(0.5 + half + 0.03, 0.26 + 0.28 * limb)
    points[25] = Point(0.45, 0.53 + 0.21 * limb)               # knees
    points[26] = Point(0.55, 0.53 + 0.21 * limb)
    points[27] = Point(0.45, 0.53 + 0.40 * limb)               # ankles
    points[28] = Point(0.55, 0.53 + 0.40 * limb)

    dx, dy = x - 0.5, y - 0.5
    return [Point(p.x + dx, p.y + dy, p.z, p.visibility) for p in points]


def detection_for(landmarks, has_face: bool = True) -> Detection:
    return Detection(
        box=pose_box(landmarks),
        signature=build_signature(landmarks),
        span=stature_span(landmarks),
        has_face=has_face,
    )


def security_context(poses, index: int = 0, width: int = 640, height: int = 480,
                     luminance: int = 140) -> FrameContext:
    """A frame context as the pipeline would hand to the security mode."""
    frame = np.full((height, width, 3), luminance, dtype=np.uint8)
    frame[height // 3:height * 2 // 3, width // 3:width * 2 // 3] = min(255, luminance + 60)
    return FrameContext(
        frame=frame, width=width, height=height,
        timestamp_ms=index * 50, now=index * 0.05, delta=0.05, frame_index=index,
        pose_landmarks=list(poses),
        face_landmarks=[make_face()] if poses else [],
    )


def drive(mode: SecurityMode, state: AppState, poses, frames: int, start: int = 0):
    """Run *frames* frames through process/draw/hud, returning the last one."""
    ctx = None
    for offset in range(frames):
        ctx = security_context(poses, start + offset)
        mode.process(ctx, state)
        ctx.ensure_drawable()
        mode.draw(ctx, state)
        mode.hud_lines(ctx, state)
        mode.status_text(ctx, state)
    return ctx


@pytest.fixture
def state() -> AppState:
    return AppState(Config())


@pytest.fixture
def mode(state) -> SecurityMode:
    return build_mode_manager(state, "security").current


# ---------------------------------------------------------------------------
# Body signatures
# ---------------------------------------------------------------------------

class TestBodySignature:
    def test_signature_is_built_from_a_full_pose(self):
        signature = build_signature(person())
        assert signature is not None
        assert signature.is_valid
        assert len(signature.ratios) >= MIN_SHARED_RATIOS

    def test_signature_ignores_position_in_frame(self):
        here = build_signature(person(x=0.2, y=0.4))
        there = build_signature(person(x=0.8, y=0.6))
        assert here.distance(there) == pytest.approx(0.0, abs=1e-6)

    def test_signature_separates_different_builds(self):
        slim = build_signature(person(build=1.0, limb=1.0))
        broad = build_signature(person(build=1.9, limb=1.35))
        same = build_signature(person(build=1.0, limb=1.0, x=0.3))
        assert slim.distance(same) < slim.distance(broad)

    def test_similarity_is_one_for_an_identical_signature(self):
        signature = build_signature(person())
        assert signature.similarity(signature) == pytest.approx(1.0)

    def test_incomparable_signatures_report_none_not_a_distance(self):
        """None means "no opinion" and must not read as "different person"."""
        full = build_signature(person())
        sparse = BodySignature({"shoulder_width": 1.0})
        assert full.distance(sparse) is None
        assert full.similarity(sparse) is None

    def test_blend_moves_toward_the_new_observation(self):
        a = BodySignature({"shoulder_width": 1.0, "hip_width": 1.0,
                           "left_thigh": 1.0, "left_shin": 1.0})
        b = BodySignature({"shoulder_width": 2.0, "hip_width": 2.0,
                           "left_thigh": 2.0, "left_shin": 2.0})
        blended = a.blend(b, 0.5)
        assert blended.ratios["shoulder_width"] == pytest.approx(1.5)

    def test_blend_adopts_ratios_it_did_not_have(self):
        a = BodySignature({"shoulder_width": 1.0})
        b = BodySignature({"hip_width": 0.8})
        assert a.blend(b, 0.2).ratios["hip_width"] == pytest.approx(0.8)

    def test_code_is_stable_and_short(self):
        signature = build_signature(person())
        assert signature.as_code() == signature.as_code()
        assert len(signature.as_code()) == 4

    def test_empty_signature_has_a_placeholder_code(self):
        assert BodySignature({}).as_code() == "----"

    @pytest.mark.parametrize("landmarks", [
        None, [], [Point(0.5, 0.5)] * 10, make_pose()[:20],
        [Point(float("nan"), 0.5)] * 33, [Point(float("inf"), 0.5)] * 33,
    ])
    def test_degenerate_poses_return_none_rather_than_raising(self, landmarks):
        assert build_signature(landmarks) is None

    def test_collapsed_pose_has_no_signature(self):
        """Every landmark at one point means a zero torso — nothing to divide by."""
        assert build_signature([Point(0.5, 0.5)] * 33) is None

    def test_describe_build_handles_missing_data(self):
        assert describe_build(None) == "desconocida"
        assert describe_build(BodySignature({})) == "desconocida"

    def test_describe_build_widens_with_shoulders(self):
        broad = BodySignature({"shoulder_width": 1.5, "hip_width": 1.0})
        narrow = BodySignature({"shoulder_width": 1.0, "hip_width": 1.0})
        assert describe_build(broad) == "ancha"
        assert describe_build(narrow) == "estrecha"


class TestPoseGeometry:
    def test_pose_box_bounds_the_body(self):
        box = pose_box(person(x=0.5, y=0.5))
        assert box is not None
        x0, y0, x1, y1 = box
        assert 0.0 <= x0 < x1 <= 1.0
        assert y0 < y1

    def test_pose_box_follows_the_subject(self):
        left = pose_box(person(x=0.2))
        right = pose_box(person(x=0.8))
        assert left[0] < right[0]

    def test_pose_box_skips_non_finite_landmarks(self):
        landmarks = list(person())
        landmarks[0] = Point(float("nan"), float("nan"))
        box = pose_box(landmarks)
        assert box is not None
        assert all(np.isfinite(v) for v in box)

    @pytest.mark.parametrize("landmarks", [None, [], [Point(0.5, 0.5)],
                                           [Point(float("nan"), 0.5)] * 33])
    def test_pose_box_and_span_tolerate_junk(self, landmarks):
        assert pose_box(landmarks) is None or isinstance(pose_box(landmarks), tuple)
        assert stature_span(landmarks) is None

    def test_stature_span_grows_with_limb_length(self):
        short = stature_span(person(limb=0.9))
        tall = stature_span(person(limb=1.4))
        assert short is not None and tall is not None
        assert tall > short


# ---------------------------------------------------------------------------
# Tracking and identification
# ---------------------------------------------------------------------------

class TestPersonTracker:
    def test_a_single_subject_keeps_one_identity(self):
        tracker = PersonTracker()
        now = 0.0
        for step in range(30):
            now += 0.05
            tracks = tracker.update(
                [detection_for(person(x=0.3 + step * 0.005))], now,
            )
        assert len(tracks) == 1
        assert tracks[0].pid == 1
        assert tracks[0].label == "SUJ-01"
        assert tracks[0].frames == 30

    def test_subject_is_dropped_after_the_forget_window(self):
        tracker = PersonTracker(forget_seconds=0.4)
        now = 0.0
        for _ in range(10):
            now += 0.05
            tracker.update([detection_for(person())], now)
        for _ in range(20):
            now += 0.05
            tracker.update([], now)
        assert tracker.active_count == 0
        assert tracker.gallery_size == 1

    def test_returning_subject_reclaims_its_identity(self):
        """The whole point of the signature: leave, come back, stay SUJ-01."""
        tracker = PersonTracker(forget_seconds=0.4, gallery_seconds=60.0)
        now = 0.0
        for _ in range(30):
            now += 0.05
            tracker.update([detection_for(person(x=0.3))], now)
        original = tracker.tracks[0].pid

        for _ in range(20):
            now += 0.05
            tracker.update([], now)

        for _ in range(10):
            now += 0.05
            tracks = tracker.update([detection_for(person(x=0.75))], now)

        assert tracks[0].pid == original
        assert tracks[0].reappearances == 1
        # No second identity was minted for the same person.
        assert tracker.total_identified == 1

    def test_a_different_person_gets_a_new_identity(self):
        tracker = PersonTracker(forget_seconds=0.3, gallery_seconds=60.0)
        now = 0.0
        for _ in range(25):
            now += 0.05
            tracker.update([detection_for(person(build=1.0, limb=1.0, x=0.3))], now)
        original = tracker.tracks[0].pid

        for _ in range(15):
            now += 0.05
            tracker.update([], now)

        for _ in range(12):
            now += 0.05
            tracks = tracker.update(
                [detection_for(person(build=2.1, limb=1.45, x=0.7))], now,
            )
        assert tracks[0].pid != original
        assert tracker.total_identified == 2

    def test_two_subjects_keep_separate_identities(self):
        tracker = PersonTracker()
        now = 0.0
        for step in range(30):
            now += 0.05
            tracks = tracker.update([
                detection_for(person(build=1.0, x=0.25 + 0.002 * step)),
                detection_for(person(build=2.0, limb=1.4, x=0.75 - 0.002 * step)),
            ], now)

        assert len(tracks) == 2
        assert len({t.pid for t in tracks}) == 2
        left = min(tracks, key=lambda t: t.centroid[0])
        right = max(tracks, key=lambda t: t.centroid[0])
        assert left.pid != right.pid

    def test_max_tracks_is_respected(self):
        tracker = PersonTracker(max_tracks=3)
        now = 0.0
        for _ in range(10):
            now += 0.05
            tracker.update(
                [detection_for(person(build=1.0 + i * 0.2, x=0.05 + i * 0.09))
                 for i in range(8)],
                now,
            )
        assert tracker.active_count <= 3

    def test_detection_without_a_signature_still_tracks(self):
        """Half a body in frame yields no signature; spatial continuity must do."""
        tracker = PersonTracker()
        now = 0.0
        for _ in range(15):
            now += 0.05
            tracks = tracker.update(
                [Detection(box=pose_box(person(x=0.4)), signature=None)], now,
            )
        assert len(tracks) == 1

    def test_distant_detections_are_never_matched(self):
        tracker = PersonTracker()
        tracker.update([detection_for(person(x=0.1, y=0.3))], 0.0)
        tracker.update([detection_for(person(x=0.9, y=0.8))], 0.05)
        # Two identities, because nothing links a body at one corner to the other.
        assert tracker.total_identified == 2

    def test_empty_update_is_harmless(self):
        tracker = PersonTracker()
        assert tracker.update([], 0.0) == []
        assert tracker.active_count == 0

    def test_reset_clears_identities_and_numbering(self):
        tracker = PersonTracker()
        tracker.update([detection_for(person())], 0.0)
        tracker.reset()
        assert tracker.active_count == 0
        assert tracker.total_identified == 0
        tracker.update([detection_for(person())], 1.0)
        assert tracker.tracks[0].pid == 1

    def test_confidence_grows_with_track_age(self):
        tracker = PersonTracker()
        now = 0.0
        tracker.update([detection_for(person())], now)
        young = tracker.tracks[0].confidence()
        for _ in range(40):
            now += 0.05
            tracker.update([detection_for(person())], now)
        assert tracker.tracks[0].confidence() > young

    def test_height_estimate_scales_with_the_reference(self):
        tracker = PersonTracker()
        tracker.update([detection_for(person())], 0.0)
        subject = tracker.tracks[0]
        assert subject.estimated_height_cm(180.0) > subject.estimated_height_cm(150.0)

    def test_height_estimate_is_none_without_a_span(self):
        tracker = PersonTracker()
        tracker.update([Detection(box=(0.4, 0.2, 0.6, 0.8), span=None)], 0.0)
        assert tracker.tracks[0].estimated_height_cm() is None

    def test_an_implausible_height_is_reported_as_unknown(self):
        """Bad framing must read as "no lo se", not as a confident wrong number."""
        tracker = PersonTracker()
        tracker.update([Detection(box=(0.1, 0.0, 0.9, 1.0), span=0.98)], 0.0)
        subject = tracker.tracks[0]
        assert subject.estimated_height_cm(230.0) is None       # too tall
        assert subject.estimated_height_cm(60.0) is None        # too short

    def test_a_plausible_height_is_reported(self):
        tracker = PersonTracker()
        tracker.update([Detection(box=(0.4, 0.1, 0.6, 0.9), span=0.82)], 0.0)
        estimate = tracker.tracks[0].estimated_height_cm(172.0)
        assert estimate == pytest.approx(172.0, abs=0.5)


# ---------------------------------------------------------------------------
# Night vision
# ---------------------------------------------------------------------------

class TestNightVision:
    SIZES = [(1, 1), (2, 2), (3, 480), (480, 3), (17, 31), (61, 81)]

    @pytest.mark.parametrize("night_mode", NIGHT_MODES)
    @pytest.mark.parametrize("luminance", [0, 8, 128, 255])
    def test_every_mode_preserves_shape_and_dtype(self, night_mode, luminance):
        frame = np.full((48, 64, 3), luminance, dtype=np.uint8)
        processor = NightVisionProcessor(mode=night_mode)
        out = processor.process(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("night_mode", NIGHT_MODES)
    def test_every_mode_survives_odd_frame_sizes(self, night_mode, size):
        height, width = size
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        out = NightVisionProcessor(mode=night_mode).process(frame)
        assert out.shape == frame.shape

    def test_processing_does_not_modify_the_input(self):
        frame = np.full((32, 32, 3), 40, dtype=np.uint8)
        original = frame.copy()
        NightVisionProcessor(mode="ir").process(frame)
        assert np.array_equal(frame, original)

    def test_auto_picks_daylight_for_a_bright_scene(self):
        processor = NightVisionProcessor(mode="auto")
        for _ in range(30):
            processor.process(np.full((32, 32, 3), 225, dtype=np.uint8))
        assert processor.effective_mode == "dia"
        assert processor.is_night is False

    def test_auto_picks_infrared_for_a_dark_scene(self):
        processor = NightVisionProcessor(mode="auto")
        for _ in range(50):
            processor.process(np.full((32, 32, 3), 6, dtype=np.uint8))
        assert processor.effective_mode == "ir"
        assert processor.is_night is True

    def test_metering_does_not_flap_inside_the_hysteresis_band(self):
        """A camera on a doorway at dusk must not oscillate day/night."""
        processor = NightVisionProcessor(mode="auto")
        for _ in range(30):
            processor.process(np.full((32, 32, 3), 205, dtype=np.uint8))

        previous = processor.effective_mode
        flips = 0
        for step in range(60):
            level = 74 + (6 if step % 2 else -6)     # jitter within 62..88
            processor.process(np.full((32, 32, 3), level, dtype=np.uint8))
            if processor.effective_mode != previous:
                flips += 1
                previous = processor.effective_mode
        assert flips <= 1

    def test_infrared_actually_brightens_a_dark_scene(self):
        dark = np.full((60, 80, 3), 14, dtype=np.uint8)
        dark[20:40, 25:55] = 46
        processor = NightVisionProcessor(mode="ir")
        for _ in range(15):
            lifted = processor.process(dark)
        assert scene_luminance(lifted) > scene_luminance(dark)

    def test_manual_gain_disables_auto_gain(self):
        processor = NightVisionProcessor()
        assert processor.auto_gain is True
        processor.adjust_gain(0.3)
        assert processor.auto_gain is False

    def test_gain_stays_within_bounds(self):
        processor = NightVisionProcessor()
        for _ in range(50):
            processor.adjust_gain(1.0)
        assert processor.gain <= 4.0
        for _ in range(100):
            processor.adjust_gain(-1.0)
        assert processor.gain >= 0.25

    def test_cycle_visits_every_mode_and_wraps(self):
        processor = NightVisionProcessor(mode=NIGHT_MODES[0])
        seen = {processor.mode}
        for _ in range(len(NIGHT_MODES)):
            seen.add(processor.cycle())
        assert seen == set(NIGHT_MODES)
        assert processor.mode == NIGHT_MODES[0]      # wrapped back round

    def test_unknown_mode_falls_back_to_auto(self):
        assert NightVisionProcessor(mode="no-existe").mode == "auto"
        assert next_night_mode("no-existe") == NIGHT_MODES[0]

    def test_set_mode_ignores_unknown_names(self):
        processor = NightVisionProcessor(mode="ir")
        processor.set_mode("tampoco-existe")
        assert processor.mode == "ir"

    def test_status_reports_the_resolved_mode(self):
        processor = NightVisionProcessor(mode="auto")
        processor.process(np.full((32, 32, 3), 10, dtype=np.uint8))
        status = processor.status()
        assert status["mode"] == "auto"
        assert status["effective"] in NIGHT_MODES
        assert 0.0 <= status["luminance"] <= 255.0

    def test_scene_luminance_handles_an_empty_frame(self):
        assert scene_luminance(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0
        assert scene_luminance(None) == 0.0

    def test_reset_restores_daylight_assumption(self):
        processor = NightVisionProcessor(mode="auto")
        for _ in range(40):
            processor.process(np.full((32, 32, 3), 4, dtype=np.uint8))
        processor.reset()
        assert processor.effective_mode == "dia"
        assert processor.applied_gain == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

class TestZones:
    def test_perimeter_covers_the_whole_frame(self):
        perimeter = ZONES[0]
        assert perimeter.box is None
        assert perimeter.contains((0.0, 0.0))
        assert perimeter.contains((1.0, 1.0))

    def test_a_bounded_zone_excludes_the_outside(self):
        zone = Zone("PRUEBA", (0.4, 0.4, 0.6, 0.6))
        assert zone.contains((0.5, 0.5))
        assert not zone.contains((0.1, 0.5))
        assert not zone.contains((0.5, 0.9))

    def test_zone_edges_are_inclusive(self):
        zone = Zone("PRUEBA", (0.4, 0.4, 0.6, 0.6))
        assert zone.contains((0.4, 0.4))
        assert zone.contains((0.6, 0.6))

    def test_every_shipped_zone_is_well_formed(self):
        for zone in ZONES:
            assert zone.name
            if zone.box is not None:
                x0, y0, x1, y1 = zone.box
                assert x0 < x1 and y0 < y1
                assert 0.0 <= x0 and x1 <= 1.0
                assert 0.0 <= y0 and y1 <= 1.0


# ---------------------------------------------------------------------------
# The mode
# ---------------------------------------------------------------------------

class TestSecurityMode:
    DECORATIVE = (
        "skeleton", "segmentation", "face_mesh", "object_detect",
        "bounding_boxes", "grid", "night_mode", "trails", "heatmap",
        "landmark_ids", "fps_graph", "debug", "help", "telemetry",
    )

    def test_entering_the_mode_disables_every_decoration(self, state):
        build_mode_manager(state, "security")
        still_on = [name for name in self.DECORATIVE if state.is_active(name)]
        assert still_on == []

    def test_face_detection_stays_on(self, state):
        """The one model the mode does want, since it flags a capturable face."""
        build_mode_manager(state, "security")
        assert state.is_active("face_detect") is True

    def test_arming_waits_for_the_delay(self, mode, state):
        mode.on_key(ord("a"), state)
        assert mode.armed is False
        drive(mode, state, [person()], 3)
        assert mode.armed is False

    def test_arming_completes_after_the_delay(self, mode, state):
        mode.on_key(ord("a"), state)
        mode._arm_at = state.uptime          # countdown reaches zero
        drive(mode, state, [person()], 2)
        assert mode.armed is True

    def test_disarming_cancels_a_pending_arm(self, mode, state):
        mode.on_key(ord("a"), state)
        assert mode._arm_at is not None
        mode.on_key(ord("a"), state)
        assert mode._arm_at is None
        assert mode.armed is False

    def test_a_subject_in_the_armed_zone_raises_an_intrusion(self, mode, state):
        mode._armed = True
        drive(mode, state, [person(x=0.5, y=0.55)], 25)
        assert any(e.kind == "INTRUSION" for e in mode._events)

    def test_a_subject_outside_the_zone_raises_nothing(self, mode, state):
        while mode.zone.name != "DERECHA":
            mode.on_key(ord("x"), state)
        mode._armed = True
        drive(mode, state, [person(x=0.10, y=0.5)], 25)
        assert not any(e.kind == "INTRUSION" for e in mode._events)

    def test_crossing_into_the_zone_raises_an_intrusion(self, mode, state):
        while mode.zone.name != "DERECHA":
            mode.on_key(ord("x"), state)
        mode._armed = True
        drive(mode, state, [person(x=0.10, y=0.5)], 20)
        assert not any(e.kind == "INTRUSION" for e in mode._events)
        drive(mode, state, [person(x=0.88, y=0.5)], 20, start=20)
        assert any(e.kind == "INTRUSION" for e in mode._events)

    def test_disarmed_system_records_no_intrusion(self, mode, state):
        drive(mode, state, [person(x=0.5)], 25)
        assert not any(e.kind == "INTRUSION" for e in mode._events)

    def test_a_new_subject_is_logged_once(self, mode, state):
        drive(mode, state, [person(x=0.5)], 20)
        altas = [e for e in mode._events if e.kind == "ALTA"]
        assert len(altas) == 1
        assert altas[0].subject == "SUJ-01"

    def test_a_departing_subject_is_logged(self, mode, state):
        drive(mode, state, [person(x=0.5)], 20)
        drive(mode, state, [], 40, start=20)
        assert any(e.kind == "BAJA" for e in mode._events)

    def test_subjects_are_published_on_the_context(self, mode, state):
        ctx = drive(mode, state, [person(x=0.5)], 10)
        assert len(ctx.extras["security_subjects"]) == 1

    def test_the_event_log_is_capped(self, mode, state):
        for index in range(mode.MAX_EVENTS + 40):
            mode._log("ALTA", state, subject=f"SUJ-{index:02d}")
        assert len(mode._events) == mode.MAX_EVENTS
        # The cap must drop the oldest, not the newest.
        assert mode._events[-1].subject == f"SUJ-{mode.MAX_EVENTS + 39:02d}"

    def test_reset_clears_log_identities_and_arm_state(self, mode, state):
        mode._armed = True
        drive(mode, state, [person()], 20)
        mode.reset(state)
        assert mode._events == []
        assert mode.tracker.active_count == 0
        assert mode.tracker.total_identified == 0
        assert mode.armed is False

    def test_night_vision_key_cycles_the_sensor(self, mode, state):
        first = mode.nightvision.mode
        mode.on_key(ord("n"), state)
        assert mode.nightvision.mode != first

    def test_zone_key_cycles_and_wraps(self, mode, state):
        names = []
        for _ in range(len(ZONES) + 1):
            names.append(mode.zone.name)
            mode.on_key(ord("x"), state)
        assert len(set(names[:len(ZONES)])) == len(ZONES)
        assert names[-1] == names[0]

    def test_channel_key_cycles(self, mode, state):
        first = mode.channel
        mode.on_key(ord("c"), state)
        assert mode.channel != first

    def test_osd_detail_cycles_through_three_levels(self, mode, state):
        levels = set()
        for _ in range(4):
            levels.add(mode._osd_detail)
            mode.on_key(ord("o"), state)
        assert levels == {0, 1, 2}

    @pytest.mark.parametrize("detail", [0, 1, 2])
    def test_every_osd_detail_level_draws(self, mode, state, detail):
        mode._osd_detail = detail
        mode._armed = True
        drive(mode, state, [person(x=0.4), person(build=1.8, x=0.7)], 8)

    def test_unhandled_keys_are_declined(self, mode, state):
        assert mode.on_key(ord("Q"), state) is False
        assert mode.on_key(0, state) is False

    @pytest.mark.parametrize("key", list("anj+-xciloez"))
    def test_every_documented_key_is_handled(self, mode, state, key):
        assert mode.on_key(ord(key), state) is True

    def test_documented_keys_match_the_handler(self, mode, state):
        """A key in the help panel that does nothing is a lie to the operator."""
        for key in mode.keys:
            code = 32 if key == "space" else ord(key[0])
            assert mode.on_key(code, state) is True, f"tecla '{key}' sin efecto"

    def test_hud_lines_are_strings(self, mode, state):
        ctx = drive(mode, state, [person(x=0.4), person(build=1.7, x=0.7)], 12)
        lines = mode.hud_lines(ctx, state)
        assert lines and all(isinstance(line, str) for line in lines)

    def test_status_text_reports_the_subject_count(self, mode, state):
        ctx = drive(mode, state, [person(x=0.5)], 12)
        assert "SUJETO" in mode.status_text(ctx, state)

    def test_status_text_when_idle(self, mode, state):
        ctx = drive(mode, state, [], 5)
        assert mode.status_text(ctx, state) == "EN ESPERA"

    def test_status_text_when_armed_and_empty(self, mode, state):
        mode._armed = True
        ctx = drive(mode, state, [], 5)
        assert mode.status_text(ctx, state) == "VIGILANDO"

    def test_the_sensor_stage_transforms_the_frame(self, mode, state):
        """A dark scene must come out of process() brighter than it went in."""
        mode.nightvision.set_mode("ir")
        ctx = security_context([person()], 0, luminance=12)
        before = scene_luminance(ctx.frame)
        mode.process(ctx, state)
        assert scene_luminance(ctx.frame) > before

    def test_export_writes_the_log(self, mode, state, tmp_path):
        state.config.analytics.export_dir = str(tmp_path)
        drive(mode, state, [person(x=0.5)], 20)
        path = mode._export(state)

        assert path is not None
        document = json.loads(open(path, encoding="utf-8").read())
        assert document["events"]
        assert document["channel"] == mode.channel
        assert document["zone"] == mode.zone.name
        assert "sensor" in document

    def test_export_of_an_empty_log_is_refused(self, mode, state, tmp_path):
        state.config.analytics.export_dir = str(tmp_path)
        assert mode._export(state) is None
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("size", [(2, 2), (1, 1), (3, 480), (481, 641)])
    def test_odd_frame_sizes_are_survivable(self, mode, state, size):
        height, width = size
        mode._armed = True
        for index in range(4):
            ctx = security_context([person()], index, width=width, height=height)
            mode.process(ctx, state)
            ctx.ensure_drawable()
            mode.draw(ctx, state)
            mode.hud_lines(ctx, state)

    def test_non_finite_landmarks_do_not_reach_the_drawing_code(self, mode, state):
        broken = list(person())
        broken[11] = Point(float("nan"), float("nan"))
        broken[12] = Point(float("inf"), float("-inf"))
        mode._armed = True
        drive(mode, state, [broken], 6)

    def test_empty_frames_are_survivable(self, mode, state):
        mode._armed = True
        drive(mode, state, [], 10)
        assert mode.tracker.active_count == 0


class TestSecurityEvent:
    def test_line_includes_every_populated_field(self):
        event = SecurityEvent("12:00:00", 1.5, "INTRUSION", "SUJ-01", "zona UMBRAL")
        line = event.line()
        assert "12:00:00" in line and "INTRUSION" in line
        assert "SUJ-01" in line and "zona UMBRAL" in line

    def test_line_omits_empty_fields(self):
        assert SecurityEvent("12:00:00", 0.0, "BAJA").line() == "12:00:00 BAJA"

    def test_to_dict_is_json_serialisable(self):
        event = SecurityEvent("12:00:00", 1.25, "ALTA", "SUJ-02", "perfil ABCD")
        payload = json.loads(json.dumps(event.to_dict()))
        assert payload["kind"] == "ALTA"
        assert payload["uptime"] == 1.25
