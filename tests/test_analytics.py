"""Tests for the analytics layer: angles, posture, reps, fitness, motion, face."""

from __future__ import annotations

import pytest

from analytics.angles import (
    PoseLandmark as L,
    body_orientation,
    compute_angle,
    compute_joint_angles,
    flag_implausible,
    limb_lengths,
    shoulder_tilt,
    trunk_inclination,
)
from analytics.face_metrics import (
    AttentionTracker,
    BlinkDetector,
    DrowsinessMonitor,
    eye_aspect_ratio,
    head_pose,
    mouth_aspect_ratio,
)
from analytics.fitness import WorkoutSession, calories_burned, estimate_intensity
from analytics.motion import (
    BalanceAnalyzer,
    MotionAnalyzer,
    StillnessDetector,
    TrajectoryRecorder,
    jump_height_estimate,
    symmetry_report,
)
from analytics.posture import PostureMonitor, analyze_posture
from analytics.reps import EXERCISES, MultiExerciseCounter, RepCounter
from analytics.session import SessionRecorder
from core.geometry import Point
from tests.conftest import make_face, make_pose


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

class TestAngles:
    def test_standing_knees_are_extended(self):
        angles = compute_joint_angles(make_pose(178.0))
        assert angles["avg_knee"] == pytest.approx(178.0, abs=1.0)

    def test_squat_knees_are_flexed(self):
        angles = compute_joint_angles(make_pose(80.0))
        assert angles["avg_knee"] == pytest.approx(80.0, abs=1.0)

    def test_upright_trunk_is_zero(self):
        assert trunk_inclination(make_pose()) == pytest.approx(0.0, abs=1.0)

    def test_leaning_trunk_is_detected(self):
        leaning = make_pose(
            LEFT_SHOULDER=Point(0.55, 0.26), RIGHT_SHOULDER=Point(0.71, 0.26),
        )
        assert trunk_inclination(leaning) > 15.0

    def test_level_shoulders_have_no_tilt(self):
        assert shoulder_tilt(make_pose()) == pytest.approx(0.0, abs=0.5)

    def test_dropped_shoulder_is_detected(self):
        tilted = make_pose(LEFT_SHOULDER=Point(0.42, 0.33))
        assert shoulder_tilt(tilted) > 10.0

    def test_occluded_joints_are_omitted_not_zeroed(self):
        pose = make_pose()
        pose[L.LEFT_KNEE] = Point(0.45, 0.74, visibility=0.0)
        angles = compute_joint_angles(pose)
        assert "left_knee" not in angles
        assert "right_knee" in angles

    def test_short_landmark_list_returns_empty(self):
        assert compute_joint_angles([Point(0, 0)] * 5) == {}

    def test_unknown_joint_raises(self):
        with pytest.raises(KeyError):
            compute_angle(make_pose(), "elbow_of_doom")

    def test_symmetric_pose_has_matching_pairs(self):
        angles = compute_joint_angles(make_pose())
        assert angles["left_knee"] == pytest.approx(angles["right_knee"], abs=0.5)

    def test_body_orientation_front(self):
        assert body_orientation(make_pose()) == "front"

    def test_limb_lengths_are_positive(self):
        lengths = limb_lengths(make_pose())
        assert lengths["left_thigh"] > 0
        assert lengths["shoulder_width"] > 0

    def test_flag_implausible_catches_out_of_range(self):
        assert "left_knee" in flag_implausible({"left_knee": 5.0})
        assert flag_implausible({"left_knee": 120.0}) == []


# ---------------------------------------------------------------------------
# Posture
# ---------------------------------------------------------------------------

class TestPosture:
    def test_good_posture_scores_high(self):
        report = analyze_posture(make_pose())
        assert report.valid
        assert report.score >= 80
        assert report.grade in ("A", "B")

    def test_slouch_is_detected(self):
        # Drop the head toward the shoulders to shorten the neck.
        slouched = make_pose(LEFT_EAR=Point(0.45, 0.23), RIGHT_EAR=Point(0.55, 0.23))
        report = analyze_posture(slouched)
        assert "slouch" in [issue.code for issue in report.issues]
        assert report.score < 80

    def test_uneven_shoulders_detected(self):
        tilted = make_pose(LEFT_SHOULDER=Point(0.42, 0.34))
        codes = [issue.code for issue in analyze_posture(tilted).issues]
        assert "shoulder_tilt" in codes

    def test_invisible_landmarks_give_invalid_report(self):
        pose = make_pose()
        for index in (L.LEFT_EAR, L.RIGHT_EAR):
            pose[index] = Point(0.5, 0.1, visibility=0.0)
        assert analyze_posture(pose).valid is False

    def test_issues_sorted_by_severity(self):
        report = analyze_posture(
            make_pose(LEFT_EAR=Point(0.45, 0.24), LEFT_SHOULDER=Point(0.42, 0.35))
        )
        severities = [issue.severity for issue in report.issues]
        assert severities == sorted(severities, reverse=True)

    def test_monitor_alerts_only_after_sustained_bad_posture(self):
        monitor = PostureMonitor(alert_score=90.0, alert_seconds=2.0)
        slouched = make_pose(LEFT_EAR=Point(0.45, 0.23), RIGHT_EAR=Point(0.55, 0.23))

        monitor.update(slouched, 0.0)
        assert monitor.should_alert(0.5) is False

        for t in range(1, 40):
            monitor.update(slouched, t * 0.2)
        assert monitor.should_alert(8.0) is True

    def test_monitor_tracks_session_stats(self):
        monitor = PostureMonitor()
        for i in range(10):
            monitor.update(make_pose(), i * 0.1)
        stats = monitor.session_stats()
        assert stats["samples"] == 10
        assert stats["average_score"] > 0


# ---------------------------------------------------------------------------
# Repetitions
# ---------------------------------------------------------------------------

def run_squats(counter: RepCounter, reps: int, start: float = 0.0, step: float = 0.2):
    """Drive a counter through *reps* full squat cycles."""
    events = []
    t = start
    for _ in range(reps):
        for angle in (170, 140, 95, 80, 95, 140, 170):
            t += step
            event = counter.update(compute_joint_angles(make_pose(angle)), t)
            if event:
                events.append(event)
    return events


class TestRepCounter:
    def test_counts_full_repetitions(self):
        counter = RepCounter("squat")
        events = run_squats(counter, 3)
        assert counter.count == 3
        assert len(events) == 3

    def test_rep_indices_increment(self):
        counter = RepCounter("squat")
        events = run_squats(counter, 3)
        assert [e.index for e in events] == [1, 2, 3]

    def test_partial_rep_does_not_count(self):
        counter = RepCounter("squat")
        t = 0.0
        for angle in (170, 150, 140, 150, 170):    # never reaches the bottom
            t += 0.2
            counter.update(compute_joint_angles(make_pose(angle)), t)
        assert counter.count == 0

    def test_too_fast_reps_are_rejected_as_noise(self):
        counter = RepCounter("squat")
        t = 0.0
        for _ in range(5):
            for angle in (170, 80, 170):
                t += 0.05                          # 0.1s per rep, below the minimum
                counter.update(compute_joint_angles(make_pose(angle)), t)
        assert counter.count == 0

    def test_depth_is_reported(self):
        counter = RepCounter("squat")
        events = run_squats(counter, 1)
        assert events[0].depth == pytest.approx(1.0, abs=0.01)

    def test_progress_tracks_current_position(self):
        counter = RepCounter("squat")
        counter.update(compute_joint_angles(make_pose(160.0)), 0.1)
        assert counter.progress == pytest.approx(0.0, abs=0.05)
        counter.update(compute_joint_angles(make_pose(100.0)), 0.3)
        assert counter.progress > 0.9

    def test_complete_set_banks_reps(self):
        counter = RepCounter("squat")
        run_squats(counter, 2)
        counter.complete_set()
        assert counter.sets == 1
        assert counter.count == 0
        assert counter.total_reps == 2       # history is preserved

    def test_switching_exercise_closes_the_set(self):
        counter = RepCounter("squat")
        run_squats(counter, 2)
        counter.set_exercise("pushup")
        assert counter.exercise == "pushup"
        assert counter.sets == 1

    def test_unknown_exercise_raises(self):
        with pytest.raises(KeyError):
            RepCounter("moonwalk")

    def test_missing_angle_is_a_no_op(self):
        counter = RepCounter("squat")
        assert counter.update({}, 1.0) is None

    def test_stats_shape(self):
        counter = RepCounter("squat")
        run_squats(counter, 2)
        stats = counter.stats()
        assert stats["total_reps"] == 2
        assert stats["exercise"] == "squat"

    def test_every_catalogued_exercise_is_constructible(self):
        for key in EXERCISES:
            assert RepCounter(key).label


class TestMultiExerciseCounter:
    def test_tracks_several_exercises(self):
        counter = MultiExerciseCounter(["squat", "pushup"])
        t = 0.0
        for _ in range(2):
            # Seven steps of 0.2s puts each rep clearly above the 0.6s minimum
            # rather than exactly on it, where float error decides the outcome.
            for angle in (170, 140, 95, 80, 95, 140, 170):
                t += 0.2
                counter.update(compute_joint_angles(make_pose(angle)), t)
        assert counter.total_reps >= 2
        assert counter.leaderboard()


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

class TestFitness:
    def test_calories_scale_with_time_and_weight(self):
        light = calories_burned(met=5.0, weight_kg=60, seconds=600)
        heavy = calories_burned(met=5.0, weight_kg=90, seconds=600)
        assert heavy > light > 0

    def test_calories_zero_for_no_time(self):
        assert calories_burned(5.0, 70, 0) == 0.0

    def test_intensity_classification(self):
        assert estimate_intensity(1, 2.0) == "ligera"
        assert estimate_intensity(30, 8.0) == "intensa"

    def test_session_accumulates_reps(self):
        session = WorkoutSession(weight_kg=70, started_at=0.0)
        counter = RepCounter("squat")
        for event in run_squats(counter, 3):
            session.record_rep(event, now=10.0)
        assert session.total_reps == 3
        assert session.calories > 0

    def test_set_closes_after_rest(self):
        session = WorkoutSession(rest_threshold=5.0, started_at=0.0)
        counter = RepCounter("squat")
        for event in run_squats(counter, 2):
            session.record_rep(event, now=1.0)
        assert session.tick(now=2.0) is None       # still resting
        closed = session.tick(now=20.0)
        assert closed is not None and closed.reps == 2

    def test_switching_exercise_closes_previous_set(self):
        session = WorkoutSession(started_at=0.0)
        squat_counter = RepCounter("squat")
        for event in run_squats(squat_counter, 1):
            session.record_rep(event, now=1.0)

        curl = RepCounter("bicep_curl")
        from analytics.reps import RepEvent
        session.record_rep(RepEvent("bicep_curl", 1, 1.5, 1.0), now=5.0)
        assert session.total_sets == 1

    def test_summary_shape(self):
        summary = WorkoutSession(started_at=0.0).summary(now=60.0)
        assert {"duration_seconds", "total_reps", "calories", "sets"} <= set(summary)


# ---------------------------------------------------------------------------
# Motion and balance
# ---------------------------------------------------------------------------

class TestMotion:
    def test_still_body_has_low_energy(self):
        analyzer = MotionAnalyzer()
        pose = make_pose()
        for i in range(20):
            metrics = analyzer.update(pose, i * 0.05)
        assert metrics["energy"] < 0.01

    def test_moving_body_has_higher_energy(self):
        analyzer = MotionAnalyzer()
        for i in range(20):
            offset = 0.02 * i
            metrics = analyzer.update(
                make_pose(LEFT_WRIST=Point(0.39 + offset, 0.54)), i * 0.05,
            )
        assert metrics["energy"] > 0.0

    def test_balance_of_upright_pose_is_stable(self):
        analyzer = BalanceAnalyzer()
        for _ in range(10):
            report = analyzer.update(make_pose())
        assert report.valid
        assert report.stability > 70
        assert report.center_of_mass is not None

    def test_balance_invalid_without_landmarks(self):
        assert BalanceAnalyzer().update([]).valid is False

    def test_symmetry_of_symmetric_pose(self):
        report = symmetry_report(compute_joint_angles(make_pose()))
        assert report["valid"]
        assert report["score"] > 95
        assert report["asymmetric"] == []

    def test_symmetry_detects_difference(self):
        angles = {"left_knee": 90.0, "right_knee": 170.0}
        report = symmetry_report(angles)
        assert "knee" in report["asymmetric"]

    def test_symmetry_without_pairs_is_invalid(self):
        assert symmetry_report({"left_knee": 90.0})["valid"] is False

    def test_stillness_requires_hold(self):
        detector = StillnessDetector(threshold=0.01, hold_seconds=1.0)
        assert detector.update(0.001, 0.0) is False
        assert detector.update(0.001, 0.5) is False
        assert detector.update(0.001, 1.5) is True

    def test_stillness_resets_on_movement(self):
        detector = StillnessDetector(threshold=0.01, hold_seconds=1.0)
        detector.update(0.001, 0.0)
        detector.update(0.5, 0.5)               # moved
        assert detector.update(0.001, 1.2) is False

    def test_trajectory_records_and_measures(self):
        recorder = TrajectoryRecorder([L.LEFT_WRIST], max_points=10)
        for i in range(5):
            recorder.update(make_pose(LEFT_WRIST=Point(0.3 + i * 0.1, 0.5)), i * 0.1)
        assert len(recorder.path(L.LEFT_WRIST)) == 5
        assert recorder.path_length(L.LEFT_WRIST) == pytest.approx(0.4, abs=0.01)

    def test_trajectory_prunes_old_points(self):
        recorder = TrajectoryRecorder([L.LEFT_WRIST])
        recorder.update(make_pose(), 0.0)
        recorder.update(make_pose(), 5.0)
        recorder.prune(now=5.0, max_age=1.0)
        assert len(recorder.path(L.LEFT_WRIST)) == 1

    def test_jump_height_estimate(self):
        height = jump_height_estimate(
            baseline_hip_y=0.6, peak_hip_y=0.5,
            body_height_normalized=0.8, real_height_cm=180,
        )
        assert height == pytest.approx(22.5, abs=0.5)

    def test_jump_height_guards_zero_scale(self):
        assert jump_height_estimate(0.6, 0.5, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Face metrics
# ---------------------------------------------------------------------------

class TestFaceMetrics:
    def test_open_eye_has_higher_ear_than_closed(self):
        open_ear = eye_aspect_ratio(make_face(eye_open=0.03), "left")
        closed_ear = eye_aspect_ratio(make_face(eye_open=0.002), "left")
        assert open_ear > closed_ear

    def test_ear_returns_none_for_short_mesh(self):
        assert eye_aspect_ratio([Point(0, 0)] * 10, "left") is None

    def test_mouth_ratio_grows_when_open(self):
        closed = mouth_aspect_ratio(make_face(mouth_open=0.005))
        wide = mouth_aspect_ratio(make_face(mouth_open=0.06))
        assert wide > closed

    def test_head_pose_neutral_face(self):
        pose = head_pose(make_face())
        assert abs(pose["yaw"]) < 10
        assert abs(pose["roll"]) < 10

    def test_head_pose_detects_turn(self):
        turned = make_face()
        turned[1] = Point(0.57, 0.30)         # nose shifted toward one cheek
        assert head_pose(turned)["yaw"] > 15

    def test_blink_detection(self):
        detector = BlinkDetector()
        assert detector.update(0.30, 0.0) is False     # open
        # A real blink spans several frames; the EAR smoothing needs more than
        # one closed sample before it crosses the threshold.
        for i in range(1, 5):
            detector.update(0.06, i * 0.03)
        assert detector.is_closed
        # Reopening is smoothed too, so the EAR needs a frame to climb back
        # over the open threshold before the blink is registered.
        detector.update(0.30, 0.27)
        assert detector.update(0.32, 0.30) is True
        assert detector.count == 1

    def test_overlong_closure_is_not_a_blink(self):
        detector = BlinkDetector()
        detector.update(0.30, 0.0)
        detector.update(0.05, 0.1)
        assert detector.update(0.30, 5.0) is False     # too slow to be a blink
        assert detector.count == 0

    def test_drowsiness_reports_perclos(self):
        monitor = DrowsinessMonitor(window_seconds=10.0)
        for i in range(50):
            report = monitor.update(ear=0.05, mar=0.1, now=i * 0.1)
        assert report.perclos > 0.9
        assert report.level == "somnolencia"

    def test_alert_face_stays_alert(self):
        monitor = DrowsinessMonitor()
        for i in range(30):
            report = monitor.update(ear=0.32, mar=0.1, now=i * 0.1)
        assert report.level == "alerta"

    def test_attention_tracks_ratio(self):
        tracker = AttentionTracker()
        facing = {"yaw": 2.0, "pitch": 1.0, "roll": 0.0}
        away = {"yaw": 70.0, "pitch": 0.0, "roll": 0.0}
        for i in range(10):
            tracker.update(facing, (0.0, 0.0), i * 0.1)
        for i in range(10, 20):
            tracker.update(away, (0.0, 0.0), i * 0.1)
        assert 0.0 < tracker.attention_ratio < 1.0
        assert tracker.stats()["distractions"] >= 1

    def test_attention_without_face_counts_as_distracted(self):
        tracker = AttentionTracker()
        tracker.update(None, None, 0.0)
        assert tracker.update(None, None, 1.0) is False


# ---------------------------------------------------------------------------
# Session export
# ---------------------------------------------------------------------------

class TestSessionRecorder:
    def test_records_and_counts(self):
        recorder = SessionRecorder(enabled=True)
        recorder.record_gesture("clap")
        recorder.record_gesture("clap")
        recorder.record_rep("squat", 1, 95.0, 1.2)
        assert recorder.counts() == {"gesture": 2, "rep": 1}
        assert recorder.label_counts("gesture")["clap"] == 2

    def test_disabled_recorder_records_nothing(self):
        recorder = SessionRecorder(enabled=False)
        recorder.record_gesture("clap")
        assert len(recorder) == 0

    def test_export_json(self, tmp_path):
        recorder = SessionRecorder(output_dir=str(tmp_path))
        recorder.record_gesture("wave")
        path = recorder.export("json")
        assert path and path.endswith(".json")

    def test_export_csv_unions_columns(self, tmp_path):
        recorder = SessionRecorder(output_dir=str(tmp_path))
        recorder.record_gesture("wave")
        recorder.record_rep("squat", 1, 90.0, 1.0)
        path = recorder.export("csv")
        header = open(path, encoding="utf-8").readline()
        assert "form_score" in header and "label" in header

    def test_export_of_empty_session_returns_none(self, tmp_path):
        assert SessionRecorder(output_dir=str(tmp_path)).export("json") is None
