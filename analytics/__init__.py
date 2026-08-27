"""
Analytics for LookThePerson.

Turns raw landmark streams into meaningful measurements: joint angles, posture
scores, exercise repetitions, calorie estimates, balance, symmetry, blink rate,
drowsiness and attention.

Like ``core``, this package is free of OpenCV and MediaPipe imports — it works
on any object exposing ``.x`` / ``.y`` / ``.visibility``, which makes every
metric here directly unit-testable.
"""

from analytics.angles import (
    PoseLandmark,
    compute_angle,
    compute_joint_angles,
    limb_lengths,
    body_orientation,
)
from analytics.face_metrics import (
    AttentionTracker,
    BlinkDetector,
    DrowsinessMonitor,
    eye_aspect_ratio,
    head_pose,
    mouth_aspect_ratio,
)
from analytics.fitness import IntensityTracker, WorkoutSession, calories_burned
from analytics.motion import (
    BalanceAnalyzer,
    MotionAnalyzer,
    StillnessDetector,
    TrajectoryRecorder,
    symmetry_report,
)
from analytics.posture import PostureMonitor, PostureReport, analyze_posture
from analytics.reps import EXERCISES, MultiExerciseCounter, RepCounter, RepEvent
from analytics.session import SessionRecorder

__all__ = [
    "PoseLandmark",
    "compute_angle",
    "compute_joint_angles",
    "limb_lengths",
    "body_orientation",
    "analyze_posture",
    "PostureMonitor",
    "PostureReport",
    "RepCounter",
    "RepEvent",
    "MultiExerciseCounter",
    "EXERCISES",
    "WorkoutSession",
    "IntensityTracker",
    "calories_burned",
    "MotionAnalyzer",
    "BalanceAnalyzer",
    "TrajectoryRecorder",
    "StillnessDetector",
    "symmetry_report",
    "BlinkDetector",
    "DrowsinessMonitor",
    "AttentionTracker",
    "eye_aspect_ratio",
    "mouth_aspect_ratio",
    "head_pose",
    "SessionRecorder",
]
