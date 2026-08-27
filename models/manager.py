"""
Model orchestration for LookThePerson.

Owns the five MediaPipe detectors and decides, every frame, which of them
actually need to run. Two mechanisms keep the frame rate up:

* **Demand** — a model only runs when the active mode or a toggle asks for it.
* **Stride** — expensive models can run every Nth frame and have their last
  result reused in between, which is invisible for slow-changing signals like
  object detection but roughly halves its cost.

Also owns landmark smoothing, so jitter is removed once, centrally, instead of
in every consumer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.face_detection import FaceDetectionModel
from models.face_mesh import FaceMeshModel
from models.hands import HandModel
from models.object_detection import ObjectDetectionModel
from models.pose import PoseModel

__all__ = ["ModelManager", "LandmarkSmoother", "MODEL_NAMES"]

MODEL_NAMES = ("pose", "hands", "face_mesh", "face_detect", "object")


class LandmarkSmoother:
    """
    Per-landmark 1€ filtering.

    Landmarks arrive with visible frame-to-frame jitter even when the subject
    is still. Filtering here means angles, gestures and drawing all benefit
    without each of them implementing smoothing separately.

    Filters are keyed by ``(track, person, index)`` so several people and both
    hands can be smoothed independently.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007):
        from core.filters import PointFilter

        self._factory = lambda: PointFilter(min_cutoff, beta)
        self._filters: Dict[Tuple[str, int, int], Any] = {}
        self.enabled = True

    def smooth(self, track: str, landmark_lists: List[Any], timestamp: float) -> List[Any]:
        """
        Return smoothed copies of the landmark lists.

        MediaPipe landmark objects are not mutable in a way we should rely on,
        so smoothed values are wrapped in lightweight points that expose the
        same attributes.
        """
        if not self.enabled or not landmark_lists:
            return landmark_lists

        from core.geometry import Point

        smoothed_lists: List[Any] = []
        for person_index, landmarks in enumerate(landmark_lists):
            smoothed: List[Point] = []
            for index, landmark in enumerate(landmarks):
                key = (track, person_index, index)
                point_filter = self._filters.get(key)
                if point_filter is None:
                    point_filter = self._factory()
                    self._filters[key] = point_filter
                x, y, z = point_filter.update(
                    landmark.x, landmark.y, getattr(landmark, "z", 0.0) or 0.0, timestamp,
                )
                smoothed.append(
                    Point(x, y, z, getattr(landmark, "visibility", 1.0))
                )
            smoothed_lists.append(smoothed)
        return smoothed_lists

    def reset(self) -> None:
        self._filters.clear()

    @property
    def filter_count(self) -> int:
        return len(self._filters)


class ModelManager:
    """
    Creates, runs and tears down the detection models.

    Args:
        config: a :class:`~core.config.Config`; detection settings are read
            from ``config.detection``.
        profiler: optional :class:`~core.metrics.StageProfiler` used to time
            each model.
    """

    def __init__(self, config: Any, profiler: Any = None):
        detection = config.detection
        self.config = config
        self.profiler = profiler

        self.pose = PoseModel(
            num_poses=detection.max_poses,
            confidence=detection.pose_confidence,
            segmentation=detection.segmentation,
        )
        self.hands = HandModel(
            num_hands=detection.max_hands,
            confidence=detection.hand_confidence,
        )
        self.face_mesh = FaceMeshModel(
            num_faces=detection.max_faces,
            confidence=detection.face_confidence,
        )
        self.face_detect = FaceDetectionModel(confidence=detection.face_confidence)
        self.objects = ObjectDetectionModel(
            confidence=detection.object_confidence,
            max_results=detection.max_objects,
        )

        self.smoother = LandmarkSmoother(
            detection.smoothing_min_cutoff, detection.smoothing_beta,
        )
        self.smoother.enabled = detection.smoothing

        self._started: Dict[str, bool] = {name: False for name in MODEL_NAMES}
        self._cache: Dict[str, Any] = {name: None for name in MODEL_NAMES}
        self._strides: Dict[str, int] = {
            "pose": 1, "hands": 1,
            "face_mesh": max(1, detection.face_mesh_stride),
            "face_detect": max(1, detection.object_stride),
            "object": max(1, detection.object_stride),
        }
        self._failures: Dict[str, str] = {}

    # -- Lifecycle ----------------------------------------------------------

    def start(self, names: Optional[Tuple[str, ...]] = None) -> List[str]:
        """
        Start the named models (all of them by default).

        Returns the names that failed, so the caller can report a missing
        model without the whole app refusing to run.
        """
        failed: List[str] = []
        for name in names or MODEL_NAMES:
            if not self._start_one(name):
                failed.append(name)
        return failed

    def _start_one(self, name: str) -> bool:
        if self._started.get(name):
            return True
        model = self._model(name)
        if model is None:
            return False
        try:
            model.start()
            self._started[name] = True
            return True
        except Exception as exc:
            self._failures[name] = str(exc)
            print(f"[models] No pude iniciar '{name}': {exc}", flush=True)
            return False

    def ensure_started(self, name: str) -> bool:
        """Start a model on first use — lazy loading for optional models."""
        return self._start_one(name)

    def stop(self) -> None:
        """Release every detector."""
        for name in MODEL_NAMES:
            model = self._model(name)
            if model is not None and self._started.get(name):
                try:
                    model.stop()
                except Exception as exc:
                    print(f"[models] Error al cerrar '{name}': {exc}", flush=True)
                self._started[name] = False

    def _model(self, name: str):
        return {
            "pose": self.pose, "hands": self.hands, "face_mesh": self.face_mesh,
            "face_detect": self.face_detect, "object": self.objects,
        }.get(name)

    # -- Inference ----------------------------------------------------------

    def detect(
        self,
        name: str,
        mp_image: Any,
        timestamp_ms: int,
        frame_index: int,
    ) -> Any:
        """
        Run one model, honouring its stride and reusing the cached result.

        Returns the model's result object, or None when it is not running.
        """
        if not self._started.get(name) and not self.ensure_started(name):
            return None

        stride = self._strides.get(name, 1)
        if stride > 1 and frame_index % stride != 0:
            return self._cache.get(name)

        model = self._model(name)
        if model is None:
            return None

        try:
            if self.profiler is not None:
                with self.profiler.stage(name):
                    result = model.detect(mp_image, timestamp_ms)
            else:
                result = model.detect(mp_image, timestamp_ms)
        except Exception as exc:
            # MediaPipe raises on out-of-order timestamps; drop the frame
            # rather than the session.
            self._failures[name] = str(exc)
            return self._cache.get(name)

        self._cache[name] = result
        return result

    def cached(self, name: str) -> Any:
        return self._cache.get(name)

    def clear_cache(self) -> None:
        for name in MODEL_NAMES:
            self._cache[name] = None

    # -- Configuration ------------------------------------------------------

    def set_stride(self, name: str, stride: int) -> None:
        """Change how often a model runs (1 = every frame)."""
        self._strides[name] = max(1, int(stride))

    def stride(self, name: str) -> int:
        return self._strides.get(name, 1)

    def set_object_confidence(self, confidence: float) -> float:
        """Rebuild the object detector at a new confidence threshold."""
        if self._started.get("object"):
            self.objects.restart_with_confidence(confidence)
        else:
            self.objects.confidence = confidence
        return self.objects.confidence

    def set_segmentation(self, enabled: bool) -> None:
        """
        Turn person masks on or off.

        The flag is baked into the detector at creation, so this restarts the
        pose model — cheap enough for a keypress, too expensive per frame.
        """
        if self.pose.segmentation == enabled:
            return
        self.pose.segmentation = enabled
        if self._started.get("pose"):
            self.pose.stop()
            self._started["pose"] = False
            self._start_one("pose")

    # -- Status -------------------------------------------------------------

    def started(self, name: str) -> bool:
        return bool(self._started.get(name))

    def active_models(self) -> List[str]:
        return [name for name in MODEL_NAMES if self._started.get(name)]

    def failures(self) -> Dict[str, str]:
        """Models that failed to start or errored, with the reason."""
        return dict(self._failures)

    def status_lines(self) -> List[str]:
        """Readable per-model status for the debug view."""
        lines = []
        for name in MODEL_NAMES:
            state = "ON " if self._started.get(name) else "off"
            stride = self._strides.get(name, 1)
            suffix = f" /{stride}" if stride > 1 else ""
            lines.append(f"{name:<12} {state}{suffix}")
        return lines
