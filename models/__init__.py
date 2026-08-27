"""
Model registry and download utilities for LookThePerson.
All MediaPipe model wrappers live in this package.
"""

import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model_assets")

# ---------------------------------------------------------------------------
# Model download catalogue
# ---------------------------------------------------------------------------

MODELS = {
    "pose": {
        "filename": "pose_landmarker_full.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/1/"
            "pose_landmarker_full.task"
        ),
    },
    "hand": {
        "filename": "hand_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/"
            "hand_landmarker.task"
        ),
    },
    "face_mesh": {
        "filename": "face_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/"
            "face_landmarker.task"
        ),
    },
    "face_detection": {
        "filename": "blaze_face_short_range.tflite",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/1/"
            "blaze_face_short_range.tflite"
        ),
    },
    "object_detection": {
        "filename": "efficientdet_lite0.tflite",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "object_detector/efficientdet_lite0/int8/1/"
            "efficientdet_lite0.tflite"
        ),
    },
}


def model_path(model_key):
    """Return the local filesystem path for a model."""
    return os.path.join(MODEL_DIR, MODELS[model_key]["filename"])


def ensure_model(model_key):
    """Download the model if it does not exist locally. Returns the path."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = model_path(model_key)
    if not os.path.exists(path):
        info = MODELS[model_key]
        print(f"[models] Descargando {info['filename']}...", flush=True)
        urllib.request.urlretrieve(info["url"], path)
        print(f"[models] {info['filename']} descargado.", flush=True)
    return path


def ensure_all_models(model_keys=None):
    """Download all requested models (or all if *model_keys* is None)."""
    keys = model_keys or list(MODELS.keys())
    for key in keys:
        ensure_model(key)
