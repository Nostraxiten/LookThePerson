"""
LookThePerson — Full Body & Gesture Detector
Cross-platform launcher (Windows + Linux)
by Nox / @nostraxiten

Detects body pose, hands, face mesh, face detection, and objects
in real-time using MediaPipe Tasks + OpenCV.

Usage:
    python looktheperson.py [--windowed] [--camera 0] [--no-calculator]
    python looktheperson.py --help
"""

import argparse
import random
import time

import cv2
import mediapipe as mp
import numpy as np

from platforms import get_platform

from models.pose import PoseModel
from models.hands import HandModel
from models.face_mesh import FaceMeshModel
from models.face_detection import FaceDetectionModel
from models.object_detection import ObjectDetectionModel

from gestures.body_gestures import (
    detect_all_body_gestures,
    hand_touches_top_of_head,
    head_circle,
)
from gestures.hand_gestures import detect_all_hand_gestures
from gestures.face_gestures import FaceGestureDetector

from actions.key_handler import KeyHandler
from actions.recording import Recorder
from actions.app_control import AppController

from ui.hud import draw_hud_panel, draw_center_point
from ui.renderer import (
    fit_frame_to_screen,
    draw_grid,
    apply_night_mode,
    draw_head_circle,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_NAME = "LookThePerson - @nostraxiten"
FPS_SMOOTHING = 0.9
CLAP_COOLDOWN_SECONDS = 0.55
POSITION_REPORT_SECONDS = 2.0

# Key codes
KEY_Q = ord("q")
KEY_ESC = 27
KEY_M = ord("m")
KEY_F = ord("f")
KEY_O = ord("o")
KEY_D = ord("d")
KEY_S = ord("s")
KEY_R = ord("r")
KEY_G = ord("g")
KEY_H = ord("h")
KEY_T = ord("t")
KEY_C = ord("c")
KEY_N = ord("n")
KEY_B = ord("b")
KEY_X = ord("x")
KEY_1 = ord("1")
KEY_2 = ord("2")
KEY_3 = ord("3")
KEY_4 = ord("4")
KEY_PLUS = ord("+")
KEY_MINUS = ord("-")


def random_body_color():
    return (random.randint(40, 255), random.randint(40, 255), random.randint(40, 255))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "LookThePerson — Detector de cuerpo completo, manos, cara y objetos "
            "con gestos interactivos. Cross-platform (Windows + Linux)."
        ),
    )
    p.add_argument("--camera", type=int, default=0, help="Indice de la camara (default: 0)")
    p.add_argument("--width", type=int, default=1280, help="Ancho de captura (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="Alto de captura (default: 720)")
    p.add_argument("--fps", type=int, default=30, help="FPS de captura (default: 30)")
    p.add_argument("--windowed", action="store_true", help="Modo ventana (no pantalla completa)")
    p.add_argument("--no-calculator", action="store_true", help="Desactiva el control de calculadora")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Display setup
# ---------------------------------------------------------------------------

def setup_display(platform_bridge, fullscreen=True):
    mx, my, mw, mh = platform_bridge.get_monitor_geometry()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, mx, my)
    cv2.resizeWindow(WINDOW_NAME, mw, mh)
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return mw, mh


# ---------------------------------------------------------------------------
# Key bindings setup
# ---------------------------------------------------------------------------

def setup_key_handler():
    kh = KeyHandler()

    # Toggle actions
    kh.register_toggle(KEY_M, "segmentation", "Mascara segmentacion", default_active=True)
    kh.register_toggle(KEY_F, "face_mesh", "Face Mesh overlay", default_active=False)
    kh.register_toggle(KEY_O, "object_detect", "Deteccion objetos", default_active=False)
    kh.register_toggle(KEY_D, "face_detect", "Deteccion caras", default_active=False)
    kh.register_toggle(KEY_G, "grid", "Cuadricula referencia", default_active=False)
    kh.register_toggle(KEY_H, "help", "Panel de ayuda", default_active=False)
    kh.register_toggle(KEY_T, "telemetry", "Telemetria detallada", default_active=True)
    kh.register_toggle(KEY_N, "night_mode", "Modo nocturno", default_active=False)
    kh.register_toggle(KEY_B, "bounding_boxes", "Bounding boxes", default_active=True)

    # One-shot actions
    kh.register_oneshot(KEY_S, "screenshot", "Captura pantalla")
    kh.register_oneshot(KEY_R, "record_toggle", "Iniciar/parar grabacion")
    kh.register_oneshot(KEY_C, "change_color", "Cambiar color esqueleto")

    # Mode presets
    kh.register_oneshot(KEY_1, "mode_full", "Modo: Completo")
    kh.register_oneshot(KEY_2, "mode_pose", "Modo: Solo Pose")
    kh.register_oneshot(KEY_3, "mode_hands", "Modo: Solo Manos")
    kh.register_oneshot(KEY_4, "mode_face", "Modo: Solo Cara")

    # Confidence adjustment
    kh.register_oneshot(KEY_PLUS, "conf_up", "Confianza +5%")
    kh.register_oneshot(KEY_MINUS, "conf_down", "Confianza -5%")

    return kh


# ---------------------------------------------------------------------------
# Mode presets
# ---------------------------------------------------------------------------

def apply_mode_preset(key_handler, mode_name):
    """Apply a detection mode preset."""
    presets = {
        "mode_full": {
            "segmentation": True, "face_mesh": True,
            "object_detect": True, "face_detect": True,
            "bounding_boxes": True,
        },
        "mode_pose": {
            "segmentation": True, "face_mesh": False,
            "object_detect": False, "face_detect": False,
            "bounding_boxes": True,
        },
        "mode_hands": {
            "segmentation": False, "face_mesh": False,
            "object_detect": False, "face_detect": False,
            "bounding_boxes": False,
        },
        "mode_face": {
            "segmentation": False, "face_mesh": True,
            "object_detect": False, "face_detect": True,
            "bounding_boxes": True,
        },
    }
    preset = presets.get(mode_name)
    if preset:
        for name, active in preset.items():
            key_handler.set_active(name, active)
        print(f"[mode] Preset aplicado: {mode_name}", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Platform
    bridge = get_platform()
    print(f"[platform] Sistema detectado: {type(bridge).__name__}", flush=True)

    # Models
    pose_model = PoseModel()
    hand_model = HandModel()
    face_mesh_model = FaceMeshModel()
    face_detect_model = FaceDetectionModel()
    object_model = ObjectDetectionModel()

    # Always start core models
    pose_model.start()
    hand_model.start()
    face_mesh_model.start()
    face_detect_model.start()
    object_model.start()

    # Systems
    key_handler = setup_key_handler()
    recorder = Recorder()
    app_controller = AppController(bridge, disabled=args.no_calculator)
    face_gesture_detector = FaceGestureDetector()

    # Camera
    backend = bridge.get_camera_backend()
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir la camara {args.camera}.", flush=True)
        return

    screen_w, screen_h = setup_display(bridge, fullscreen=not args.windowed)
    print(
        "Camara activa — Pulsa H para ver controles, Q/Esc para salir.",
        flush=True,
    )

    # State
    body_color = random_body_color()
    last_clap_time = 0.0
    was_clapping = False
    last_position_report = 0.0
    start_time = time.monotonic()
    last_frame_time = start_time
    smoothed_fps = 0.0
    last_gesture_text = ""
    confidence_level = 0.35

    # -----------------------------------------------------------------------
    # Loop
    # -----------------------------------------------------------------------

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.monotonic()
            timestamp_ms = int((now - start_time) * 1000)

            # FPS
            dt = max(now - last_frame_time, 1e-6)
            last_frame_time = now
            ifps = 1.0 / dt
            smoothed_fps = ifps if smoothed_fps == 0.0 else smoothed_fps * FPS_SMOOTHING + ifps * (1 - FPS_SMOOTHING)

            # Convert to MediaPipe image
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            # ----- Pose detection (always on) -----
            pose_result = pose_model.detect(mp_img, timestamp_ms)
            pose_detected = bool(pose_result and pose_result.pose_landmarks)

            center = None
            touching_head = False
            body_gesture_status = ""
            hand_status = ""
            clapped = False

            if pose_detected:
                primary = pose_result.pose_landmarks[0]
                center = PoseModel.body_center(primary, width, height)

                for pi, landmarks in enumerate(pose_result.pose_landmarks):
                    person_touch = hand_touches_top_of_head(landmarks)
                    touching_head = touching_head or person_touch

                    # Segmentation tint
                    if (
                        key_handler.is_active("segmentation")
                        and pose_result.segmentation_masks
                        and pi < len(pose_result.segmentation_masks)
                    ):
                        PoseModel.tint_body(frame, pose_result.segmentation_masks[pi], body_color)

                    # Skeleton
                    PoseModel.draw_skeleton(frame, landmarks, width, height, body_color)

                    # Head circle
                    draw_head_circle(frame, landmarks, width, height, person_touch)

                # Body gestures
                body_gestures = detect_all_body_gestures(primary)

                # Clap -> change color
                is_clapping = body_gestures["clap"]
                if is_clapping and not was_clapping and now - last_clap_time >= CLAP_COOLDOWN_SECONDS:
                    body_color = random_body_color()
                    last_clap_time = now
                    clapped = True
                    last_gesture_text = "APLAUSO"
                was_clapping = is_clapping

                # T-pose
                if body_gestures.get("t_pose"):
                    last_gesture_text = "T-POSE"

                # Squat
                if body_gestures.get("squat"):
                    last_gesture_text = "SQUAT"

                # App control (calculator, YouTube)
                body_gesture_status = app_controller.update_body_gestures(body_gestures, now)

                # Center point
                if key_handler.is_active("telemetry"):
                    draw_center_point(frame, center)
            else:
                was_clapping = False

            # ----- Hand detection (always on) -----
            hand_result = hand_model.detect(mp_img, timestamp_ms)
            hand_count = 0

            if hand_result and hand_result.hand_landmarks:
                hand_count = len(hand_result.hand_landmarks)
                hand_info = detect_all_hand_gestures(hand_result.hand_landmarks)

                for hlm in hand_result.hand_landmarks:
                    HandModel.draw_skeleton(frame, hlm, width, height)

                # Hand gesture names
                for g in hand_info.get("gestures", []):
                    if g and g not in ("fist", "open_palm"):
                        last_gesture_text = g.upper()

                hand_status = app_controller.update_hand_input(hand_info, now)

            # ----- Face Mesh (toggleable) -----
            face_count = 0
            expressions = {}

            if key_handler.is_active("face_mesh"):
                fm_result = face_mesh_model.detect(mp_img, timestamp_ms)
                if fm_result and fm_result.face_landmarks:
                    face_count = len(fm_result.face_landmarks)
                    for fl in fm_result.face_landmarks:
                        FaceMeshModel.draw_mesh(frame, fl, width, height)
                        FaceMeshModel.draw_gaze_indicator(frame, fl, width, height)

                        # Expression detection
                        face_info = face_gesture_detector.update(fl)
                        expressions = face_info.get("raw", {})
                        for triggered in face_info.get("triggered", []):
                            last_gesture_text = f"FACE: {triggered.upper()}"

            # ----- Face Detection (toggleable) -----
            if key_handler.is_active("face_detect"):
                fd_result = face_detect_model.detect(mp_img, timestamp_ms)
                if fd_result:
                    FaceDetectionModel.draw_detections(frame, fd_result, width, height)
                    if fd_result.detections:
                        face_count = max(face_count, len(fd_result.detections))

            # ----- Object Detection (toggleable) -----
            object_count = 0
            if key_handler.is_active("object_detect"):
                obj_result = object_model.detect(mp_img, timestamp_ms)
                if obj_result and obj_result.detections:
                    object_count = len(obj_result.detections)
                    ObjectDetectionModel.draw_detections(frame, obj_result, width, height)

            # ----- Grid overlay -----
            if key_handler.is_active("grid"):
                draw_grid(frame)

            # ----- Night mode -----
            if key_handler.is_active("night_mode"):
                frame = apply_night_mode(frame)

            # ----- Build status text -----
            if touching_head:
                status_text = "TOCANDO CABEZA"
            elif body_gesture_status:
                status_text = body_gesture_status
            elif hand_status:
                status_text = hand_status
            elif clapped:
                status_text = "APLAUSO!"
            elif pose_detected:
                status_text = "CUERPO DETECTADO"
            else:
                status_text = "BUSCANDO CUERPO..."

            # ----- HUD -----
            active_models = []
            if pose_detected:
                active_models.append("POSE")
            if hand_count > 0:
                active_models.append("HANDS")
            if face_count > 0 and key_handler.is_active("face_mesh"):
                active_models.append("FACE")
            if key_handler.is_active("object_detect"):
                active_models.append("OBJECTS")
            if key_handler.is_active("face_detect"):
                active_models.append("FACE-DET")

            stats = {
                "fps": smoothed_fps,
                "status_text": status_text,
                "pose_count": len(pose_result.pose_landmarks) if pose_detected else 0,
                "hand_count": hand_count,
                "face_count": face_count,
                "object_count": object_count,
                "recording": recorder.is_recording,
                "last_gesture": last_gesture_text,
                "body_center": center,
                "active_models": active_models,
                "expressions": expressions,
            }

            show_help = key_handler.is_active("help")
            if key_handler.is_active("telemetry"):
                draw_hud_panel(frame, key_handler, stats, show_help=show_help)
            elif show_help:
                draw_hud_panel(frame, key_handler, stats, show_help=True)
            else:
                # Minimal HUD
                cv2.putText(frame, status_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                fps_text = f"{smoothed_fps:.0f} FPS"
                fs, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.putText(frame, fps_text, (max(12, width - fs[0] - 12), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            # ----- Recording -----
            if recorder.is_recording:
                recorder.write_frame(frame)

            # ----- Position report -----
            if now - last_position_report >= POSITION_REPORT_SECONDS:
                if center:
                    print(f"position: {center[0]},{center[1]}", flush=True)
                else:
                    print("position: 0", flush=True)
                last_position_report = now

            # ----- Display -----
            fullscreen_frame = fit_frame_to_screen(frame, screen_w, screen_h)
            cv2.imshow(WINDOW_NAME, fullscreen_frame)

            # ----- Key handling -----
            key = cv2.waitKey(1) & 0xFF

            if key in (KEY_Q, KEY_ESC):
                break

            if key == KEY_X:
                locked = app_controller.toggle_lock()
                print(f"calc: {'bloqueada' if locked else 'permitida'}", flush=True)
                continue

            result = key_handler.process_key(key)
            if result:
                action_name, state = result

                if action_name == "screenshot":
                    recorder.take_screenshot(frame)

                elif action_name == "record_toggle":
                    recorder.toggle_recording(frame)

                elif action_name == "change_color":
                    body_color = random_body_color()

                elif action_name.startswith("mode_"):
                    apply_mode_preset(key_handler, action_name)

                elif action_name == "conf_up":
                    confidence_level = min(0.95, confidence_level + 0.05)
                    object_model.restart_with_confidence(confidence_level)
                    print(f"[config] Confianza: {confidence_level:.0%}", flush=True)

                elif action_name == "conf_down":
                    confidence_level = max(0.1, confidence_level - 0.05)
                    object_model.restart_with_confidence(confidence_level)
                    print(f"[config] Confianza: {confidence_level:.0%}", flush=True)

                elif action_name in ("face_mesh", "face_detect", "object_detect"):
                    print(f"[toggle] {action_name}: {'ON' if state else 'OFF'}", flush=True)

    finally:
        recorder.cleanup()
        pose_model.stop()
        hand_model.stop()
        face_mesh_model.stop()
        face_detect_model.stop()
        object_model.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Salida solicitada por el usuario (KeyboardInterrupt). Cerrando...")
