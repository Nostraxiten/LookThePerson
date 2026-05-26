"""
Full Body Detector - Nox/@nostraxiten
Compatible con mediapipe >= 0.10 (Tasks API)

Requisitos:
    pip install opencv-python mediapipe

La primera vez descarga los modelos de cuerpo y manos automaticamente.
"""

import argparse
import ctypes
import os
import random
import subprocess
import time
import unicodedata
import urllib.request
import webbrowser
from ctypes import wintypes

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR

MODEL_PATH_POSE = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL_POSE = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
MODEL_PATH_HAND = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL_HAND = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

WINDOW_NAME = "Full Body Detector - @nostraxiten"


def download_model(model_path, model_url, model_name):
    if not os.path.exists(model_path):
        print(f"Descargando modelo MediaPipe {model_name}...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Modelo {model_name} descargado.")


BODY_COLOR = (0, 220, 255)
POINT_COLOR = (0, 255, 80)
TEXT_COLOR = (255, 255, 255)
MIN_VISIBILITY = 0.35
GESTURE_VISIBILITY = 0.2
CLAP_DISTANCE = 0.11
CLAP_COOLDOWN_SECONDS = 0.55
MASK_ALPHA = 0.55
MASK_THRESHOLD = 0.35
POSITION_REPORT_SECONDS = 2.0
CENTER_POINT_COLOR = (255, 0, 255)
FPS_SMOOTHING = 0.9
ARMS_OPEN_SECONDS = 0.45
ARMS_CLOSED_SECONDS = 0.45
ARMS_OPEN_MIN_SHOULDER_RATIO = 1.75
ARMS_OPEN_OUTSIDE_SHOULDER_MARGIN = 0.25
ARMS_OPEN_MAX_VERTICAL_DIFF = 0.38
ARMS_CLOSED_MAX_WRIST_DISTANCE = 0.24
ARMS_CLOSED_TORSO_MARGIN = 0.08
CALCULATOR_TITLES = ("calculadora", "calculator")
CALCULATOR_WINDOW_WIDTH = 420
CALCULATOR_WINDOW_HEIGHT = 620
CALCULATOR_INPUT_STABLE_SECONDS = 0.45
CALCULATOR_INPUT_COOLDOWN_SECONDS = 0.8
CALCULATOR_CLEAR_STABLE_SECONDS = 0.35
CALCULATOR_CLEAR_COOLDOWN_SECONDS = 1.0
YOUTUBE_URL = "https://www.youtube.com"
YOUTUBE_OPEN_SECONDS = 0.45
YOUTUBE_OPEN_COOLDOWN_SECONDS = 2.0

HAND_COLOR = (255, 180, 0)
HEAD_COLOR = (0, 255, 0)
HEAD_TOUCH_COLOR = (0, 180, 255)
MAX_POSES = 4

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


class Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", Rect),
        ("rcWork", Rect),
        ("dwFlags", ctypes.c_ulong),
    ]


def get_monitor_geometry():
    monitors = []
    collect_monitors(monitors)

    if not monitors:
        return 0, 0, 1280, 720

    monitors.sort(key=lambda item: (item[0], item[1]))
    return monitors[1] if len(monitors) > 1 else monitors[0]


def collect_monitors(monitors):
    monitor_enum_proc_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(Rect),
        wintypes.LPARAM,
    )

    def callback(monitor, _dc, _rect, _data):
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(MonitorInfo)
        ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info))
        rect = info.rcMonitor
        monitors.append((
            rect.left,
            rect.top,
            rect.right - rect.left,
            rect.bottom - rect.top,
        ))
        return True

    try:
        monitor_enum_proc = monitor_enum_proc_type(callback)
        ctypes.windll.user32.EnumDisplayMonitors(
            0,
            0,
            monitor_enum_proc,
            0,
        )
    except Exception:
        monitors.clear()


def get_upper_monitor_geometry():
    monitors = []
    collect_monitors(monitors)

    if not monitors:
        return 0, 0, 1280, 720

    monitors.sort(key=lambda item: (item[1], item[0]))
    return monitors[0]


def setup_display_window(fullscreen=True):
    monitor_x, monitor_y, monitor_width, monitor_height = get_monitor_geometry()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, monitor_x, monitor_y)
    cv2.resizeWindow(WINDOW_NAME, monitor_width, monitor_height)
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return monitor_width, monitor_height


def move_window_to_upper_monitor(hwnd, width=None, height=None):
    monitor_x, monitor_y, monitor_width, monitor_height = get_upper_monitor_geometry()
    window_width = width or monitor_width
    window_height = height or monitor_height
    window_x = monitor_x + max(0, (monitor_width - window_width) // 2)
    window_y = monitor_y + max(0, (monitor_height - window_height) // 2)

    ctypes.windll.user32.ShowWindow(hwnd, 1)
    ctypes.windll.user32.MoveWindow(
        hwnd,
        window_x,
        window_y,
        window_width,
        window_height,
        True,
    )
    ctypes.windll.user32.ShowWindow(hwnd, 1)


def move_windows_by_title(title_parts):
    title_parts = [normalize_text(part) for part in title_parts if part]
    moved = False

    enum_windows_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _data):
        nonlocal moved
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True

        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = normalize_text(buffer.value)

        if title and any(part in title for part in title_parts):
            move_window_to_upper_monitor(
                hwnd,
                CALCULATOR_WINDOW_WIDTH,
                CALCULATOR_WINDOW_HEIGHT,
            )
            moved = True

        return True

    ctypes.windll.user32.EnumWindows(enum_windows_proc_type(callback), 0)
    return moved


def find_window_by_title(title_parts):
    title_parts = [normalize_text(part) for part in title_parts if part]
    found_hwnd = None
    enum_windows_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _data):
        nonlocal found_hwnd
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True

        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = normalize_text(buffer.value)

        if title and any(part in title for part in title_parts):
            found_hwnd = hwnd
            return False

        return True

    ctypes.windll.user32.EnumWindows(enum_windows_proc_type(callback), 0)
    return found_hwnd


def move_windows_by_pid(pid):
    moved = False
    enum_windows_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _data):
        nonlocal moved
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True

        window_pid = wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid:
            move_window_to_upper_monitor(
                hwnd,
                CALCULATOR_WINDOW_WIDTH,
                CALCULATOR_WINDOW_HEIGHT,
            )
            moved = True

        return True

    ctypes.windll.user32.EnumWindows(enum_windows_proc_type(callback), 0)
    return moved


def close_windows_by_title(title_parts):
    title_parts = [normalize_text(part) for part in title_parts if part]
    closed = False
    wm_close = 0x0010
    enum_windows_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _data):
        nonlocal closed
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True

        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = normalize_text(buffer.value)

        if title and any(part in title for part in title_parts):
            ctypes.windll.user32.PostMessageW(hwnd, wm_close, 0, 0)
            closed = True

        return True

    ctypes.windll.user32.EnumWindows(enum_windows_proc_type(callback), 0)
    return closed


def fit_frame_to_screen(frame, screen_width, screen_height):
    frame_height, frame_width, _ = frame.shape
    scale = min(screen_width / frame_width, screen_height / frame_height)
    resized_width = max(1, int(frame_width * scale))
    resized_height = max(1, int(frame_height * scale))

    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    x = (screen_width - resized_width) // 2
    y = (screen_height - resized_height) // 2
    output[y:y + resized_height, x:x + resized_width] = resized
    return output


def random_body_color():
    return (
        random.randint(40, 255),
        random.randint(40, 255),
        random.randint(40, 255),
    )


def normalize_text(text):
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(text.split())


def open_calculator():
    try:
        return subprocess.Popen(["calc.exe"])
    except OSError as exc:
        print(f"gesture: no pude abrir la calculadora: {exc}", flush=True)
        return None


def open_youtube():
    try:
        webbrowser.open_new_tab(YOUTUBE_URL)
        print("gesture: youtube abierta", flush=True)
    except Exception as exc:
        print(f"gesture: no pude abrir YouTube: {exc}", flush=True)


def both_hands_raised(landmarks):
    required = (
        landmarks[13], landmarks[14], landmarks[15], landmarks[16]
    )
    if not all(landmark_is_visible_for_gesture(landmark) for landmark in required):
        return False

    circle = head_circle(landmarks, 1, 1)
    if circle is None:
        return False

    _, center_y, radius, _norm_x, _norm_y, _norm_radius = circle
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]

    return (
        left_wrist.y < left_elbow.y
        and right_wrist.y < right_elbow.y
        and left_wrist.y <= center_y - radius * 0.15
        and right_wrist.y <= center_y - radius * 0.15
    )


def close_calculator(calculator_process):
    closed = close_windows_by_title(CALCULATOR_TITLES)
    if not closed and calculator_process and calculator_process.poll() is None:
        calculator_process.terminate()


def send_key_to_calculator(key):
    hwnd = find_window_by_title(CALCULATOR_TITLES)
    if not hwnd:
        return False

    vk = 0x6B if key == "+" else ord(key)
    keyeventf_keyup = 0x0002
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk, 0, keyeventf_keyup, 0)
    return True


def clear_calculator():
    hwnd = find_window_by_title(CALCULATOR_TITLES)
    if not hwnd:
        return False

    vk_escape = 0x1B
    keyeventf_keyup = 0x0002
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    ctypes.windll.user32.keybd_event(vk_escape, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_escape, 0, keyeventf_keyup, 0)
    return True


def landmark_is_visible(landmark):
    return getattr(landmark, "visibility", 1.0) >= MIN_VISIBILITY


def landmark_is_visible_for_gesture(landmark):
    return getattr(landmark, "visibility", 1.0) >= GESTURE_VISIBILITY


def wrists_are_clapping(landmarks):
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    if not landmark_is_visible(left_wrist) or not landmark_is_visible(right_wrist):
        return False

    dx = left_wrist.x - right_wrist.x
    dy = left_wrist.y - right_wrist.y
    distance = (dx * dx + dy * dy) ** 0.5
    return distance <= CLAP_DISTANCE


def arms_are_open(landmarks):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    required = (left_shoulder, right_shoulder, left_wrist, right_wrist)
    if not all(landmark_is_visible_for_gesture(landmark) for landmark in required):
        return False

    wrist_min_x = min(left_wrist.x, right_wrist.x)
    wrist_max_x = max(left_wrist.x, right_wrist.x)
    shoulder_min_x = min(left_shoulder.x, right_shoulder.x)
    shoulder_max_x = max(left_shoulder.x, right_shoulder.x)
    shoulder_distance = max(shoulder_max_x - shoulder_min_x, 0.01)
    wrist_distance = wrist_max_x - wrist_min_x
    wrist_vertical_diff = abs(left_wrist.y - right_wrist.y)
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    outside_margin = shoulder_distance * ARMS_OPEN_OUTSIDE_SHOULDER_MARGIN
    wrists_outside_shoulders = (
        wrist_min_x <= shoulder_min_x - outside_margin
        and wrist_max_x >= shoulder_max_x + outside_margin
    )
    wrists_near_shoulders = (
        abs(left_wrist.y - shoulder_center_y) <= ARMS_OPEN_MAX_VERTICAL_DIFF
        and abs(right_wrist.y - shoulder_center_y) <= ARMS_OPEN_MAX_VERTICAL_DIFF
    )

    return (
        wrists_outside_shoulders
        and wrist_distance / shoulder_distance >= ARMS_OPEN_MIN_SHOULDER_RATIO
        and wrist_vertical_diff <= ARMS_OPEN_MAX_VERTICAL_DIFF
        and wrists_near_shoulders
    )


def arms_are_closed(landmarks):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    required = (left_shoulder, right_shoulder, left_wrist, right_wrist)
    if not all(landmark_is_visible(landmark) for landmark in required):
        return False

    dx = left_wrist.x - right_wrist.x
    dy = left_wrist.y - right_wrist.y
    distance = (dx * dx + dy * dy) ** 0.5

    shoulder_min_x = min(left_shoulder.x, right_shoulder.x)
    shoulder_max_x = max(left_shoulder.x, right_shoulder.x)
    shoulder_width = max(shoulder_max_x - shoulder_min_x, 0.01)
    inner_min_x = shoulder_min_x + shoulder_width * ARMS_CLOSED_TORSO_MARGIN
    inner_max_x = shoulder_max_x - shoulder_width * ARMS_CLOSED_TORSO_MARGIN
    wrists_inside_torso = (
        inner_min_x <= left_wrist.x <= inner_max_x
        and inner_min_x <= right_wrist.x <= inner_max_x
    )

    return distance <= ARMS_CLOSED_MAX_WRIST_DISTANCE and wrists_inside_torso


def distance_2d(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def count_extended_fingers(hand_landmarks):
    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    extended = 0

    if distance_2d(thumb_tip, wrist) > distance_2d(thumb_ip, wrist) + 0.035:
        extended += 1

    for tip_id, pip_id in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if hand_landmarks[tip_id].y < hand_landmarks[pip_id].y - 0.02:
            extended += 1

    return extended


def hand_calculator_gesture(hand_landmarks):
    finger_count = count_extended_fingers(hand_landmarks)
    if finger_count == 0:
        return "+"
    if 1 <= finger_count <= 4:
        return str(finger_count)
    if finger_count == 5:
        return "+"
    return None


def draw_hand_skeleton(frame, hand_landmarks, width, height):
    points = [(int(point.x * width), int(point.y * height)) for point in hand_landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], HAND_COLOR, 2)

    for point in points:
        cv2.circle(frame, point, 4, HAND_COLOR, -1)


def body_center(landmarks, width, height):
    center_landmark_ids = (11, 12, 23, 24)
    visible_points = []

    for landmark_id in center_landmark_ids:
        landmark = landmarks[landmark_id]
        if landmark_is_visible(landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            if 0 <= x < width and 0 <= y < height:
                visible_points.append((x, y))

    if not visible_points:
        return None

    center_x = sum(point[0] for point in visible_points) // len(visible_points)
    center_y = sum(point[1] for point in visible_points) // len(visible_points)
    return center_x, center_y


def head_circle(landmarks, width, height):
    head_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    points = []

    for landmark_id in head_ids:
        landmark = landmarks[landmark_id]
        if landmark_is_visible_for_gesture(landmark):
            points.append((landmark.x, landmark.y))

    if len(points) < 2:
        return None

    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    radius = max(
        ((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2) ** 0.5
        for point in points
    )

    if landmark_is_visible_for_gesture(landmarks[11]) and landmark_is_visible_for_gesture(landmarks[12]):
        shoulder_width = abs(landmarks[11].x - landmarks[12].x)
        radius = max(radius, shoulder_width * 0.28)

    radius = min(max(radius * 1.35, 0.045), 0.16)
    center_y -= radius * 0.15

    return (
        int(center_x * width),
        int(center_y * height),
        int(radius * max(width, height)),
        center_x,
        center_y,
        radius,
    )


def hand_touches_top_of_head(landmarks):
    circle = head_circle(landmarks, 1, 1)
    if circle is None:
        return False

    _px, _py, _pr, center_x, center_y, radius = circle
    for wrist_id in (15, 16):
        wrist = landmarks[wrist_id]
        if not landmark_is_visible_for_gesture(wrist):
            continue

        dx = wrist.x - center_x
        dy = wrist.y - center_y
        inside_circle = dx * dx + dy * dy <= radius * radius
        in_top_zone = wrist.y <= center_y + radius * 0.15
        if inside_circle and in_top_zone:
            return True

    return False


def draw_head_circle(frame, landmarks, width, height, touching):
    circle = head_circle(landmarks, width, height)
    if circle is None:
        return

    center_x, center_y, radius, _norm_x, _norm_y, _norm_radius = circle
    color = HEAD_TOUCH_COLOR if touching else HEAD_COLOR
    cv2.circle(frame, (center_x, center_y), radius, color, 2)
    cv2.line(
        frame,
        (center_x - radius, int(center_y + radius * 0.15)),
        (center_x + radius, int(center_y + radius * 0.15)),
        color,
        1,
    )


def tint_body(frame, segmentation_mask, color):
    mask = segmentation_mask.numpy_view()
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    body_mask = mask > MASK_THRESHOLD
    color_layer = np.zeros_like(frame)
    color_layer[:] = color
    tinted = cv2.addWeighted(frame, 1.0 - MASK_ALPHA, color_layer, MASK_ALPHA, 0)
    frame[body_mask] = tinted[body_mask]


def draw_body_skeleton(frame, landmarks, width, height, color):
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        visible = landmark_is_visible(landmark)
        in_frame = 0 <= x < width and 0 <= y < height
        points.append((x, y, visible and in_frame))

    for start, end in POSE_CONNECTIONS:
        x1, y1, visible_start = points[start]
        x2, y2, visible_end = points[end]
        if visible_start and visible_end:
            cv2.line(frame, (x1, y1), (x2, y2), color, 3)

    for x, y, visible in points:
        if visible:
            cv2.circle(frame, (x, y), 5, POINT_COLOR, -1)
            cv2.circle(frame, (x, y), 7, (20, 20, 20), 1)


def draw_center_point(frame, center):
    if center is None:
        return

    cv2.circle(frame, center, 9, CENTER_POINT_COLOR, -1)
    cv2.circle(frame, center, 13, (20, 20, 20), 2)
    cv2.putText(
        frame,
        f"{center[0]}, {center[1]}",
        (center[0] + 14, center[1] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        CENTER_POINT_COLOR,
        2,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detector de cuerpo y manos con gestos para calculadora.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="indice de la camara que se va a usar",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="ancho solicitado para la camara",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="alto solicitado para la camara",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS solicitados para la camara",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="abre la vista en ventana en vez de pantalla completa",
    )
    parser.add_argument(
        "--no-calculator",
        action="store_true",
        help="desactiva los gestos que abren o controlan la calculadora",
    )
    return parser.parse_args()


def ensure_models():
    download_model(MODEL_PATH_POSE, MODEL_URL_POSE, "Pose Landmarker")
    download_model(MODEL_PATH_HAND, MODEL_URL_HAND, "Hand Landmarker")


def main():
    args = parse_args()
    ensure_models()

    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH_POSE),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_segmentation_masks=True,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir la camara {args.camera}.")
        return

    screen_width, screen_height = setup_display_window(fullscreen=not args.windowed)
    print("Camara activa - pulsa Q o Esc para salir. X bloquea/desbloquea la calculadora.")

    body_color = random_body_color()
    was_clapping = False
    last_clap_time = 0.0
    last_position_report_time = 0.0
    calculator_process = None
    calculator_is_open = False
    calculator_move_attempts = 0
    arms_open_since = None
    arms_closed_since = None
    both_hands_up_since = None
    last_youtube_open_time = 0.0
    current_hand_gesture = None
    hand_gesture_since = 0.0
    last_sent_hand_gesture = None
    last_sent_hand_gesture_time = 0.0
    both_hands_open_since = None
    last_clear_time = 0.0
    calculator_controls_locked = args.no_calculator
    start_time = time.monotonic()
    last_frame_time = start_time
    smoothed_fps = 0.0

    with (
        vision.PoseLandmarker.create_from_options(pose_options) as pose_detector,
        vision.HandLandmarker.create_from_options(hand_options) as hand_detector,
    ):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            timestamp_ms = int((time.monotonic() - start_time) * 1000)

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            pose_result = pose_detector.detect_for_video(mp_img, timestamp_ms)
            hand_result = hand_detector.detect_for_video(mp_img, timestamp_ms)
            pose_detected = bool(pose_result and pose_result.pose_landmarks)
            clapped_now = False
            center = None
            touching_head = False
            calculator_gesture_text = ""
            now = time.monotonic()
            frame_seconds = max(now - last_frame_time, 1e-6)
            last_frame_time = now
            instant_fps = 1.0 / frame_seconds
            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0.0
                else smoothed_fps * FPS_SMOOTHING + instant_fps * (1.0 - FPS_SMOOTHING)
            )

            if pose_detected:
                primary_landmarks = pose_result.pose_landmarks[0]
                center = body_center(primary_landmarks, width, height)

                for pose_index, landmarks in enumerate(pose_result.pose_landmarks):
                    person_touches_head = hand_touches_top_of_head(landmarks)
                    touching_head = touching_head or person_touches_head

                    if pose_result.segmentation_masks and pose_index < len(pose_result.segmentation_masks):
                        tint_body(frame, pose_result.segmentation_masks[pose_index], body_color)

                    draw_body_skeleton(frame, landmarks, width, height, body_color)
                    draw_head_circle(frame, landmarks, width, height, person_touches_head)

                is_clapping = wrists_are_clapping(primary_landmarks)
                arms_open = arms_are_open(primary_landmarks)
                arms_closed = arms_are_closed(primary_landmarks)

                if is_clapping and not was_clapping and now - last_clap_time >= CLAP_COOLDOWN_SECONDS:
                    body_color = random_body_color()
                    last_clap_time = now
                    clapped_now = True

                was_clapping = is_clapping

                if arms_open:
                    arms_open_since = arms_open_since or now
                    arms_closed_since = None
                    calculator_gesture_text = "BRAZOS ABIERTOS..."
                elif arms_closed:
                    arms_closed_since = arms_closed_since or now
                    arms_open_since = None
                    if calculator_is_open:
                        calculator_gesture_text = "CERRANDO..."
                else:
                    arms_open_since = None
                    arms_closed_since = None

                both_hands_up = both_hands_raised(primary_landmarks)
                if both_hands_up:
                    if both_hands_up_since is None:
                        both_hands_up_since = now
                    calculator_gesture_text = "YOUTUBE..."
                else:
                    both_hands_up_since = None

                if (
                    both_hands_up_since
                    and now - both_hands_up_since >= YOUTUBE_OPEN_SECONDS
                    and now - last_youtube_open_time >= YOUTUBE_OPEN_COOLDOWN_SECONDS
                ):
                    open_youtube()
                    last_youtube_open_time = now
                    calculator_gesture_text = "YOUTUBE ABIERTA"
                    both_hands_up_since = None

                if (
                    not calculator_controls_locked
                    and arms_open_since
                    and not calculator_is_open
                    and now - arms_open_since >= ARMS_OPEN_SECONDS
                ):
                    calculator_process = open_calculator()
                    if calculator_process is not None:
                        calculator_is_open = True
                        calculator_move_attempts = 20
                        calculator_gesture_text = "CALCULADORA ABIERTA"
                        print("gesture: calculadora abierta", flush=True)

                if (
                    not calculator_controls_locked
                    and arms_closed_since
                    and calculator_is_open
                    and now - arms_closed_since >= ARMS_CLOSED_SECONDS
                ):
                    close_calculator(calculator_process)
                    calculator_process = None
                    calculator_is_open = False
                    calculator_move_attempts = 0
                    calculator_gesture_text = "CALCULADORA CERRADA"
                    print("gesture: calculadora cerrada", flush=True)

                draw_center_point(frame, center)
            else:
                was_clapping = False
                arms_open_since = None
                arms_closed_since = None

            hand_gesture = None
            if hand_result and hand_result.hand_landmarks:
                finger_counts = []
                for hand_landmarks in hand_result.hand_landmarks:
                    finger_counts.append(count_extended_fingers(hand_landmarks))
                    draw_hand_skeleton(frame, hand_landmarks, width, height)

                if len(finger_counts) >= 2 and all(count == 5 for count in finger_counts[:2]):
                    both_hands_open_since = both_hands_open_since or now
                    hand_gesture = None
                else:
                    both_hands_open_since = None
                    hand_gesture = hand_calculator_gesture(hand_result.hand_landmarks[0])
            else:
                both_hands_open_since = None

            if hand_gesture != current_hand_gesture:
                current_hand_gesture = hand_gesture
                hand_gesture_since = now

            stable_hand_gesture = (
                current_hand_gesture
                and now - hand_gesture_since >= CALCULATOR_INPUT_STABLE_SECONDS
            )
            calculator_available = (
                not args.no_calculator
                and (calculator_is_open or bool(find_window_by_title(CALCULATOR_TITLES)))
            )
            if calculator_is_open and not calculator_available:
                calculator_is_open = False
                calculator_process = None
                calculator_move_attempts = 0
            clear_is_stable = (
                both_hands_open_since
                and now - both_hands_open_since >= CALCULATOR_CLEAR_STABLE_SECONDS
                and now - last_clear_time >= CALCULATOR_CLEAR_COOLDOWN_SECONDS
            )

            if (
                not calculator_controls_locked
                and calculator_available
                and clear_is_stable
                and clear_calculator()
            ):
                last_clear_time = now
                both_hands_open_since = None
                current_hand_gesture = None
                last_sent_hand_gesture = None
                calculator_gesture_text = "CALC: BORRAR"
                print("gesture: calc borrar", flush=True)

            can_send_hand_gesture = (
                not calculator_controls_locked
                and stable_hand_gesture
                and calculator_available
                and not clear_is_stable
                and (
                    current_hand_gesture != last_sent_hand_gesture
                    or now - last_sent_hand_gesture_time >= CALCULATOR_INPUT_COOLDOWN_SECONDS
                )
            )

            if can_send_hand_gesture and send_key_to_calculator(current_hand_gesture):
                last_sent_hand_gesture = current_hand_gesture
                last_sent_hand_gesture_time = now
                calculator_gesture_text = f"CALC: {current_hand_gesture}"
                print(f"gesture: calc {current_hand_gesture}", flush=True)

            if not args.no_calculator and calculator_move_attempts > 0:
                moved_by_pid = (
                    calculator_process
                    and move_windows_by_pid(calculator_process.pid)
                )
                moved_by_title = move_windows_by_title(CALCULATOR_TITLES)
                calculator_move_attempts = 0 if moved_by_pid or moved_by_title else calculator_move_attempts - 1

            now = time.monotonic()
            if now - last_position_report_time >= POSITION_REPORT_SECONDS:
                if center is None:
                    print("position: 0", flush=True)
                else:
                    print(f"position: {center[0]},{center[1]}", flush=True)
                last_position_report_time = now

            if touching_head:
                status_text = "TOCANDO CABEZA"
            elif args.no_calculator:
                status_text = "CALC DESACTIVADA"
            elif calculator_controls_locked:
                status_text = "CALC BLOQUEADA"
            else:
                status_text = calculator_gesture_text or (
                    "APLAUSO!" if clapped_now else (
                        "CUERPO DETECTADO" if pose_detected else "BUSCANDO CUERPO..."
                    )
                )
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                HEAD_TOUCH_COLOR if touching_head else TEXT_COLOR,
                2,
            )
            cv2.putText(
                frame,
                "Q/Esc = salir | X = bloquear calc",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                1,
            )
            fps_text = f"{smoothed_fps:.0f} FPS"
            fps_size, _baseline = cv2.getTextSize(
                fps_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1,
            )
            cv2.putText(
                frame,
                fps_text,
                (max(10, width - fps_size[0] - 10), 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                1,
            )

            fullscreen_frame = fit_frame_to_screen(frame, screen_width, screen_height)
            cv2.imshow(WINDOW_NAME, fullscreen_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("x"), ord("X")):
                if args.no_calculator:
                    print("calc: desactivada por --no-calculator", flush=True)
                    continue
                calculator_controls_locked = not calculator_controls_locked
                state = "bloqueada" if calculator_controls_locked else "permitida"
                print(f"calc: {state}", flush=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
