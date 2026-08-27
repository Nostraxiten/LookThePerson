"""
Application pipeline for LookThePerson.

:class:`Application` owns the frame loop and wires the subsystems together:
camera -> models -> analytics -> active mode -> overlays -> HUD -> display.

The loop itself contains no feature logic. Everything a user can see or
trigger lives in a mode, an action or an analytics class, which is what makes
the app extensible without this file growing.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from actions.bindings import ActionRegistry, GestureBindings
from actions.key_handler import KeyHandler
from analytics.angles import compute_joint_angles
from analytics.motion import MotionAnalyzer
from analytics.session import SessionRecorder
from core.config import Config, save_config
from core.events import Events
from core.state import AppState, FrameContext
from core.theme import next_theme
from gestures.body_gestures import detect_every_gesture
from gestures.face_gestures import FaceGestureDetector
from gestures.hand_gestures import HandGestureTracker, detect_all_hand_gestures
from io_utils.camera import CameraSource
from io_utils.capture import MediaRecorder
from models.face_mesh import FaceMeshModel
from models.hands import HandModel
from models.manager import ModelManager
from models.pose import PoseModel
from modes import build_mode_manager
from platforms import get_platform
from ui.hud import HUD
from ui.renderer import (
    apply_night_mode,
    draw_bounding_boxes,
    draw_grid,
    draw_head_circle,
    fit_frame_to_screen,
)
from ui.widgets import ToastManager

__all__ = ["Application", "WINDOW_NAME"]

WINDOW_NAME = "LookThePerson - @nostraxiten"

# Global key bindings. Mode-specific keys are consulted first, so a mode can
# claim any of these for itself.
KEYS = {
    "quit": (ord("q"), 27),
    "help": ord("h"),
    "telemetry": ord("t"),
    "picker": 9,               # TAB
    "next_mode": ord("]"),
    "prev_mode": ord("["),
    "screenshot": ord("s"),
    "record": ord("r"),
    "gif": ord("g"),
    "burst": ord("B"),
    "theme": ord("v"),
    "grid": ord("#"),
    "night": ord("n"),
    "segmentation": ord("m"),
    "face_mesh": ord("f"),
    "face_detect": ord("d"),
    "objects": ord("o"),
    "skeleton": ord("k"),
    "mirror": ord("w"),
    "fps_graph": ord("y"),
    "debug": ord("`"),
    "pause": ord("p"),
    "save_config": ord("S"),
    "conf_up": ord("+"),
    "conf_down": ord("-"),
    "reset_mode": ord("0"),
}

# Number keys jump straight to these modes.
QUICK_MODES = {
    ord("1"): "full", ord("2"): "pose", ord("3"): "hands", ord("4"): "face",
    ord("5"): "objects", ord("6"): "reps", ord("7"): "posture",
    ord("8"): "silhouette", ord("9"): "debug",
}


class Application:
    """The complete application: state, subsystems and the frame loop."""

    def __init__(self, config: Config):
        self.config = config
        self.state = AppState(config)
        self.bridge = get_platform()
        self.state.note("platform_bridge", self.bridge)

        self.models = ModelManager(config, profiler=self.state.profiler)
        self.camera = CameraSource(
            index=config.camera.index,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            backend=self.bridge.get_camera_backend(),
            mirror=config.camera.mirror,
            buffer_size=config.camera.buffer_size,
            source=config.camera.source,
            loop=config.camera.loop_source,
            auto_reconnect=config.camera.auto_reconnect,
            reconnect_delay=config.camera.reconnect_delay,
            max_reconnect_attempts=config.camera.max_reconnect_attempts,
        )
        self.recorder = MediaRecorder(
            output_dir=config.recording.output_dir,
            video_fps=config.recording.video_fps,
            codec=config.recording.video_codec,
            extension=config.recording.video_extension,
            screenshot_format=config.recording.screenshot_format,
        )
        self.session = SessionRecorder(
            output_dir=config.analytics.export_dir,
            enabled=config.analytics.session_log,
        )

        self.toasts = ToastManager()
        self.hud = HUD(self.toasts)
        self.keys = KeyHandler()
        self.actions = ActionRegistry()
        self.bindings = GestureBindings.from_config(self.actions, config.gestures)

        self.motion = MotionAnalyzer()
        self.face_gestures = FaceGestureDetector()
        self.hand_gestures = HandGestureTracker(
            stable_seconds=config.gestures.stable_seconds,
            cooldown_seconds=config.gestures.cooldown_seconds,
        )

        self.modes = build_mode_manager(self.state, config.mode)
        self._register_actions()
        self._register_keys()
        self._subscribe_events()

        self.screen_size: Tuple[int, int] = (config.camera.width, config.camera.height)
        self._start_time = time.monotonic()
        self._last_frame_time = self._start_time
        self._hip_baseline: Optional[float] = None
        self._window_ready = False

    # -----------------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------------

    def _register_actions(self) -> None:
        """Register everything a gesture or key can trigger."""
        register = self.actions.register

        register("screenshot", "Captura de pantalla",
                 lambda state, **_: self._take_screenshot(), cooldown=1.0, category="captura")
        register("start_recording", "Iniciar grabacion",
                 lambda state, **_: self._start_recording(), cooldown=2.0, category="captura")
        register("stop_recording", "Parar grabacion",
                 lambda state, **_: self._stop_recording(), cooldown=2.0, category="captura")
        register("capture_gif", "Grabar GIF",
                 lambda state, **_: self.recorder.start_gif(), cooldown=3.0, category="captura")

        register("change_color", "Cambiar color del esqueleto",
                 lambda state, **_: self._random_color(), cooldown=0.6, category="visual")
        register("next_theme", "Siguiente tema",
                 lambda state, **_: state.set_theme(next_theme(state.theme_name)),
                 cooldown=1.0, category="visual")
        register("toggle_grid", "Cuadricula",
                 lambda state, **_: state.toggle("grid"), cooldown=1.0, category="visual")
        register("toggle_help", "Panel de ayuda",
                 lambda state, **_: state.toggle("help"), cooldown=1.2, category="visual")

        register("next_mode", "Siguiente modo",
                 lambda state, **_: self.modes.next_mode(), cooldown=1.2, category="modos")
        register("prev_mode", "Modo anterior",
                 lambda state, **_: self.modes.previous_mode(), cooldown=1.2, category="modos")

        register("open_calculator", "Abrir calculadora",
                 lambda state, **_: self.bridge.open_calculator(),
                 permission="allow_calculator", cooldown=3.0, category="sistema")
        register("close_calculator", "Cerrar calculadora",
                 lambda state, **_: self.bridge.close_calculator(None),
                 permission="allow_calculator", cooldown=3.0, category="sistema")
        register("open_browser", "Abrir navegador",
                 lambda state, **_: self.bridge.open_url("https://www.youtube.com"),
                 permission="allow_browser", cooldown=5.0, category="sistema")
        register("media_play_pause", "Play/pausa",
                 lambda state, **_: self.bridge.send_media_key("play_pause"),
                 permission="allow_media_keys", cooldown=1.0, category="sistema")

    def _register_keys(self) -> None:
        """Register the global key bindings shown in the help panel."""
        toggle_specs = [
            (KEYS["telemetry"], "telemetry", "Telemetria / HUD", True, "hud"),
            (KEYS["help"], "help", "Panel de ayuda", False, "hud"),
            (KEYS["grid"], "grid", "Cuadricula", False, "hud"),
            (KEYS["fps_graph"], "fps_graph", "Grafica de FPS", False, "hud"),
            (KEYS["debug"], "debug", "Info de depuracion", False, "hud"),
            (KEYS["segmentation"], "segmentation", "Mascara de segmentacion", True, "modelos"),
            (KEYS["face_mesh"], "face_mesh", "Malla facial", False, "modelos"),
            (KEYS["face_detect"], "face_detect", "Deteccion de caras", False, "modelos"),
            (KEYS["objects"], "object_detect", "Deteccion de objetos", False, "modelos"),
            (KEYS["skeleton"], "skeleton", "Esqueleto", True, "modelos"),
            (KEYS["night"], "night_mode", "Modo nocturno", False, "visual"),
            (KEYS["mirror"], "mirror", "Espejo", True, "visual"),
        ]
        for code, name, description, default, group in toggle_specs:
            self.keys.register_toggle(
                code, name, description,
                default_active=self.state.is_active(name) or default, group=group,
            )

        oneshot_specs = [
            (KEYS["screenshot"], "screenshot", "Captura de pantalla", "captura"),
            (KEYS["record"], "record_toggle", "Grabar video", "captura"),
            (KEYS["gif"], "gif", "Grabar GIF", "captura"),
            (KEYS["burst"], "burst", "Rafaga de fotos", "captura"),
            (KEYS["theme"], "theme", "Cambiar tema", "visual"),
            (KEYS["picker"], "picker", "Selector de modos", "modos"),
            (KEYS["next_mode"], "next_mode", "Siguiente modo", "modos"),
            (KEYS["prev_mode"], "prev_mode", "Modo anterior", "modos"),
            (KEYS["reset_mode"], "reset_mode", "Reiniciar modo", "modos"),
            (KEYS["pause"], "pause", "Pausar", "general"),
            (KEYS["save_config"], "save_config", "Guardar configuracion", "general"),
            (KEYS["conf_up"], "conf_up", "Confianza +5%", "modelos"),
            (KEYS["conf_down"], "conf_down", "Confianza -5%", "modelos"),
        ]
        for code, name, description, group in oneshot_specs:
            self.keys.register_oneshot(code, name, description, group=group)

    def _subscribe_events(self) -> None:
        """Connect the event bus to the toast queue and the session log."""
        self.state.bus.subscribe(Events.NOTIFY, self.toasts.handle_event)

        def log_event(event) -> None:
            if self.session.enabled:
                self.session.record_event(event.name, **{
                    k: v for k, v in event.payload.items()
                    if isinstance(v, (str, int, float, bool))
                })

        self.state.bus.subscribe_many(
            [Events.GESTURE, Events.REP_COMPLETED, Events.POSTURE_ALERT,
             Events.MOTION_ALERT, Events.ACTION_TRIGGERED, Events.MODE_CHANGED],
            log_event,
        )

        # Modes ask for screenshots through the bus rather than holding the
        # recorder themselves.
        def handle_action(event) -> None:
            if event.get("name") == "screenshot":
                self._take_screenshot()

        self.state.bus.subscribe(Events.ACTION_TRIGGERED, handle_action)

        # Keep the mode catalogue available to the HUD picker.
        self.state.note("mode_catalog", {
            category: [m.key for m in modes]
            for category, modes in self.modes.categories().items()
        })

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def setup_display(self) -> None:
        """
        Create the output window sized to the preferred monitor.

        In headless mode no window is created at all, which is what lets the
        pipeline run on a server or under a test harness.
        """
        if self.config.display.headless:
            self.screen_size = (self.config.camera.width, self.config.camera.height)
            print("[app] Modo headless: sin ventana.", flush=True)
            return

        try:
            mx, my, mw, mh = self.bridge.get_monitor_geometry()
        except Exception:
            mx, my, mw, mh = 0, 0, self.config.camera.width, self.config.camera.height

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WINDOW_NAME, mx, my)
        cv2.resizeWindow(WINDOW_NAME, mw, mh)
        if self.config.display.fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
            )
        self.screen_size = (mw, mh)
        self._window_ready = True

    def start(self) -> bool:
        """Open the camera and start the models the current mode needs."""
        if not self.camera.open():
            print(f"[ERROR] {self.camera.last_error}", flush=True)
            return False

        required = set(self.modes.required_models()) | {"pose", "hands"}
        failed = self.models.start(tuple(required))
        if "pose" in failed:
            print("[ERROR] El modelo de pose no arranco; no puedo continuar.", flush=True)
            return False

        self.setup_display()
        self.session.set_metadata(
            mode=self.state.mode_name,
            resolution=f"{self.camera.width}x{self.camera.height}",
        )
        print(
            f"[app] Listo — modo '{self.state.mode_name}', "
            f"{len(self.modes)} modos disponibles. H=ayuda, TAB=modos, Q=salir.",
            flush=True,
        )
        return True

    # -----------------------------------------------------------------------
    # Frame pipeline
    # -----------------------------------------------------------------------

    def build_context(self, frame: np.ndarray, now: float) -> FrameContext:
        """Run detection and analysis, returning the context for this frame."""
        height, width = frame.shape[:2]
        delta = max(now - self._last_frame_time, 1e-6)
        self._last_frame_time = now

        ctx = FrameContext(
            frame=frame, width=width, height=height,
            timestamp_ms=int((now - self._start_time) * 1000),
            now=now, delta=delta, frame_index=self.state.frame_index,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        self._run_models(ctx, mp_image)
        self._analyze(ctx)
        return ctx

    def _run_models(self, ctx: FrameContext, mp_image: Any) -> None:
        """Run whichever models the mode and toggles call for."""
        index = ctx.frame_index
        smoother = self.models.smoother
        timestamp = ctx.now

        ctx.pose_result = self.models.detect("pose", mp_image, ctx.timestamp_ms, index)
        if ctx.pose_result and ctx.pose_result.pose_landmarks:
            ctx.pose_landmarks = smoother.smooth(
                "pose", list(ctx.pose_result.pose_landmarks), timestamp,
            )

        if self.modes.requires("hands") or self.state.is_active("gestures"):
            ctx.hand_result = self.models.detect("hands", mp_image, ctx.timestamp_ms, index)
            if ctx.hand_result and ctx.hand_result.hand_landmarks:
                ctx.hand_landmarks = smoother.smooth(
                    "hand", list(ctx.hand_result.hand_landmarks), timestamp,
                )
                ctx.handedness = [
                    categories[0].category_name
                    for categories in (getattr(ctx.hand_result, "handedness", None) or [])
                    if categories
                ]

        if self.state.is_active("face_mesh") or self.modes.requires("face_mesh"):
            ctx.face_mesh_result = self.models.detect(
                "face_mesh", mp_image, ctx.timestamp_ms, index,
            )
            if ctx.face_mesh_result and ctx.face_mesh_result.face_landmarks:
                ctx.face_landmarks = list(ctx.face_mesh_result.face_landmarks)

        if self.state.is_active("face_detect") or self.modes.requires("face_detect"):
            ctx.face_detect_result = self.models.detect(
                "face_detect", mp_image, ctx.timestamp_ms, index,
            )

        if self.state.is_active("object_detect") or self.modes.requires("object"):
            ctx.object_result = self.models.detect(
                "object", mp_image, ctx.timestamp_ms, index,
            )

    def _analyze(self, ctx: FrameContext) -> None:
        """Derive angles, gestures and motion from the raw detections."""
        with self.state.profiler.stage("analytics"):
            if ctx.has_pose:
                primary = ctx.primary_pose
                ctx.body_center = PoseModel.body_center(primary, ctx.width, ctx.height)
                ctx.angles = compute_joint_angles(primary)
                ctx.motion = self.motion.update(primary, ctx.now)

                # Slow-moving hip baseline, so jump detection survives the
                # person simply standing somewhere else.
                hip_y = (primary[23].y + primary[24].y) / 2.0
                self._hip_baseline = (
                    hip_y if self._hip_baseline is None
                    else self._hip_baseline * 0.98 + hip_y * 0.02
                )
                ctx.body_gestures = detect_every_gesture(primary, self._hip_baseline)

            if ctx.has_hands:
                ctx.hand_info = detect_all_hand_gestures(ctx.hand_landmarks)

            if ctx.has_face:
                face_info = self.face_gestures.update(ctx.primary_face)
                from analytics.face_metrics import head_pose
                face_info["head_pose"] = head_pose(ctx.primary_face)
                ctx.face_info = face_info

    def _dispatch_gestures(self, ctx: FrameContext) -> None:
        """Fire the actions bound to whatever gestures are active."""
        if not self.state.is_active("gestures"):
            return

        fired = self.bindings.dispatch(ctx.body_gestures, self.state, ctx.now)

        for name in self.hand_gestures.update(ctx.hand_info, ctx.now):
            if self.bindings.fire(name, self.state, ctx.now):
                fired.append(name)

        for name in ctx.face_info.get("triggered", []):
            if self.bindings.fire(name, self.state, ctx.now):
                fired.append(name)

        for name in fired:
            self.state.set_gesture(name.upper(), ctx.now)
            self.state.bus.emit(Events.GESTURE, name=name)
            if self.session.enabled:
                self.session.record_gesture(name)

    def _draw_standard_overlays(self, ctx: FrameContext) -> None:
        """Draw the shared overlays the toggles control."""
        theme = self.state.theme

        if ctx.has_pose:
            for index, landmarks in enumerate(ctx.pose_landmarks):
                color = (
                    self.state.body_color if index == 0
                    else theme.category_color(index)
                )
                if (
                    self.state.is_active("segmentation")
                    and ctx.pose_result
                    and getattr(ctx.pose_result, "segmentation_masks", None)
                    and index < len(ctx.pose_result.segmentation_masks)
                ):
                    PoseModel.tint_body(
                        ctx.frame, ctx.pose_result.segmentation_masks[index], color,
                    )
                if self.state.is_active("skeleton"):
                    PoseModel.draw_skeleton(
                        ctx.frame, landmarks, ctx.width, ctx.height, color,
                    )
                    draw_head_circle(
                        ctx.frame, landmarks, ctx.width, ctx.height,
                        ctx.gesture("head_touch"),
                    )

        if ctx.has_hands and self.state.is_active("skeleton"):
            for landmarks in ctx.hand_landmarks:
                HandModel.draw_skeleton(
                    ctx.frame, landmarks, ctx.width, ctx.height, theme.hand,
                )

        if ctx.face_landmarks and self.state.is_active("face_mesh"):
            for landmarks in ctx.face_landmarks:
                FaceMeshModel.draw_mesh(ctx.frame, landmarks, ctx.width, ctx.height)
                FaceMeshModel.draw_gaze_indicator(ctx.frame, landmarks, ctx.width, ctx.height)

        if self.state.is_active("bounding_boxes"):
            if ctx.face_detect_result and self.state.is_active("face_detect"):
                draw_bounding_boxes(
                    ctx.frame, ctx.face_detect_result, ctx.width, ctx.height,
                    theme.box, "", "corner", theme,
                )
            if ctx.object_result and self.state.is_active("object_detect"):
                draw_bounding_boxes(
                    ctx.frame, ctx.object_result, ctx.width, ctx.height,
                    theme.box, "", "corner", theme,
                )

        if self.state.is_active("grid"):
            draw_grid(ctx.frame, color=theme.grid)

        if self.state.is_active("night_mode"):
            ctx.frame = apply_night_mode(ctx.frame)

    def _update_status(self, ctx: FrameContext) -> None:
        """Compose the status line, letting the mode override it."""
        override = self.modes.status_text(ctx)
        if override is not None:
            self.state.status_text = override
            return

        if ctx.gesture("head_touch"):
            self.state.status_text = "TOCANDO CABEZA"
        elif ctx.gesture("t_pose"):
            self.state.status_text = "T-POSE"
        elif ctx.gesture("both_hands_raised"):
            self.state.status_text = "MANOS ARRIBA"
        elif ctx.has_pose:
            self.state.status_text = "CUERPO DETECTADO"
        elif ctx.has_hands:
            self.state.status_text = "MANOS DETECTADAS"
        else:
            self.state.status_text = "BUSCANDO..."

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Run until the user quits or the source stops delivering frames."""
        if not self.start():
            return

        try:
            while self.state.running:
                self.state.profiler.begin_frame()

                ok, frame = self.camera.read()
                if not ok or frame is None:
                    print("[app] Fin del origen de video.", flush=True)
                    break

                now = time.monotonic()
                self.state.fps.tick(now)
                self.state.frame_index += 1

                if not self.state.paused:
                    ctx = self.build_context(frame, now)

                    with self.state.profiler.stage("gestures"):
                        self._dispatch_gestures(ctx)
                    with self.state.profiler.stage("mode"):
                        self.modes.process(ctx)
                        # A mode may have replaced the frame with a crop or a
                        # view; normalise before anything tries to draw on it.
                        ctx.ensure_drawable()
                    with self.state.profiler.stage("draw"):
                        self._draw_standard_overlays(ctx)
                        ctx.ensure_drawable()
                        self.modes.draw(ctx)

                    self._update_status(ctx)
                    self._last_context = ctx
                else:
                    ctx = getattr(self, "_last_context", None)
                    if ctx is None:
                        continue
                    ctx.frame = frame

                if self.config.display.max_frames and (
                    self.state.frame_index >= self.config.display.max_frames
                ):
                    print(
                        f"[app] Limite de {self.config.display.max_frames} frames alcanzado.",
                        flush=True,
                    )
                    break

                with self.state.profiler.stage("hud"):
                    self.keys.sync_from(self.state.active_toggles())
                    self.hud.draw(
                        ctx, self.state,
                        mode_lines=self.modes.hud_lines(ctx),
                        key_bindings=self.keys.help_rows(),
                        mode_keys=self.modes.key_help(),
                        overlay_owner=self.modes.owns_overlay,
                    )

                self._handle_capture(ctx)

                if not self.config.display.headless:
                    display = fit_frame_to_screen(ctx.frame, *self.screen_size)
                    cv2.imshow(WINDOW_NAME, display)

                self.state.profiler.end_frame()

                if not self.config.display.headless and not self._handle_input():
                    break
        except KeyboardInterrupt:
            print("\n[app] Interrumpido por el usuario.", flush=True)
        finally:
            self.shutdown()

    def _handle_capture(self, ctx: FrameContext) -> None:
        """Feed the frame to whatever capture is running."""
        frame = ctx.frame
        self.state.recording = self.recorder.is_recording

        if self.recorder.is_recording:
            self.recorder.write_frame(frame)
            limit = self.config.recording.auto_stop_seconds
            if limit > 0 and self.recorder.recording_duration() >= limit:
                self._stop_recording()

        if self.recorder.burst_active:
            self.recorder.update_burst(frame, ctx.now)

        if self.recorder.gif_recording:
            self.recorder.update_gif(frame)

    # -----------------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------------

    def _handle_input(self) -> bool:
        """Process one keypress. Returns False to exit the loop."""
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return True

        if key in KEYS["quit"]:
            return False

        # Modes get first refusal on every key.
        if self.modes.handle_key(key):
            return True

        if key in QUICK_MODES:
            self.modes.switch(QUICK_MODES[key])
            return True

        result = self.keys.process_key(key)
        if result is None:
            return True

        name, value = result
        self._apply_key_action(name, value)
        return True

    def _apply_key_action(self, name: str, value: Any) -> None:
        """Carry out a global key action."""
        state = self.state

        # Toggles that mirror straight into application state.
        if name in ("telemetry", "help", "grid", "fps_graph", "debug", "skeleton",
                    "face_mesh", "face_detect", "object_detect", "night_mode"):
            state.set_toggle(name, bool(value))
            return

        if name == "segmentation":
            state.set_toggle(name, bool(value))
            self.models.set_segmentation(bool(value))
            return

        if name == "mirror":
            state.set_toggle(name, bool(value))
            self.camera.mirror = bool(value)
            return

        handlers = {
            "screenshot": lambda: self._take_screenshot(),
            "record_toggle": lambda: self._toggle_recording(),
            "gif": lambda: self._toggle_gif(),
            "burst": lambda: self.recorder.start_burst(
                self.config.recording.burst_count,
                self.config.recording.burst_interval,
                time.monotonic(),
            ),
            "theme": lambda: state.set_theme(next_theme(state.theme_name)),
            "picker": lambda: self.hud.toggle_picker(),
            "next_mode": lambda: self.modes.next_mode(),
            "prev_mode": lambda: self.modes.previous_mode(),
            "reset_mode": lambda: self._reset_mode(),
            "pause": lambda: self._toggle_pause(),
            "save_config": lambda: self._save_config(),
            "conf_up": lambda: self._adjust_confidence(0.05),
            "conf_down": lambda: self._adjust_confidence(-0.05),
        }
        handler = handlers.get(name)
        if handler:
            handler()

    # -- Key action implementations -----------------------------------------

    def _take_screenshot(self) -> None:
        context = getattr(self, "_last_context", None)
        if context is None:
            return
        path = self.recorder.screenshot(context.frame)
        if path:
            self.state.increment("screenshots")
            self.state.notify("Captura guardada", "good")

    def _toggle_recording(self) -> None:
        if self.recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        context = getattr(self, "_last_context", None)
        if context is None:
            return
        started, path = self.recorder.start_recording(context.frame)
        self.state.recording = started
        if started:
            self.state.notify("Grabando", "danger")
            self.state.bus.emit(Events.RECORDING_STARTED, path=path or "")

    def _stop_recording(self) -> None:
        if not self.recorder.is_recording:
            return
        _running, path = self.recorder.stop_recording()
        self.state.recording = False
        self.state.increment("recordings")
        self.state.notify("Grabacion guardada", "good")
        self.state.bus.emit(Events.RECORDING_STOPPED, path=path or "")

    def _toggle_gif(self) -> None:
        if self.recorder.gif_recording:
            path = self.recorder.finish_gif()
            self.state.notify("GIF guardado" if path else "GIF descartado",
                              "good" if path else "warn")
        else:
            self.recorder.start_gif()
            self.state.notify("Capturando GIF...", "info")

    def _random_color(self) -> None:
        import random
        self.state.body_color = (
            random.randint(40, 255), random.randint(40, 255), random.randint(40, 255),
        )

    def _reset_mode(self) -> None:
        mode = self.modes.current
        if mode is not None:
            mode.reset(self.state)
            self.state.notify(f"{mode.label} reiniciado")

    def _toggle_pause(self) -> None:
        self.state.paused = not self.state.paused
        self.state.notify("Pausado" if self.state.paused else "Reanudado")

    def _save_config(self) -> None:
        try:
            path = save_config(self.config)
            self.state.notify("Configuracion guardada", "good")
            print(f"[app] Configuracion guardada en {path}", flush=True)
        except OSError as exc:
            self.state.notify("No pude guardar la configuracion", "danger")
            print(f"[app] Error guardando configuracion: {exc}", flush=True)

    def _adjust_confidence(self, delta: float) -> None:
        current = self.config.detection.object_confidence
        updated = max(0.1, min(0.95, current + delta))
        self.config.detection.object_confidence = updated
        self.models.set_object_confidence(updated)
        self.state.notify(f"Confianza objetos: {updated:.0%}")

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release everything and print the session summary."""
        self.recorder.cleanup()
        self.modes.shutdown()
        self.models.stop()
        self.camera.close()
        if self._window_ready and not self.config.display.headless:
            cv2.destroyAllWindows()

        if self.session.enabled and len(self.session):
            self.session.set_metadata(**self.state.summary())
            self.session.export(self.config.analytics.export_format)

        self.state.shutdown()
        self._print_summary()

    def _print_summary(self) -> None:
        summary = self.state.summary()
        print("\n" + "=" * 46, flush=True)
        print("  Resumen de la sesion", flush=True)
        print("=" * 46, flush=True)
        print(f"  Duracion:     {self.state.uptime_text()}", flush=True)
        print(f"  Frames:       {summary['frames']}", flush=True)
        print(f"  FPS medio:    {summary['average_fps']:.1f}", flush=True)
        print(f"  Modo final:   {summary['mode']}", flush=True)

        counters = summary.get("counters", {})
        if counters:
            print("  Contadores:", flush=True)
            for name, value in sorted(counters.items()):
                print(f"    {name}: {value}", flush=True)

        files = self.recorder.counts()
        if files:
            print("  Archivos:", flush=True)
            for kind, count in sorted(files.items()):
                print(f"    {kind}: {count}", flush=True)
        print("=" * 46 + "\n", flush=True)
