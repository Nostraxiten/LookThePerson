"""
LookThePerson — Full Body & Gesture Detector
Cross-platform launcher (Windows + Linux)
by Nox / @nostraxiten

Real-time body, hand, face and object tracking with a mode system: fitness
coaching, wellness monitoring, creative filters, gesture interaction and
utility tools, all driven by the webcam.

Usage:
    python looktheperson.py                     # default mode
    python looktheperson.py --mode workout      # start in a specific mode
    python looktheperson.py --list-modes        # see everything available
    python looktheperson.py --windowed --camera 1
    python looktheperson.py --source video.mp4  # run on a file instead
    python looktheperson.py --help
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

__all__ = ["main", "parse_args", "build_overrides"]

BANNER = r"""
  _              _    _____ _         ____
 | |    ___  ___| | _|_   _| |__   ___|  _ \ ___ _ __ ___  ___  _ __
 | |   / _ \/ _ \ |/ / | | | '_ \ / _ \ |_) / _ \ '__/ __|/ _ \| '_ \
 | |__| (_) | (_) |   <  | | | | | |  __/  __/  __/ |  \__ \ (_) | | | |
 |_____\___/ \___/|_|\_\ |_| |_| |_|\___|_|   \___|_|  |___/\___/|_| |_|
"""


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="looktheperson",
        description=(
            "LookThePerson — deteccion de cuerpo, manos, cara y objetos con "
            "sistema de modos, analitica y gestos. Multiplataforma."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python looktheperson.py --mode workout --theme neon\n"
            "  python looktheperson.py --list-modes\n"
            "  python looktheperson.py --source clip.mp4 --windowed\n"
            "  python looktheperson.py --profile gimnasio --save-config\n"
        ),
    )

    camera = parser.add_argument_group("camara")
    camera.add_argument("--camera", type=int, default=None, metavar="N",
                        help="indice de la camara (por defecto 0)")
    camera.add_argument("--width", type=int, default=None, metavar="PX",
                        help="ancho de captura")
    camera.add_argument("--height", type=int, default=None, metavar="PX",
                        help="alto de captura")
    camera.add_argument("--fps", type=int, default=None, metavar="N",
                        help="FPS solicitados a la camara")
    camera.add_argument("--source", type=str, default=None, metavar="RUTA",
                        help="usa un video o carpeta de imagenes en vez de la camara")
    camera.add_argument("--no-mirror", action="store_true",
                        help="no invertir la imagen horizontalmente")

    display = parser.add_argument_group("pantalla")
    display.add_argument("--windowed", action="store_true",
                         help="modo ventana en vez de pantalla completa")
    display.add_argument("--mode", type=str, default=None, metavar="NOMBRE",
                         help="modo inicial (ver --list-modes)")
    display.add_argument("--theme", type=str, default=None, metavar="NOMBRE",
                         help="tema de color inicial")
    display.add_argument("--no-hud", action="store_true",
                         help="arranca con la telemetria oculta")
    display.add_argument("--headless", action="store_true",
                         help="ejecuta sin ventana (servidores, benchmarks)")
    display.add_argument("--max-frames", type=int, default=None, metavar="N",
                         help="procesa como mucho N frames y sale")

    detection = parser.add_argument_group("deteccion")
    detection.add_argument("--max-poses", type=int, default=None, metavar="N",
                           help="maximo de personas a seguir")
    detection.add_argument("--no-segmentation", action="store_true",
                           help="desactiva las mascaras de segmentacion (mas rapido)")
    detection.add_argument("--no-smoothing", action="store_true",
                           help="desactiva el filtrado de landmarks")
    detection.add_argument("--object-stride", type=int, default=None, metavar="N",
                           help="ejecuta los modelos pesados 1 de cada N frames")

    behaviour = parser.add_argument_group("comportamiento")
    behaviour.add_argument("--no-calculator", action="store_true",
                           help="prohibe el control de la calculadora")
    behaviour.add_argument("--no-gestures", action="store_true",
                           help="desactiva las acciones por gestos")
    behaviour.add_argument("--allow-mouse", action="store_true",
                           help="permite controlar el raton con la mano")
    behaviour.add_argument("--allow-media", action="store_true",
                           help="permite enviar teclas multimedia")
    behaviour.add_argument("--session-log", action="store_true",
                           help="registra la sesion y la exporta al salir")
    behaviour.add_argument("--weight", type=float, default=None, metavar="KG",
                           help="peso corporal para estimar calorias")
    behaviour.add_argument("--height-cm", type=float, default=None, metavar="CM",
                           help="altura para las medidas corporales")

    config_group = parser.add_argument_group("configuracion")
    config_group.add_argument("--profile", type=str, default="default", metavar="NOMBRE",
                              help="perfil de configuracion a cargar")
    config_group.add_argument("--config", type=str, default=None, metavar="RUTA",
                              help="archivo de configuracion adicional")
    config_group.add_argument("--save-config", action="store_true",
                              help="guarda la configuracion resultante y sale")

    info = parser.add_argument_group("informacion")
    info.add_argument("--list-modes", action="store_true",
                      help="lista todos los modos disponibles y sale")
    info.add_argument("--list-cameras", action="store_true",
                      help="busca camaras disponibles y sale")
    info.add_argument("--list-themes", action="store_true",
                      help="lista los temas de color y sale")
    info.add_argument("--show-config", action="store_true",
                      help="muestra la configuracion efectiva y sale")
    info.add_argument("--download-models", action="store_true",
                      help="descarga todos los modelos y sale")
    info.add_argument("--version", action="store_true", help="muestra la version y sale")

    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Translate CLI flags into a nested config override dict.

    Only flags the user actually passed appear, so unset options never
    override a configured value.
    """
    camera: Dict[str, Any] = {
        "index": args.camera,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "source": args.source,
    }
    if args.no_mirror:
        camera["mirror"] = False

    display: Dict[str, Any] = {"theme": args.theme, "max_frames": args.max_frames}
    if args.windowed:
        display["fullscreen"] = False
    if args.no_hud:
        display["show_hud"] = False
    if args.headless:
        display["headless"] = True

    detection: Dict[str, Any] = {
        "max_poses": args.max_poses,
        "object_stride": args.object_stride,
    }
    if args.no_segmentation:
        detection["segmentation"] = False
    if args.no_smoothing:
        detection["smoothing"] = False

    gestures: Dict[str, Any] = {}
    if args.no_gestures:
        gestures["enabled"] = False
    if args.no_calculator:
        gestures["allow_calculator"] = False
    if args.allow_mouse:
        gestures["allow_mouse_control"] = True
    if args.allow_media:
        gestures["allow_media_keys"] = True

    analytics: Dict[str, Any] = {
        "user_weight_kg": args.weight,
        "user_height_cm": args.height_cm,
    }
    if args.session_log:
        analytics["session_log"] = True

    overrides: Dict[str, Any] = {
        "camera": _prune(camera),
        "display": _prune(display),
        "detection": _prune(detection),
        "gestures": gestures,
        "analytics": _prune(analytics),
        "mode": args.mode,
    }
    return _prune(overrides)


def _prune(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values and empty sections."""
    return {
        key: value for key, value in data.items()
        if value is not None and value != {}
    }


# ---------------------------------------------------------------------------
# Informational commands
# ---------------------------------------------------------------------------

def _print_modes() -> None:
    """List every mode, grouped by category."""
    from modes import all_modes

    modes = all_modes()
    grouped: Dict[str, List[Any]] = {}
    for mode in modes:
        grouped.setdefault(mode.category, []).append(mode)

    print(f"\nLookThePerson — {len(modes)} modos disponibles\n")
    for category, items in grouped.items():
        print(f"  {category.upper()}")
        for mode in items:
            print(f"    {mode.key:<14} {mode.description}")
            if mode.keys:
                keys = "  ".join(f"[{k}] {v}" for k, v in mode.keys.items())
                print(f"    {'':<14} teclas: {keys}")
        print()
    print("  Uso: python looktheperson.py --mode <nombre>\n")


def _print_cameras() -> None:
    """Probe and report the available cameras."""
    from io_utils.camera import list_cameras, probe_resolutions

    print("\nBuscando camaras (puede tardar unos segundos)...\n")
    found = list_cameras(maximum=6)
    if not found:
        print("  No se encontro ninguna camara.\n")
        return

    for index in found:
        resolutions = probe_resolutions(index)
        modes = ", ".join(f"{w}x{h}" for w, h in resolutions) or "desconocidas"
        print(f"  Camara {index}: {modes}")
    print(f"\n  Uso: python looktheperson.py --camera {found[0]}\n")


def _print_themes() -> None:
    from core.theme import THEMES

    print("\nTemas disponibles:\n")
    for name, theme in THEMES.items():
        print(f"  {name:<10} acento BGR{theme.accent}")
    print("\n  Uso: python looktheperson.py --theme neon\n")


def _print_config(config) -> None:
    print("\nConfiguracion efectiva:\n")
    for line in config.describe():
        print(f"  {line}")
    print()


def _download_models() -> None:
    from models import MODELS, ensure_all_models

    print(f"\nDescargando {len(MODELS)} modelos...\n")
    try:
        ensure_all_models()
        print("\nTodos los modelos estan listos.\n")
    except Exception as exc:
        print(f"\n[ERROR] Fallo la descarga: {exc}\n")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, handle info commands and otherwise run the app."""
    args = parse_args(argv)

    if args.version:
        print("LookThePerson 2.0")
        return 0

    if args.list_modes:
        _print_modes()
        return 0

    if args.list_themes:
        _print_themes()
        return 0

    if args.list_cameras:
        _print_cameras()
        return 0

    if args.download_models:
        _download_models()
        return 0

    from core.config import load_config, save_config

    config = load_config(
        cli_overrides=build_overrides(args),
        profile=args.profile,
        extra_path=args.config,
    )

    if args.show_config:
        _print_config(config)
        return 0

    if args.save_config:
        path = save_config(config, profile=None if args.profile == "default" else args.profile)
        print(f"Configuracion guardada en {path}")
        return 0

    # Validate the requested mode before opening any hardware.
    from modes import DEFAULT_MODE, mode_keys

    if config.mode not in mode_keys():
        print(
            f"[ERROR] Modo desconocido: '{config.mode}'. "
            f"Usa --list-modes para ver los {len(mode_keys())} disponibles.",
            file=sys.stderr,
        )
        return 2

    print(BANNER)
    print(f"  Modo inicial: {config.mode}  |  Tema: {config.display.theme}")
    print(f"  {len(mode_keys())} modos disponibles — pulsa TAB para el selector\n")

    from app import Application

    application = Application(config)
    application.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Salida solicitada por el usuario. Cerrando...")
        raise SystemExit(130)
