"""
Action layer for LookThePerson.

* ``actions.key_handler`` — keyboard bindings (toggles and one-shots).
* ``actions.bindings`` — gesture-to-action mapping with permissions and
  cooldowns, so nothing reaches outside the app without an explicit allow.
* ``actions.recording`` — the original recorder, kept for compatibility;
  new code should use ``io_utils.capture.MediaRecorder``.
* ``actions.app_control`` — calculator and browser control via the platform
  bridge.
"""

from actions.bindings import ActionRegistry, GestureBindings
from actions.key_handler import KeyAction, KeyHandler

__all__ = ["KeyHandler", "KeyAction", "ActionRegistry", "GestureBindings"]
