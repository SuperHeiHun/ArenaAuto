"""Helpers for launching child processes without flashing console windows."""

from __future__ import annotations

import subprocess
import sys


def no_console_window_options() -> dict[str, int]:
    """Prevent a console executable from allocating a window on Windows."""
    if sys.platform != "win32":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW}
