"""Shared ANSI color helper for zl_bench plugins and scripts.

Color policy, decided once at import (env is fixed before python starts):
1. FLAGGEMS_PERF_COLOR=always|never wins — run_pytest.sh sets `always` when
   *its* stdout is a tty, because the tee pipeline hides the tty from python;
2. FORCE_COLOR forces on (or off when set to 0/false, node semantics), taking
   precedence over NO_COLOR;
3. NO_COLOR (non-empty) forces off;
4. otherwise: on iff stdout is a tty.

run.log stays plain either way: run_pytest.sh strips ANSI from the log file
after the run.
"""

from __future__ import annotations

import os
import sys

_FALSY = ("", "0", "false", "no", "off", "none")


def _color_enabled() -> bool:
    mode = os.environ.get("FLAGGEMS_PERF_COLOR", "").strip().lower()
    if mode in ("always", "1", "yes", "on"):
        return True
    if mode in ("never", "0", "no", "off"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        return force.strip().lower() not in _FALSY
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


COLOR = _color_enabled()

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BRIGHT_BLUE = "\033[94m"


def paint(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes when color is enabled; identity otherwise."""
    if not COLOR or not codes:
        return text
    return "".join(codes) + text + RESET


def tag(label: str) -> str:
    """Bright-blue '[xxx-plugin]' prefix for plugin log lines."""
    return paint(label, BRIGHT_BLUE)
