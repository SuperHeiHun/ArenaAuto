#!/usr/bin/env python3
"""ArenaAuto development entrypoint: python main.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run() -> int:
    from arena_auto.app import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
