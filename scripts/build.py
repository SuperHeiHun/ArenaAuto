#!/usr/bin/env python3
"""Build Windows EXE with PyInstaller: python scripts/build.py"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    spec = ROOT / "ArenaAuto.spec"
    if not spec.is_file():
        print(f"找不到 spec: {spec}", file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", str(spec)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode == 0:
        dist = ROOT / "dist"
        print(f"构建完成，输出目录: {dist}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
