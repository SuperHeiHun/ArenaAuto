"""Use the GPU-capable Python environment; saved Config selects OCR backend."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arena_auto.app import main

if __name__ == "__main__":
    raise SystemExit(main())
