#!/usr/bin/env python3
"""OCR / template performance benchmark.

Usage:
  python scripts/benchmark_ocr.py [--image path] [--repeat 5] [--full]

Measures real timings for:
  - full-screen OCR
  - single ROI OCR (power-like)
  - 5 ROI sequential OCR (one frame)
  - OCR cache hit
  - template matching
Does not invent numbers — prints measured Avg / P95 only.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arena_auto.config.manager import ConfigManager  # noqa: E402
from arena_auto.recognition.detector import (  # noqa: E402
    ChallengeCountReader,
    PowerReader,
    ScreenDetector,
)
from arena_auto.recognition.ocr import create_base_ocr_provider  # noqa: E402
from arena_auto.recognition.ocr_engine import OcrEngine  # noqa: E402
from arena_auto.recognition.screen import crop_region  # noqa: E402
from arena_auto.recognition.template import TemplateRecognizer  # noqa: E402


def _percentile(samples: Sequence[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return float(ordered[idx])


def bench(fn: Callable[[], object], repeat: int) -> Tuple[float, float]:
    samples: List[float] = []
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.fmean(samples), _percentile(samples, 0.95)


def load_screenshot(path: Path | None) -> np.ndarray:
    candidates: List[Path] = []
    if path is not None:
        candidates.append(path)
    testimg = ROOT / "testimg"
    if testimg.is_dir():
        # Prefer arena/select frame: challenge + power ROIs actually present.
        preferred = ["选择对手", "挑战次数购买", "去获胜", "胜利"]
        all_imgs = sorted(testimg.glob("*.jpg")) + sorted(testimg.glob("*.png"))
        ordered: List[Path] = []
        for stem in preferred:
            ordered.extend(p for p in all_imgs if stem in p.stem)
        ordered.extend(p for p in all_imgs if p not in ordered)
        candidates.extend(ordered)
    for p in candidates:
        if p.is_file():
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    # synthetic fallback so the script always runs
    img = np.zeros((720, 1600, 3), dtype=np.uint8)
    cv2.putText(img, "1234567", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(img, "4 / 5", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def load_config():
    cal = ROOT / "config" / "config.calibrated.yaml"
    if cal.is_file():
        manager = ConfigManager(path=cal)
        manager.load()
        return manager.config, str(cal.name)
    manager = ConfigManager.load_or_create()
    return manager.config, "config.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="ArenaAuto OCR benchmark")
    parser.add_argument("--image", type=Path, default=None, help="screenshot path")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--full", action="store_true", help="also init EasyOCR full-screen path"
    )
    args = parser.parse_args()

    config, config_name = load_config()
    image = load_screenshot(args.image)
    print(f"config: {config_name}")
    print(f"image: {image.shape[1]}x{image.shape[0]} repeat={args.repeat}")
    print(f"provider={config.ocr.provider} device={config.ocr.device} scale={config.ocr.scale} timeout_ms={config.ocr.timeout_ms}")
    print()

    rows: List[Tuple[str, float, float]] = []

    # Template matching (always)
    templates = TemplateRecognizer(config.templates)
    t_avg, t_p95 = bench(lambda: templates.find(image, "go_win"), args.repeat)
    rows.append(("Template matching (go_win)", t_avg, t_p95))

    # ROI extraction only (no OCR)
    def _crop_all() -> None:
        for opp in config.opponents:
            if opp.power_region[2] > 0 and opp.power_region[3] > 0:
                try:
                    crop_region(image, opp.power_region)
                except ValueError:
                    pass
        try:
            crop_region(image, config.challenge_region)
        except ValueError:
            pass

    c_avg, c_p95 = bench(_crop_all, args.repeat)
    rows.append(("Crop 5 power + challenge ROIs", c_avg, c_p95))

    # Real OCR if provider available
    try:
        provider = create_base_ocr_provider(config.ocr, allow_null=False)
    except Exception as exc:  # noqa: BLE001
        print(f"OCR provider unavailable: {exc}")
        provider = None

    if provider is not None:
        engine = OcrEngine(provider=provider, settings=config.ocr, use_shared=False)
        engine.clear_cache()

        f_avg, f_p95 = bench(lambda: engine.recognize(image), args.repeat)
        rows.append(("Full OCR (engine.recognize)", f_avg, f_p95))

        regions = [o.power_region for o in config.opponents if o.power_region[2] > 0]
        # map to scaled coords using 1:1 (assume image matches reference-ish)
        from arena_auto.recognition.screen import CoordinateScaler

        scaler = CoordinateScaler(config.screen.reference_width, config.screen.reference_height)
        scaler.maybe_update_from_image(image)
        scaled_regions = [scaler.scale_region(r) for r in regions]

        def _five_sequential() -> None:
            for i, reg in enumerate(scaled_regions):
                engine.read_roi(image, reg, allowlist="0123456789,.", cache_key=None, force=True)

        s_avg, s_p95 = bench(_five_sequential, max(1, args.repeat // 2 or 1))
        rows.append(("5 ROI sequential OCR (force)", s_avg, s_p95))

        # cache warm then hit
        if scaled_regions:
            engine.read_roi(
                image,
                scaled_regions[0],
                allowlist="0123456789,.",
                cache_key="bench0",
                force=False,
            )
            hit_avg, hit_p95 = bench(
                lambda: engine.read_roi(
                    image,
                    scaled_regions[0],
                    allowlist="0123456789,.",
                    cache_key="bench0",
                    force=False,
                ),
                max(5, args.repeat),
            )
            rows.append(("Cache hit (same ROI)", hit_avg, hit_p95))

        # power reader path (uses config ROIs + one frame each call)
        power_reader = PowerReader(engine, config, config.recognition, scaler, None)
        shots = {"n": 0}

        def _factory() -> np.ndarray:
            shots["n"] += 1
            return image

        p_avg, p_p95 = bench(
            lambda: power_reader.read_all(_factory, config.opponents),
            max(1, args.repeat // 2 or 1),
        )
        rows.append((f"PowerReader.read_all (screenshots={shots['n']})", p_avg, p_p95))

        challenge = ChallengeCountReader(engine, config, scaler, None)
        ch_avg, ch_p95 = bench(lambda: challenge.read(image), args.repeat)
        rows.append(("Challenge ROI OCR", ch_avg, ch_p95))

        detector = ScreenDetector(engine, config, templates, scaler)
        d_avg, d_p95 = bench(lambda: detector.detect(image), args.repeat)
        rows.append(("ScreenDetector.detect", d_avg, d_p95))

        stats = engine.get_stats()
        print("OcrEngine stats:")
        for key in (
            "calls",
            "cache_hits",
            "cache_hit_rate",
            "avg_ms",
            "p50_ms",
            "p95_ms",
            "max_ms",
            "provider",
            "device",
        ):
            val = stats.get(key)
            if isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")
        print()

        if args.full:
            # force a real base engine already created above
            print("(--full: engine already included in full OCR row)")

    print(f"{'Method':<42} {'Avg':>10} {'P95':>10}")
    print("-" * 64)
    for name, avg, p95 in rows:
        print(f"{name:<42} {avg:>8.1f}ms {p95:>8.1f}ms")
    print()
    print("Notes:")
    print("- Numbers are from this machine/run only (not fabricated).")
    print("- Enable config.logging.performance=true for [PERF] logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
