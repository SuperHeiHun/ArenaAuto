"""Offline OCR device/accuracy benchmark. Never connects to or clicks a device."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from arena_auto.models.config import AppConfig
from arena_auto.recognition.auto_targeting import AutoTargeting
from arena_auto.recognition.detector import ScreenDetector
from arena_auto.recognition.ocr import create_base_ocr_provider
from arena_auto.recognition.ocr_engine import OcrEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--device", choices=("directml", "cpu", "auto"), default="directml")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--targeting", action="store_true", help="Also test opponent selection and challenge count")
    parser.add_argument("--threshold", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, default=ROOT / "debug" / "gpu_ocr_report.json")
    args = parser.parse_args()
    if args.gpu_id < 0 or args.repeat < 1:
        parser.error("gpu-id 必须 >=0，repeat 必须 >=1")
    config = AppConfig()
    config.ocr.provider = "rapidocr"
    config.ocr.device = args.device
    config.ocr.gpu_device_id = args.gpu_id
    config.ocr.gpu_fallback_to_cpu = False  # Diagnostics must not silently benchmark CPU as GPU.
    started = time.perf_counter()
    provider = create_base_ocr_provider(config.ocr, allow_fallback=False, allow_null=False)
    init_seconds = time.perf_counter() - started
    ocr = OcrEngine(provider=provider, settings=config.ocr, use_shared=False)
    report = {"python": sys.executable, "provider": provider.name, "device": provider.device_label,
              "session_providers": provider.session_providers, "initialization_seconds": round(init_seconds, 3),
              "device_actions": 0, "images": []}
    for path in args.images:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取截图: {path}")
        timings = []
        boxes = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            boxes = ocr.recognize(image)
            timings.append(round(time.perf_counter() - started, 3))
        detector = ScreenDetector(ocr, config)
        entry = {"file": str(path), "ocr_seconds": timings, "box_count": len(boxes),
                 "outcome": detector.detect_outcome(image, boxes=boxes),
                 "texts": [b.text for b in boxes]}
        if args.targeting:
            targeting = AutoTargeting(ocr, config)
            started = time.perf_counter()
            try:
                opponents, selected = targeting.select(lambda: image.copy(), args.threshold)
                entry["selection"] = {"seconds": round(time.perf_counter() - started, 3),
                                      "powers": [o.power for o in opponents],
                                      "selected_id": selected.id if selected else None,
                                      "click": selected.click_position if selected else None}
            except Exception as exc:  # noqa: BLE001
                entry["selection"] = {"error": str(exc)}
            count = targeting.read_challenge(image)
            entry["challenge_count"] = str(count) if count else None
        report["images"].append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
