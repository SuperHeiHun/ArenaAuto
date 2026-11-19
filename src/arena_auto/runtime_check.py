"""Offline smoke test for the packaged OCR DLLs/models; never uses ADB or GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = {"ok": False, "device_actions": 0, "frozen": bool(getattr(sys, "frozen", False))}
    try:
        import cv2
        import numpy as np

        from arena_auto.models.config import OcrSettings
        from arena_auto.config.manager import ConfigManager
        from arena_auto.recognition.rapidocr_provider import RapidOCRProvider

        # Hosted CI runners do not need a GPU: load and run the exact bundled
        # models with CPU EP, rather than merely checking that DLLs exist.
        provider = RapidOCRProvider(OcrSettings(provider="rapidocr", device="cpu", gpu_fallback_to_cpu=False))
        image = np.full((100, 360, 3), 255, dtype=np.uint8)
        cv2.putText(image, "12345", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        boxes = provider.recognize(image)
        if not boxes:
            raise RuntimeError("打包模型推理没有返回文字")
        if not all(path.is_file() for path in provider._model_paths):
            raise RuntimeError("打包 OCR 模型文件缺失")
        defaults = ConfigManager.initial_config()
        report.update(ok=True, provider=provider.name, device=provider.device_label,
                      session_providers=provider.session_providers, texts=[b.text for b in boxes],
                      model_count=len(provider._model_paths),
                      default_auto_initialize=defaults.automation.auto_targeting,
                      default_ocr_provider=defaults.ocr.provider)
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if sys.stdout is not None:
        print(json.dumps(report, ensure_ascii=True))
    return 0 if report["ok"] else 1
