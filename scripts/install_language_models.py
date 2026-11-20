"""Install/cache RapidOCR and EasyOCR Korean/Japanese models locally."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arena_auto.paths import get_writable_path
from arena_auto.recognition.model_store import EASYOCR_DETECTION_MODEL, EASYOCR_MODELS, KOREAN_MODELS, download_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=("korean", "japan"),
                        default=["korean", "japan"])
    parser.add_argument("--engines", nargs="+", choices=("rapidocr", "easyocr"),
                        default=["rapidocr", "easyocr"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    installed = []

    if "rapidocr" in args.engines and "korean" in args.languages:
        root = get_writable_path("resources/models/rapidocr")
        for filename, url, checksum in KOREAN_MODELS:
            path = download_model(root / filename, url, checksum)
            installed.append({"engine": "rapidocr", "language": "korean",
                              "path": str(path), "bytes": path.stat().st_size})
    if "rapidocr" in args.engines and "japan" in args.languages:
        from arena_auto.models.config import OcrSettings
        from arena_auto.recognition.rapidocr_provider import RapidOCRProvider

        provider = RapidOCRProvider(OcrSettings(provider="rapidocr", lang="japan", device="cpu"))
        installed.append({"engine": "rapidocr", "language": "japan",
                          "model": "bundled PP-OCRv6 multilingual", "device": provider.device_label})

    if "easyocr" in args.engines:
        try:
            import easyocr  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            parser.error(f"EasyOCR 不可用；请先在当前环境安装 EasyOCR: {exc}")
        root = get_writable_path("resources/models/easyocr")
        det_filename, det_url, det_checksum = EASYOCR_DETECTION_MODEL
        det_path = download_model(root / det_filename, det_url, det_checksum,
                                  algorithm="md5", archive_member=det_filename)
        installed.append({"engine": "easyocr", "language": "shared-detector",
                          "path": str(det_path), "bytes": det_path.stat().st_size,
                          "md5": hashlib.md5(det_path.read_bytes()).hexdigest()})
        for language in args.languages:
            filename, url, checksum = EASYOCR_MODELS[language]
            path = download_model(root / filename, url, checksum,
                                  algorithm="md5", archive_member=filename)
            installed.append({"engine": "easyocr", "language": language,
                              "path": str(path), "bytes": path.stat().st_size,
                              "md5": hashlib.md5(path.read_bytes()).hexdigest()})

    print(f"模型安装完成: {installed}")
    if "easyocr" in args.engines:
        print("EasyOCR CRAFT 检测模型与语言识别模型已安装到 resources/models/easyocr。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
