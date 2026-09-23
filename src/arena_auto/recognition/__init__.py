"""Recognition package."""

from arena_auto.recognition.detector import (
    ButtonFinder,
    ChallengeCountReader,
    PowerReader,
    ScreenDetector,
)
from arena_auto.recognition.ocr import (
    EasyOCRProvider,
    OCRProvider,
    PaddleOCRProvider,
    TesseractOCRProvider,
    create_ocr_provider,
    parse_power,
    preprocess_for_ocr,
)
from arena_auto.recognition.ocr_engine import (
    OcrEngine,
    OcrPerformanceMonitor,
    get_ocr_engine,
    set_performance_logging,
)
from arena_auto.recognition.screen import CoordinateScaler, crop_region
from arena_auto.recognition.template import TemplateRecognizer

__all__ = [
    "ButtonFinder",
    "ChallengeCountReader",
    "CoordinateScaler",
    "EasyOCRProvider",
    "OCRProvider",
    "OcrEngine",
    "OcrPerformanceMonitor",
    "PaddleOCRProvider",
    "PowerReader",
    "ScreenDetector",
    "TemplateRecognizer",
    "TesseractOCRProvider",
    "create_ocr_provider",
    "crop_region",
    "get_ocr_engine",
    "parse_power",
    "preprocess_for_ocr",
    "set_performance_logging",
]
