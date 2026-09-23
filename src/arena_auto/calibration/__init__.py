"""Auto-calibration package: automatic / semi-auto / manual configuration."""

from arena_auto.calibration.analyzer import CalibrationAnalyzer
from arena_auto.calibration.button_detector import (
    BUTTON_DISPLAY,
    BUTTON_PAGES,
    BUTTON_PAGE_ORDER,
    ButtonDetector,
    button_specs,
    page_targets,
    pages_for_button,
)
from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.models import (
    SOURCE_CALIBRATED,
    SOURCE_DEFAULT,
    SOURCE_MANUAL,
    SOURCE_OCR,
    SOURCE_STRUCTURE,
    SOURCE_TEMPLATE,
    CalibrationBundle,
    CalibrationDetection,
    CalibrationPoint,
    CalibrationRegion,
)
from arena_auto.calibration.opponent_detector import OpponentDetector
from arena_auto.calibration.roi_detector import RoiDetector
from arena_auto.calibration.screen_analyzer import ScreenAnalyzer
from arena_auto.calibration.template_generator import TemplateGenerator

__all__ = [
    "CalibrationAnalyzer",
    "CalibrationBundle",
    "CalibrationDetection",
    "CalibrationPoint",
    "CalibrationRegion",
    "CoordinateMapper",
    "ButtonDetector",
    "OpponentDetector",
    "RoiDetector",
    "ScreenAnalyzer",
    "TemplateGenerator",
    "SOURCE_CALIBRATED",
    "SOURCE_DEFAULT",
    "SOURCE_MANUAL",
    "SOURCE_OCR",
    "SOURCE_STRUCTURE",
    "SOURCE_TEMPLATE",
    "button_specs",
    "BUTTON_DISPLAY",
    "BUTTON_PAGES",
    "BUTTON_PAGE_ORDER",
    "page_targets",
    "pages_for_button",
]
