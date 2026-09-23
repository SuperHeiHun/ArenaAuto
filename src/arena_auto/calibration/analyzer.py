"""Orchestrates one-pass calibration analysis, debug dumps, and verification."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from arena_auto.calibration.button_detector import ButtonDetector
from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.models import (
    SOURCE_CALIBRATED,
    CalibrationBundle,
    CalibrationDetection,
    CalibrationRegion,
)
from arena_auto.calibration.opponent_detector import OpponentDetector
from arena_auto.calibration.roi_detector import RoiDetector
from arena_auto.calibration.screen_analyzer import ScreenAnalyzer
from arena_auto.calibration.template_generator import TemplateGenerator
from arena_auto.models.config import AppConfig, ButtonGeometry, OpponentConfig
from arena_auto.models.state import ScreenKind
from arena_auto.paths import get_debug_dir
from arena_auto.recognition.detector import (
    ButtonFinder,
    ChallengeCountReader,
    PowerReader,
    ScreenDetector,
)
from arena_auto.recognition.ocr import OCRProvider
from arena_auto.recognition.screen import CoordinateScaler
from arena_auto.recognition.template import TemplateRecognizer

logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """Full auto-calibration pipeline built on existing recognition modules."""

    def __init__(
        self,
        config: AppConfig,
        ocr: OCRProvider,
        templates: Optional[TemplateRecognizer] = None,
    ) -> None:
        self.config = config
        self.ocr = ocr
        self.templates = templates or TemplateRecognizer(config.templates)
        self.mapper = CoordinateMapper(
            config.screen.reference_width, config.screen.reference_height
        )
        self.screen_analyzer = ScreenAnalyzer(ocr, config)
        self.opponent_detector = OpponentDetector(config, self.mapper)
        self.roi_detector = RoiDetector(config, self.mapper)
        self.button_detector = ButtonDetector(config, self.mapper)
        self.template_generator = TemplateGenerator(config, self.mapper)

    # --------------------------------------------------------------- analyze
    def analyze(self, screenshot: np.ndarray) -> CalibrationBundle:
        if screenshot is None or screenshot.size == 0:
            raise ValueError("截图为空，无法标定")
        height, width = screenshot.shape[:2]
        self.mapper.update_actual(width, height)
        logger.info("[CALIBRATION] 开始自动标定")
        logger.info("[CALIBRATION] 屏幕：%sx%s", width, height)
        logger.info("[CALIBRATION] 检测竞技场页面")

        self._save_debug("original.png", screenshot)

        boxes = self.screen_analyzer.recognize(screenshot)
        ocr_overlay = self.screen_analyzer.draw_ocr_overlay(screenshot, boxes)
        self._save_debug("ocr_overlay.png", ocr_overlay)

        kind, arena_ok, arena_conf = self.screen_analyzer.detect_screen(
            screenshot, boxes
        )
        bundle = CalibrationBundle(
            screen_width=width,
            screen_height=height,
            reference_width=self.config.screen.reference_width,
            reference_height=self.config.screen.reference_height,
            arena_detected=arena_ok,
            arena_confidence=arena_conf,
            screen_kind=kind.value,
            ocr_box_count=len(boxes),
        )
        if not arena_ok:
            bundle.notes.append(
                f"当前页面识别为 {kind.value}，请确认处于竞技场页面"
            )
            bundle.warnings.append("竞技场页面未自动确认")

        slot_regions: List[CalibrationRegion] = []
        opponents, notes = self.opponent_detector.detect(boxes)
        bundle.opponents = opponents
        bundle.notes.extend(notes)
        slot_regions = [d.region for d in opponents]
        opponent_overlay = self._draw_opponents(screenshot, opponents)
        self._save_debug("opponent_detection.png", opponent_overlay)
        self._save_debug("power_regions.png", opponent_overlay)

        challenge = self.roi_detector.detect_challenge(boxes, slot_regions)
        bundle.challenge = challenge
        challenge_overlay = screenshot.copy()
        if challenge is not None:
            self._draw_region(challenge_overlay, challenge.region, (255, 200, 0), "挑战次数")
            logger.info(
                "[CALIBRATION] 找到挑战次数 %s", challenge.detail or "OK"
            )
        self._save_debug("challenge_count.png", challenge_overlay)

        buttons = self.button_detector.detect(boxes)
        bundle.buttons = buttons
        button_overlay = self._draw_buttons(screenshot, buttons)
        self._save_debug("buttons.png", button_overlay)

        if buttons and self.config.calibration.generate_templates:
            regions = {name: det.region for name, det in buttons.items()}
            bundle.generated_templates = self.template_generator.generate_all(
                screenshot, regions
            )

        missing_slots = 5 - len([d for d in opponents if d.region.width > 0])
        if missing_slots > 0:
            bundle.warnings.append(f"对手缺失 {missing_slots} 个槽位")
        low = bundle.low_confidence_items
        if low:
            bundle.warnings.append(
                "低置信度项: " + ", ".join(d.name for d in low)
            )

        final = self.draw_bundle(screenshot, bundle)
        self._save_debug("final_calibration.png", final)
        logger.info("[CALIBRATION] 自动标定完成")
        return bundle

    # ------------------------------------------------------- page-wise buttons
    def detect_page_buttons(
        self, screenshot: np.ndarray, page: str
    ) -> Dict[str, object]:
        """Detect only the buttons that belong to one guided page.

        Used by wizard step 5: user walks through arena → purchase →
        victory → defeat screens one at a time. Results merge across pages.
        Never taps the device.
        """
        from arena_auto.calibration.button_detector import page_targets

        if screenshot is None or screenshot.size == 0:
            raise ValueError("截图为空，无法检测按钮")
        height, width = screenshot.shape[:2]
        self.mapper.update_actual(width, height)
        targets = page_targets(page)
        logger.info("[CALIBRATION] 分步按钮检测 page=%s targets=%s", page, targets)

        boxes = self.screen_analyzer.recognize(screenshot)
        detected = self.button_detector.detect(boxes, names=targets)
        kind, _, _ = self.screen_analyzer.detect_screen(screenshot, boxes)
        overlay = self._draw_buttons(screenshot, detected)
        self._save_debug("buttons.png", overlay)

        templates: Dict[str, str] = {}
        if detected and self.config.calibration.generate_templates:
            regions = {name: det.region for name, det in detected.items()}
            templates = self.template_generator.generate_all(screenshot, regions)

        missing = [t for t in targets if t not in detected]
        if missing:
            logger.info("[CALIBRATION] 本页未找到: %s", ", ".join(missing))
        return {
            "image": screenshot,
            "overlay": overlay,
            "detected": detected,
            "targets": list(targets),
            "missing": missing,
            "templates": templates,
            "kind": kind.value,
            "page": page,
        }

    # ---------------------------------------------------------------- verify
    def verify(
        self,
        screenshot_factory: Callable[[], np.ndarray],
        bundle: Optional[CalibrationBundle] = None,
    ) -> Dict[str, object]:
        """Re-screenshot and validate recognition. NEVER taps the device."""
        cfg = copy.deepcopy(self.config)
        if bundle is not None:
            self.apply_bundle(cfg, bundle)

        scaler = CoordinateScaler(
            cfg.screen.reference_width, cfg.screen.reference_height
        )
        first = screenshot_factory()
        scaler.maybe_update_from_image(first)

        detector = ScreenDetector(self.ocr, cfg, self.templates, scaler)
        power_reader = PowerReader(
            self.ocr, cfg, cfg.recognition, scaler, debug=None
        )
        challenge_reader = ChallengeCountReader(self.ocr, cfg, scaler, debug=None)
        buttons = ButtonFinder(self.templates, self.ocr, cfg, debug=None)

        checks: List[Dict[str, object]] = []
        kind = detector.detect(first)
        checks.append(
            {
                "name": "竞技场",
                "ok": kind == ScreenKind.ARENA,
                "detail": kind.value,
            }
        )

        opponents = power_reader.read_all(screenshot_factory, cfg.opponents)
        for opp in opponents:
            checks.append(
                {
                    "name": f"#{opp.id}",
                    "ok": opp.power is not None,
                    "detail": (
                        f"{opp.power:,}" if opp.power is not None else "OCR_FAILED"
                    ),
                }
            )

        count = challenge_reader.read(screenshot_factory())
        checks.append(
            {
                "name": "挑战次数",
                "ok": count is not None,
                "detail": str(count) if count else "OCR_FAILED",
            }
        )

        fallback_map = {
            "go_win": cfg.profile.go_win_text,
            "exit": cfg.profile.exit_text,
        }
        for name, text in fallback_map.items():
            result = buttons.find(screenshot_factory(), name, text)
            checks.append(
                {
                    "name": name,
                    "ok": result.found,
                    "detail": f"conf={result.confidence:.2f}" if result.found else "未找到",
                }
            )

        passed = sum(1 for c in checks if c["ok"])
        summary = {
            "checks": checks,
            "passed": passed,
            "total": len(checks),
            "all_ok": passed == len(checks),
        }
        logger.info(
            "[CALIBRATION] 配置验证 %s/%s",
            passed,
            len(checks),
        )
        return summary

    # ------------------------------------------------------------ config I/O
    def apply_bundle(
        self,
        config: AppConfig,
        bundle: CalibrationBundle,
        overwrite_manual: bool = False,
    ) -> AppConfig:
        """Write bundle (actual px) into config as reference coordinates."""
        mapper = CoordinateMapper(
            config.screen.reference_width,
            config.screen.reference_height,
            bundle.screen_width,
            bundle.screen_height,
        )

        by_id: Dict[int, CalibrationDetection] = {}
        for det in bundle.opponents:
            try:
                slot = int(det.name.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                continue
            by_id[slot] = det

        for opponent in config.opponents:
            det = by_id.get(opponent.id)
            if det is None or det.region.width <= 0:
                continue
            if opponent.source == "manual" and not overwrite_manual:
                continue
            opponent.power_region = mapper.actual_region_to_reference(det.region)
            # click uses card center derived in actual px
            click_actual = det.click_point
            opponent.click = mapper.actual_point_to_reference(click_actual)
            opponent.source = (
                "manual"
                if det.source == "manual"
                else SOURCE_CALIBRATED
            )
            opponent.confidence = float(det.confidence)

        if bundle.challenge is not None and bundle.challenge.region.width > 0:
            config.challenge_region = mapper.actual_region_to_reference(
                bundle.challenge.region
            )

        for name, det in bundle.buttons.items():
            geo = ButtonGeometry(
                region=mapper.actual_region_to_reference(det.region),
                click=mapper.actual_point_to_reference(det.click_point),
                confidence=float(det.confidence),
                source=(
                    "manual" if det.source == "manual" else SOURCE_CALIBRATED
                ),
            )
            config.buttons[name] = geo
            if name == "buy_challenge":
                config.purchase.buy_button = geo
            elif name == "confirm_buy":
                config.purchase.confirm_button = geo
            elif name == "victory":
                config.result.victory = geo
            elif name == "defeat":
                config.result.defeat = geo
            elif name == "exit":
                config.result.exit = geo

            if name in bundle.generated_templates:
                entry = config.templates.get(name)
                if entry is None:
                    from arena_auto.models.config import TemplateEntry

                    entry = TemplateEntry(path="", threshold=0.82)
                    config.templates[name] = entry
                entry.path = bundle.generated_templates[name]

        config.version = 2
        return config

    # ----------------------------------------------------------- rendering
    def draw_bundle(
        self, screenshot: np.ndarray, bundle: CalibrationBundle
    ) -> np.ndarray:
        overlay = self._draw_opponents(screenshot, bundle.opponents)
        if bundle.challenge is not None:
            self._draw_region(
                overlay, bundle.challenge.region, (255, 200, 0), "挑战次数"
            )
        for name, det in bundle.buttons.items():
            color = self._button_color(det.confidence)
            self._draw_region(overlay, det.region, color, name)
            cv2.circle(
                overlay,
                (det.click_point.x, det.click_point.y),
                5,
                color,
                2,
            )
        for det in bundle.opponents:
            cv2.circle(
                overlay,
                (det.click_point.x, det.click_point.y),
                6,
                (255, 0, 255),
                2,
            )
        return overlay

    def _draw_opponents(
        self, screenshot: np.ndarray, opponents: List[CalibrationDetection]
    ) -> np.ndarray:
        overlay = screenshot.copy()
        for det in opponents:
            color = self._button_color(det.confidence)
            label = det.name.replace("opponent_", "#")
            value = f"{det.value:,}" if det.value else "OCR缺失"
            self._draw_region(
                overlay,
                det.region,
                color,
                f"{label} {value} {int(det.confidence * 100)}%",
            )
        return overlay

    def _draw_buttons(
        self, screenshot: np.ndarray, buttons: Dict[str, CalibrationDetection]
    ) -> np.ndarray:
        overlay = screenshot.copy()
        for name, det in buttons.items():
            color = self._button_color(det.confidence)
            self._draw_region(
                overlay,
                det.region,
                color,
                f"{name} {int(det.confidence * 100)}%",
            )
            cv2.circle(
                overlay,
                (det.click_point.x, det.click_point.y),
                5,
                color,
                2,
            )
        return overlay

    @staticmethod
    def _button_color(confidence: float) -> tuple:
        if confidence >= 0.90:
            return (0, 220, 0)
        if confidence >= 0.75:
            return (0, 200, 255)
        return (0, 0, 255)

    @staticmethod
    def _draw_region(
        image: np.ndarray,
        region: CalibrationRegion,
        color: tuple,
        label: str,
    ) -> None:
        h, w = image.shape[:2]
        x0 = max(0, int(region.x))
        y0 = max(0, int(region.y))
        x1 = min(w, int(region.x + region.width))
        y1 = min(h, int(region.y + region.height))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        if label:
            cv2.putText(
                image,
                label,
                (x0, max(14, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    # ---------------------------------------------------------------- debug
    def _save_debug(self, name: str, image: np.ndarray) -> Optional[Path]:
        if not (
            self.config.debug.enabled and self.config.calibration.debug_overlays
        ):
            return None
        try:
            folder = get_debug_dir() / "calibration"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / name
            ok, buf = cv2.imencode(".png", image)
            if ok:
                buf.tofile(str(path))
            logger.debug("[CALIBRATION] 调试图: %s", path)
            return path
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CALIBRATION] 保存调试图失败: %s", exc)
            return None
