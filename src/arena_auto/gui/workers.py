"""Background workers for automation and device discovery."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from arena_auto.adb.controller import AdbController
from arena_auto.automation.controller import AutomationController, ControllerCallbacks
from arena_auto.models.recognition import Opponent
from arena_auto.models.state import AutomationState, ScreenKind, SessionStatistics

logger = logging.getLogger(__name__)


class DeviceListWorker(QThread):
    """Enumerate ADB devices without blocking the GUI."""

    devices_found = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, adb: AdbController, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.adb = adb

    def run(self) -> None:
        try:
            devices: List[str] = self.adb.list_devices()
            self.devices_found.emit(devices)
        except Exception as exc:  # noqa: BLE001
            logger.warning("设备列表失败: %s", exc)
            self.error_occurred.emit(str(exc))


class AutomationWorker(QObject):
    """Runs AutomationController on a QThread with Qt signals."""

    state_changed = Signal(AutomationState)
    screen_detected = Signal(ScreenKind)
    log_message = Signal(str, str)
    powers_detected = Signal(list)
    target_changed = Signal(int)
    challenge_changed = Signal(object)
    statistics_changed = Signal(object)
    screenshot_updated = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        controller: AutomationController,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._wire_callbacks()

    def _wire_callbacks(self) -> None:
        cb = ControllerCallbacks(
            on_state=lambda s: self.state_changed.emit(s),
            on_log=lambda level, msg: self.log_message.emit(level, msg),
            on_powers=lambda powers: self.powers_detected.emit(powers),
            on_target=lambda tid: self.target_changed.emit(tid),
            on_challenge=lambda c: self.challenge_changed.emit(c),
            on_statistics=lambda stats: self.statistics_changed.emit(stats),
            on_screenshot=lambda img: self.screenshot_updated.emit(img),
            on_screen=lambda kind: self.screen_detected.emit(kind),
            on_error=lambda msg: self.error_occurred.emit(msg),
            on_finished=lambda: self.finished.emit(),
        )
        self.controller.callbacks = cb

    def run(self) -> None:
        try:
            self.controller.run()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Worker 异常")
            self.error_occurred.emit(str(exc))
            self.finished.emit()

    def request_stop(self) -> None:
        self.controller.request_stop()


class ScreenshotWorker(QThread):
    """One-shot screenshot for refresh / debug tools."""

    captured = Signal(object)
    failed = Signal(str)

    def __init__(self, adb: AdbController, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.adb = adb

    def run(self) -> None:
        try:
            image = self.adb.screenshot()
            self.captured.emit(image)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class DebugTestWorker(QThread):
    """Run a named recognition test against a fresh screenshot."""

    result_ready = Signal(str, object, object)
    # test_name, annotated_image, payload(dict)

    TESTS = (
        "screenshot",
        "ocr",
        "arena",
        "go_win",
        "exit",
        "buy",
        "confirm",
        "victory",
        "defeat",
        "challenge",
        "powers",
    )

    def __init__(
        self,
        adb: AdbController,
        controller: AutomationController,
        test_name: str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.adb = adb
        self.controller = controller
        self.test_name = test_name

    def run(self) -> None:
        payload: dict = {}
        try:
            image = self.adb.screenshot()
            ctrl = self.controller
            ctrl.scaler.maybe_update_from_image(image)
            annotated = image.copy()
            name = self.test_name

            if name == "screenshot":
                payload["info"] = "OK"
            elif name == "ocr":
                boxes = ctrl.ocr.recognize(image)
                payload["texts"] = [b.text for b in boxes]
                for b in boxes:
                    x, y, w, h = b.bbox
                    cv2_rect(annotated, x, y, w, h)
                    put_label(annotated, f"{b.text} {b.confidence:.2f}", x, max(0, y - 8))
            elif name == "arena":
                kind = ctrl.detector.detect(image)
                payload["screen"] = kind.value
            elif name == "powers":
                opponents = ctrl.power_reader.read_all(lambda: self.adb.screenshot())
                payload["powers"] = [
                    {
                        "id": o.id,
                        "power": o.power,
                        "region": list(o.power_region),
                        "raw": getattr(o, "raw_ocr", ""),
                    }
                    for o in opponents
                ]
                for o in opponents:
                    x, y, w, h = ctrl.scaler.scale_region(o.power_region)
                    cv2_rect(annotated, x, y, w, h)
                    label = str(o.power) if o.power is not None else "OCR_FAILED"
                    put_label(annotated, f"#{o.id} {label}", x, max(0, y - 8))
            elif name == "challenge":
                count = ctrl.challenge_reader.read(image)
                payload["count"] = None if count is None else [count.current, count.maximum]
                payload["raw"] = getattr(ctrl.challenge_reader, "last_raw", "")
                x, y, w, h = ctrl.scaler.scale_region(ctrl.config.challenge_region)
                cv2_rect(annotated, x, y, w, h)
            elif name in ("go_win", "exit", "buy", "confirm", "victory", "defeat"):
                fallback_map = {
                    "go_win": ctrl.config.profile.go_win_text,
                    "exit": ctrl.config.profile.exit_text,
                    "buy": ctrl.config.profile.buy_text,
                    "confirm": ctrl.config.profile.confirm_text,
                    "victory": ctrl.config.profile.victory_text,
                    "defeat": ctrl.config.profile.defeat_text,
                }
                result = ctrl.buttons.find(image, name, fallback_map[name])
                payload["found"] = result.found
                payload["confidence"] = result.confidence
                payload["bbox"] = list(result.bbox)
                if result.found:
                    x, y, w, h = result.bbox
                    cv2_rect(annotated, x, y, w, h)
                    put_label(
                        annotated,
                        f"{name} {result.confidence:.2f}",
                        x,
                        max(0, y - 8),
                    )

            self.result_ready.emit(name, annotated, payload)
        except Exception as exc:  # noqa: BLE001
            payload["error"] = str(exc)
            self.result_ready.emit(self.test_name, None, payload)


def cv2_rect(image: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    import cv2

    height, width = image.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(width, int(x + w)), min(height, int(y + h))
    if x1 > x0 and y1 > y0:
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)


def put_label(image: np.ndarray, text: str, x: int, y: int) -> None:
    import cv2

    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
