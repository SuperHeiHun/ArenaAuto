"""7-step calibration wizard: device → arena → opponents → ROIs → buttons → test → save."""

from __future__ import annotations

import copy
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QThread, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from arena_auto.adb.controller import AdbController
from arena_auto.calibration.analyzer import CalibrationAnalyzer
from arena_auto.calibration.button_detector import (
    BUTTON_DISPLAY,
    BUTTON_PAGES,
    BUTTON_PAGE_ORDER,
    page_targets,
    pages_for_button,
)
from arena_auto.calibration.models import (
    SOURCE_MANUAL,
    CalibrationBundle,
    CalibrationDetection,
    CalibrationPoint,
    CalibrationRegion,
)
from arena_auto.config.manager import ConfigManager
from arena_auto.gui.widgets import ndarray_to_qpixmap
from arena_auto.models.config import AppConfig
from arena_auto.paths import get_calibrated_config_path
from arena_auto.recognition.ocr import OCRProvider, create_ocr_provider
from arena_auto.recognition.template import TemplateRecognizer

logger = logging.getLogger(__name__)

STEP_TITLES = (
    "步骤 1：连接设备",
    "步骤 2：确认竞技场页面",
    "步骤 3：分析对手列表",
    "步骤 4：确认战力区域",
    "步骤 5：分步检测按钮",
    "步骤 6：测试识别",
    "步骤 7：保存配置",
)


class _OcrHolder:
    """Lazy OCR cache so engine load happens once, off the GUI thread."""

    def __init__(
        self, config: AppConfig, existing: Optional[OCRProvider] = None
    ) -> None:
        self._config = config
        self._ocr = existing

    def get(self) -> OCRProvider:
        if self._ocr is None:
            self._ocr = create_ocr_provider(self._config.ocr)
        return self._ocr


@dataclass
class CanvasItem:
    label: str
    region: CalibrationRegion
    point: CalibrationPoint
    color: Tuple[int, int, int] = (0, 220, 0)
    detection: Optional[CalibrationDetection] = None
    confidence: float = 1.0


class CalibrationCanvas(QWidget):
    """Zoom / pan / drag canvas for semi-automatic ROI adjustment."""

    region_changed = Signal(int)
    point_changed = Signal(int)
    selection_changed = Signal(int)

    HANDLE = 7
    POINT_HIT = 9

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setStyleSheet("background:#1b1b1b;")
        self._pixmap = None
        self._image_size: Tuple[int, int] = (0, 0)
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._items: List[CanvasItem] = []
        self._selected = -1
        self._mode = ""
        self._resize_handle = ""
        self._press_widget = QPointF()
        self._press_image = QPointF()
        self._start_region: Optional[CalibrationRegion] = None
        self._start_point: Optional[CalibrationPoint] = None
        self._band_end = QPointF()
        self._pan_start = QPointF()

    # ------------------------------------------------------------------ data
    def set_image(self, image: Optional[np.ndarray]) -> None:
        self._pixmap = ndarray_to_qpixmap(image) if image is not None else None
        if image is None:
            self._image_size = (0, 0)
        else:
            h, w = image.shape[:2]
            self._image_size = (w, h)
        self.fit_to_view()
        self.update()

    def set_items(self, items: List[CanvasItem], select: int = -1) -> None:
        self._items = list(items)
        self._selected = select if 0 <= select < len(self._items) else -1
        self.update()

    def items(self) -> List[CanvasItem]:
        return self._items

    def selected(self) -> int:
        return self._selected

    def select(self, index: int) -> None:
        if index != self._selected:
            self._selected = index if 0 <= index < len(self._items) else -1
            self.selection_changed.emit(self._selected)
            self.update()

    def fit_to_view(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        iw, ih = self._image_size
        if iw <= 0 or ih <= 0:
            return
        sx = self.width() / float(iw)
        sy = self.height() / float(ih)
        self._scale = max(0.02, min(sx, sy) * 0.97)
        self._offset = QPointF(
            (self.width() - iw * self._scale) / 2.0,
            (self.height() - ih * self._scale) / 2.0,
        )
        self.update()

    # ----------------------------------------------------------- transforms
    def _to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(
            x * self._scale + self._offset.x(),
            y * self._scale + self._offset.y(),
        )

    def _to_image(self, pos: QPointF) -> QPointF:
        if self._scale <= 0:
            return QPointF(-1, -1)
        return QPointF(
            (pos.x() - self._offset.x()) / self._scale,
            (pos.y() - self._offset.y()) / self._scale,
        )

    def _region_rect(self, region: CalibrationRegion) -> QRectF:
        top_left = self._to_widget(region.x, region.y)
        return QRectF(
            top_left,
            QPointF(
                top_left.x() + region.width * self._scale,
                top_left.y() + region.height * self._scale,
            ),
        )

    def _handle_rects(self, rect: QRectF) -> Dict[str, QRectF]:
        s = self.HANDLE
        return {
            "nw": QRectF(rect.left() - s, rect.top() - s, 2 * s, 2 * s),
            "ne": QRectF(rect.right() - s, rect.top() - s, 2 * s, 2 * s),
            "sw": QRectF(rect.left() - s, rect.bottom() - s, 2 * s, 2 * s),
            "se": QRectF(rect.right() - s, rect.bottom() - s, 2 * s, 2 * s),
        }

    def _clamp_region(self, region: CalibrationRegion) -> CalibrationRegion:
        iw, ih = self._image_size
        w = max(4, min(int(region.width), iw))
        h = max(4, min(int(region.height), ih))
        x = max(0, min(int(region.x), iw - w))
        y = max(0, min(int(region.y), ih - h))
        return CalibrationRegion(x=x, y=y, width=w, height=h)

    def _clamp_point(self, point: CalibrationPoint) -> CalibrationPoint:
        iw, ih = self._image_size
        return CalibrationPoint(
            x=max(0, min(int(point.x), iw)),
            y=max(0, min(int(point.y), ih)),
        )

    # ---------------------------------------------------------------- events
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_scale = max(0.02, min(8.0, self._scale * factor))
        cursor = event.position()
        image_pos = self._to_image(cursor)
        self._scale = new_scale
        self._offset = QPointF(
            cursor.x() - image_pos.x() * self._scale,
            cursor.y() - image_pos.y() * self._scale,
        )
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        pos = event.position()
        image_pos = self._to_image(pos)

        if event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            self._mode = "pan"
            self._pan_start = pos
            return

        if event.button() != Qt.LeftButton:
            return

        if 0 <= self._selected < len(self._items):
            rect = self._region_rect(self._items[self._selected].region)
            for name, handle in self._handle_rects(rect).items():
                if handle.contains(pos):
                    self._mode = "resize"
                    self._resize_handle = name
                    self._press_image = image_pos
                    self._start_region = copy.copy(
                        self._items[self._selected].region
                    )
                    return
            point = self._items[self._selected].point
            p_pos = self._to_widget(point.x, point.y)
            if (pos - p_pos).manhattanLength() <= self.POINT_HIT:
                self._mode = "point"
                self._start_point = copy.copy(point)
                self._press_image = image_pos
                return

        hit = self._hit_item(image_pos)
        if hit >= 0:
            self.select(hit)
            self._mode = "move"
            self._press_image = image_pos
            self._start_region = copy.copy(self._items[hit].region)
            return

        if event.modifiers() & Qt.ShiftModifier and 0 <= self._selected < len(
            self._items
        ):
            self._mode = "band"
            self._press_image = image_pos
            self._band_end = image_pos
            return

        self._mode = "pan"
        self._pan_start = pos

    def _hit_item(self, image_pos: QPointF) -> int:
        for index in range(len(self._items) - 1, -1, -1):
            if self._items[index].region.contains(
                int(image_pos.x()), int(image_pos.y())
            ):
                return index
        return -1

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        image_pos = self._to_image(pos)

        if self._mode == "pan":
            delta = pos - self._pan_start
            self._offset += delta
            self._pan_start = pos
            self.update()
            return

        if self._mode == "move" and self._start_region is not None:
            dx = int(round(image_pos.x() - self._press_image.x()))
            dy = int(round(image_pos.y() - self._press_image.y()))
            moved = CalibrationRegion(
                x=self._start_region.x + dx,
                y=self._start_region.y + dy,
                width=self._start_region.width,
                height=self._start_region.height,
            )
            self._items[self._selected].region = self._clamp_region(moved)
            self._sync_item(self._selected)
            self.update()
            return

        if self._mode == "resize" and self._start_region is not None:
            start = self._start_region
            x0, y0 = start.x, start.y
            x1, y1 = start.x + start.width, start.y + start.height
            nx = int(round(image_pos.x()))
            ny = int(round(image_pos.y()))
            if "w" in self._resize_handle:
                x0 = nx
            if "e" in self._resize_handle:
                x1 = nx
            if "n" in self._resize_handle:
                y0 = ny
            if "s" in self._resize_handle:
                y1 = ny
            if x1 - x0 < 6:
                x1 = x0 + 6
            if y1 - y0 < 6:
                y1 = y0 + 6
            self._items[self._selected].region = self._clamp_region(
                CalibrationRegion(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
            )
            self._sync_item(self._selected)
            self.update()
            return

        if self._mode == "point" and self._start_point is not None:
            self._items[self._selected].point = self._clamp_point(
                CalibrationPoint(
                    x=int(round(image_pos.x())), y=int(round(image_pos.y()))
                )
            )
            self._sync_item(self._selected)
            self.update()
            return

        if self._mode == "band":
            self._band_end = image_pos
            self.update()
            return

        hover_handle = ""
        if 0 <= self._selected < len(self._items):
            rect = self._region_rect(self._items[self._selected].region)
            for name, handle in self._handle_rects(rect).items():
                if handle.contains(pos):
                    hover_handle = name
                    break
        if hover_handle in ("nw", "se"):
            self.setCursor(Qt.SizeFDiagCursor)
        elif hover_handle in ("ne", "sw"):
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        mode = self._mode
        self._mode = ""
        if mode == "band" and 0 <= self._selected < len(self._items):
            x0 = int(min(self._press_image.x(), self._band_end.x()))
            y0 = int(min(self._press_image.y(), self._band_end.y()))
            x1 = int(max(self._press_image.x(), self._band_end.x()))
            y1 = int(max(self._press_image.y(), self._band_end.y()))
            if x1 - x0 >= 6 and y1 - y0 >= 6:
                self._items[self._selected].region = self._clamp_region(
                    CalibrationRegion(
                        x=x0, y=y0, width=x1 - x0, height=y1 - y0
                    )
                )
                self._sync_item(self._selected)
                self.region_changed.emit(self._selected)
            self.update()
            return
        if mode == "move" or mode == "resize":
            self.region_changed.emit(self._selected)
        elif mode == "point":
            self.point_changed.emit(self._selected)
        self.update()

    def _sync_item(self, index: int) -> None:
        item = self._items[index]
        if item.detection is not None:
            item.detection.region = copy.copy(item.region)
            item.detection.click_point = copy.copy(item.point)
            item.detection.source = SOURCE_MANUAL
            item.detection.confidence = 1.0
        item.confidence = 1.0

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap is not None and self._scale == 1.0:
            self.fit_to_view()

    # ----------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1b1b1b"))
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignCenter, "无截图")
            return
        iw, ih = self._image_size
        target = QRectF(
            self._offset.x(),
            self._offset.y(),
            iw * self._scale,
            ih * self._scale,
        )
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._pixmap, QRectF(0, 0, iw, ih))

        font = QFont("Segoe UI", max(8, int(9 * max(self._scale, 0.6))))
        painter.setFont(font)
        for index, item in enumerate(self._items):
            rect = self._region_rect(item.region)
            selected = index == self._selected
            color = QColor(*item.color)
            pen = QPen(color, 3 if selected else 2)
            if selected:
                pen.setColor(QColor("#ffffff"))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

            label = item.label
            if selected:
                for handle in self._handle_rects(rect).values():
                    painter.fillRect(handle, color)
                    painter.setPen(QPen(QColor("#222"), 1))
                    painter.drawRect(handle)
                    painter.setPen(pen)

            point = item.point
            center = self._to_widget(point.x, point.y)
            painter.setPen(QPen(QColor("#ff00ff"), 2))
            painter.setBrush(QColor("#ff00ff"))
            painter.drawEllipse(center, 5, 5)
            painter.setBrush(Qt.NoBrush)

            text_rect = QRectF(rect.left(), rect.top() - 20, rect.width(), 18)
            painter.fillRect(text_rect, QColor(0, 0, 0, 160))
            painter.setPen(QColor("#fff"))
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, label)

        if self._mode == "band":
            band = QRectF(
                self._to_widget(self._press_image.x(), self._press_image.y()),
                self._to_widget(self._band_end.x(), self._band_end.y()),
            ).normalized()
            painter.setPen(QPen(QColor("#00ffff"), 1, Qt.DashLine))
            painter.setBrush(QColor(0, 255, 255, 40))
            painter.drawRect(band)


class CalibrationWorker(QThread):
    """Runs calibration tasks off the GUI thread."""

    result_ready = Signal(str, object)
    failed = Signal(str, str)
    progress = Signal(str)

    def __init__(
        self,
        task: str,
        adb: AdbController,
        config_manager: ConfigManager,
        ocr_holder: _OcrHolder,
        templates: Optional[TemplateRecognizer] = None,
        bundle: Optional[CalibrationBundle] = None,
        page: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.task = task
        self.adb = adb
        self.config_manager = config_manager
        self.ocr_holder = ocr_holder
        self.templates = templates
        self.bundle = bundle
        self.page = page

    def run(self) -> None:
        try:
            if self.task == "devices":
                devices = self.adb.list_devices()
                self.result_ready.emit("devices", {"devices": devices})
                return

            if self.task == "screenshot":
                image = self.adb.screenshot()
                self.result_ready.emit("screenshot", {"image": image})
                return

            if self.task == "arena":
                self.progress.emit("截图并识别页面...")
                image = self.adb.screenshot()
                config = self.config_manager.config
                analyzer = CalibrationAnalyzer(
                    config, self.ocr_holder.get(), self.templates
                )
                boxes = analyzer.screen_analyzer.recognize(image)
                kind, ok, conf = analyzer.screen_analyzer.detect_screen(image, boxes)
                overlay = analyzer.screen_analyzer.draw_ocr_overlay(image, boxes)
                self.result_ready.emit(
                    "arena",
                    {
                        "image": image,
                        "overlay": overlay,
                        "kind": kind.value,
                        "ok": ok,
                        "confidence": conf,
                    },
                )
                return

            if self.task == "analyze":
                self.progress.emit("截图中...")
                image = self.adb.screenshot()
                self.progress.emit("全屏 OCR 分析中...")
                config = self.config_manager.config
                analyzer = CalibrationAnalyzer(
                    config, self.ocr_holder.get(), self.templates
                )
                bundle = analyzer.analyze(image)
                annotated = analyzer.draw_bundle(image, bundle)
                self.result_ready.emit(
                    "analyze",
                    {
                        "bundle": bundle,
                        "image": image,
                        "annotated": annotated,
                    },
                )
                return

            if self.task == "buttons":
                self.progress.emit("截图并检测当前页面按钮...")
                image = self.adb.screenshot()
                config = self.config_manager.config
                analyzer = CalibrationAnalyzer(
                    config, self.ocr_holder.get(), self.templates
                )
                payload = analyzer.detect_page_buttons(image, self.page or "arena")
                self.result_ready.emit("buttons", payload)
                return

            if self.task == "verify":
                self.progress.emit("验证配置（不会点击）...")
                config = self.config_manager.config
                analyzer = CalibrationAnalyzer(
                    config, self.ocr_holder.get(), self.templates
                )
                summary = analyzer.verify(
                    lambda: self.adb.screenshot(), bundle=self.bundle
                )
                self.result_ready.emit("verify", {"summary": summary})
                return

            self.failed.emit(self.task, f"未知任务: {self.task}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("CalibrationWorker 任务失败: %s", self.task)
            self.failed.emit(self.task, str(exc))


class CalibrationWizard(QDialog):
    """Guided auto-calibration: 7 steps, visualization, manual override, safe save."""

    def __init__(
        self,
        config_manager: ConfigManager,
        adb: AdbController,
        ocr_provider: Optional[OCRProvider] = None,
        templates: Optional[TemplateRecognizer] = None,
        debug=None,
        recalibrate: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "ArenaAuto - 重新标定" if recalibrate else "ArenaAuto - 自动配置"
        )
        self.resize(1280, 860)
        self.config_manager = config_manager
        self.adb = adb
        self.templates = templates
        self.debug = debug
        self.recalibrate = recalibrate
        self.ocr_holder = _OcrHolder(config_manager.config, ocr_provider)

        self.bundle: Optional[CalibrationBundle] = None
        self.screenshot: Optional[np.ndarray] = None
        self.annotated: Optional[np.ndarray] = None
        self.arena_ok = False
        self.arena_manual_ok = False
        self.verify_summary: Optional[dict] = None
        self.saved_path: Optional[Path] = None
        self.devices: List[str] = []
        self._workers: List[QThread] = []
        self._mode = "power"  # power | button | challenge

        # page-wise button detection state
        self.button_pages_attempted: set = set()
        self.button_found_page: Dict[str, str] = {}
        self._button_row_names: List[str] = []
        self._button_list_to_canvas: List[int] = []
        self._canvas_to_list: Dict[int, int] = {}

        self._build_ui()
        self._goto(0)

    # ------------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.step_label = QLabel(STEP_TITLES[0])
        self.step_label.setStyleSheet(
            "font-size:16px;font-weight:bold;color:#2a6;padding:4px;"
        )
        layout.addWidget(self.step_label)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, stretch=1)

        self.stack.addWidget(self._build_device_step())
        self.stack.addWidget(self._build_arena_step())
        self.stack.addWidget(self._build_opponents_step())
        self.stack.addWidget(self._build_power_step())
        self.stack.addWidget(self._build_buttons_step())
        self.stack.addWidget(self._build_test_step())
        self.stack.addWidget(self._build_save_step())

        nav = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#666;")
        nav.addWidget(self.status_label, stretch=1)
        self.prev_btn = QPushButton("上一步")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn = QPushButton("下一步")
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)

    def _build_device_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        hint = QLabel(
            "自动获取 adb devices。单设备自动选择；多设备请手动选择；"
            "未检测到设备无法继续。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        box = QGroupBox("Android 设备")
        form = QFormLayout(box)
        self.device_combo = QComboBox()
        form.addRow("设备", self.device_combo)
        self.device_status = QLabel("检测中...")
        form.addRow("状态", self.device_status)
        layout.addWidget(box)

        row = QHBoxLayout()
        self.refresh_devices_btn = QPushButton("刷新设备")
        self.refresh_devices_btn.clicked.connect(self._start_devices)
        row.addWidget(self.refresh_devices_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self.device_log = QPlainTextEdit()
        self.device_log.setReadOnly(True)
        self.device_log.setMaximumBlockCount(500)
        layout.addWidget(self.device_log, stretch=1)
        return widget

    def _build_arena_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        hint = QLabel(
            "确认当前游戏画面是竞技场页面（综合检测：胜利者的竞技场 / "
            "请选择对手 / 可挑战次数）。识别失败可手动确认后继续。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.arena_image = CalibrationCanvas()
        layout.addWidget(self.arena_image, stretch=1)

        row = QHBoxLayout()
        self.arena_detect_btn = QPushButton("截图并检测")
        self.arena_detect_btn.clicked.connect(lambda: self._start_task("arena"))
        self.arena_manual_check = QCheckBox("我已确认当前是竞技场页面")
        self.arena_manual_check.toggled.connect(self._on_arena_manual)
        row.addWidget(self.arena_detect_btn)
        row.addWidget(self.arena_manual_check)
        row.addStretch(1)
        layout.addLayout(row)

        self.arena_result = QLabel("尚未检测")
        self.arena_result.setWordWrap(True)
        layout.addWidget(self.arena_result)
        return widget

    def _build_opponents_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        hint = QLabel("自动分析 1~5 对手的重复卡片结构与战力区域。")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)
        self.opponent_image = CalibrationCanvas()
        splitter.addWidget(self.opponent_image)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.opponent_list = QListWidget()
        right_layout.addWidget(self.opponent_list, stretch=1)
        self.analyze_btn = QPushButton("重新分析")
        self.analyze_btn.clicked.connect(lambda: self._start_task("analyze"))
        right_layout.addWidget(self.analyze_btn)
        self.opponent_notes = QPlainTextEdit()
        self.opponent_notes.setReadOnly(True)
        self.opponent_notes.setMaximumBlockCount(300)
        right_layout.addWidget(self.opponent_notes)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)
        return widget

    def _build_power_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        hint = QLabel(
            "拖动 ROI 调整战力区域，拖动粉色点设置点击位置；"
            "Shift+拖动可框选，滚轮缩放，右键拖动平移。手动修改将标记为 manual。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)
        self.power_canvas = CalibrationCanvas()
        self.power_canvas.region_changed.connect(
            lambda i: self._on_manual_change(i, "power")
        )
        self.power_canvas.point_changed.connect(
            lambda i: self._on_manual_change(i, "power")
        )
        self.power_canvas.selection_changed.connect(
            lambda i: self.power_list.setCurrentRow(i)
        )
        splitter.addWidget(self.power_canvas)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.power_list = QListWidget()
        self.power_list.currentRowChanged.connect(self._on_power_row)
        right_layout.addWidget(self.power_list, stretch=1)
        self.power_detail = QLabel("选择一项查看详细信息")
        self.power_detail.setWordWrap(True)
        right_layout.addWidget(self.power_detail)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)
        return widget

    def _build_buttons_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        hint = QLabel(
            "按钮按页面分步检测：先在竞技场页检测，再依次进入购买弹窗"
            "（挑战次数为 0 时点对手弹出）/ 胜利结算 / 失败结算页检测，"
            "结果会合并保存。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666;")
        layout.addWidget(hint)

        page_row = QHBoxLayout()
        page_row.addWidget(QLabel("检测页面:"))
        self.button_page_combo = QComboBox()
        for key in BUTTON_PAGE_ORDER:
            self.button_page_combo.addItem(BUTTON_PAGES[key]["title"], key)
        self.button_page_combo.currentIndexChanged.connect(
            self._on_button_page_changed
        )
        page_row.addWidget(self.button_page_combo, stretch=1)
        self.detect_page_btn = QPushButton("检测当前页面")
        self.detect_page_btn.clicked.connect(
            lambda: self._start_task("buttons")
        )
        self.detect_page_btn.setStyleSheet(
            "background:#2a6;color:#fff;font-weight:bold;padding:6px 14px;"
        )
        page_row.addWidget(self.detect_page_btn)
        self.reanalyze_btn = QPushButton("重新分析竞技场")
        self.reanalyze_btn.setToolTip(
            "重新截图分析对手/战力/挑战次数（已检出的其它页面按钮会保留）"
        )
        self.reanalyze_btn.clicked.connect(lambda: self._start_task("analyze"))
        page_row.addWidget(self.reanalyze_btn)
        layout.addLayout(page_row)

        self.button_hint = QLabel(BUTTON_PAGES["arena"]["hint"])  # type: ignore[arg-type]
        self.button_hint.setWordWrap(True)
        self.button_hint.setStyleSheet(
            "color:#2a6;background:#1e2a1e;padding:6px;border-radius:4px;"
        )
        layout.addWidget(self.button_hint)

        splitter = QSplitter(Qt.Horizontal)
        self.button_canvas = CalibrationCanvas()
        self.button_canvas.region_changed.connect(
            lambda i: self._on_manual_change(i, "button")
        )
        self.button_canvas.point_changed.connect(
            lambda i: self._on_manual_change(i, "button")
        )
        self.button_canvas.selection_changed.connect(
            lambda i: self._select_button_list_row(i)
        )
        splitter.addWidget(self.button_canvas)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.button_list = QListWidget()
        self.button_list.currentRowChanged.connect(self._on_button_row)
        right_layout.addWidget(self.button_list, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)
        return widget

    def _on_button_page_changed(self, _index: int) -> None:
        key = self.button_page_combo.currentData() or "arena"
        self.button_hint.setText(BUTTON_PAGES[key]["hint"])  # type: ignore[arg-type]

    def _select_button_list_row(self, canvas_index: int) -> None:
        row = self._canvas_to_list.get(canvas_index, -1)
        if 0 <= row < self.button_list.count():
            if self.button_list.currentRow() != row:
                self.button_list.blockSignals(True)
                self.button_list.setCurrentRow(row)
                self.button_list.blockSignals(False)

    def _build_test_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        hint = QLabel(
            "重新截图并按标定结果识别（读取战力 / 挑战次数 / 按钮）。"
            "测试过程禁止真正点击。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        self.test_btn = QPushButton("运行测试")
        self.test_btn.clicked.connect(lambda: self._start_task("verify"))
        row.addWidget(self.test_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self.test_result = QPlainTextEdit()
        self.test_result.setReadOnly(True)
        layout.addWidget(self.test_result, stretch=1)
        return widget

    def _build_save_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("自动配置完成情况")
        grid = QGridLayout(group)
        self.summary_labels: Dict[str, QLabel] = {}
        rows = (
            "arena",
            "opponents",
            "powers",
            "challenge",
            "go_win",
            "exit",
            "victory",
            "defeat",
            "buy",
            "confirm",
            "verify",
        )
        titles = {
            "arena": "竞技场",
            "opponents": "对手",
            "powers": "战力区域",
            "challenge": "挑战次数",
            "go_win": "去获胜",
            "exit": "退出",
            "victory": "胜利",
            "defeat": "失败",
            "buy": "购买",
            "confirm": "确认",
            "verify": "配置验证",
        }
        for row_index, key in enumerate(rows):
            title = QLabel(titles[key])
            value = QLabel("-")
            value.setWordWrap(True)
            self.summary_labels[key] = value
            grid.addWidget(title, row_index, 0)
            grid.addWidget(value, row_index, 1)
        layout.addWidget(group)

        self.save_notes = QPlainTextEdit()
        self.save_notes.setReadOnly(True)
        self.save_notes.setMaximumBlockCount(400)
        layout.addWidget(self.save_notes, stretch=1)

        row = QHBoxLayout()
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self._save_config)
        self.restart_btn = QPushButton("重新标定")
        self.restart_btn.clicked.connect(self._restart)
        row.addWidget(self.save_btn)
        row.addWidget(self.restart_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self.save_status = QLabel("")
        self.save_status.setWordWrap(True)
        layout.addWidget(self.save_status)
        return widget

    # ------------------------------------------------------------- navigation
    def _goto(self, index: int) -> None:
        index = max(0, min(index, self.stack.count() - 1))
        self.stack.setCurrentIndex(index)
        self.step_label.setText(
            f"{STEP_TITLES[index]}  （{index + 1} / {self.stack.count()}）"
        )
        self.prev_btn.setEnabled(index > 0)
        self._refresh_next_state()
        if index == 0:
            self._start_devices()
        elif index == 1 and self.screenshot is None:
            self._start_task("arena")
        elif index == 2 and self.bundle is None:
            self._start_task("analyze")
        elif index == 2:
            self._render_opponents()
        elif index == 3:
            self._render_power()
        elif index == 4:
            self._render_buttons()
        elif index == 6:
            self._render_summary()

    def _prev(self) -> None:
        self._goto(self.stack.currentIndex() - 1)

    def _next(self) -> None:
        if not self.next_btn.isEnabled():
            return
        self._goto(self.stack.currentIndex() + 1)

    def _refresh_next_state(self) -> None:
        index = self.stack.currentIndex()
        enabled = True
        if index == 0:
            enabled = bool(self.devices) and self.device_combo.count() > 0
        elif index == 1:
            enabled = self.arena_ok or self.arena_manual_ok
        elif index == 2:
            enabled = self.bundle is not None
        self.next_btn.setEnabled(enabled)

    # ---------------------------------------------------------------- workers
    def _start_devices(self) -> None:
        self.device_status.setText("检测中...")
        self._start_task("devices")

    def _start_task(self, task: str) -> None:
        self.status_label.setText(f"运行: {task} ...")
        page = "arena"
        if hasattr(self, "button_page_combo"):
            page = self.button_page_combo.currentData() or "arena"
        worker = CalibrationWorker(
            task=task,
            adb=self.adb,
            config_manager=self.config_manager,
            ocr_holder=self.ocr_holder,
            templates=self.templates,
            bundle=self.bundle,
            page=page,
            parent=self,
        )
        worker.result_ready.connect(self._on_worker_result)
        worker.failed.connect(self._on_worker_failed)
        worker.progress.connect(
            lambda msg: self.status_label.setText(msg)
        )
        worker.finished.connect(
            lambda w=worker: self._cleanup_worker(w)
        )
        self._workers.append(worker)
        worker.start()

    def _cleanup_worker(self, worker: QThread) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()
        self.status_label.setText("就绪")

    def _on_worker_failed(self, task: str, message: str) -> None:
        self.status_label.setText(f"失败: {message}")
        self.device_log.appendPlainText(f"[{task}] 失败: {message}")
        QMessageBox.warning(self, "自动配置", f"{task} 失败:\n{message}")

    def _on_worker_result(self, task: str, payload: dict) -> None:
        if task == "devices":
            self.devices = list(payload.get("devices") or [])
            self.device_combo.clear()
            if not self.devices:
                self.device_status.setText(
                    "未检测到 Android 设备，请确认 ADB 已连接"
                )
                self.device_log.appendPlainText("未检测到 Android 设备")
            else:
                for serial in self.devices:
                    self.device_combo.addItem(serial)
                if len(self.devices) == 1:
                    self.device_status.setText(f"自动选择设备: {self.devices[0]}")
                else:
                    self.device_status.setText(
                        f"检测到 {len(self.devices)} 台设备，请选择"
                    )
            self._refresh_next_state()
            return

        if task == "screenshot":
            image = payload.get("image")
            if image is not None:
                self.arena_image.set_image(image)
            return

        if task == "arena":
            self.screenshot = payload.get("image")
            overlay = payload.get("overlay")
            if overlay is None:
                overlay = payload.get("image")
            self.arena_image.set_image(overlay)
            ok = bool(payload.get("ok"))
            conf = float(payload.get("confidence") or 0.0)
            kind = payload.get("kind") or "unknown"
            self.arena_ok = ok
            if ok:
                self.arena_result.setText(
                    f"✓ 识别为竞技场页面 (kind={kind}, confidence={conf:.2f})"
                )
                self.arena_result.setStyleSheet("color:#2a6;font-weight:bold;")
            else:
                self.arena_result.setText(
                    f"⚠ 未自动确认竞技场 (kind={kind})，"
                    "请确认画面后勾选手动确认"
                )
                self.arena_result.setStyleSheet("color:#c60;font-weight:bold;")
            self._refresh_next_state()
            return

        if task == "analyze":
            new_bundle = payload.get("bundle")
            # keep buttons/templates discovered on other pages
            if self.bundle is not None and new_bundle is not None:
                merged = dict(self.bundle.buttons)
                merged.update(new_bundle.buttons)
                new_bundle.buttons = merged
                if new_bundle.challenge is None:
                    new_bundle.challenge = self.bundle.challenge
                for key, value in self.bundle.generated_templates.items():
                    new_bundle.generated_templates.setdefault(key, value)
            self.bundle = new_bundle
            self.screenshot = payload.get("image")
            self.annotated = payload.get("annotated")
            self.verify_summary = None
            if self.bundle is not None:
                self._mark_pages_attempted_from_kind(self.bundle.screen_kind)
            self.arena_ok = bool(self.bundle and self.bundle.arena_detected)
            if self.bundle and self.bundle.arena_detected:
                self.arena_manual_check.setChecked(True)
            self._render_opponents()
            self._render_power()
            self._render_buttons()
            self._refresh_next_state()
            if self.bundle:
                low = self.bundle.low_confidence_items
                if low:
                    self.status_label.setText(
                        f"⚠ {len(low)} 项置信度较低，请在后续步骤调整"
                    )
                else:
                    self.status_label.setText("分析完成")
            return

        if task == "buttons":
            page = payload.get("page") or "arena"
            image = payload.get("image")
            if image is not None:
                self.screenshot = image
                self.annotated = payload.get("overlay")
            detected = payload.get("detected") or {}
            if self.bundle is None:
                height, width = (image.shape[:2] if image is not None else (0, 0))
                self.bundle = CalibrationBundle(
                    screen_width=width,
                    screen_height=height,
                    reference_width=self.config_manager.config.screen.reference_width,
                    reference_height=self.config_manager.config.screen.reference_height,
                )
            for name, det in detected.items():
                self.bundle.buttons[name] = det
                self.button_found_page[name] = page
            for name, path in (payload.get("templates") or {}).items():
                self.bundle.generated_templates[name] = path
            self.button_pages_attempted.add(page)
            self._mark_pages_attempted_from_kind(payload.get("kind") or "")
            self._render_buttons()
            found = len(detected)
            missing = payload.get("missing") or []
            if missing:
                self.status_label.setText(
                    f"[{page}] 检出 {found} 项，未找到: {', '.join(missing)}"
                )
            else:
                self.status_label.setText(f"[{page}] 本页按钮全部检出")
            return

        if task == "verify":
            self.verify_summary = payload.get("summary")
            self.test_result.setPlainText(self._format_verify(self.verify_summary))
            self.status_label.setText("测试完成（未点击）")
            return

    def _format_verify(self, summary: Optional[dict]) -> str:
        if not summary:
            return "尚未测试"
        lines = ["配置测试结果", ""]
        for check in summary.get("checks") or []:
            mark = "✓" if check.get("ok") else "✗"
            lines.append(f"{mark} {check.get('name')}: {check.get('detail')}")
        lines.append("")
        lines.append(
            f"通过 {summary.get('passed', 0)}/{summary.get('total', 0)}"
        )
        if not summary.get("all_ok"):
            lines.append("⚠ 配置可能存在问题")
        return "\n".join(lines)

    # --------------------------------------------------------------- render
    def _color_for(self, confidence: float) -> Tuple[int, int, int]:
        cal = self.config_manager.config.calibration
        if confidence >= cal.high_confidence:
            return (0, 220, 0)
        if confidence >= cal.low_confidence:
            return (0, 200, 255)
        return (220, 60, 0)

    def _power_items(self) -> List[CanvasItem]:
        items: List[CanvasItem] = []
        if self.bundle is None:
            return items
        for det in self.bundle.opponents:
            label = det.name.replace("opponent_", "#")
            value = f"{det.value:,}" if det.value else "OCR缺失"
            items.append(
                CanvasItem(
                    label=f"{label} 战力：{value} {int(det.confidence * 100)}%",
                    region=det.region,
                    point=det.click_point,
                    color=self._color_for(det.confidence),
                    detection=det,
                    confidence=det.confidence,
                )
            )
        return items

    def _button_items(self) -> Tuple[List[CanvasItem], List[int]]:
        """Canvas items for found buttons (display order) + challenge.

        Returns (items, list_row_to_canvas_index) — -1 when a list row has
        no canvas counterpart (not yet detected).
        """
        items: List[CanvasItem] = []
        mapping: List[int] = []
        bundle = self.bundle
        for name, _home in BUTTON_DISPLAY:
            det = bundle.buttons.get(name) if bundle else None
            if det is None:
                mapping.append(-1)
                continue
            mapping.append(len(items))
            items.append(
                CanvasItem(
                    label=f"{name} {int(det.confidence * 100)}%",
                    region=det.region,
                    point=det.click_point,
                    color=self._color_for(det.confidence),
                    detection=det,
                    confidence=det.confidence,
                )
            )
        if bundle is not None and bundle.challenge is not None:
            det = bundle.challenge
            mapping.append(len(items))  # challenge row (appended after buttons)
            items.append(
                CanvasItem(
                    label=f"challenge {det.detail} {int(det.confidence * 100)}%",
                    region=det.region,
                    point=det.click_point,
                    color=self._color_for(det.confidence),
                    detection=det,
                    confidence=det.confidence,
                )
            )
        return items, mapping

    def _mark_pages_attempted_from_kind(self, kind: str) -> None:
        if kind == "arena":
            self.button_pages_attempted.add("arena")
        elif kind == "purchase":
            self.button_pages_attempted.add("purchase")
        elif kind == "result" and self.bundle is not None:
            # victory vs defeat is ambiguous — mark only what was actually found
            if "victory" in self.bundle.buttons:
                self.button_pages_attempted.add("victory")
            if "defeat" in self.bundle.buttons:
                self.button_pages_attempted.add("defeat")

    def _button_missed(self, name: str) -> bool:
        """True when some attempted page should have contained this button."""
        if self.bundle is not None and name in self.bundle.buttons:
            return False
        for page in self.button_pages_attempted:
            if name in page_targets(page):
                return True
        return False

    def _render_opponents(self) -> None:
        if self.bundle is None:
            return
        image = self.annotated if self.annotated is not None else self.screenshot
        if image is not None:
            self.opponent_image.set_image(image)
            self.opponent_image.set_items(self._power_items())
        self.opponent_list.clear()
        for det in self.bundle.opponents:
            label = det.name.replace("opponent_", "#")
            value = f"{det.value:,}" if det.value else "OCR_FAILED"
            conf = int(det.confidence * 100)
            if det.confidence < self.config_manager.config.calibration.low_confidence:
                mark = "⚠"
            else:
                mark = "✓"
            text = (
                f"{label}  战力：{value}  {conf}%  {mark}\n"
                f"    ROI: {list(det.region.as_tuple())}  来源: {det.source}"
            )
            item = QListWidgetItem(text)
            if mark == "⚠":
                item.setForeground(QColor("#c60"))
            self.opponent_list.addItem(item)
        found = self.bundle.opponents_found
        self.opponent_list.addItem(
            QListWidgetItem(f"合计: {found}/5 个对手")
        )
        notes = list(self.bundle.notes) + [f"⚠ {w}" for w in self.bundle.warnings]
        self.opponent_notes.setPlainText("\n".join(notes) if notes else "无")

    def _render_power(self) -> None:
        if self.screenshot is None:
            return
        self.power_canvas.set_image(self.screenshot)
        items = self._power_items()
        self.power_canvas.set_items(items, select=0)
        self.power_list.clear()
        for item in items:
            self.power_list.addItem(item.label)
        self._update_power_detail(0 if items else -1)

    def _render_buttons(self) -> None:
        if self.screenshot is not None:
            self.button_canvas.set_image(self.screenshot)
        items, mapping = self._button_items()
        self.button_canvas.set_items(items, select=0 if items else -1)
        self._button_list_to_canvas = mapping
        self._canvas_to_list = {
            canvas_index: row
            for row, canvas_index in enumerate(mapping)
            if canvas_index >= 0
        }

        self.button_list.clear()
        self._button_row_names = []
        bundle = self.bundle
        cal = self.config_manager.config.calibration

        for name, home in BUTTON_DISPLAY:
            det = bundle.buttons.get(name) if bundle else None
            found_page = self.button_found_page.get(name, home)
            page_title = BUTTON_PAGES[found_page]["title"]  # type: ignore[index]
            self._button_row_names.append(name)
            if det is not None:
                mark = (
                    "⚠"
                    if det.confidence < cal.low_confidence
                    else "✓"
                )
                text = (
                    f"{page_title}  {name}: {mark} "
                    f"{int(det.confidence * 100)}%  "
                    f"ROI={list(det.region.as_tuple())}"
                )
                item = QListWidgetItem(text)
                item.setForeground(
                    QColor("#c60") if mark == "⚠" else QColor("#2a6")
                )
            elif self._button_missed(name):
                others = [
                    BUTTON_PAGES[p]["title"]  # type: ignore[index]
                    for p in pages_for_button(name)
                    if p not in self.button_pages_attempted
                ]
                retry = f"，也可到 {'/'.join(others)} 重试" if others else ""
                text = (
                    f"{page_title}  {name}: ○ 已检测未找到{retry}"
                )
                item = QListWidgetItem(text)
                item.setForeground(QColor("#c60"))
            else:
                text = (
                    f"{page_title}  {name}: · 未检测 — "
                    "请切换到该页面后点「检测当前页面」"
                )
                item = QListWidgetItem(text)
                item.setForeground(QColor("#888"))
            self.button_list.addItem(item)

        # challenge row (arena page)
        self._button_row_names.append("challenge_count")
        if bundle is not None and bundle.challenge is not None:
            det = bundle.challenge
            item = QListWidgetItem(
                f"{BUTTON_PAGES['arena']['title']}  challenge_count: ✓ "  # type: ignore[index]
                f"{det.detail} {int(det.confidence * 100)}%"
            )
            item.setForeground(QColor("#2a6"))
        elif "arena" in self.button_pages_attempted:
            item = QListWidgetItem(
                f"{BUTTON_PAGES['arena']['title']}  challenge_count: ○ 未找到"  # type: ignore[index]
            )
            item.setForeground(QColor("#c60"))
        else:
            item = QListWidgetItem(
                f"{BUTTON_PAGES['arena']['title']}  challenge_count: · 未检测"  # type: ignore[index]
            )
            item.setForeground(QColor("#888"))
        self.button_list.addItem(item)

    def _render_summary(self) -> None:
        bundle = self.bundle
        cal = self.config_manager.config.calibration
        if bundle is None:
            for label in self.summary_labels.values():
                label.setText("无分析数据")
            return

        self.summary_labels["arena"].setText(
            "✓" if bundle.arena_detected else "⚠ 未自动确认"
        )
        found = bundle.opponents_found
        self.summary_labels["opponents"].setText(f"{found} / 5")
        powers_ok = sum(
            1
            for d in bundle.opponents
            if d.region.width > 0 and d.confidence >= cal.low_confidence
        )
        self.summary_labels["powers"].setText(
            f"{powers_ok} / 5"
            + ("" if powers_ok == 5 else "  （低置信度项请调整）")
        )
        self.summary_labels["challenge"].setText(
            "✓" if bundle.challenge is not None else "✗ 未找到"
        )
        mapping = {
            "go_win": "go_win",
            "exit": "exit",
            "victory": "victory",
            "defeat": "defeat",
            "buy": "buy_challenge",
            "confirm": "confirm_buy",
        }
        for key, name in mapping.items():
            det = bundle.buttons.get(name)
            if det is None:
                if self._button_missed(name):
                    self.summary_labels[key].setText("○ 已检测未找到")
                else:
                    self.summary_labels[key].setText("· 未检测（步骤5）")
            elif det.confidence < cal.low_confidence:
                self.summary_labels[key].setText(
                    f"⚠ {int(det.confidence * 100)}%"
                )
            else:
                self.summary_labels[key].setText(
                    f"✓ {int(det.confidence * 100)}%"
                )
        if self.verify_summary is None:
            self.summary_labels["verify"].setText("未测试（建议先测试）")
        elif self.verify_summary.get("all_ok"):
            self.summary_labels["verify"].setText("✓ 配置验证成功")
        else:
            passed = self.verify_summary.get("passed", 0)
            total = self.verify_summary.get("total", 0)
            self.summary_labels["verify"].setText(
                f"⚠ 配置可能存在问题 ({passed}/{total})"
            )

        notes = list(bundle.notes) + [f"⚠ {w}" for w in bundle.warnings]
        if bundle.generated_templates:
            notes.append(
                "生成模板: " + ", ".join(sorted(bundle.generated_templates))
            )
        notes.append("调试图: debug/calibration/")
        self.save_notes.setPlainText("\n".join(notes))

    # ------------------------------------------------------- manual adjustment
    def _on_power_row(self, row: int) -> None:
        if row >= 0:
            self.power_canvas.select(row)
            self._update_power_detail(row)

    def _update_power_detail(self, row: int) -> None:
        if self.bundle is None or not (0 <= row < len(self.bundle.opponents)):
            self.power_detail.setText("选择一项查看详细信息")
            return
        det = self.bundle.opponents[row]
        value = f"{det.value:,}" if det.value else "OCR_FAILED"
        self.power_detail.setText(
            f"#{row + 1}\n"
            f"战力：{value}\n"
            f"ROI：{list(det.region.as_tuple())}\n"
            f"点击：{list(det.click_point.as_tuple())}\n"
            f"Confidence：{det.confidence:.2f}\n"
            f"来源：{det.source}"
        )

    def _on_button_row(self, row: int) -> None:
        if 0 <= row < len(self._button_list_to_canvas):
            canvas_index = self._button_list_to_canvas[row]
            if canvas_index >= 0:
                self.button_canvas.select(canvas_index)

    def _on_manual_change(self, index: int, mode: str) -> None:
        if self.bundle is None or index < 0:
            return
        if mode == "power":
            if index < len(self.bundle.opponents):
                det = self.bundle.opponents[index]
                det.source = SOURCE_MANUAL
                det.confidence = 1.0
                item = self.power_canvas.items()[index]
                item.confidence = 1.0
                if index < self.power_list.count():
                    value = f"{det.value:,}" if det.value else "OCR缺失"
                    self.power_list.item(index).setText(
                        f"#{index + 1}  {value}  100%  ✓\n"
                        f"    ROI: {list(det.region.as_tuple())}  来源: manual"
                    )
                self._update_power_detail(index)
                self.power_list.item(index).setForeground(QColor("#2a6"))
        else:
            items = self.button_canvas.items()
            if index >= len(items):
                return
            det = items[index].detection
            if det is not None:
                det.source = SOURCE_MANUAL
                det.confidence = 1.0
            list_row = self._canvas_to_list.get(index, -1)
            if (
                0 <= list_row < self.button_list.count()
                and list_row < len(self._button_row_names)
            ):
                name = self._button_row_names[list_row]
                det2 = self.bundle.buttons.get(name) if self.bundle else None
                if det2 is not None:
                    page_key = self.button_found_page.get(name, "arena")
                    title = BUTTON_PAGES[page_key]["title"]  # type: ignore[index]
                    self.button_list.item(list_row).setText(
                        f"{title}  {name}: ✓ 100%  "
                        f"ROI={list(det2.region.as_tuple())}"
                    )
                self.button_list.item(list_row).setForeground(QColor("#2a6"))

    def _on_arena_manual(self, checked: bool) -> None:
        self.arena_manual_ok = checked
        self._refresh_next_state()

    # ------------------------------------------------------------------ save
    def _save_config(self) -> None:
        if self.bundle is None:
            QMessageBox.warning(self, "保存", "尚未分析，请先完成自动分析")
            return

        main_path = self.config_manager.path
        exists = main_path.is_file()
        overwrite = False
        if exists:
            box = QMessageBox(self)
            box.setWindowTitle("发现已有配置")
            box.setText(
                f"发现已有配置。\n\n{main_path}\n\n"
                "默认另存为新配置（config.calibrated.yaml），"
                "不会覆盖现有 config.yaml。"
            )
            save_as = box.addButton(
                "另存为新配置", QMessageBox.AcceptRole
            )
            overwrite_btn = box.addButton(
                "覆盖现有配置", QMessageBox.DestructiveRole
            )
            box.addButton("取消", QMessageBox.RejectRole)
            box.setDefaultButton(save_as)  # type: ignore[arg-type]
            box.exec()
            clicked = box.clickedButton()
            if clicked == save_as:
                overwrite = False
            elif clicked == overwrite_btn:
                overwrite = True
            else:
                self.save_status.setText("已取消保存")
                return

        try:
            new_config = copy.deepcopy(self.config_manager.config)
            analyzer = CalibrationAnalyzer(
                new_config, self.ocr_holder.get(), self.templates
            )
            analyzer.apply_bundle(
                new_config, self.bundle, overwrite_manual=overwrite or self.recalibrate
            )
            new_config.version = 2

            if overwrite:
                target = main_path
                manager = ConfigManager(path=target, config=new_config)
                manager.save()
                self.config_manager.config = manager.config
                self.saved_path = target
            else:
                target = get_calibrated_config_path()
                manager = ConfigManager(path=target, config=new_config)
                manager.save()
                self.saved_path = target

            self.save_status.setText(f"✓ 配置已保存: {self.saved_path}")
            self.save_status.setStyleSheet("color:#2a6;font-weight:bold;")
            logger.info("[CALIBRATION] 配置已保存: %s", self.saved_path)
            QMessageBox.information(
                self,
                "自动配置",
                f"自动配置完成\n\n配置文件:\n{self.saved_path}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("保存配置失败")
            QMessageBox.critical(self, "保存配置", f"保存失败:\n{exc}")

    def _restart(self) -> None:
        self.bundle = None
        self.annotated = None
        self.verify_summary = None
        self.saved_path = None
        self.arena_ok = False
        self.button_pages_attempted.clear()
        self.button_found_page.clear()
        self._button_row_names = []
        self._button_list_to_canvas = []
        self._canvas_to_list = {}
        if hasattr(self, "button_page_combo"):
            self.button_page_combo.setCurrentIndex(0)
        self.arena_manual_check.setChecked(False)
        self.test_result.clear()
        self.save_status.setText("")
        self._goto(1)

    # ----------------------------------------------------------------- misc
    def open_debug_calibration(self) -> None:
        from arena_auto.paths import get_debug_dir

        path = get_debug_dir() / "calibration"
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def closeEvent(self, event) -> None:  # noqa: N802
        for worker in self._workers:
            try:
                worker.wait(2000)
            except Exception:  # noqa: BLE001
                pass
        super().closeEvent(event)
