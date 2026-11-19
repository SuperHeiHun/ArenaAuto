"""Opponent click-point editing independent of power ROI and device taps."""

import copy
from unittest.mock import MagicMock

import numpy as np
import pytest

from arena_auto.calibration.analyzer import CalibrationAnalyzer
from arena_auto.calibration.models import (
    CalibrationBundle, CalibrationDetection, CalibrationPoint, CalibrationRegion,
)
from arena_auto.config.manager import ConfigManager
from arena_auto.models.config import AppConfig
from arena_auto.recognition.ocr import NullOCRProvider


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def detection(index=1):
    return CalibrationDetection(
        name=f"opponent_{index}", region=CalibrationRegion(120, 20 + (index - 1) * 40, 40, 20),
        click_point=CalibrationPoint(80, 30 + (index - 1) * 40),
        confidence=0.9, value=123456,
    )


def mouse_event(canvas, x, y):
    from PySide6.QtCore import Qt
    event = MagicMock()
    event.position.return_value = canvas._to_widget(x, y)
    event.button.return_value = Qt.LeftButton
    event.modifiers.return_value = Qt.NoModifier
    return event


def make_canvas():
    from arena_auto.calibration.wizard import CalibrationCanvas, CanvasItem
    canvas = CalibrationCanvas()
    canvas.resize(800, 600)
    canvas.set_image(np.zeros((100, 200, 3), dtype=np.uint8))
    dets = [detection(1), detection(2)]
    canvas.set_items([CanvasItem(d.name, copy.copy(d.region), copy.copy(d.click_point), detection=d) for d in dets])
    return canvas, dets


@pytest.mark.parametrize("selected", [-1, 0])
def test_unselected_point_outside_roi_can_be_dragged(qt_app, selected):
    canvas, dets = make_canvas()
    canvas.select(selected)
    original_roi = copy.copy(dets[1].region)
    changed = []
    canvas.point_changed.connect(changed.append)
    canvas.mousePressEvent(mouse_event(canvas, 80, 70))
    assert canvas.selected() == 1
    assert canvas._mode == "point"
    canvas.mouseMoveEvent(mouse_event(canvas, 50, 65))
    canvas.mouseReleaseEvent(mouse_event(canvas, 50, 65))
    assert dets[1].click_point == CalibrationPoint(50, 65)
    assert dets[1].region == original_roi
    assert dets[1].source == "manual"
    assert changed == [1]
    assert dets[0].click_point == CalibrationPoint(80, 30)


def test_point_takes_priority_over_resize_handle(qt_app):
    canvas, dets = make_canvas()
    canvas.select(0)
    canvas.items()[0].point = CalibrationPoint(120, 20)
    canvas.mousePressEvent(mouse_event(canvas, 120, 20))
    assert canvas._mode == "point"


def test_place_point_is_one_shot_and_does_not_change_roi(qt_app):
    canvas, dets = make_canvas()
    canvas.select(1)
    original_roi = copy.copy(dets[1].region)
    canvas.set_point_placement(True)
    canvas.mousePressEvent(mouse_event(canvas, 30, 80))
    assert dets[1].click_point == CalibrationPoint(30, 80)
    assert dets[1].region == original_roi
    assert not canvas._placing_point
    assert canvas._mode == ""


def test_place_point_rejects_letterbox_and_drag_clamps_to_last_pixel(qt_app):
    canvas, dets = make_canvas()
    canvas.select(0)
    canvas.set_point_placement(True)
    canvas.mousePressEvent(mouse_event(canvas, -20, 50))
    assert dets[0].click_point == CalibrationPoint(80, 30)
    assert canvas._placing_point
    canvas.set_point_placement(False)
    canvas.mousePressEvent(mouse_event(canvas, 80, 30))
    canvas.mouseMoveEvent(mouse_event(canvas, 400, 300))
    canvas.mouseReleaseEvent(mouse_event(canvas, 400, 300))
    assert dets[0].click_point == CalibrationPoint(199, 99)


def test_moving_roi_does_not_move_click_point(qt_app):
    canvas, dets = make_canvas()
    canvas.mousePressEvent(mouse_event(canvas, 140, 30))
    assert canvas._mode == "move"
    canvas.mouseMoveEvent(mouse_event(canvas, 145, 35))
    canvas.mouseReleaseEvent(mouse_event(canvas, 145, 35))
    assert dets[0].click_point == CalibrationPoint(80, 30)
    assert dets[0].region == CalibrationRegion(125, 25, 40, 20)


def test_wizard_opponent_selection_point_edit_and_scaled_save(qt_app, monkeypatch, tmp_path):
    from arena_auto.calibration.wizard import CalibrationWizard
    monkeypatch.setattr(CalibrationWizard, "_goto", lambda *args: None)
    config = AppConfig()
    config.screen.reference_width = 400
    config.screen.reference_height = 200
    manager = ConfigManager(tmp_path / "config.yaml", config)
    adb = MagicMock()
    wizard = CalibrationWizard(manager, adb, NullOCRProvider())
    wizard.screenshot = np.zeros((100, 200, 3), dtype=np.uint8)
    wizard.bundle = CalibrationBundle(screen_width=200, screen_height=100, opponents=[detection(1), detection(2)])
    wizard._render_opponents()
    wizard.opponent_list.setCurrentRow(1)
    assert wizard.opponent_image.selected() == 1
    wizard.opponent_point_btn.click()
    wizard.opponent_image.mousePressEvent(mouse_event(wizard.opponent_image, 40, 75))
    assert wizard.bundle.opponents[1].click_point == CalibrationPoint(40, 75)
    assert not wizard.opponent_point_btn.isChecked()
    assert "[40, 75]" in wizard.opponent_list.item(1).text()
    assert wizard.bundle.opponents_found == 2  # manual edits still count as found
    wizard._render_power()
    wizard.power_list.setCurrentRow(1)
    assert wizard.power_canvas.selected() == 1
    assert wizard.power_canvas.items()[1].point == CalibrationPoint(40, 75)
    analyzer = CalibrationAnalyzer(config, NullOCRProvider())
    analyzer.apply_bundle(config, wizard.bundle)
    assert config.opponents[1].click == (80, 150)
    assert config.opponents[1].power_region == (240, 120, 80, 40)
    assert config.opponents[1].source == "manual"
    manager.save()
    assert manager.load().opponents[1].click == (80, 150)
    adb.tap.assert_not_called()
    wizard.close()


def test_new_manual_click_overrides_old_manual_but_auto_detection_does_not():
    config = AppConfig()
    config.screen.reference_width = 200
    config.screen.reference_height = 100
    opponent = config.opponents[0]
    opponent.source = "manual"
    opponent.click = (70, 25)
    opponent.power_region = (115, 15, 35, 20)
    det = detection()
    bundle = CalibrationBundle(screen_width=200, screen_height=100, opponents=[det])
    analyzer = CalibrationAnalyzer(config, NullOCRProvider())
    analyzer.apply_bundle(config, bundle)
    assert opponent.click == (70, 25)
    assert opponent.power_region == (115, 15, 35, 20)
    det.source = "manual"
    det.click_point = CalibrationPoint(40, 35)
    analyzer.apply_bundle(config, bundle)
    assert opponent.click == (40, 35)
    assert opponent.power_region == (120, 20, 40, 20)
    assert opponent.source == "manual"


def test_click_marker_has_visible_outer_ring_and_larger_hit_area(qt_app):
    canvas, _ = make_canvas()
    canvas.select(0)
    image = canvas.grab().toImage()
    center = canvas._to_widget(80, 30)
    cx, cy = int(round(center.x())), int(round(center.y()))
    pink_outer_pixels = 0
    for x in range(cx - 18, cx + 19):
        for y in range(cy - 18, cy + 19):
            color = image.pixelColor(x, y)
            if (x - cx) ** 2 + (y - cy) ** 2 > 8 ** 2:
                if color.red() > 180 and color.blue() > 140 and color.green() < 120:
                    pink_outer_pixels += 1
    assert pink_outer_pixels > 50
    from PySide6.QtCore import QPointF
    assert canvas._hit_point(center + QPointF(16, 0)) == 0
