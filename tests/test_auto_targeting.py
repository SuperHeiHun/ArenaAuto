"""Coordinate-free targeting tests using synthetic screenshots and fake OCR."""

import threading
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController
from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import OCRFailedError, StoppedError
from arena_auto.models.config import AppConfig
from arena_auto.models.recognition import ChallengeCount, DetectionResult, OCRBox, Opponent
from arena_auto.models.state import AutomationState, ScreenKind
from arena_auto.recognition.auto_targeting import AutoTargeting


def scene(scale=1.0):
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    boxes = [OCRBox("请选择对手", 0.95, (340, 60, 150, 25))]
    for i, power in enumerate([67499, 68685, 3170348, 65564]):
        top = 100 + i * 145
        cv2.rectangle(image, (330, top), (930, top + 130), (220, 220, 220), -1)
        cv2.circle(image, (390, top + 80), 30, (0, 140, 0), 4)
        boxes.extend([
            OCRBox(f"Player{i + 1}", 0.95, (340, top + 10, 100, 20)),
            OCRBox(f"战斗力：{power}", 0.95, (710, top + 10, 160, 25)),
        ])
    # Own defense power must never be mistaken for an opponent.
    boxes.append(OCRBox("防御卡组战斗力：4575674", 0.95, (50, 450, 220, 30)))
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
        boxes = [OCRBox(b.text, b.confidence, tuple(int(v * scale) for v in b.bbox)) for b in boxes]
    return image, boxes


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.5])
def test_complete_cards_detected_partial_card_excluded(scale):
    image, boxes = scene(scale)
    targeting = AutoTargeting(MagicMock(), AppConfig())
    cards = targeting.detect_cards(image, boxes)
    assert [c.opponent.power for c in cards] == [67499, 68685, 3170348]
    for card in cards:
        x, y, w, h = card.region
        px, py = card.opponent.click_position
        assert x < px < x + w * 0.3
        assert y < py < y + h
        assert y + h < image.shape[0] - 3


@pytest.mark.parametrize("remove", ["anchor", "first_power", "second_power", "names"])
def test_missing_anchors_or_earlier_headers_do_not_select(remove):
    image, boxes = scene()
    if remove == "anchor":
        boxes = boxes[1:]
    elif remove == "first_power":
        boxes = [b for b in boxes if "67499" not in b.text]
    elif remove == "second_power":
        boxes = [b for b in boxes if "68685" not in b.text]
    else:
        boxes = [b for b in boxes if not b.text.startswith("Player")]
    targeting = AutoTargeting(MagicMock(), AppConfig())
    assert not targeting.detect_cards(image, boxes)


def test_no_visual_card_edges_no_coordinate_fallback():
    image, boxes = scene()
    image[:] = 0
    assert not AutoTargeting(MagicMock(), AppConfig()).detect_cards(image, boxes)


def test_stable_selection_uses_two_fresh_frames_and_actual_pixels():
    image, boxes = scene()
    config = AppConfig()
    config.recognition.stable_reads = 1  # Auto mode still requires >=2 frames.
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, config)
    factory = MagicMock(side_effect=lambda: image.copy())
    opponents, selected = targeting.select(factory, 68000)
    assert factory.call_count == 2
    assert selected.id == 1 and selected.power == 67499
    assert len(opponents) == 3
    assert selected.click_position[0] < 500  # Not converted to 1920-wide reference coordinates.


def test_refresh_never_clicks_an_unstable_target():
    image, boxes = scene()
    changed = [OCRBox("PlayerNEW" if b.text == "Player1" else b.text, b.confidence, b.bbox) for b in boxes]
    ocr = MagicMock()
    ocr.recognize.side_effect = [boxes, changed, boxes]
    targeting = AutoTargeting(ocr, AppConfig())
    with pytest.raises(OCRFailedError, match="不稳定"):
        targeting.select(lambda: image.copy(), 68000)


def test_no_opponent_under_threshold_returns_no_target():
    image, boxes = scene()
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    opponents, selected = AutoTargeting(ocr, AppConfig()).select(lambda: image.copy(), 50000)
    assert len(opponents) == 3
    assert selected is None


def test_stop_before_ocr():
    event = threading.Event()
    event.set()
    ocr = MagicMock()
    with pytest.raises(StoppedError):
        AutoTargeting(ocr, AppConfig()).select(MagicMock(), 100000, event)
    ocr.recognize.assert_not_called()


def test_challenge_requires_label_and_rejects_ambiguous_counts():
    ocr = MagicMock()
    image, _ = scene()
    targeting = AutoTargeting(ocr, AppConfig())
    label = OCRBox("可挑战次数", 0.95, (50, 350, 130, 30))
    count = OCRBox("4 / 5", 0.95, (220, 350, 70, 30))
    ocr.recognize.return_value = [count]
    assert targeting.read_challenge(image) is None
    ocr.recognize.return_value = [label, count]
    assert targeting.read_challenge(image) == ChallengeCount(4, 5)
    ocr.recognize.return_value = [label, count, OCRBox("2/5", 0.95, (230, 355, 60, 20))]
    assert targeting.read_challenge(image) is None


def controller(tmp_path):
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.automation.auto_targeting = True
    manager.config.automation.click_delay = 0
    manager.config.debug.enabled = False
    manager.config.opponents[0].click = (999, 888)  # Must never be used.
    adb = MagicMock()
    adb.screenshot.return_value = np.zeros((600, 1000, 3), dtype=np.uint8)
    templates = MagicMock()
    templates.find.return_value = DetectionResult(found=False)
    ctrl = AutomationController(manager, adb=adb, ocr=MagicMock(), templates=templates)
    ctrl.auto_targeting.select = MagicMock(return_value=([Opponent(1, 67499, click_position=(390, 180))], Opponent(1, 67499, click_position=(390, 180))))
    ctrl.auto_targeting.verify_target = MagicMock(return_value=True)
    ctrl.power_reader.read_until_at_or_below = MagicMock()
    ctrl._wait = MagicMock(return_value=False)
    return ctrl


def test_controller_auto_path_does_not_use_saved_coordinates(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.sm.transition(AutomationState.SELECT_OPPONENT)
    ctrl._state_select_opponent()
    ctrl.adb.tap.assert_called_once_with(390, 180, delay=0)
    ctrl.power_reader.read_until_at_or_below.assert_not_called()
    assert ctrl.sm.state == AutomationState.WAIT_PREPARE
    assert ctrl.config.opponents[0].click == (999, 888)


def test_auto_failure_stops_without_tap_or_manual_fallback(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.auto_targeting.select.side_effect = OCRFailedError("没有完整卡片")
    ctrl.sm.transition(AutomationState.SELECT_OPPONENT)
    ctrl._state_select_opponent()
    assert ctrl.sm.state == AutomationState.STOPPED
    ctrl.adb.tap.assert_not_called()
    ctrl.power_reader.read_until_at_or_below.assert_not_called()


def test_failed_click_attempts_are_bounded(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.ctx.auto_selection_attempts = 2
    ctrl._state_select_opponent()
    assert ctrl.sm.state == AutomationState.STOPPED
    ctrl.auto_targeting.select.assert_not_called()
    ctrl.adb.tap.assert_not_called()


def test_prepare_transition_confirms_and_resets_attempts(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.ctx.auto_selection_attempts = 1
    ctrl.detector.detect = MagicMock(return_value=ScreenKind.PREPARE)
    ctrl.sm.transition(AutomationState.WAIT_PREPARE)
    ctrl._state_wait_prepare()
    assert ctrl.sm.state == AutomationState.PREPARE
    assert ctrl.ctx.auto_selection_attempts == 0


def test_auto_purchase_does_not_bypass_authorization(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.config.purchase.enabled = False
    ctrl.sm.transition(AutomationState.BUY_CHALLENGE)
    ctrl._state_buy_challenge()
    assert ctrl.sm.state == AutomationState.STOPPED
    ctrl.adb.tap.assert_not_called()


def test_auto_config_roundtrip(tmp_path):
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.automation.auto_targeting = True
    manager.save()
    assert manager.load().automation.auto_targeting
    assert not ConfigManager.from_dict({}).automation.auto_targeting


def test_stop_after_ocr_before_any_selection():
    image, boxes = scene()
    event = threading.Event()
    ocr = MagicMock()
    def recognize(_image):
        event.set()
        return boxes
    ocr.recognize.side_effect = recognize
    with pytest.raises(StoppedError):
        AutoTargeting(ocr, AppConfig()).select(lambda: image.copy(), 100000, event)


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_debug_auto_targeting_is_read_only(qt_app, tmp_path):
    from arena_auto.gui.workers import DebugTestWorker
    ctrl = controller(tmp_path)
    image, boxes = scene()
    ctrl.adb.screenshot.return_value = image
    ctrl.ocr.recognize.return_value = boxes
    worker = DebugTestWorker(ctrl.adb, ctrl, "auto_targeting")
    payloads = []
    worker.result_ready.connect(lambda name, image, payload: payloads.append(payload))
    worker.run()
    assert len(payloads[0]["cards"]) == 3
    ctrl.adb.tap.assert_not_called()
    assert ctrl.config.opponents[0].click == (999, 888)


def test_gui_auto_targeting_toggle(qt_app, monkeypatch, tmp_path):
    from arena_auto.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    manager = ConfigManager(tmp_path / "config.yaml")
    window = MainWindow(manager)
    try:
        assert not window.auto_target_check.isChecked()
        window.auto_target_check.setChecked(True)
        assert window._apply_form_to_config()
        manager.save()
        assert manager.load().automation.auto_targeting
    finally:
        window.close()


def test_just_before_tap_visual_verification():
    image, boxes = scene()
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, AppConfig())
    cards = targeting.scan(image)
    opponent = cards[0].opponent
    assert targeting.verify_target(image.copy(), opponent)
    changed = image.copy()
    x, y, w, h = cards[0].region
    changed[y:y + h, x:x + w] = (100, 10, 20)
    assert not targeting.verify_target(changed, opponent)
    assert not targeting.verify_target(cv2.resize(image, (500, 300)), opponent)


def test_controller_cancels_if_card_changes_while_ocr_runs(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.auto_targeting.verify_target.return_value = False
    ctrl.sm.transition(AutomationState.SELECT_OPPONENT)
    ctrl._state_select_opponent()
    assert ctrl.sm.state == AutomationState.STOPPED
    ctrl.adb.tap.assert_not_called()
