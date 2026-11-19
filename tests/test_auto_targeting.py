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
        assert x < px < x + w
        assert y < py < y + h
        assert card.target_source == "半自动配置：战力 ROI 左侧"
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
    ocr.recognize.side_effect = lambda frame: boxes if frame.shape == image.shape else [OCRBox("67,499", 0.9, (0, 0, 100, 30))]
    targeting = AutoTargeting(ocr, config)
    factory = MagicMock(side_effect=lambda: image.copy())
    opponents, selected = targeting.select(factory, 68000)
    assert factory.call_count == 3  # One discovery, two fresh numeric ROI confirmations.
    assert selected.id == 1 and selected.power == 67499
    assert len(opponents) == 1  # First qualifying opponent short-circuits later slots.
    assert ocr.recognize.call_count == 3
    assert 600 < selected.click_position[0] < 900  # Actual 1000-wide screenshot, not 1920-wide reference.


def test_refresh_never_clicks_an_unstable_target():
    image, boxes = scene()
    refreshed = image.copy()
    refreshed[100:230, 330:930] = (20, 30, 150)
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, AppConfig())
    factory = MagicMock(side_effect=[image.copy(), refreshed.copy()] * 3)
    with pytest.raises(OCRFailedError, match="画面已变化"):
        targeting.select(factory, 68000)


def test_no_opponent_under_threshold_returns_no_target():
    image, boxes = scene()
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, AppConfig())
    targeting._read_power = MagicMock(side_effect=[67499, 67499, 68685, 68685, 3170348, 3170348])
    opponents, selected = targeting.select(lambda: image.copy(), 50000)
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
    assert ConfigManager.from_dict({}).automation.auto_targeting
    assert not ConfigManager.from_dict({"automation": {"auto_targeting": False}}).automation.auto_targeting


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


def test_layout_uses_calibration_style_raw_then_preprocessed_retry():
    image, boxes = scene()
    ocr = MagicMock()
    scaled = [OCRBox(b.text, b.confidence, tuple(v * 3 for v in b.bbox)) for b in boxes]
    ocr.recognize.side_effect = [[], scaled]
    targeting = AutoTargeting(ocr, AppConfig())
    cards = targeting.scan(image)
    assert len(cards) == 3
    assert targeting.input_mode == "预处理截图"
    assert ocr.recognize.call_args_list[0].args[0].shape == image.shape
    assert ocr.recognize.call_args_list[1].args[0].ndim == 2  # Preprocessed frame.
    ocr.recognize.side_effect = None
    ocr.recognize.return_value = boxes
    assert targeting.scan(image)
    assert targeting.input_mode == "原始截图"
    assert ocr.recognize.call_count == 3  # No preprocessed pass when raw succeeds.


def test_power_roi_falls_back_to_raw_labeled_retry():
    image, _ = scene()
    ocr = MagicMock()
    ocr.recognize.return_value = []
    targeting = AutoTargeting(ocr, AppConfig())
    assert targeting._read_power(image, (710, 110, 160, 25)) is None
    assert ocr.recognize.call_count == 2  # Preprocessed then raw labeled retry.
    ocr.recognize.reset_mock()
    empty, labeled = [], [OCRBox("战斗力：67,499", 0.9, (0, 0, 120, 25))]
    ocr.recognize.side_effect = [empty, labeled]
    assert targeting._read_power(image, (710, 110, 160, 25)) == 67499
    assert ocr.recognize.call_count == 2


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


def split_power_boxes(boxes, low_second=False, missing_first=False):
    result = []
    for box in boxes:
        if box.text.startswith("战斗力："):
            x, y, w, h = box.bbox
            result.append(OCRBox("战斗力", 0.95, (x, y, 50, h)))
            if missing_first and "67499" in box.text:
                continue
            confidence = 0.498 if low_second and "68685" in box.text else 0.95
            result.append(OCRBox(box.text.split("：")[1], confidence, (x + 60, y, w - 60, h)))
        else:
            result.append(box)
    return result


def test_later_low_confidence_number_does_not_invalidate_first_target():
    image, boxes = scene()
    boxes = split_power_boxes(boxes, low_second=True)
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, AppConfig())
    targeting._read_power = MagicMock(return_value=67499)
    read, selected = targeting.select(lambda: image.copy(), 68000)
    assert selected.id == 1
    assert len(read) == 1
    assert targeting._read_power.call_args.args[0].shape == image.shape
    assert targeting._read_power.call_args.args[1] == targeting.last_cards[0].opponent.power_region
    assert targeting._read_power.call_count == 2
    ocr.recognize.assert_called_once()  # No full-list stability comparison.


def test_missing_raw_number_keeps_geometry_but_requires_roi_confirmation():
    image, boxes = scene()
    boxes = split_power_boxes(boxes, missing_first=True)
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, AppConfig())
    cards = targeting.scan(image)
    assert cards[0].opponent.id == 1
    assert cards[0].opponent.power is None  # Never fabricate zero or a known power.
    targeting._read_power = MagicMock(return_value=67499)
    _, selected = targeting.select(lambda: image.copy(), 68000)
    assert selected.id == 1 and selected.power == 67499


def test_numeric_roi_overrides_incorrect_full_frame_number_after_two_reads():
    image, boxes = scene()
    boxes = split_power_boxes(boxes, low_second=True)
    boxes = [OCRBox("174861" if b.text == "68685" else b.text, b.confidence, b.bbox) for b in boxes]
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, AppConfig())
    targeting._read_power = MagicMock(side_effect=[3000000, 3000000, 114861, 114861])
    read, selected = targeting.select(lambda: image.copy(), 200000)
    assert selected.id == 2 and selected.power == 114861
    assert [o.power for o in read] == [3000000, 114861]


def test_missing_later_name_does_not_block_eligible_first_opponent():
    image, boxes = scene()
    boxes = [b for b in boxes if b.text != "Player2"]
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, AppConfig())
    targeting._read_power = MagicMock(return_value=67499)
    _, selected = targeting.select(lambda: image.copy(), 68000)
    assert selected.id == 1


def test_unknown_earlier_card_cannot_be_skipped_for_later_target():
    image, boxes = scene()
    boxes = [b for b in boxes if b.text != "Player2"]
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, AppConfig())
    targeting._read_power = MagicMock(return_value=3000000)
    with pytest.raises(OCRFailedError, match="前序"):
        targeting.select(lambda: image.copy(), 2000000)
    assert targeting._read_power.call_count == 6  # Only #1, never the unknown #2 or #3.


def test_unstable_numeric_reads_never_select_even_when_below_threshold():
    image, boxes = scene()
    ocr = MagicMock()
    ocr.recognize.return_value = boxes
    targeting = AutoTargeting(ocr, AppConfig())
    targeting._read_power = MagicMock(side_effect=[67499, 68499, 67499] * 3)
    with pytest.raises(OCRFailedError, match="不稳定"):
        targeting.select(lambda: image.copy(), 100000)


def test_transient_empty_layout_retries_before_stopping():
    image, boxes = scene()
    ocr = MagicMock()
    ocr.recognize.side_effect = [[], [], boxes]  # First frame: raw + processed empty; next frame succeeds.
    targeting = AutoTargeting(ocr, AppConfig())
    targeting._read_power = MagicMock(return_value=67499)
    _, selected = targeting.select(lambda: image.copy(), 68000)
    assert selected.id == 1
    assert ocr.recognize.call_count == 3


def test_roi_reads_use_allowlist_and_bypass_cache():
    from arena_auto.recognition.ocr_engine import OcrEngine
    config = AppConfig()
    provider = MagicMock()
    provider.supports_read_kwargs = True
    provider.recognize.return_value = [OCRBox("67,499", 0.9, (0, 0, 100, 30))]
    engine = OcrEngine(provider=provider, settings=config.ocr, lazy=False, use_shared=False)
    targeting = AutoTargeting(engine, config)
    image, _ = scene()
    assert targeting._read_power(image, (710, 110, 160, 25)) == 67499
    assert targeting._read_power(image, (710, 110, 160, 25)) == 67499
    assert provider.recognize.call_count == 2
    assert provider.recognize.call_args.kwargs["allowlist"] == config.ocr.power_allowlist
    provider.recognize.return_value = [OCRBox("67,499", 0.1, (0, 0, 100, 30))]
    assert targeting._read_power(image, (710, 110, 160, 25)) is None


def test_worker_logs_are_delivered_without_worker_event_loop(qt_app, monkeypatch, tmp_path):
    import logging
    from arena_auto.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    root = logging.getLogger("arena_auto")
    old_level = root.level
    root.setLevel(logging.INFO)
    ui = MainWindow(ConfigManager(tmp_path / "config.yaml"))
    ui.log_view.clear()
    try:
        thread = threading.Thread(target=lambda: root.info("worker-progress-without-timer"))
        thread.start()
        thread.join()
        qt_app.processEvents()
        assert "worker-progress-without-timer" in ui.log_view.toPlainText()
    finally:
        ui.close()
        root.setLevel(old_level)


def test_missing_first_power_label_can_use_aligned_numeric_column():
    image, boxes = scene()
    boxes = split_power_boxes(boxes)
    boxes = [b for b in boxes if not (b.text == "战斗力" and b.bbox[1] == 110)]
    targeting = AutoTargeting(MagicMock(), AppConfig())
    cards = targeting.detect_cards(image, boxes)
    assert [c.opponent.id for c in cards] == [1, 2, 3]
    assert cards[0].opponent.power == 67499


def test_numeric_header_recovery_does_not_include_currency_above_panel():
    image, boxes = scene()
    boxes = split_power_boxes(boxes)
    boxes = [b for b in boxes if not (b.text == "战斗力" and b.bbox[1] == 110)]
    boxes.append(OCRBox("999999", 0.99, (770, 30, 100, 25)))
    cards = AutoTargeting(MagicMock(), AppConfig()).detect_cards(image, boxes)
    assert cards[0].opponent.power == 67499
    assert len(cards) == 3


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.5])
@pytest.mark.parametrize("reference", [(1920, 1080), (1600, 720)])
def test_automatic_clicks_equal_calibration_clicks(scale, reference):
    from arena_auto.calibration.coordinate_mapper import CoordinateMapper
    from arena_auto.calibration.opponent_detector import OpponentDetector
    image, boxes = scene(scale)
    config = AppConfig()
    config.screen.reference_width, config.screen.reference_height = reference
    config.calibration.power_region_padding.x = 24
    config.calibration.card_extend_left = 420
    h, w = image.shape[:2]
    mapper = CoordinateMapper(reference[0], reference[1], w, h)
    calibration, _ = OpponentDetector(config, mapper).detect(boxes)
    actual = AutoTargeting(MagicMock(), config).detect_cards(image, boxes)
    by_id = {int(d.name.split("_")[-1]): d for d in calibration}
    assert len(actual) == 3
    for card in actual:
        # Compare to the actual calibration detector, not a second offset formula.
        expected = by_id[card.opponent.id]
        assert card.opponent.click_position == expected.click_point.as_tuple()
        assert card.opponent.click_position[0] <= expected.region.x + 1


def test_calibration_geometry_preserves_the_existing_left_roi_formula():
    from arena_auto.calibration.coordinate_mapper import CoordinateMapper
    from arena_auto.calibration.opponent_detector import OpponentDetector
    mapper = CoordinateMapper(1920, 1080, 1600, 720)
    detector = OpponentDetector(AppConfig(), mapper)
    roi, click = detector.power_geometry_from_bbox((1295, 145, 168, 28), 169)
    assert roi.as_tuple() == (1278, 132, 202, 54)
    assert click.as_tuple() == (1221, 159)


def test_click_point_does_not_depend_on_avatar_animation_or_nickname_width():
    image, boxes = scene()
    original = AutoTargeting(MagicMock(), AppConfig()).detect_cards(image, boxes)
    changed = image.copy()
    cv2.rectangle(changed, (350, 140), (430, 220), (200, 50, 200), -1)
    renamed = [OCRBox("NewName" if b.text == "Player1" else b.text, b.confidence,
                      (340, 110, 75, 20) if b.text == "Player1" else b.bbox) for b in boxes]
    updated = AutoTargeting(MagicMock(), AppConfig()).detect_cards(changed, renamed)
    assert original[0].opponent.click_position == updated[0].opponent.click_position


def test_auto_wait_prepare_does_not_retry_on_first_arena_frame(tmp_path):
    import time
    ctrl = controller(tmp_path)
    ctrl.detector.detect = MagicMock(return_value=ScreenKind.ARENA)
    ctrl.sm.transition(AutomationState.WAIT_PREPARE)
    ctrl.sm._entered_at = time.monotonic()
    ctrl._state_wait_prepare()
    assert ctrl.sm.state == AutomationState.WAIT_PREPARE
    ctrl.adb.tap.assert_not_called()
    assert ctrl.ctx.opponent_wait_snapshot_saved
    ctrl.sm._entered_at = time.monotonic() - 4
    ctrl._state_wait_prepare()
    assert ctrl.sm.state == AutomationState.ARENA


def test_auto_wait_prepare_accepts_prepare_after_early_arena_frame(tmp_path):
    ctrl = controller(tmp_path)
    ctrl.detector.detect = MagicMock(side_effect=[ScreenKind.ARENA, ScreenKind.PREPARE])
    ctrl.ctx.auto_selection_attempts = 1
    ctrl.sm.transition(AutomationState.WAIT_PREPARE)
    ctrl._state_wait_prepare()
    assert ctrl.sm.state == AutomationState.WAIT_PREPARE
    ctrl._state_wait_prepare()
    assert ctrl.sm.state == AutomationState.PREPARE
    assert ctrl.ctx.auto_selection_attempts == 0
    ctrl.adb.tap.assert_not_called()
