"""Guarded reuse across slow OCR must retain fresh-frame click protections."""

import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController
from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import OCRFailedError, StoppedError
from arena_auto.models.config import AppConfig
from arena_auto.models.recognition import DetectionResult, OCRBox
from arena_auto.models.state import AutomationState, ScreenKind
from arena_auto.recognition.auto_targeting import AutoTargeting
from arena_auto.recognition.frame_guard import regions_unchanged
from arena_auto.recognition.ocr_pipeline import FrameOCRCache


def arena_scene():
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    boxes = [OCRBox("请选择对手", .95, (340, 60, 150, 25))]
    for index in range(3):
        top = 100 + index * 145
        cv2.rectangle(image, (330, top), (930, top + 130), (220, 220, 220), -1)
        boxes += [OCRBox(f"Player{index}", .95, (340, top + 10, 100, 20)),
                  OCRBox("战斗力：121235", .95, (710, top + 10, 160, 25))]
    return image, boxes


def targeting(image, boxes):
    provider = MagicMock()
    provider.recognize.return_value = boxes
    auto = AutoTargeting(provider, AppConfig())
    auto.frame_cache = FrameOCRCache(provider)
    auto.frame_cache.raw(image)  # Same evidence used in the challenge stage.
    return auto, provider


def test_guarded_layout_reuse_uses_live_frame_as_first_numeric_read():
    old, boxes = arena_scene()
    auto, provider = targeting(old, boxes)
    fresh = [old.copy(), old.copy()]
    factory = MagicMock(side_effect=fresh)
    read = MagicMock(return_value=121235)
    auto._read_power = read
    confirmed, selected = auto.select(factory, 700000, layout_hint=old)
    assert selected.id == 1 and selected.power == 121235
    assert len(confirmed) == 1 and read.call_count == 2
    assert read.call_args_list[0].args[0] is fresh[0]
    assert read.call_args_list[1].args[0] is fresh[1]
    assert factory.call_count == 2  # No extra discovery screenshot.
    assert provider.recognize.call_count == 1  # No repeated full-frame OCR.


@pytest.mark.parametrize("location", ["title", "later_card", "tail"])
def test_layout_guard_checks_more_than_the_selected_card(location):
    old, boxes = arena_scene()
    auto, provider = targeting(old, boxes)
    changed = old.copy()
    if location == "title":
        changed[30:60, 600:650] = 255
    elif location == "later_card":
        changed[430:450, 350:450] = 0
    else:
        changed[560:590, 600:650] = 255
    auto._read_power = MagicMock(return_value=121235)
    factory = MagicMock(side_effect=[changed, changed.copy()])
    confirmed, selected = auto.select(factory, 700000, layout_hint=old)
    assert selected.id == 1
    assert provider.recognize.call_count == 2  # Fresh full OCR after failed guard.
    assert provider.recognize.call_args.args[0] is changed
    assert auto._read_power.call_count == 2


def test_changed_resolution_falls_back_to_fresh_discovery():
    old, boxes = arena_scene()
    auto, _ = targeting(old, boxes)
    changed = np.zeros((300, 500, 3), dtype=np.uint8)
    scan = MagicMock(side_effect=[auto.detect_cards(old, boxes), []])
    auto.scan = scan
    auto.config.recognition.retry_count = 1
    with pytest.raises(OCRFailedError):
        auto.select(lambda: changed, 700000, layout_hint=old)
    assert scan.call_args.args[0] is changed


def test_guarded_reuse_still_rejects_unstable_numeric_reads():
    old, boxes = arena_scene()
    auto, _ = targeting(old, boxes)
    auto.config.recognition.retry_count = 1
    auto._read_power = MagicMock(side_effect=[121235, 999999])
    with pytest.raises(OCRFailedError):
        auto.select(lambda: old.copy(), 700000, layout_hint=old)


def test_guarded_layout_never_skips_missing_first_opponent():
    old, boxes = arena_scene()
    auto, _ = targeting(old, boxes)
    auto.config.recognition.retry_count = 1
    cards = auto.detect_cards(old, boxes)[1:]
    auto.scan = MagicMock(return_value=cards)
    auto._read_power = MagicMock(return_value=121235)
    with pytest.raises(OCRFailedError):
        auto.select(lambda: old.copy(), 700000, layout_hint=old)
    auto._read_power.assert_not_called()


def test_prepare_changed_page_cannot_use_old_button_for_click(tmp_path):
    ctrl, provider = controller(tmp_path)
    old = np.full((200, 400, 3), 100, dtype=np.uint8)
    provider.recognize.return_value = [OCRBox("去获胜", .9, (280, 130, 80, 30))]
    ctrl.frame_cache.raw(old)
    ctrl.ctx.prepare_frame = old
    ctrl.ctx.prepare_frame_at = time.monotonic() - 2
    ctrl.adb.screenshot.return_value = np.zeros_like(old)
    provider.recognize.return_value = []
    # Cached old evidence still has the button, but the fresh page does not.
    ctrl._state_prepare()
    ctrl.adb.tap.assert_not_called()


def test_stop_during_live_layout_validation_prevents_roi_reads():
    import threading
    old, boxes = arena_scene()
    auto, _ = targeting(old, boxes)
    stop = threading.Event()
    auto._read_power = MagicMock()

    def factory():
        stop.set()
        return old.copy()

    with pytest.raises(StoppedError):
        auto.select(factory, 700000, stop, layout_hint=old)
    auto._read_power.assert_not_called()


def controller(tmp_path):
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.automation.auto_targeting = True
    manager.config.automation.click_delay = 0
    manager.config.debug.enabled = False
    provider = MagicMock()
    adb = MagicMock()
    templates = MagicMock()
    templates.find.return_value = DetectionResult(False)
    ctrl = AutomationController(manager, adb=adb, ocr=provider, templates=templates)
    return ctrl, provider


def test_cached_button_uses_same_full_frame_matching_rules(tmp_path):
    ctrl, provider = controller(tmp_path)
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    provider.recognize.return_value = [OCRBox("去获胜", .9, (280, 130, 80, 30))]
    ctrl.detector._recognize_all(image)
    result = ctrl.buttons.find_go_win(image)
    assert result.found and result.bbox == (280, 130, 80, 30)
    provider.recognize.assert_called_once()


def test_cached_buy_does_not_accept_long_confirmation_sentence(tmp_path):
    ctrl, provider = controller(tmp_path)
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    ctrl.config.profile.confirm_text = "确认"
    provider.recognize.return_value = [OCRBox("确认对手布置", .9, (20, 20, 160, 30))]
    ctrl.frame_cache.raw(image)
    assert not ctrl.buttons._full_ocr_result(image, "确认", "confirm_buy")


@pytest.mark.parametrize("changed", [False, True])
def test_prepare_reads_button_from_old_evidence_but_always_takes_new_frame(tmp_path, changed):
    ctrl, provider = controller(tmp_path)
    old = np.full((200, 400, 3), 100, dtype=np.uint8)
    fresh = old.copy()
    if changed:
        fresh[50:100, 30:100] = 220  # Even an overlay away from the button rejects reuse.
    provider.recognize.return_value = [OCRBox("去获胜", .9, (280, 130, 80, 30))]
    ctrl.frame_cache.raw(old)
    ctrl.ctx.prepare_frame = old
    ctrl.ctx.prepare_frame_at = time.monotonic() - 2
    ctrl.adb.screenshot.return_value = fresh
    ctrl._state_prepare()
    assert ctrl.adb.screenshot.call_count == 1
    assert ctrl.adb.tap.call_count == 1
    assert ctrl.sm.state == AutomationState.WAIT_RESULT
    if not changed:
        assert provider.recognize.call_count == 1
    else:
        assert provider.recognize.call_count > 1


def test_frame_guard_rejects_local_replacement_with_same_global_average():
    old = np.full((300, 600, 3), 100, dtype=np.uint8)
    changed = old.copy()
    changed[20:40, 100:120] = 180
    changed[40:60, 100:120] = 20
    assert old.mean() == changed.mean()
    assert not regions_unchanged(old, changed, [(0, 0, 600, 300)])
    assert not regions_unchanged(old, old.copy(), [(0, 0, 601, 300)])


def test_frame_peek_never_runs_ocr_or_returns_other_frame_evidence():
    provider = MagicMock()
    provider.recognize.return_value = [OCRBox("去获胜", .9, (10, 10, 20, 20))]
    cache = FrameOCRCache(provider)
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    assert cache.peek(image) is None
    provider.recognize.assert_not_called()
    cache.raw(image)
    assert cache.peek(image)
    assert cache.peek(image.copy()) is None
    assert provider.recognize.call_count == 1


def test_prepare_without_cached_ocr_does_not_recognize_old_frame(tmp_path):
    ctrl, provider = controller(tmp_path)
    old = np.full((200, 400, 3), 100, dtype=np.uint8)
    ctrl.ctx.prepare_frame = old
    ctrl.ctx.prepare_frame_at = time.monotonic() - 2
    fresh = old.copy()
    ctrl.adb.screenshot.return_value = fresh
    provider.recognize.return_value = [OCRBox("去获胜", .9, (280, 130, 80, 30))]
    ctrl._state_prepare()
    assert provider.recognize.call_args.args[0] is fresh
    provider.recognize.assert_called_once()


def test_cached_button_outside_manual_roi_keeps_roi_first_behavior(tmp_path):
    from arena_auto.models.config import ButtonGeometry
    ctrl, provider = controller(tmp_path)
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    ctrl.config.screen.reference_width = 400
    ctrl.config.screen.reference_height = 200
    ctrl.config.buttons["go_win"] = ButtonGeometry(region=(0, 0, 100, 50), click=(50, 25))
    provider.recognize.return_value = [OCRBox("去获胜", .9, (280, 130, 80, 30))]
    ctrl.frame_cache.raw(image)
    assert ctrl.buttons.cached_go_win(image) is None
    provider.recognize.assert_called_once()  # Cache-only method cannot OCR an old ROI.
    # The manual ROI is searched before falling back to the cached full frame.
    finder = ctrl.buttons._find_ocr_boxes
    ctrl.buttons._find_ocr_boxes = MagicMock(wraps=finder)
    ctrl.buttons.find_go_win(image)
    assert ctrl.buttons._find_ocr_boxes.call_args_list[1].args[0].shape == (50, 100, 3)
