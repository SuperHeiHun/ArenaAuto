"""Stable PREPARE panels may be reused, but never through dialog/identity changes."""

import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController
from arena_auto.config.manager import ConfigManager
from arena_auto.models.recognition import DetectionResult, OCRBox
from arena_auto.models.state import AutomationState
from arena_auto.recognition.prepare_guard import PrepareStructure


def scene(scale=1):
    image = np.full((360, 800, 3), 30, dtype=np.uint8)
    cv2.rectangle(image, (568, 43), (795, 354), (240, 250, 245), -1)
    cv2.rectangle(image, (350, 300), (560, 354), (200, 230, 220), -1)
    cv2.rectangle(image, (623, 316), (739, 347), (40, 190, 245), -1)
    button = (654, 323, 57, 15)
    boxes = [OCRBox("备战攻击队设置", .95, (65, 8, 193, 26)),
             OCRBox("对方防御队构成", .95, (355, 302, 63, 12)),
             OCRBox("战斗力 138307", .95, (485, 302, 68, 12)),
             OCRBox("去获胜", .95, button)]
    for i in range(6):
        x, y = 585 + i % 3 * 68, 90 + i // 3 * 85
        cv2.rectangle(image, (x, y), (x + 48, y + 48), (80, 130, 170), -1)
        cv2.rectangle(image, (x, y + 54), (x + 34, y + 61), (60, 60, 60), -1)
        boxes.append(OCRBox(str(120000 + i), .95, (x, y + 53, 35, 10)))
    cv2.rectangle(image, (66, 10), (256, 28), (80, 90, 130), -1)
    cv2.circle(image, (220, 170), 28, (180, 70, 100), -1)
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        boxes = [OCRBox(b.text, b.confidence, tuple(round(v * scale) for v in b.bbox)) for b in boxes]
        button = tuple(round(v * scale) for v in button)
    return image, boxes, button


@pytest.mark.parametrize("scale", [1, 1.5, 2])
def test_dynamic_panels_accept_character_animation_without_new_ocr(scale):
    old, boxes, button = scene(scale)
    structure = PrepareStructure.discover(old, boxes, button)
    assert structure is not None
    fresh = old.copy()
    cv2.circle(fresh, (round(220 * scale), round(170 * scale)), round(30 * scale), (30, 30, 30), -1)
    cv2.circle(fresh, (round(280 * scale), round(190 * scale)), round(35 * scale), (70, 180, 120), -1)
    report, blocked = structure.validate(old, fresh)
    assert report.unchanged and not blocked


@pytest.mark.parametrize("part", ["button", "summary", "title", "card", "edge"])
def test_stable_identity_and_button_regions_still_reject_local_changes(part):
    old, boxes, button = scene()
    structure = PrepareStructure.discover(old, boxes, button)
    assert structure is not None
    fresh = old.copy()
    x, y = {"button": (660, 325), "summary": (490, 305), "title": (140, 16),
            "card": (590, 100), "edge": (570, 230)}[part]
    fresh[y:y + 10, x:x + 12] = (10, 10, 200)
    report, _ = structure.validate(old, fresh)
    assert not report.unchanged


@pytest.mark.parametrize("mode", ["dialog", "small_dialog", "low_contrast", "transparent_border", "dimming", "existing_dialog"])
def test_dialog_or_dimming_blocks_click_not_just_reuse(mode):
    old, boxes, button = scene()
    fresh = old.copy()
    if mode == "dimming":
        fresh = (fresh.astype(np.float32) * .5).astype(np.uint8)
    elif mode == "small_dialog":
        cv2.rectangle(fresh, (300, 120), (375, 165), (220, 220, 220), -1)
    elif mode == "low_contrast":
        cv2.rectangle(fresh, (300, 120), (375, 165), (36, 36, 36), -1)
    else:
        target = old if mode == "existing_dialog" else fresh
        cv2.rectangle(target, (170, 110), (450, 250), (220, 220, 220),
                      3 if mode == "transparent_border" else -1)
    structure = PrepareStructure.discover(old, boxes, button)
    assert structure is not None
    report, blocked = structure.validate(old, fresh)
    assert not report.unchanged and blocked


@pytest.mark.parametrize("missing", ["title", "numbers", "summary", "pixels", "button_location"])
def test_incomplete_or_unrecognized_structure_falls_back(missing):
    old, boxes, button = scene()
    if missing == "title":
        boxes = [b for b in boxes if b.bbox[1] > 30]
    elif missing == "numbers":
        boxes = [b for b in boxes if not b.text.isdigit()]
    elif missing == "summary":
        boxes = [b for b in boxes if b.bbox[1] != 302]
    elif missing == "pixels":
        old[:] = 30
    else:
        button = (10, 10, 60, 20)
    assert PrepareStructure.discover(old, boxes, button) is None


def test_resolution_or_panel_shift_rejects_geometry_reuse():
    old, boxes, button = scene()
    structure = PrepareStructure.discover(old, boxes, button)
    assert structure is not None
    report, _ = structure.validate(old, cv2.resize(old, (400, 180)))
    assert not report.unchanged
    shifted = np.roll(old, 8, axis=1)
    report, _ = structure.validate(old, shifted)
    assert not report.unchanged


def controller(tmp_path):
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.automation.auto_targeting = True
    manager.config.automation.click_delay = 0
    manager.config.debug.enabled = False
    templates = MagicMock()
    templates.find.return_value = DetectionResult(False)
    provider = MagicMock()
    ctrl = AutomationController(manager, adb=MagicMock(), ocr=provider, templates=templates)
    return ctrl, provider


def test_controller_reuses_after_one_fresh_capture_with_no_second_ocr(tmp_path, caplog):
    import logging
    ctrl, provider = controller(tmp_path)
    old, boxes, _ = scene()
    provider.recognize.return_value = boxes
    ctrl.frame_cache.raw(old)
    ctrl.ctx.prepare_frame = old
    ctrl.ctx.prepare_frame_at = time.monotonic()
    fresh = old.copy()
    cv2.circle(fresh, (260, 170), 40, (70, 180, 120), -1)
    ctrl.adb.screenshot.return_value = fresh
    with caplog.at_level(logging.INFO):
        ctrl._state_prepare()
    provider.recognize.assert_called_once()
    ctrl.adb.screenshot.assert_called_once()
    ctrl.adb.tap.assert_called_once()
    assert ctrl.sm.state == AutomationState.WAIT_RESULT
    assert "新截图验证稳定结构" in caplog.text


def test_controller_stops_on_overlay_without_fallback_click(tmp_path):
    ctrl, provider = controller(tmp_path)
    old, boxes, _ = scene()
    provider.recognize.return_value = boxes  # Fresh OCR could still see button behind the popup.
    ctrl.frame_cache.raw(old)
    ctrl.ctx.prepare_frame = old
    ctrl.ctx.prepare_frame_at = time.monotonic()
    fresh = old.copy()
    cv2.rectangle(fresh, (170, 110), (450, 250), (220, 220, 220), -1)
    ctrl.adb.screenshot.return_value = fresh
    ctrl._state_prepare()
    ctrl.adb.tap.assert_not_called()
    provider.recognize.assert_called_once()
    assert ctrl.sm.state == AutomationState.STOPPED


def test_heading_outline_allows_only_tiny_motion_not_different_text():
    from arena_auto.recognition.prepare_guard import _heading_outline_matches

    old = np.zeros((720, 1600, 3), dtype=np.uint8)
    cv2.putText(old, "PREPARE FORMATION", (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    region = (120, 15, 380, 45)
    moved = np.roll(old, 1, axis=0)
    assert _heading_outline_matches(old, moved, region)
    changed = np.zeros_like(old)
    cv2.putText(changed, "RESULT FORMATION", (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    assert not _heading_outline_matches(old, changed, region)
    assert not _heading_outline_matches(old, np.roll(old, 8, axis=0), region)


def test_stop_after_capture_still_prevents_reused_click(tmp_path):
    from arena_auto.exceptions import StoppedError
    ctrl, provider = controller(tmp_path)
    old, boxes, _ = scene()
    provider.recognize.return_value = boxes
    ctrl.frame_cache.raw(old)
    ctrl.ctx.prepare_frame = old
    ctrl.ctx.prepare_frame_at = time.monotonic()

    def shot():
        ctrl.stop_event.set()
        return old.copy()

    ctrl.adb.screenshot.side_effect = shot
    with pytest.raises(StoppedError):
        ctrl._state_prepare()
    ctrl.adb.tap.assert_not_called()


def test_evidence_expiring_during_capture_falls_back_to_fresh_detection(tmp_path, caplog):
    import logging
    ctrl, provider = controller(tmp_path)
    old, boxes, _ = scene()
    provider.recognize.return_value = boxes
    ctrl.frame_cache.raw(old)
    ctrl.ctx.prepare_frame = old
    ctrl.ctx.prepare_frame_at = time.monotonic()

    def shot():
        ctrl.ctx.prepare_frame_at = time.monotonic() - 6
        return old.copy()

    ctrl.adb.screenshot.side_effect = shot
    with caplog.at_level(logging.INFO):
        ctrl._state_prepare()
    assert provider.recognize.call_count > 1
    assert "在新截图获取后过期" in caplog.text


def test_opencv_panel_threshold_is_identical_to_original_channel_minimum():
    rng = np.random.default_rng(6)
    image = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    assert np.array_equal(cv2.inRange(image, (170, 170, 170), (255, 255, 255)) > 0,
                          image.min(axis=2) >= 170)
