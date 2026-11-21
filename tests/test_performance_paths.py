"""Performance shortcuts may eliminate duplicate work, not verification."""

import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController
from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import StoppedError
from arena_auto.models.config import OcrSettings, TemplateEntry
from arena_auto.models.recognition import ChallengeCount, OCRBox
from arena_auto.models.state import AutomationState, ScreenKind
from arena_auto.recognition.ocr_pipeline import FrameOCRCache
from arena_auto.recognition.scene_change import SettledSceneChange
from arena_auto.recognition.template import TemplateRecognizer


def test_frame_cache_shares_exact_frame_but_not_similar_fresh_frame():
    provider = MagicMock()
    provider.recognize.return_value = [OCRBox("战力 12345", .9, (5, 5, 40, 20))]
    cache = FrameOCRCache(provider)
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    assert cache.raw(image) == cache.recognize(image, OcrSettings()).boxes
    assert provider.recognize.call_count == 1
    cache.raw(image.copy())
    assert provider.recognize.call_count == 2
    cache.clear()
    cache.raw(image)
    assert provider.recognize.call_count == 3


def test_frame_cache_shares_processed_retry_with_original_coordinates():
    provider = MagicMock()
    provider.recognize.side_effect = [[], [OCRBox("12345", .9, (15, 15, 90, 30))]]
    cache = FrameOCRCache(provider)
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    result = cache.recognize(image, OcrSettings(scale=3))
    assert result.boxes[0].bbox == (5, 5, 30, 10)
    assert result.input_mode == "预处理截图"
    assert cache.recognize(image, OcrSettings(scale=3)) == result
    assert provider.recognize.call_count == 2


def test_frame_cache_raw_exception_retries_but_stop_is_not_swallowed():
    provider = MagicMock()
    provider.recognize.side_effect = [ValueError("bad raw"), [OCRBox("12345", .9, (0, 0, 15, 15))]]
    cache = FrameOCRCache(provider)
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    assert cache.recognize(image, OcrSettings()).boxes
    cache.clear()
    provider.recognize.side_effect = StoppedError("stop")
    with pytest.raises(StoppedError):
        cache.recognize(image, OcrSettings())


def controller(tmp_path):
    manager = ConfigManager(tmp_path / "config.yaml")
    manager.config.debug.enabled = False
    manager.config.automation.auto_targeting = True
    manager.config.automation.dry_run = False
    manager.config.automation.click_delay = 0
    image = np.zeros((300, 600, 3), dtype=np.uint8)
    adb = MagicMock()
    adb.screenshot.side_effect = lambda: image.copy()
    ocr = MagicMock()
    ctrl = AutomationController(manager, adb=adb, ocr=ocr, templates=MagicMock())
    return ctrl, ocr


def test_page_challenge_and_layout_share_one_frame_ocr(tmp_path):
    ctrl, provider = controller(tmp_path)
    image = np.zeros((300, 600, 3), dtype=np.uint8)
    provider.recognize.return_value = [
        OCRBox("可挑战次数", .99, (50, 30, 100, 20)),
        OCRBox("4 / 5", .99, (155, 30, 50, 20)),
    ]
    ctrl.detector._recognize_all(image)
    assert ctrl.auto_targeting.read_challenge(image) == ChallengeCount(4, 5)
    ctrl.auto_targeting.detect_cards = MagicMock(return_value=[])
    ctrl.auto_targeting.scan(image)
    assert provider.recognize.call_count == 1


def test_controller_shares_detection_arena_and_discovery_frame(tmp_path):
    ctrl, provider = controller(tmp_path)
    ctrl.templates.find.return_value.found = False
    provider.recognize.return_value = [
        OCRBox("胜利者的竞技场", .99, (20, 5, 200, 20)),
        OCRBox("可挑战次数", .99, (50, 30, 100, 20)),
        OCRBox("4 / 5", .99, (155, 30, 50, 20)),
    ]
    ctrl._state_detect_screen()
    frame = ctrl.ctx.arena_frame
    assert ctrl.sm.state == AutomationState.ARENA
    ctrl._state_arena()
    assert ctrl.ctx.layout_frame is frame
    ctrl.auto_targeting.detect_cards = MagicMock(return_value=[])

    def select(factory, threshold, stop_event, **kwargs):
        ctrl.auto_targeting.scan(factory())
        return [], None

    ctrl.auto_targeting.select = select
    ctrl._wait = MagicMock(return_value=False)
    ctrl._state_select_opponent()
    assert provider.recognize.call_count == 1
    assert ctrl.adb.screenshot.call_count == 1
    ctrl.adb.tap.assert_not_called()


def test_slow_challenge_ocr_does_not_make_an_old_frame_fresh(tmp_path, monkeypatch):
    ctrl, _ = controller(tmp_path)
    clock = [100]
    monkeypatch.setattr("arena_auto.automation.controller.time.monotonic", lambda: clock[0])

    def challenge(image):
        clock[0] = 102
        return ChallengeCount(4, 5)

    ctrl._read_challenge = challenge
    ctrl._state_arena()
    assert ctrl.ctx.layout_frame_at == 100
    frame = ctrl.ctx.layout_frame
    seen = []

    def select(factory, threshold, stop_event, **kwargs):
        seen.append(factory())
        return [], None

    ctrl.auto_targeting.select = select
    ctrl._wait = MagicMock(return_value=False)
    ctrl._state_select_opponent()
    assert seen[0] is not frame


@pytest.mark.parametrize("expired", [False, True])
def test_arena_frame_reuse_is_layout_only_and_expires(tmp_path, expired):
    ctrl, _ = controller(tmp_path)
    ctrl._read_challenge = MagicMock(return_value=ChallengeCount(4, 5))
    ctrl._state_arena()
    original = ctrl.ctx.layout_frame
    if expired:
        ctrl.ctx.layout_frame_at = time.monotonic() - 2
    observed = []

    def select(factory, threshold, stop_event, **kwargs):
        observed.extend([factory(), factory(), factory()])
        return [], None

    ctrl.auto_targeting.select = select
    ctrl._wait = MagicMock(return_value=False)
    ctrl._state_select_opponent()
    assert (observed[0] is original) is (not expired)
    assert observed[1] is not observed[0] and observed[2] is not observed[1]
    assert ctrl.ctx.layout_frame is None
    ctrl.adb.tap.assert_not_called()


def test_tap_and_reload_invalidate_frame_evidence(tmp_path):
    ctrl, provider = controller(tmp_path)
    image = np.zeros((300, 600, 3), dtype=np.uint8)
    provider.recognize.return_value = [OCRBox("button", .9, (10, 10, 20, 20))]
    ctrl.frame_cache.raw(image)
    ctrl.ctx.layout_frame = image
    ctrl._safe_tap(10, 10)
    assert ctrl.ctx.layout_frame is None
    ctrl.frame_cache.raw(image)
    assert provider.recognize.call_count == 2
    ctrl.reload_config()
    ctrl.frame_cache.raw(image)
    assert provider.recognize.call_count == 3


def test_performance_records_state_and_wait_even_on_failure(tmp_path, monkeypatch):
    ctrl, _ = controller(tmp_path)
    perf_log = MagicMock()
    monkeypatch.setattr("arena_auto.recognition.ocr_engine.perf_log", perf_log)
    ctrl.sm.transition(AutomationState.ARENA)
    ctrl._state_arena = MagicMock(side_effect=StoppedError("stop"))
    with pytest.raises(StoppedError):
        ctrl._step()
    assert perf_log.call_args.args[0] == "State ARENA"
    assert perf_log.call_args.args[1] >= 0
    ctrl.stop_event.set()
    assert ctrl._wait(1)
    assert perf_log.call_args.args[0] == "Interruptible wait"


def template_scene():
    random = np.random.default_rng(42)
    tile = random.integers(20, 230, (24, 36, 3), dtype=np.uint8)
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    image[150:174, 200:236] = tile
    recognizer = TemplateRecognizer({"exit": TemplateEntry(path="missing.png", threshold=.8)})
    assert recognizer.register_image("exit", tile)
    return recognizer, image, tile


def test_template_same_frame_cache_and_local_search(monkeypatch):
    recognizer, image, _ = template_scene()
    match = MagicMock(wraps=cv2.matchTemplate)
    monkeypatch.setattr(cv2, "matchTemplate", match)
    first = recognizer.find(image, "exit")
    assert first.found
    calls = match.call_count
    assert recognizer.find(image, "exit") == first
    assert match.call_count == calls
    assert recognizer.find(image.copy(), "exit").found
    assert match.call_args.args[0].size < image.shape[0] * image.shape[1]


def test_local_template_miss_falls_back_when_button_moves():
    recognizer, image, tile = template_scene()
    assert recognizer.find(image, "exit").found
    changed = np.zeros_like(image)
    changed[300:324, 600:636] = tile
    assert recognizer.find(changed, "exit").bbox == (600, 300, 36, 24)


def test_template_registration_clears_matches_and_scaled_cache():
    recognizer, image, tile = template_scene()
    assert recognizer.find(image, "exit").found
    assert recognizer._scaled_templates
    assert recognizer.register_image("exit", np.flip(tile, axis=0))
    assert not recognizer._scaled_templates
    assert not recognizer.find(image, "exit").found
    recognizer.set_entries({"exit": TemplateEntry(path="another.png", threshold=.8)})
    assert not recognizer._last_hits
    assert not recognizer._match_cache


def test_scaled_templates_are_reused_on_fresh_frames(monkeypatch):
    recognizer, image, _ = template_scene()
    resize = MagicMock(wraps=cv2.resize)
    monkeypatch.setattr(cv2, "resize", resize)
    recognizer.find(image, "exit")
    calls = resize.call_count
    assert calls == 4
    recognizer.find(image.copy(), "exit")
    assert resize.call_count == calls


def test_frame_match_cache_respects_threshold_and_resolution():
    recognizer, image, tile = template_scene()
    assert recognizer.find(image, "exit").found
    assert not recognizer.find(image, "exit", threshold=1.1).found
    smaller = np.zeros((100, 200, 3), dtype=np.uint8)
    smaller[40:64, 80:116] = tile
    assert recognizer.find(smaller, "exit").bbox == (80, 40, 36, 24)


def test_scene_change_requires_distributed_change_and_settling():
    signal = SettledSceneChange()
    base = np.zeros((360, 640, 3), dtype=np.uint8)
    assert not signal.update(base)
    small = base.copy()
    small[30:60, 30:60] = 255
    assert not signal.update(small)
    changed = np.full_like(base, 180)
    assert not signal.update(changed)
    assert signal.update(changed.copy())
    assert not signal.update(base)
    assert not signal.update(base.copy())


def test_scene_animation_never_settles():
    signal = SettledSceneChange()
    for value in (0, 100, 200, 50, 200, 80):
        assert not signal.update(np.full((200, 400, 3), value, dtype=np.uint8))


def test_scene_probe_config_default_and_roundtrip(tmp_path):
    assert ConfigManager.from_dict({}).battle.result_scene_probe
    config = ConfigManager.from_dict({"battle": {"result_scene_probe": False}})
    manager = ConfigManager(tmp_path / "config.yaml", config)
    manager.save()
    assert not manager.load().battle.result_scene_probe


def test_scene_change_does_not_bypass_five_second_probe_limit(tmp_path, monkeypatch):
    ctrl, _ = controller(tmp_path)
    ctrl.ctx.result_ocr_last_at = 99
    ctrl.config.battle.result_ocr_interval = 1
    monkeypatch.setattr("arena_auto.automation.controller.time.monotonic", lambda: 103)
    ctrl.detector.detect_screen_and_outcome = MagicMock()
    image = np.zeros((300, 600, 3), dtype=np.uint8)
    assert ctrl._probe_result_ocr(image, scene_changed=True) is None
    ctrl.detector.detect_screen_and_outcome.assert_not_called()


def test_scene_probe_shortens_cooldown_only_once_and_does_not_guess(tmp_path, monkeypatch):
    ctrl, _ = controller(tmp_path)
    ctrl.templates.find.return_value.found = False
    ctrl.detector.outcome_from_templates = MagicMock(return_value=None)
    ctrl.detector.detect_screen_and_outcome = MagicMock(return_value=(ScreenKind.UNKNOWN, None))
    ctrl._wait = MagicMock(return_value=False)
    ctrl.ctx.battle_started_at = 90
    ctrl.ctx.result_ocr_last_at = 99
    clock = [100]
    monkeypatch.setattr("arena_auto.automation.controller.time.monotonic", lambda: clock[0])
    ctrl.sm.transition(AutomationState.WAIT_RESULT)
    frames = [np.zeros((300, 600, 3), dtype=np.uint8), np.full((300, 600, 3), 180, dtype=np.uint8)]
    ctrl.adb.screenshot.side_effect = [frames[0], frames[1], frames[1].copy(), frames[0], frames[0].copy()]
    ctrl._state_wait_result()
    clock[0] = 102
    ctrl._state_wait_result()
    clock[0] = 105
    ctrl._state_wait_result()
    assert ctrl.detector.detect_screen_and_outcome.call_count == 1
    clock[0] = 106
    ctrl._state_wait_result()
    clock[0] = 111
    ctrl._state_wait_result()
    assert ctrl.detector.detect_screen_and_outcome.call_count == 1
    assert ctrl.ctx.outcome is None
    assert ctrl.sm.state == AutomationState.WAIT_RESULT
    ctrl.adb.tap.assert_not_called()
