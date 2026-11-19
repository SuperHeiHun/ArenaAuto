"""ArenaAuto Fast Recognition: template-only battle loop + stop/short-circuit."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController, ControllerCallbacks
from arena_auto.config.manager import ConfigManager
from arena_auto.debug.recorder import DebugRecorder
from arena_auto.exceptions import OCRFailedError, StoppedError
from arena_auto.models.config import OpponentConfig
from arena_auto.models.recognition import DetectionResult, Opponent
from arena_auto.models.state import AutomationState, ScreenKind
from arena_auto.recognition.detector import (
    PowerReader,
    ScreenDetector,
    is_placeholder_power_region,
)
from arena_auto.recognition.ocr import OCRBox, NullOCRProvider


def _make_controller(tmp_path: Path) -> AutomationController:
    manager = ConfigManager(path=tmp_path / "config.yaml")
    manager.config = manager.config.__class__()
    manager.config.automation.dry_run = True
    manager.config.debug.enabled = False
    manager.config.debug.allow_ocr_fallback = False

    adb = MagicMock()
    adb.screenshot.return_value = np.zeros((720, 1600, 3), dtype=np.uint8)
    ocr = MagicMock()
    templates = MagicMock()
    templates.preload_all.return_value = []
    templates.find.return_value = DetectionResult(found=False)

    ctrl = AutomationController(
        config_manager=manager,
        adb=adb,
        ocr=ocr,
        templates=templates,
        callbacks=ControllerCallbacks(),
        debug=DebugRecorder(base_dir=tmp_path / "debug"),
    )
    ctrl.config = manager.config
    return ctrl


def _img(w: int = 1600, h: int = 720) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class _Stop:
    def __init__(self, set_now: bool = False) -> None:
        self._e = threading.Event()
        if set_now:
            self._e.set()

    def is_set(self) -> bool:
        return self._e.is_set()

    def set(self) -> None:
        self._e.set()

    def clear(self) -> None:
        self._e.clear()

    def wait(self, s: float) -> bool:
        return self._e.wait(s)


# ---------------------------------------------------------------- WAIT_RESULT
def test_wait_result_ocr_zero_normal_path(tmp_path: Path) -> None:
    """Normal WAIT_RESULT: template only — detector.detect / OCR must not run."""
    ctrl = _make_controller(tmp_path)
    ctrl.config.battle.poll_interval = 0.0
    ctrl.ctx.battle_started_at = 1e12  # not timed out
    ctrl.ctx.battle_started_at = __import__("time").monotonic()
    ctrl.config.battle.max_duration = 9999.0
    ctrl.detector.detect = MagicMock()
    ctrl.detector.detect_screen_and_outcome = MagicMock()
    ctrl.detector.outcome_from_templates = MagicMock(return_value=None)
    ctrl.templates.find = MagicMock(return_value=DetectionResult(found=False))

    ctrl.sm.transition(AutomationState.WAIT_RESULT)
    ctrl._state_wait_result()

    ctrl.detector.detect.assert_not_called()
    ctrl.detector.detect_screen_and_outcome.assert_not_called()
    assert ctrl.sm.state == AutomationState.WAIT_RESULT
    assert ctrl.ctx.wait_result_template_fails == 1
    assert not getattr(ctrl.config.debug, "allow_ocr_fallback", False)


def test_wait_result_victory_template_to_result(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.config.battle.poll_interval = 0.0
    ctrl.ctx.battle_started_at = __import__("time").monotonic()
    ctrl.detector.outcome_from_templates = MagicMock(return_value="WIN")
    ctrl.detector.detect = MagicMock()

    ctrl.sm.transition(AutomationState.WAIT_RESULT)
    ctrl._state_wait_result()

    assert ctrl.sm.state == AutomationState.RESULT
    assert ctrl.ctx.outcome == "WIN"
    ctrl.detector.detect.assert_not_called()


def test_wait_result_exit_template_without_outcome(tmp_path: Path) -> None:
    """exit template hit without victory/defeat → RESULT, outcome left None."""
    ctrl = _make_controller(tmp_path)
    ctrl.config.battle.poll_interval = 0.0
    ctrl.ctx.battle_started_at = __import__("time").monotonic()
    ctrl.detector.outcome_from_templates = MagicMock(return_value=None)
    ctrl.templates.find = MagicMock(
        side_effect=lambda img, name, **kw: DetectionResult(
            found=(name == "exit"),
            confidence=0.9,
            bbox=(10, 10, 40, 20),
            template_name=name,
        )
    )

    ctrl.sm.transition(AutomationState.WAIT_RESULT)
    ctrl._state_wait_result()
    assert ctrl.sm.state == AutomationState.RESULT
    assert ctrl.ctx.outcome is None


def test_wait_result_stop_raises_immediately(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.stop_event.set()
    ctrl.sm.transition(AutomationState.WAIT_RESULT)
    with pytest.raises(StoppedError):
        ctrl._state_wait_result()


def test_wait_result_ocr_fallback_gated_and_needs_failures(tmp_path: Path) -> None:
    """allow_ocr_fallback=False → never OCR even after many misses."""
    ctrl = _make_controller(tmp_path)
    ctrl.config.battle.poll_interval = 0.0
    ctrl.ctx.battle_started_at = __import__("time").monotonic()
    ctrl.ctx.wait_result_template_fails = 99
    ctrl.detector.outcome_from_templates = MagicMock(return_value=None)
    ctrl.templates.find = MagicMock(return_value=DetectionResult(found=False))
    ctrl.detector.detect_screen_and_outcome = MagicMock()

    ctrl.sm.transition(AutomationState.WAIT_RESULT)
    ctrl._state_wait_result()
    ctrl.detector.detect_screen_and_outcome.assert_not_called()


def test_wait_result_ocr_fallback_runs_when_opted_in(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.config.battle.poll_interval = 0.0
    ctrl.config.debug.allow_ocr_fallback = True
    ctrl.config.debug.ocr_fallback_min_failures = 2
    ctrl.ctx.battle_started_at = __import__("time").monotonic()
    ctrl.ctx.wait_result_template_fails = 3
    ctrl.detector.outcome_from_templates = MagicMock(return_value=None)
    ctrl.templates.find = MagicMock(return_value=DetectionResult(found=False))
    ctrl.detector.detect_screen_and_outcome = MagicMock(
        return_value=(ScreenKind.RESULT, "WIN")
    )

    ctrl.sm.transition(AutomationState.WAIT_RESULT)
    ctrl._state_wait_result()
    ctrl.detector.detect_screen_and_outcome.assert_called_once()
    assert ctrl.sm.state == AutomationState.RESULT
    assert ctrl.ctx.outcome == "WIN"


# ---------------------------------------------------------------- RESULT / EXIT
def test_result_no_detect_no_full_ocr(tmp_path: Path) -> None:
    """RESULT: template outcome + exit only — never detector.detect()."""
    ctrl = _make_controller(tmp_path)
    ctrl.detector.detect = MagicMock()
    ctrl.detector.outcome_from_templates = MagicMock(return_value=None)
    ctrl.templates.find = MagicMock(
        side_effect=lambda img, name, **kw: DetectionResult(
            found=(name == "exit"),
            confidence=0.9,
            bbox=(10, 10, 40, 20),
            template_name=name,
        )
    )
    ctrl.buttons.find_exit = MagicMock(
        return_value=DetectionResult(found=True, confidence=0.9, bbox=(1, 2, 30, 20))
    )

    ctrl.sm.transition(AutomationState.RESULT)
    ctrl._state_result()

    ctrl.detector.detect.assert_not_called()
    assert ctrl.sm.state == AutomationState.EXIT_RESULT
    assert ctrl.ctx.outcome is None  # unknown outcome still proceeds to exit


def test_result_victory_counts_stats(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.detector.detect = MagicMock()
    ctrl.detector.outcome_from_templates = MagicMock(return_value="WIN")
    ctrl.templates.find = MagicMock(return_value=DetectionResult(found=True))

    ctrl.sm.transition(AutomationState.RESULT)
    ctrl._state_result()
    assert ctrl.sm.state == AutomationState.EXIT_RESULT
    assert ctrl.statistics.total_battles == 1
    assert ctrl.statistics.victories == 1


def test_exit_result_find_exit_full_ocr_false(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.buttons.find_exit = MagicMock(
        return_value=DetectionResult(found=False, confidence=0.0)
    )
    ctrl.sm.transition(AutomationState.EXIT_RESULT)
    ctrl._state_exit_result()
    ctrl.buttons.find_exit.assert_called_once()
    _, kwargs = ctrl.buttons.find_exit.call_args
    assert kwargs.get("full_ocr") is False
    assert ctrl.sm.state == AutomationState.EXIT_RESULT  # wait next tick


def test_exit_result_stop_raises(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.stop_event.set()
    ctrl.sm.transition(AutomationState.EXIT_RESULT)
    with pytest.raises(StoppedError):
        ctrl._state_exit_result()


# ---------------------------------------------------------------- short-circuit
def test_is_placeholder_power_region() -> None:
    assert is_placeholder_power_region((0, 0, 100, 50))
    assert not is_placeholder_power_region((1614, 202, 184, 68))


def _power_reader(cfg=None) -> PowerReader:
    from arena_auto.models.config import AppConfig

    config = cfg or AppConfig()
    ocr = MagicMock()
    reader = PowerReader(ocr, config, config.recognition)
    return reader


def test_read_until_first_slot_match_skips_rest() -> None:
    reader = _power_reader()
    reader.recognition.stable_reads = 1  # isolate short-circuit (no stability re-read)
    cfgs = [
        OpponentConfig(id=1, power_region=(10, 10, 50, 30), click=(100, 200)),
        OpponentConfig(id=2, power_region=(10, 100, 50, 30), click=(100, 400)),
    ]
    reader._read_once_debug = MagicMock(return_value=(100_000, "100000"))
    shots = []

    def factory():
        shots.append(1)
        return _img()

    results, selected = reader.read_until_at_or_below(
        factory, 2_000_000, cfgs, stop_event=None
    )
    assert selected is not None and selected.id == 1
    assert len(results) == 1
    assert reader._read_once_debug.call_count == 1  # short-circuit
    assert len(shots) == 1


def test_read_until_placeholder_first_raises_without_ocr() -> None:
    reader = _power_reader()
    cfgs = [OpponentConfig(id=5, power_region=(0, 0, 100, 50), click=(0, 0))]
    reader._read_once_debug = MagicMock()
    with pytest.raises(OCRFailedError, match="自动标定"):
        reader.read_until_at_or_below(lambda: _img(), 2_000_000, cfgs)
    reader._read_once_debug.assert_not_called()


def test_read_until_placeholder_tail_stops_after_real_slots() -> None:
    """#1-4 real above threshold + #5 placeholder → stop cleanly, no OCR_FAILED."""
    reader = _power_reader()
    reader.recognition.stable_reads = 1
    cfgs = [
        OpponentConfig(id=1, power_region=(10, 10, 50, 30), click=(1, 1)),
        OpponentConfig(id=5, power_region=(0, 0, 100, 50), click=(0, 0)),
    ]
    reader._read_once_debug = MagicMock(return_value=(5_000_000, "5000000"))
    results, selected = reader.read_until_at_or_below(
        lambda: _img(), 2_000_000, cfgs
    )
    assert selected is None
    assert len(results) == 1
    assert reader._read_once_debug.call_count == 1


def test_read_until_none_power_raises() -> None:
    reader = _power_reader()
    cfgs = [OpponentConfig(id=1, power_region=(10, 10, 50, 30), click=(1, 1))]
    reader._read_once_debug = MagicMock(return_value=(None, ""))
    with pytest.raises(OCRFailedError, match="OCR_FAILED"):
        reader.read_until_at_or_below(lambda: _img(), 2_000_000, cfgs)


def test_read_until_stop_event_aborts_before_ocr() -> None:
    reader = _power_reader()
    cfgs = [OpponentConfig(id=1, power_region=(10, 10, 50, 30), click=(1, 1))]
    reader._read_once_debug = MagicMock(return_value=(100_000, "1"))
    stop = threading.Event()
    stop.set()
    with pytest.raises(OCRFailedError, match="停止"):
        reader.read_until_at_or_below(
            lambda: _img(), 2_000_000, cfgs, stop_event=stop
        )
    reader._read_once_debug.assert_not_called()


# ---------------------------------------------------------------- OCR cancel
def test_ocr_engine_cancel_short_circuits() -> None:
    from arena_auto.recognition.ocr_engine import OcrEngine

    base = MagicMock()
    base.recognize = MagicMock(return_value=[])
    engine = OcrEngine(provider=base, settings=None, lazy=True, use_shared=False)
    engine.set_cancel_check(lambda: True)
    boxes = engine.recognize(_img(100, 100))
    assert boxes == []
    base.recognize.assert_not_called()


def test_request_stop_installs_ocr_cancel(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    engines = []

    class FakeEngine:
        def __init__(self) -> None:
            self.fn = None

        def set_cancel_check(self, fn) -> None:
            self.fn = fn

    eng = FakeEngine()
    ctrl.power_reader.engine = eng
    engines.append(eng)
    ctrl.request_stop()
    assert ctrl.stop_event.is_set()
    assert eng.fn is not None
    assert eng.fn() is True


# ---------------------------------------------------------------- ScreenDetector
def test_detect_screen_and_outcome_template_only_no_ocr() -> None:
    config = ConfigManager.from_dict(
        __import__("arena_auto.config.manager", fromlist=["default_config_dict"]).default_config_dict()
    )
    ocr = MagicMock()
    ocr.recognize = MagicMock(
        return_value=[OCRBox(text="胜利", confidence=0.9, bbox=(0, 0, 10, 10))]
    )
    detector = ScreenDetector(ocr, config, templates=None)
    image = _img()
    kind, outcome = detector.detect_screen_and_outcome(image, allow_ocr=False)
    assert kind == ScreenKind.UNKNOWN
    assert outcome is None
    ocr.recognize.assert_not_called()


def test_find_exit_full_ocr_false_skips_full_frame_ocr(tmp_path: Path) -> None:
    from arena_auto.models.config import AppConfig, TemplateEntry
    from arena_auto.recognition.detector import ButtonFinder
    from arena_auto.recognition.template import TemplateRecognizer

    config = AppConfig()
    config.templates = {
        "exit": TemplateEntry(path=str(tmp_path / "no.png"), threshold=0.82)
    }
    config.profile.exit_text = "退出"
    recognizer = TemplateRecognizer(config.templates, base_dir=tmp_path)
    ocr = MagicMock()
    ocr.recognize = MagicMock(return_value=[])
    finder = ButtonFinder(recognizer, ocr, config, None)

    # region hint missing → with full_ocr=False must not full-frame OCR
    result = finder.find_exit(_img(), full_ocr=False)
    assert not result.found
    ocr.recognize.assert_not_called()
