"""Purchase flow tests: challenge==0 -> select opponent -> dialog -> buy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from arena_auto.automation.controller import AutomationController, ControllerCallbacks
from arena_auto.config.manager import ConfigManager
from arena_auto.debug.recorder import DebugRecorder
from arena_auto.models.recognition import ChallengeCount, DetectionResult, Opponent
from arena_auto.models.state import AutomationState, ScreenKind


def _make_controller(tmp_path: Path) -> AutomationController:
    manager = ConfigManager(path=tmp_path / "config.yaml")
    manager.config = manager.config.__class__()
    manager.config.automation.dry_run = True
    manager.config.debug.enabled = False

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


def test_zero_challenge_goes_select_opponent(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.config.purchase.enabled = True
    ctrl.challenge_reader.read = MagicMock(return_value=ChallengeCount(0, 5))
    ctrl.sm.transition(AutomationState.ARENA)
    ctrl._state_arena()
    assert ctrl.ctx.pending_purchase is True
    assert ctrl.sm.state == AutomationState.SELECT_OPPONENT


def test_zero_challenge_not_purchasable_stops(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.config.purchase.enabled = False
    ctrl.challenge_reader.read = MagicMock(return_value=ChallengeCount(0, 5))
    ctrl.sm.transition(AutomationState.ARENA)
    ctrl._state_arena()
    assert ctrl.sm.state == AutomationState.STOPPED


def test_zero_challenge_over_max_stops(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.config.purchase.enabled = True
    ctrl.config.purchase.max_per_session = 0
    ctrl.challenge_reader.read = MagicMock(return_value=ChallengeCount(0, 5))
    ctrl.sm.transition(AutomationState.ARENA)
    ctrl._state_arena()
    assert ctrl.sm.state == AutomationState.STOPPED


def test_positive_challenge_clears_pending(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.ctx.pending_purchase = True
    ctrl.challenge_reader.read = MagicMock(return_value=ChallengeCount(3, 5))
    ctrl.sm.transition(AutomationState.ARENA)
    ctrl._state_arena()
    assert ctrl.ctx.pending_purchase is False
    assert ctrl.sm.state == AutomationState.SELECT_OPPONENT


def _opponents() -> list[Opponent]:
    return [
        Opponent(id=1, power=2_500_000, click_position=(100, 200)),
        Opponent(id=2, power=1_800_000, click_position=(100, 400)),
    ]


def test_opponent_tap_with_pending_goes_buy(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.ctx.pending_purchase = True
    ctrl.power_reader.read_until_at_or_below = MagicMock(
        return_value=(_opponents(), _opponents()[1])
    )
    ctrl.sm.transition(AutomationState.SELECT_OPPONENT)
    ctrl._state_select_opponent()
    assert ctrl.sm.state == AutomationState.BUY_CHALLENGE
    assert ctrl.ctx.pending_purchase is True


def test_opponent_tap_without_pending_goes_wait_prepare(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.ctx.pending_purchase = False
    ctrl.power_reader.read_until_at_or_below = MagicMock(
        return_value=(_opponents(), _opponents()[1])
    )
    ctrl.sm.transition(AutomationState.SELECT_OPPONENT)
    ctrl._state_select_opponent()
    assert ctrl.sm.state == AutomationState.WAIT_PREPARE


def test_opponent_short_circuit_skips_later_slots(tmp_path: Path) -> None:
    """First slot <= threshold must not OCR #2..#5 (read_until contract)."""
    ctrl = _make_controller(tmp_path)
    first = Opponent(id=1, power=188_508, click_position=(100, 200))
    ctrl.power_reader.read_until_at_or_below = MagicMock(return_value=([first], first))
    ctrl.sm.transition(AutomationState.SELECT_OPPONENT)
    ctrl._state_select_opponent()
    assert ctrl.sm.state == AutomationState.WAIT_PREPARE
    assert ctrl.ctx.target is not None and ctrl.ctx.target.id == 1
    assert len(ctrl.ctx.opponents) == 1  # only #1 was read
    ctrl.power_reader.read_until_at_or_below.assert_called_once()


def test_opponent_none_power_goes_recovery(tmp_path: Path) -> None:
    """OCR failure (None power) must not silently select; recovery, not RECOVERY-skip."""
    from arena_auto.exceptions import OCRFailedError

    ctrl = _make_controller(tmp_path)
    ctrl.power_reader.read_until_at_or_below = MagicMock(
        side_effect=OCRFailedError("对手 #1 战力 OCR_FAILED，禁止继续选择")
    )
    ctrl.sm.transition(AutomationState.SELECT_OPPONENT)
    ctrl._state_select_opponent()
    assert ctrl.sm.state == AutomationState.RECOVERY
    assert ctrl.ctx.target is None
    assert ctrl.statistics.recognition_failures >= 1


def test_buy_finds_button_goes_confirm(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.buttons.find_buy = MagicMock(
        return_value=DetectionResult(found=True, confidence=0.9, bbox=(10, 20, 80, 40))
    )
    ctrl.buttons.find_confirm = MagicMock(return_value=DetectionResult(found=False))
    ctrl.buttons.find_go_win = MagicMock(return_value=DetectionResult(found=False))
    ctrl.sm.transition(AutomationState.BUY_CHALLENGE)
    ctrl._state_buy_challenge()
    assert ctrl.sm.state == AutomationState.CONFIRM_PURCHASE


def test_buy_dialog_missing_but_prepare_goes_prepare(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.ctx.pending_purchase = True
    ctrl.buttons.find_buy = MagicMock(return_value=DetectionResult(found=False))
    ctrl.buttons.find_confirm = MagicMock(return_value=DetectionResult(found=False))
    ctrl.buttons.find_go_win = MagicMock(
        return_value=DetectionResult(found=True, confidence=0.9, bbox=(1, 2, 30, 20))
    )
    ctrl.sm.transition(AutomationState.BUY_CHALLENGE)
    ctrl._state_buy_challenge()
    assert ctrl.sm.state == AutomationState.PREPARE
    assert ctrl.ctx.pending_purchase is False


def test_buy_timeout_goes_recovery(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.buttons.find_buy = MagicMock(return_value=DetectionResult(found=False))
    ctrl.buttons.find_confirm = MagicMock(return_value=DetectionResult(found=False))
    ctrl.buttons.find_go_win = MagicMock(return_value=DetectionResult(found=False))
    ctrl.config.timeouts.purchase = 0.001
    ctrl.sm.transition(AutomationState.BUY_CHALLENGE)
    import time as _time

    ctrl.sm._entered_at = _time.monotonic() - 1
    ctrl._state_buy_challenge()
    assert ctrl.sm.state == AutomationState.RECOVERY


def test_confirm_single_click_success_routes_prepare(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.ctx.pending_purchase = True
    ctrl.ctx.challenge = ChallengeCount(0, 5)
    ctrl.buttons.find_confirm = MagicMock(return_value=DetectionResult(found=False))
    ctrl.challenge_reader.read = MagicMock(return_value=ChallengeCount(1, 5))
    ctrl.detector.detect = MagicMock(return_value=ScreenKind.PREPARE)
    ctrl.sm.transition(AutomationState.CONFIRM_PURCHASE)
    ctrl._state_confirm_purchase()
    assert ctrl.statistics.purchases == 1
    assert ctrl.statistics.session_purchase_count == 1
    assert ctrl.ctx.pending_purchase is False
    assert ctrl.sm.state == AutomationState.PREPARE


def test_confirm_button_success_routes_arena(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.ctx.pending_purchase = True
    ctrl.ctx.challenge = ChallengeCount(0, 5)
    ctrl.config.purchase.auto_confirm = True
    ctrl.buttons.find_confirm = MagicMock(
        return_value=DetectionResult(found=True, confidence=0.9, bbox=(5, 5, 40, 30))
    )
    ctrl.challenge_reader.read = MagicMock(return_value=ChallengeCount(1, 5))
    ctrl.detector.detect = MagicMock(return_value=ScreenKind.ARENA)
    ctrl.sm.transition(AutomationState.CONFIRM_PURCHASE)
    ctrl._state_confirm_purchase()
    assert ctrl.statistics.purchases == 1
    assert ctrl.ctx.pending_purchase is False
    assert ctrl.sm.state == AutomationState.ARENA


def test_confirm_auto_confirm_off_with_button_stops(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.config.purchase.auto_confirm = False
    ctrl.buttons.find_confirm = MagicMock(
        return_value=DetectionResult(found=True, confidence=0.9, bbox=(5, 5, 40, 30))
    )
    ctrl.sm.transition(AutomationState.CONFIRM_PURCHASE)
    ctrl._state_confirm_purchase()
    assert ctrl.sm.state == AutomationState.STOPPED
    assert ctrl.statistics.purchases == 0


def test_detect_screen_purchase_goes_buy(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.detector.detect = MagicMock(return_value=ScreenKind.PURCHASE)
    ctrl.sm.transition(AutomationState.DETECT_SCREEN)
    ctrl._state_detect_screen()
    assert ctrl.sm.state == AutomationState.BUY_CHALLENGE


def test_recovery_purchase_goes_buy(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path)
    ctrl.detector.detect = MagicMock(return_value=ScreenKind.PURCHASE)
    ctrl.sm.transition(AutomationState.RECOVERY)
    ctrl._state_recovery()
    assert ctrl.sm.state == AutomationState.BUY_CHALLENGE
    assert ctrl.ctx.recovery_attempts == 0
