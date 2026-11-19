"""State machine and opponent selection tests."""

from __future__ import annotations

import time

import pytest

from arena_auto.automation.selection import select_first_at_or_below, select_opponent
from arena_auto.automation.state_machine import StateMachine
from arena_auto.exceptions import OCRFailedError
from arena_auto.models.recognition import Opponent
from arena_auto.models.state import AutomationState


def test_transitions() -> None:
    sm = StateMachine()
    assert sm.state == AutomationState.IDLE
    sm.transition(AutomationState.CHECK_DEVICE)
    sm.transition(AutomationState.DETECT_SCREEN)
    sm.transition(AutomationState.ARENA)
    assert sm.state == AutomationState.ARENA
    assert sm.previous == AutomationState.DETECT_SCREEN
    names = [s.name for s, _ in sm.history]
    assert names[-1] == "ARENA"


def test_timeout_tracking() -> None:
    sm = StateMachine()
    sm.transition(AutomationState.DETECT_SCREEN)
    assert not sm.has_timed_out(10)
    sm._entered_at = time.monotonic() - 11  # force elapsed
    assert sm.has_timed_out(10)
    assert not sm.has_timed_out(0)  # disabled


def test_select_first_at_or_below() -> None:
    # threshold 2,000,000 -> expect index 1 (opponent #2)
    powers = [2_500_000, 1_800_000, 1_900_000, 2_100_000, 1_500_000]
    assert select_first_at_or_below(powers, 2_000_000) == 1


def test_select_none_when_all_exceed() -> None:
    powers = [2_500_000, 2_800_000, 3_000_000, 3_500_000, 4_000_000]
    assert select_first_at_or_below(powers, 2_000_000) is None


def test_select_skips_none_raises() -> None:
    powers = [None, 1_000_000, 1_500_000]
    with pytest.raises(OCRFailedError):
        select_first_at_or_below(powers, 2_000_000)


def test_select_first_including_boundary() -> None:
    powers = [2_000_000, 1_000_000]
    assert select_first_at_or_below(powers, 2_000_000) == 0


def test_select_opponent_returns_object() -> None:
    opponents = [
        Opponent(id=1, power=2_500_000),
        Opponent(id=2, power=1_800_000),
    ]
    selected = select_opponent(opponents, 2_000_000)
    assert selected is not None
    assert selected.id == 2


def test_comparison_rules() -> None:
    assert 2_500_000 > 2_000_000
    assert 1_800_000 <= 2_000_000
