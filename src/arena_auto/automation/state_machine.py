"""Explicit state machine with per-state timeout tracking."""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Tuple

from arena_auto.models.state import AutomationState

logger = logging.getLogger(__name__)

# Optional validator: return True if transition is allowed
TransitionHook = Callable[[AutomationState, AutomationState], bool]


class StateMachine:
    """Tracks current automation state, entry time, and history."""

    def __init__(
        self,
        initial: AutomationState = AutomationState.IDLE,
        on_transition: Optional[Callable[[AutomationState, AutomationState], None]] = None,
    ) -> None:
        self._state = initial
        self._entered_at = time.monotonic()
        self._previous = initial
        self._history: List[Tuple[AutomationState, float]] = [(initial, self._entered_at)]
        self._on_transition = on_transition

    @property
    def state(self) -> AutomationState:
        return self._state

    @property
    def previous(self) -> AutomationState:
        return self._previous

    @property
    def history(self) -> List[Tuple[AutomationState, float]]:
        return list(self._history)

    def transition(self, new_state: AutomationState) -> AutomationState:
        """Move to new_state and reset entry timer."""
        if new_state == self._state:
            # refresh timer on self-loop only when explicitly re-entered
            return self._state
        old = self._state
        self._previous = old
        self._state = new_state
        self._entered_at = time.monotonic()
        self._history.append((new_state, self._entered_at))
        if len(self._history) > 100:
            self._history = self._history[-100:]
        logger.debug("状态切换: %s -> %s", old.name, new_state.name)
        if self._on_transition is not None:
            try:
                self._on_transition(old, new_state)
            except Exception as exc:  # noqa: BLE001
                logger.warning("状态回调失败: %s", exc)
        return self._state

    def reenter(self) -> None:
        """Reset the timeout timer without changing state."""
        self._entered_at = time.monotonic()

    def time_in_state(self) -> float:
        return time.monotonic() - self._entered_at

    def has_timed_out(self, timeout_seconds: float) -> bool:
        if timeout_seconds <= 0:
            return False
        return self.time_in_state() >= timeout_seconds

    def reset(self) -> None:
        self._previous = self._state
        self._state = AutomationState.IDLE
        self._entered_at = time.monotonic()
        self._history.append((AutomationState.IDLE, self._entered_at))
