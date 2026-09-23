"""Automation package."""

from arena_auto.automation.controller import AutomationController, ControllerCallbacks
from arena_auto.automation.selection import select_first_at_or_below
from arena_auto.automation.state_machine import StateMachine
from arena_auto.automation.states import AutomationState, ScreenKind

__all__ = [
    "AutomationController",
    "AutomationState",
    "ControllerCallbacks",
    "ScreenKind",
    "StateMachine",
    "select_first_at_or_below",
]
