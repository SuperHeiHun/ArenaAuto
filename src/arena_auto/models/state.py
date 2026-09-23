"""Automation state and session statistics models."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict


class AutomationState(Enum):
    """Top-level automation state machine states."""

    IDLE = auto()
    CHECK_DEVICE = auto()
    DETECT_SCREEN = auto()
    ARENA = auto()
    SELECT_OPPONENT = auto()
    WAIT_PREPARE = auto()
    PREPARE = auto()
    START_BATTLE = auto()
    WAIT_RESULT = auto()
    RESULT = auto()
    EXIT_RESULT = auto()
    CHECK_CHALLENGE_COUNT = auto()
    BUY_CHALLENGE = auto()
    CONFIRM_PURCHASE = auto()
    RECOVERY = auto()
    STOPPED = auto()
    ERROR = auto()


class ScreenKind(Enum):
    """Detected game screen category."""

    ARENA = "arena"
    PREPARE = "prepare"
    BATTLE = "battle"
    RESULT = "result"
    PURCHASE = "purchase"
    UNKNOWN = "unknown"


@dataclass
class SessionStatistics:
    """Runtime counters shown in the GUI and persisted as JSON."""

    total_battles: int = 0
    victories: int = 0
    defeats: int = 0
    purchases: int = 0
    recognition_failures: int = 0
    recoveries: int = 0
    session_purchase_count: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def uptime_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["uptime_seconds"] = self.uptime_seconds
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionStatistics":
        known = {f: data.get(f) for f in cls.__dataclass_fields__ if f in data}
        known = {k: v for k, v in known.items() if v is not None}
        return cls(**known)

    def merge_cumulative(self, historical: Dict[str, Any]) -> None:
        """Add persisted lifetime totals into this session object for display."""
        self.total_battles += int(historical.get("total_battles", 0) or 0)
        self.victories += int(historical.get("victories", 0) or 0)
        self.defeats += int(historical.get("defeats", 0) or 0)
        self.purchases += int(historical.get("purchases", 0) or 0)
        self.recognition_failures += int(historical.get("recognition_failures", 0) or 0)
        self.recoveries += int(historical.get("recoveries", 0) or 0)
