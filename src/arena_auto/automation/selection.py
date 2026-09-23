"""Opponent selection: first slot at or below threshold, order 1..5."""

from __future__ import annotations

from typing import List, Optional, Sequence

from arena_auto.exceptions import OCRFailedError
from arena_auto.models.recognition import Opponent


def select_first_at_or_below(
    powers: Sequence[Optional[int]],
    threshold: int,
    *,
    start_index: int = 0,
) -> Optional[int]:
    """Return 0-based index of first power <= threshold.

    - Iterates strictly in list order (opponents 1 -> 5).
    - None entries are OCR failures and raise OCRFailedError: they are never
      treated as 0 and never skipped silently (skipping would risk attacking
      an unreadable high-power target later while an earlier unknown slot
      might have been eligible).
    - Returns None when every readable power exceeds threshold.
    """
    if threshold < 0:
        raise ValueError("threshold 不能为负")

    for index in range(start_index, len(powers)):
        value = powers[index]
        if value is None:
            raise OCRFailedError(f"对手 #{index + 1} 战力 OCR_FAILED，禁止继续选择")
        if value <= threshold:
            return index
    return None


def select_opponent(
    opponents: Sequence[Opponent],
    threshold: int,
) -> Optional[Opponent]:
    """Select first Opponent with power <= threshold. Raises OCRFailedError on None power."""
    powers: List[Optional[int]] = [opp.power for opp in opponents]
    index = select_first_at_or_below(powers, threshold)
    if index is None:
        return None
    return opponents[index]
