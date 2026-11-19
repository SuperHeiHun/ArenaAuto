"""Power / challenge text parsing tests."""

from __future__ import annotations

import pytest

from arena_auto.recognition.ocr import parse_challenge_count, parse_power


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,234,567", 1234567),
        ("1 234 567", 1234567),
        ("1.234.567", 1234567),
        ("1234567", 1234567),
        ("2,500,000", 2500000),
        ("I234567", 1234567),
        ("l234567", 1234567),
        ("  1,800,000  ", 1800000),
        ("战斗力：1,234,567", 1234567),
        ("战斗力:1234567", 1234567),
        ("战力 4575674", 4575674),
        ("战力：2,500,000", 2500000),
        ("Power: 1200000", 1200000),
    ],
)
def test_parse_power_valid(raw: str, expected: int) -> None:
    assert parse_power(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "abc",
        "12A45",
        "0",
        "-5",
        "???",
        "1,2,3,x",
        "战斗力：",
        "战力 abc",
        "战斗力：12A45",
    ],
)
def test_parse_power_invalid_returns_none(raw) -> None:
    assert parse_power(raw) is None


def test_parse_power_never_returns_zero_on_failure() -> None:
    assert parse_power("failed") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4 / 5", (4, 5)),
        ("0/5", (0, 5)),
        ("可挑战次数 3 / 5", (3, 5)),
        ("1/10", (1, 10)),
    ],
)
def test_parse_challenge_count(raw: str, expected: tuple) -> None:
    assert parse_challenge_count(raw) == expected


def test_parse_challenge_invalid() -> None:
    assert parse_challenge_count("no numbers") is None
    assert parse_challenge_count("9 / 5") is None
