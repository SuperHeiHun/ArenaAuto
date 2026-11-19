"""Coordinate scaling and crop tests."""

from __future__ import annotations

import numpy as np
import pytest

from arena_auto.recognition.screen import CoordinateScaler, crop_region


def test_scale_point_1280x720() -> None:
    scaler = CoordinateScaler(reference_width=1920, reference_height=1080)
    scaler.update_actual(1280, 720)
    assert scaler.scale_point(960, 540) == (640, 360)
    assert scaler.scale_x == pytest.approx(1280 / 1920)
    assert scaler.scale_y == pytest.approx(720 / 1080)


def test_scale_region() -> None:
    scaler = CoordinateScaler(1920, 1080)
    scaler.update_actual(960, 540)
    x, y, w, h = scaler.scale_region((100, 200, 40, 60))
    assert (x, y, w, h) == (50, 100, 20, 30)


def test_scale_identity_when_no_actual() -> None:
    scaler = CoordinateScaler(1920, 1080)
    assert scaler.scale_point(10, 20) == (10, 20)


def test_crop_region_clamps() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = crop_region(image, (90, 90, 30, 30))
    assert roi.shape[0] == 10
    assert roi.shape[1] == 10


def test_crop_region_invalid() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_region(image, (0, 0, 0, 5))
    with pytest.raises(ValueError):
        crop_region(image, (100, 100, 10, 10))
