"""No real ADB: timing diagnostics and single-use connection evidence."""

import logging
import subprocess
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from arena_auto.adb.controller import AdbController
from arena_auto.exceptions import AdbDisconnectedError, AdbError, ScreenshotError
from arena_auto.models.config import AdbConfig, DebugSettings
from arena_auto.debug.recorder import DebugRecorder
from arena_auto.recognition.frame_guard import compare_regions, regions_unchanged, save_guard_evidence


def fake_adb(monkeypatch):
    adb = AdbController(AdbConfig(serial="fake-device"))
    adb._connected = True
    monkeypatch.setattr(adb, "resolve_executable", lambda: "fake-adb")
    frame = np.full((80, 160, 3), 100, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    raw = encoded.tobytes()
    run = MagicMock(side_effect=lambda args, **kwargs: raw if "exec-out" in args else "ok")
    monkeypatch.setattr(adb, "_run", run)
    return adb, run, frame


def test_recent_screenshot_skips_one_echo_but_not_next_tap(monkeypatch, caplog):
    adb, run, frame = fake_adb(monkeypatch)
    with caplog.at_level(logging.INFO):
        assert np.array_equal(adb.screenshot(), frame)
        adb.tap(10, 20)
        adb.tap(30, 40)
    assert [call.args[0][-1] for call in run.call_args_list] == ["-p", "input tap 10 20", "echo ok", "input tap 30 40"]
    assert "命令/设备编码/传输" in caplog.text and "PNG修复/解码" in caplog.text
    assert "复用近期截图" in caplog.text and "主动探测" in caplog.text


@pytest.mark.parametrize("age", [3.01, -1.0])
def test_expired_or_future_capture_still_probes(monkeypatch, age):
    adb, run, _ = fake_adb(monkeypatch)
    adb.screenshot()
    adb._screenshot_at = 100.0
    monkeypatch.setattr("arena_auto.adb.controller.time.monotonic", lambda: 100.0 + age)
    adb.tap(1, 2)
    assert run.call_args_list[-2].args[0][-1] == "echo ok"


@pytest.mark.parametrize("action", ["update", "disconnect", "device", "connect", "swipe", "keyevent", "tcpip"])
def test_config_device_connection_or_other_action_invalidates_capture(monkeypatch, action):
    adb, run, _ = fake_adb(monkeypatch)
    adb.screenshot()
    if action == "update":
        adb.update_config(AdbConfig(serial="fake-device"))
    elif action == "disconnect":
        adb.disconnect()
        adb._connected = True  # Even a later connection cannot revive the evidence.
    elif action == "device":
        adb._serial = "other-device"
    elif action == "connect":
        adb.connect("fake-device")
    elif action == "swipe":
        adb.swipe(1, 2, 3, 4)
    elif action == "keyevent":
        adb.keyevent(4)
    else:
        adb.enable_tcpip()
    run.reset_mock(side_effect=False)
    adb.tap(1, 2)
    assert run.call_args_list[0].args[0][-1] == "echo ok"


def test_successful_capture_does_not_revive_disconnected_state(monkeypatch):
    adb, run, _ = fake_adb(monkeypatch)
    adb._connected = False
    adb.screenshot()
    with pytest.raises(AdbDisconnectedError):
        adb.tap(1, 2)
    assert run.call_count == 1


@pytest.mark.parametrize("failure", [b"", b"not a png", AdbError("timeout")])
def test_failed_screenshot_cannot_leave_old_connection_evidence(monkeypatch, failure):
    adb, run, _ = fake_adb(monkeypatch)
    adb.screenshot()
    run.side_effect = failure if isinstance(failure, Exception) else None
    run.return_value = failure
    with pytest.raises((ScreenshotError, AdbError)):
        adb.screenshot()
    assert adb._screenshot_at is None
    run.side_effect = None
    run.return_value = "ok"
    adb.tap(1, 2)
    assert run.call_args_list[-2].args[0][-1] == "echo ok"


def test_tap_failure_is_not_retried_or_probed_after_sending(monkeypatch):
    adb, run, _ = fake_adb(monkeypatch)
    adb.screenshot()
    run.reset_mock()
    run.side_effect = AdbError("transport timeout")
    with pytest.raises(AdbError, match="点击失败"):
        adb.tap(1, 2)
    run.assert_called_once()
    assert not adb._connected and adb._screenshot_at is None
    with pytest.raises(AdbDisconnectedError):
        adb.tap(1, 2)
    assert run.call_count == 1


@pytest.mark.parametrize("failure", ["timeout", "offline", "generic", "unchecked"])
def test_actual_command_failure_invalidates_screenshot_token(monkeypatch, failure):
    adb = AdbController(AdbConfig(serial="fake-device"))
    adb._connected = True
    adb._screenshot_at = 100.0
    adb._screenshot_serial = adb.serial
    if failure == "timeout":
        run = MagicMock(side_effect=subprocess.TimeoutExpired("fake-adb", 1))
    else:
        run = MagicMock(return_value=subprocess.CompletedProcess([], 1, b"", b"device offline" if failure == "offline" else b"error"))
    monkeypatch.setattr("arena_auto.adb.controller.subprocess.run", run)
    if failure == "unchecked":
        adb._run(["fake-adb"], check=False)
    else:
        with pytest.raises(AdbError):
            adb._run(["fake-adb"])
    assert adb._screenshot_at is None


def test_guard_report_identifies_absolute_local_block_without_weakening_thresholds():
    old = np.full((80, 160, 3), 100, dtype=np.uint8)
    fresh = old.copy()
    fresh[16:32, 80:96] = 150
    report = compare_regions(old, fresh, [(64, 0, 96, 80)])
    assert not report.unchanged and not regions_unchanged(old, fresh, [(64, 0, 96, 80)])
    assert report.max_block > 3 and report.changed_blocks > 0
    assert (80, 16, 16, 16) in report.blocks and len(report.blocks) <= 5
    assert "最大变化位置" in report.describe()
    assert compare_regions(old, old.copy(), [(0, 0, 160, 80)]).unchanged


@pytest.mark.parametrize("kind,reason", [("missing", "缺少截图"), ("shape", "截图尺寸变化"), ("regions", "缺少校验区域"), ("invalid", "校验区域无效")])
def test_guard_reports_invalid_inputs(kind, reason):
    old = np.zeros((80, 160, 3), dtype=np.uint8)
    fresh = None if kind == "missing" else np.zeros((40, 80, 3), dtype=np.uint8) if kind == "shape" else old.copy()
    regions = [] if kind == "regions" else [(-1, 0, 10, 10)] if kind == "invalid" else [(0, 0, 160, 80)]
    report = compare_regions(old, fresh, regions)
    assert not report.unchanged and report.reason == reason


def test_guard_evidence_optional_bounded_and_does_not_modify_frames(tmp_path):
    old = np.full((80, 160, 3), 100, dtype=np.uint8)
    fresh = old.copy()
    fresh[16:32, 80:96] = 150
    original = fresh.copy()
    report = compare_regions(old, fresh, [(0, 0, 160, 80)])
    recorder = DebugRecorder(DebugSettings(enabled=False), base_dir=tmp_path)
    save_guard_evidence(recorder, old, fresh, report, "test")
    assert not list(tmp_path.rglob("*.png"))
    recorder.settings.enabled = True
    recorder.settings.save_screenshots = True
    save_guard_evidence(recorder, old, fresh, report, "test")
    assert len(list(tmp_path.rglob("*.png"))) == 3
    assert np.array_equal(fresh, original)
    broken = MagicMock()
    broken.enabled = True
    broken.save.side_effect = OSError("disk full")
    save_guard_evidence(broken, old, fresh, report, "test")  # Diagnostics are nonfatal.


def test_guard_decisions_match_previous_threshold_algorithm():
    def original(previous, current, regions):
        for x, y, w, h in regions:
            diff = cv2.absdiff(cv2.GaussianBlur(previous[y:y+h, x:x+w], (3, 3), 0),
                               cv2.GaussianBlur(current[y:y+h, x:x+w], (3, 3), 0))
            if float(diff.mean()) > 1.5:
                return False
            for top in range(0, h, 16):
                for left in range(0, w, 16):
                    if float(diff[top:top+16, left:left+16].mean()) > 3.0:
                        return False
        return True

    rng = np.random.default_rng(17)
    old = rng.integers(30, 220, (80, 160, 3), dtype=np.uint8)
    regions = [(1, 3, 73, 47), (80, 0, 80, 80)]
    for magnitude in range(10):
        for _ in range(10):
            noise = rng.integers(-magnitude, magnitude + 1, old.shape)
            fresh = np.clip(old.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            assert regions_unchanged(old, fresh, regions) == original(old, fresh, regions)
