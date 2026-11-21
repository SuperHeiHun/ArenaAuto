"""Refresh status signals and end audio are isolated from ADB and real speakers."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from arena_auto.automation.controller import ControllerCallbacks
from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import ConfigError
from arena_auto.models.config import CompletionSoundSettings
from arena_auto.models.state import AutomationState


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QApplication, "beep", MagicMock())
    return application


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    from arena_auto.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(MainWindow, "_attach_log_handler", lambda self: None)
    manager = ConfigManager.load_or_create(tmp_path / "config.yaml")
    ui = MainWindow(manager)
    ui._sound_player = MagicMock()
    yield ui
    ui.close()
    app.processEvents()


def test_refresh_count_is_displayed_and_preserved_on_limit_change(window):
    assert "0 / 3" in window.refresh_count_label.text()
    window._on_refresh_count(2, 3)
    assert "2 / 3" in window.refresh_count_label.text()
    window.refresh_limit_spin.setValue(5)
    assert window._apply_form_to_config()
    assert "2 / 5" in window.refresh_count_label.text()
    window.refresh_limit_spin.setValue(0)
    assert window._apply_form_to_config()
    assert "2" in window.refresh_count_label.text() and "已关闭" in window.refresh_count_label.text()
    window._sound_player.play.assert_not_called()


def test_controller_reports_increments_and_eligible_reset(tmp_path):
    from test_opponent_refresh import controller
    ctrl, opponents, _ = controller(tmp_path)
    callback = MagicMock()
    ctrl.callbacks = ControllerCallbacks(on_refresh_count=callback)
    for _ in range(3):
        ctrl._refresh_opponents(opponents)
    assert callback.call_args_list == [call(1, 3), call(2, 3), call(3, 3)]
    opponents[0].power = ctrl.config.automation.power_threshold
    ctrl.auto_targeting.select.return_value = ([opponents[0]], opponents[0])
    ctrl._state_select_opponent()
    assert callback.call_args == call(0, 3)


def test_new_run_reports_zero_without_device_work(tmp_path):
    from test_opponent_refresh import controller
    ctrl, _, _ = controller(tmp_path)
    callback = MagicMock()
    ctrl.callbacks = ControllerCallbacks(on_refresh_count=callback)
    ctrl.ctx.consecutive_refreshes = 3
    ctrl._step = MagicMock(side_effect=lambda: ctrl.sm.transition(AutomationState.STOPPED))
    ctrl._save_statistics = MagicMock()
    ctrl.run()
    callback.assert_called_once_with(0, 3)
    ctrl.adb.tap.assert_not_called()
    ctrl.adb.screenshot.assert_not_called()


def test_status_callback_failure_does_not_change_refresh_limit_logic(tmp_path):
    from test_opponent_refresh import controller
    ctrl, opponents, _ = controller(tmp_path)
    ctrl.callbacks = ControllerCallbacks(on_refresh_count=MagicMock(side_effect=RuntimeError("GUI closed")))
    for _ in range(3):
        ctrl._refresh_opponents(opponents)
    assert ctrl.ctx.consecutive_refreshes == 3 and ctrl.sm.state == AutomationState.STOPPED


def test_worker_wires_refresh_count_with_no_log_parsing(window):
    from arena_auto.gui.workers import AutomationWorker
    ctrl = MagicMock()
    worker = AutomationWorker(ctrl)
    window._automation_worker = worker
    worker.refresh_count_changed.connect(window._on_refresh_count)
    ctrl.callbacks.on_refresh_count(1, 3)
    assert "1 / 3" in window.refresh_count_label.text()
    ctrl.callbacks.on_refresh_count(0, 3)
    assert "0 / 3" in window.refresh_count_label.text()
    window._automation_worker = None


def test_sound_form_save_and_preview_do_not_apply_preview_config(window, monkeypatch, tmp_path):
    from arena_auto.gui.main_window import QFileDialog
    chosen = str(tmp_path / "自定义 提示音.mp3")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", MagicMock(return_value=(chosen, "audio")))
    window._choose_completion_sound()
    assert window.sound_path_edit.text() == chosen
    window.completion_sound_check.setChecked(False)
    original = copy.deepcopy(window.config.completion_sound)
    window._preview_completion_sound()
    window._sound_player.play.assert_called_once_with(CompletionSoundSettings(True, chosen))
    assert window.config.completion_sound == original
    assert window._apply_form_to_config()
    window.config_manager.save()
    restored = window.config_manager.load()
    assert restored.completion_sound == CompletionSoundSettings(False, chosen)


def test_cancel_file_choice_preserves_existing_path(window, monkeypatch):
    from arena_auto.gui.main_window import QFileDialog
    window.sound_path_edit.setText("kept.wav")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", MagicMock(return_value=("", "")))
    window._choose_completion_sound()
    assert window.sound_path_edit.text() == "kept.wav"


def test_cancel_start_never_arms_or_plays_completion_sound(window):
    window._ensure_controller = MagicMock()
    window._confirm_automation_start = MagicMock(return_value=False)
    window.start_automation()
    assert not window._completion_pending
    window._on_worker_finished()
    window._sound_player.play.assert_not_called()


def test_finished_sound_plays_once_and_uses_run_snapshot(window):
    chosen = CompletionSoundSettings(True, "run-snapshot.wav")
    window._completion_settings = copy.deepcopy(chosen)
    window._completion_pending = True
    window.config.completion_sound.path = "changed-later.mp3"
    for control in (window.completion_sound_check, window.sound_path_edit, window.sound_browse_btn):
        control.setEnabled(False)
    window._on_worker_finished()
    window._on_worker_finished()
    window._sound_player.play.assert_called_once_with(chosen)
    assert window.start_btn.isEnabled()
    assert window.sound_path_edit.isEnabled()


def test_manual_stop_sounds_only_when_worker_finishes(window):
    worker = MagicMock()
    window._automation_worker = worker
    window._completion_pending = True
    window._completion_settings = CompletionSoundSettings()
    window.stop_automation()
    worker.request_stop.assert_called_once()
    window._sound_player.play.assert_not_called()
    window._on_worker_finished()
    window._sound_player.play.assert_called_once()


def test_disabled_snapshot_does_not_play_even_if_form_changes(window):
    window._completion_pending = True
    window._completion_settings = CompletionSoundSettings(False, "disabled.wav")
    window.config.completion_sound.enabled = True
    window._on_worker_finished()
    window._sound_player.play.assert_not_called()


def test_sound_failure_cannot_break_gui_finished_cleanup(window):
    window._completion_pending = True
    window._sound_player.play.side_effect = RuntimeError("audio backend missing")
    window._on_worker_finished()
    assert window.start_btn.isEnabled() and not window._completion_pending
    assert window._automation_worker is None


def test_close_suppresses_completion_and_stops_preview(window):
    window._completion_pending = True
    window.close()
    window._on_worker_finished()
    assert window._closing and not window._completion_pending
    window._sound_player.play.assert_not_called()
    window._sound_player.stop.assert_called()


def test_old_worker_events_cannot_end_or_update_new_run(window):
    from arena_auto.gui.workers import AutomationWorker
    old, active = AutomationWorker(MagicMock()), AutomationWorker(MagicMock())
    window._automation_worker = active
    window._completion_pending = True
    for worker in (old, active):
        worker.finished.connect(window._on_worker_finished)
        worker.refresh_count_changed.connect(window._on_refresh_count)
    old.refresh_count_changed.emit(99, 3)
    old.finished.emit()
    assert "0 / 3" in window.refresh_count_label.text()
    assert window._completion_pending and window._automation_worker is active
    window._sound_player.play.assert_not_called()
    active.refresh_count_changed.emit(3, 3)
    active.finished.emit()
    active.finished.emit()
    assert "3 / 3" in window.refresh_count_label.text()
    window._sound_player.play.assert_called_once()


def test_relay_keeps_run_identity_when_qt_sender_is_unavailable(window, monkeypatch):
    from arena_auto.gui.main_window import _AutomationStatusRelay
    from arena_auto.gui.workers import AutomationWorker
    old, active = AutomationWorker(MagicMock()), AutomationWorker(MagicMock())
    window._automation_worker = active
    window._completion_pending = True
    monkeypatch.setattr(window, "sender", lambda: None)
    stale = _AutomationStatusRelay(window, old)
    stale.refresh(99, 3)
    stale.finish()
    assert "0 / 3" in window.refresh_count_label.text()
    assert window._completion_pending and window._automation_worker is active
    window._sound_player.play.assert_not_called()
    current = _AutomationStatusRelay(window, active)
    current.refresh(3, 3)
    current.finish()
    assert "3 / 3" in window.refresh_count_label.text()
    window._sound_player.play.assert_called_once()


def test_real_worker_thread_delivers_status_and_finish_audio_on_gui_thread(window, app):
    from PySide6.QtCore import QEventLoop, QThread, QTimer
    controller = MagicMock()

    def run():
        controller.callbacks.on_refresh_count(1, 3)
        controller.callbacks.on_finished()
        controller.callbacks.on_finished()  # Defensive duplicate-finish coverage.

    controller.run.side_effect = run
    window._ensure_controller = MagicMock(return_value=controller)
    window._confirm_automation_start = MagicMock(return_value=True)
    loop = QEventLoop()
    gui_threads = []

    def played(_settings):
        gui_threads.append(QThread.currentThread() == app.thread())
        loop.quit()

    window._sound_player.play.side_effect = played
    window.start_automation()
    assert not window.sound_path_edit.isEnabled()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(3000)
    loop.exec()
    timer.stop()
    # Never leave a test thread alive if the expected signal was not delivered.
    if window._automation_thread is not None:
        window._automation_thread.quit()
        window._automation_thread.wait(3000)
        window._automation_thread = None
    assert gui_threads == [True]
    assert "1 / 3" in window.refresh_count_label.text()
    assert window.sound_path_edit.isEnabled() and not window._completion_pending


@pytest.mark.parametrize("raw", [None, "file.wav", {"enabled": "false"}, {"path": 42}])
def test_invalid_sound_config_rejected(raw):
    with pytest.raises(ConfigError, match="completion_sound"):
        ConfigManager.from_dict({"completion_sound": raw})


def test_old_config_uses_system_sound_and_null_path_can_clear_file():
    assert ConfigManager.from_dict({}).completion_sound == CompletionSoundSettings(True, "")
    assert ConfigManager.from_dict({"completion_sound": {"path": None}}).completion_sound.path == ""


@pytest.fixture
def media(app, monkeypatch):
    class Signal:
        def __init__(self):
            self.slots = []

        def connect(self, slot):
            self.slots.append(slot)

        def emit(self, *args):
            for slot in list(self.slots):
                slot(*args)

    players, audios, options = [], [], {}

    class Player:
        MediaStatus = SimpleNamespace(InvalidMedia="invalid")

        def __init__(self, parent):
            self.errorOccurred, self.mediaStatusChanged = Signal(), Signal()
            self.play, self.stop, self.deleteLater = MagicMock(), MagicMock(), MagicMock()
            self.setAudioOutput = MagicMock()
            players.append(self)

        def setSource(self, source):
            self.source = source
            if options.get("invalid_on_source"):
                self.mediaStatusChanged.emit(self.MediaStatus.InvalidMedia)

    class Audio:
        def __init__(self, parent):
            self.deleteLater = MagicMock()
            audios.append(self)

    module = SimpleNamespace(QMediaPlayer=Player, QAudioOutput=Audio)
    monkeypatch.setitem(sys.modules, "PySide6.QtMultimedia", module)
    from arena_auto.gui.completion_sound import CompletionSoundPlayer
    player = CompletionSoundPlayer()
    monkeypatch.setattr(player, "_beep", MagicMock())
    yield player, players, audios, options, module
    player.stop()


def test_disabled_sound_creates_no_audio_backend(media, tmp_path):
    service, players, _, _, _ = media
    service.play(CompletionSoundSettings(False, str(tmp_path / "missing.mp3")))
    assert not players
    service._beep.assert_not_called()


def test_default_and_missing_custom_file_use_system_sound(media, tmp_path):
    service, players, _, _, _ = media
    service.play(CompletionSoundSettings())
    service.play(CompletionSoundSettings(True, str(tmp_path / "missing.wav")))
    assert not players and service._beep.call_count == 2


def test_custom_unicode_relative_file_loads_asynchronously(media, monkeypatch, tmp_path):
    from arena_auto.gui import completion_sound
    service, players, audios, _, _ = media
    monkeypatch.setattr(completion_sound, "get_app_root", lambda: tmp_path)
    path = tmp_path / "我的 提示.mp3"
    path.write_bytes(b"mock audio, never decoded")
    service.play(CompletionSoundSettings(True, path.name))
    assert Path(players[0].source.toLocalFile()).resolve() == path.resolve()
    players[0].setAudioOutput.assert_called_once_with(audios[0])
    players[0].play.assert_called_once()
    service._beep.assert_not_called()


def test_decode_error_falls_back_exactly_once(media, tmp_path):
    service, players, _, _, _ = media
    path = tmp_path / "bad.wav"
    path.write_bytes(b"invalid bytes")
    service.play(CompletionSoundSettings(True, str(path)))
    players[0].errorOccurred.emit("decoder error")
    players[0].mediaStatusChanged.emit("invalid")
    service._beep.assert_called_once()


def test_synchronous_media_error_does_not_then_start_bad_audio(media, tmp_path):
    service, players, _, options, _ = media
    path = tmp_path / "bad.wav"
    path.write_bytes(b"invalid bytes")
    options["invalid_on_source"] = True
    service.play(CompletionSoundSettings(True, str(path)))
    players[0].play.assert_not_called()
    service._beep.assert_called_once()


def test_late_error_from_old_preview_cannot_interrupt_new_sound(media, tmp_path):
    service, players, _, _, _ = media
    path = tmp_path / "sound.wav"
    path.write_bytes(b"mock audio")
    service.play(CompletionSoundSettings(True, str(path)))
    service.play(CompletionSoundSettings(True, str(path)))
    players[0].errorOccurred.emit("late old error")
    service._beep.assert_not_called()
    players[1].errorOccurred.emit("current error")
    service._beep.assert_called_once()


def test_stop_invalidates_pending_audio_errors(media, tmp_path):
    service, players, _, _, _ = media
    path = tmp_path / "sound.wav"
    path.write_bytes(b"mock audio")
    service.play(CompletionSoundSettings(True, str(path)))
    service.stop()
    players[0].errorOccurred.emit("error after window closed")
    service._beep.assert_not_called()


def test_missing_audio_backend_falls_back_without_raising(media, tmp_path):
    service, _, _, _, module = media
    path = tmp_path / "sound.wav"
    path.write_bytes(b"mock audio")
    module.QMediaPlayer = MagicMock(side_effect=RuntimeError("backend unavailable"))
    service.play(CompletionSoundSettings(True, str(path)))
    service._beep.assert_called_once()


def test_partial_audio_construction_is_cleaned_up(media, tmp_path):
    service, players, _, _, module = media
    path = tmp_path / "sound.wav"
    path.write_bytes(b"mock audio")
    module.QAudioOutput = MagicMock(side_effect=RuntimeError("no audio output"))
    service.play(CompletionSoundSettings(True, str(path)))
    players[0].stop.assert_called_once()
    players[0].deleteLater.assert_called_once()
    assert service._player is None
    service._beep.assert_called_once()
