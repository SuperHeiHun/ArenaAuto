"""GUI-owned asynchronous end-of-run audio; never starts external processes."""

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QUrl
from PySide6.QtWidgets import QApplication

from arena_auto.models.config import CompletionSoundSettings
from arena_auto.paths import get_app_root

logger = logging.getLogger(__name__)


class CompletionSoundPlayer(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = None
        self._audio = None
        self._generation = 0

    @staticmethod
    def _beep():
        try:
            QApplication.beep()
        except Exception as exc:  # noqa: BLE001
            logger.warning("系统提示音不可用: %s", exc)

    def stop(self):
        self._generation += 1
        player, audio = self._player, self._audio
        self._player = self._audio = None
        if player is not None:
            try:
                player.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("停止音效失败: %s", exc)
            try:
                player.deleteLater()
            except Exception:  # noqa: BLE001
                pass
        if audio is not None:
            try:
                audio.deleteLater()
            except Exception:  # noqa: BLE001
                pass

    def play(self, settings: CompletionSoundSettings):
        if not settings.enabled:
            return
        fallback_played = False
        try:
            self.stop()
            if not settings.path.strip():
                self._beep()
                return
            path = Path(settings.path).expanduser()
            if not path.is_absolute():
                path = get_app_root() / path
            if not path.is_file():
                logger.warning("结束音效文件不存在，改用系统提示音: %s", path)
                self._beep()
                return
            # Lazy import: ordinary startup and default beeps need no decoder or
            # audio device. Media loading/decoding is asynchronous in Qt.
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            player = QMediaPlayer(self)
            self._player = player
            audio = QAudioOutput(self)
            self._audio = audio
            player.setAudioOutput(audio)
            generation = self._generation
            failed = False

            def fallback(*_args):
                nonlocal failed, fallback_played
                if generation != self._generation or failed:
                    return
                failed = True
                fallback_played = True
                logger.warning("结束音效无法播放，改用系统提示音: %s", path)
                self.stop()
                self._beep()

            player.errorOccurred.connect(fallback)
            player.mediaStatusChanged.connect(
                lambda status: fallback() if status == QMediaPlayer.MediaStatus.InvalidMedia else None
            )
            player.setSource(QUrl.fromLocalFile(str(path.resolve())))
            if generation == self._generation and self._player is player:
                player.play()
        except Exception as exc:  # noqa: BLE001
            logger.warning("结束音效播放失败，改用系统提示音: %s", exc)
            try:
                self.stop()
            except Exception:  # noqa: BLE001
                pass
            if not fallback_played:
                self._beep()
