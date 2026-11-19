"""Central logging setup for file + GUI callback sinks."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

from arena_auto.paths import get_log_dir

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


class CallbackHandler(logging.Handler):
    """Forward log records to a callable (GUI signal bridge).

    Formats as ``[LEVEL] message`` only — the GUI prepends ``HH:MM:SS``.
    """

    def __init__(self, callback: Callable[[str, str], None], level: int = logging.INFO):
        super().__init__(level=level)
        self._callback = callback
        self.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self._callback(record.levelname, message)
        except Exception:  # noqa: BLE001 - never let logging break the app
            self.handleError(record)


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.DEBUG,
    performance: bool = False,
) -> logging.Logger:
    """Configure root ArenaAuto logger with rotating file handler.

    performance=True also writes [PERF] lines to logs/performance.log
    (config.logging.performance).
    """
    global _configured
    root = logging.getLogger("arena_auto")
    root.setLevel(level)

    log_dir = log_dir or get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    if performance:
        try:
            from arena_auto.recognition.ocr_engine import set_performance_logging

            set_performance_logging(True)
        except Exception:  # noqa: BLE001
            pass
        perf_file = log_dir / "performance.log"
        perf_logger = logging.getLogger("arena_auto.performance")
        perf_logger.setLevel(logging.INFO)
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == str(perf_file)
            for h in perf_logger.handlers
        ):
            ph = RotatingFileHandler(
                perf_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
            )
            ph.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
            perf_logger.addHandler(ph)
        perf_logger.propagate = False

    if _configured:
        return root

    log_file = log_dir / "arena_auto.log"

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    root.addHandler(stream_handler)

    root.propagate = False
    _configured = True
    root.info("日志文件: %s", log_file)
    if performance:
        root.info("性能日志: %s", log_dir / "performance.log")
    return root


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("arena_auto"):
        name = f"arena_auto.{name}"
    return logging.getLogger(name)
