"""Domain-specific exceptions."""

from __future__ import annotations


class ArenaAutoError(Exception):
    """Base error for ArenaAuto."""


class ConfigError(ArenaAutoError):
    """Invalid or unreadable configuration."""


class AdbError(ArenaAutoError):
    """ADB execution failure."""


class AdbNotFoundError(AdbError):
    """ADB executable missing."""


class AdbDisconnectedError(AdbError):
    """Device connection lost."""


class ScreenshotError(AdbError):
    """Screenshot capture failed."""


class OCRProviderError(ArenaAutoError):
    """No usable OCR engine / OCR engine failed to initialize."""


class RecognitionError(ArenaAutoError):
    """Screen / button / power recognition failed safely."""


class OCRFailedError(RecognitionError):
    """OCR could not determine a value - never coerce to 0."""


class TemplateMissingError(RecognitionError):
    """Template file is absent."""


class StateTimeoutError(ArenaAutoError):
    """A state exceeded its configured timeout."""


class RecoveryFailedError(ArenaAutoError):
    """Recovery could not restore a known page."""


class StoppedError(ArenaAutoError):
    """User requested stop."""
