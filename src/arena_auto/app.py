"""Application bootstrap."""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Optional

from arena_auto.config.manager import ConfigManager
from arena_auto.exceptions import ConfigError
from arena_auto.logging.logger import setup_logging

logger = logging.getLogger(__name__)


def main(*, ocr_provider: Optional[str] = None, ocr_device: Optional[str] = None) -> int:
    """Entry point used by main.py and python -m arena_auto."""
    try:
        config_manager = ConfigManager.load_or_create()
    except ConfigError as exc:
        setup_logging()
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication(sys.argv)
            QMessageBox.critical(None, "配置错误", f"发生错误：\n{exc}")
        except Exception:  # noqa: BLE001
            print(f"配置错误: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        setup_logging()
        logger.exception("配置加载失败")
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication(sys.argv)
            QMessageBox.critical(None, "错误", f"发生错误：\n{exc}")
        except Exception:  # noqa: BLE001
            print(f"错误: {exc}", file=sys.stderr)
        return 1

    # Dedicated launcher overrides only the current session until the user saves.
    if ocr_provider is not None:
        config_manager.config.ocr.provider = ocr_provider
    if ocr_device is not None:
        config_manager.config.ocr.device = ocr_device

    setup_logging(performance=bool(getattr(config_manager.config.logging, "performance", False)))

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from PySide6.QtGui import QIcon

        app = QApplication(sys.argv)
        app.setApplicationName("ArenaAuto")
        app.setOrganizationName("ArenaAuto")

        from arena_auto.paths import get_resource_path

        icon_path = get_resource_path("resources/app_icon.png")
        if icon_path.is_file():
            app.setWindowIcon(QIcon(str(icon_path)))

        from arena_auto.gui.main_window import MainWindow

        window = MainWindow(config_manager)
        window.show()
        return app.exec()
    except Exception as exc:  # noqa: BLE001
        logger.exception("GUI 启动失败")
        message = f"发生错误：\n{exc}\n\n{traceback.format_exc()}"
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is None:
                QApplication(sys.argv)
            QMessageBox.critical(None, "ArenaAuto 错误", message)
        except Exception:  # noqa: BLE001
            print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
