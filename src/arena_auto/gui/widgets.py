"""Reusable GUI widgets: image view and template capture dialog."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


def ndarray_to_qpixmap(image: np.ndarray) -> Optional[QPixmap]:
    """Convert BGR ndarray to QPixmap."""
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        h, w = image.shape[:2]
        qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())
    rgb = image[:, :, ::-1].copy()
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ImageView(QLabel):
    """Scroll-free image label that scales while keeping aspect ratio."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#202020;color:#aaa;")
        self.setText("无截图")
        self._pixmap: Optional[QPixmap] = None

    def set_image(self, image: Optional[np.ndarray]) -> None:
        pixmap = ndarray_to_qpixmap(image) if image is not None else None
        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.setText("无截图")
            self.setPixmap(QPixmap())
            return
        self.setText("")
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        super().setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)


class RubberBandImageLabel(ImageView):
    """Image label supporting rectangular rubber-band selection."""

    region_selected = Signal(int, int, int, int)  # x, y, w, h in image coords

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._origin = QPoint()
        self._band_end = QPoint()
        self._selecting = False
        self._image_size: Tuple[int, int] = (0, 0)  # w, h

    def set_image(self, image: Optional[np.ndarray]) -> None:
        if image is not None:
            h, w = image.shape[:2]
            self._image_size = (w, h)
        else:
            self._image_size = (0, 0)
        super().set_image(image)

    def _mapped_point(self, pos: QPoint) -> QPoint:
        pixmap = self._pixmap
        if pixmap is None or pixmap.isNull():
            return QPoint(-1, -1)
        # QLabel centers pixmap
        pw, ph = pixmap.width(), pixmap.height()
        lw, lh = self.width(), self.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        x = pos.x() - ox
        y = pos.y() - oy
        if x < 0 or y < 0 or x > pw or y > ph:
            return QPoint(-1, -1)
        img_w, img_h = self._image_size
        if img_w <= 0 or img_h <= 0 or pw <= 0 or ph <= 0:
            return QPoint(-1, -1)
        return QPoint(int(x * img_w / pw), int(y * img_h / ph))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            self._origin = event.position().toPoint()
            self._band_end = self._origin
            self._selecting = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._selecting:
            self._band_end = event.position().toPoint()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._selecting:
            self._selecting = False
            p0 = self._mapped_point(self._origin)
            p1 = self._mapped_point(event.position().toPoint())
            if p0.x() >= 0 and p1.x() >= 0:
                x0, x1 = sorted((p0.x(), p1.x()))
                y0, y1 = sorted((p0.y(), p1.y()))
                if x1 - x0 >= 4 and y1 - y0 >= 4:
                    self.region_selected.emit(x0, y0, x1 - x0, y1 - y0)
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._selecting and self._pixmap is not None:
            painter = QPainter(self)
            painter.setPen(Qt.red)
            rect = QRect(self._origin, self._band_end).normalized()
            painter.drawRect(rect)


class TemplateCaptureDialog(QDialog):
    """Crop a region from screenshot, name it, save as template PNG."""

    def __init__(
        self,
        image: np.ndarray,
        template_dir: Path,
        suggested_name: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("从当前截图创建模板")
        self.resize(900, 640)
        self.template_dir = Path(template_dir)
        self.selected_region: Optional[Tuple[int, int, int, int]] = None
        self.saved_path: Optional[Path] = None

        layout = QVBoxLayout(self)
        hint = QLabel("在截图上拖拽框选按钮区域，然后输入模板名称保存。")
        layout.addWidget(hint)

        self.image_label = RubberBandImageLabel()
        self.image_label.region_selected.connect(self._on_region)
        self.image_label.set_image(image)
        layout.addWidget(self.image_label, stretch=1)

        form = QHBoxLayout()
        form.addWidget(QLabel("模板名称:"))
        self.name_edit = QLineEdit(suggested_name)
        self.name_edit.setPlaceholderText("例如 go_win / exit / victory")
        form.addWidget(self.name_edit)
        self.region_label = QLabel("未选择区域")
        form.addWidget(self.region_label)
        layout.addLayout(form)

        self.status = QLabel("")
        layout.addWidget(self.status)

        buttons = QDialogButtonBox()
        self.save_btn = buttons.addButton("保存模板", QDialogButtonBox.AcceptRole)
        self.open_btn = buttons.addButton("打开目录", QDialogButtonBox.ActionRole)
        buttons.addButton(QDialogButtonBox.Close)
        layout.addWidget(buttons)

        self.save_btn.clicked.connect(self._save)
        self.open_btn.clicked.connect(self._open_dir)
        buttons.rejected.connect(self.reject)

        self._image = image

    def _on_region(self, x: int, y: int, w: int, h: int) -> None:
        self.selected_region = (x, y, w, h)
        self.region_label.setText(f"ROI: [{x}, {y}, {w}, {h}]")

    def _save(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "模板", "请输入模板名称")
            return
        if self.selected_region is None:
            QMessageBox.warning(self, "模板", "请先框选区域")
            return
        try:
            import cv2

            x, y, w, h = self.selected_region
            crop = self._image[y : y + h, x : x + w]
            self.template_dir.mkdir(parents=True, exist_ok=True)
            path = self.template_dir / f"{name}.png"
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                raise RuntimeError("cv2.imencode 失败")
            buf.tofile(str(path))
            self.saved_path = path
            self.status.setText(f"已保存: {path}")
            # also write config snippet hint
            snip = self.template_dir / f"{name}_region.txt"
            snip.write_text(f"{name}: [{x}, {y}, {w}, {h}]\n", encoding="utf-8")
            QMessageBox.information(
                self,
                "模板",
                f"已保存模板:\n{path}\n\n请在 config.yaml 的 templates 中确认路径与阈值。",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "模板", f"保存失败:\n{exc}")

    def _open_dir(self) -> None:
        import os

        self.template_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.template_dir))  # Windows; guarded by platform usage
