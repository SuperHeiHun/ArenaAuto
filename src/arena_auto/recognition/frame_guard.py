"""Fresh-frame visual validation for previously observed geometry only."""

from dataclasses import dataclass
import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameGuardReport:
    unchanged: bool
    reason: str
    region: tuple = ()
    mean: float = 0.0
    max_block: float = 0.0
    changed_blocks: int = 0
    # At most five largest changed blocks, in absolute screenshot coordinates.
    blocks: tuple = ()

    def describe(self):
        return (f"{self.reason}；区域={self.region}；平均差={self.mean:.2f}/1.50；"
                f"最大块差={self.max_block:.2f}/3.00；变化块={self.changed_blocks}；"
                f"最大变化位置={self.blocks}")


def regions_unchanged(previous: np.ndarray, current: np.ndarray, regions) -> bool:
    return compare_regions(previous, current, regions).unchanged


def compare_regions(previous: np.ndarray, current: np.ndarray, regions) -> FrameGuardReport:
    """Require near-identical patches, including local changes within a patch.

    No OCR evidence or power value is produced here. Invalid geometry, changed
    resolution, a moved label/card or a changed local block rejects reuse.
    """
    if previous is None or current is None:
        return FrameGuardReport(False, "缺少截图")
    if previous.shape != current.shape:
        return FrameGuardReport(False, "截图尺寸变化")
    height, width = current.shape[:2]
    if not regions:
        return FrameGuardReport(False, "缺少校验区域")
    for x, y, w, h in regions:
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            return FrameGuardReport(False, "校验区域无效", (x, y, w, h))
        old = previous[y:y + h, x:x + w]
        new = current[y:y + h, x:x + w]
        # Keep colour: similar-luminance replacement portraits must also reject
        # the hint. A small blur tolerates isolated compression noise only.
        diff = cv2.absdiff(cv2.GaussianBlur(old, (3, 3), 0), cv2.GaussianBlur(new, (3, 3), 0))
        mean = float(diff.mean())
        max_block = 0.0
        changed = []
        for top in range(0, h, 16):
            for left in range(0, w, 16):
                value = float(diff[top:top + 16, left:left + 16].mean())
                max_block = max(max_block, value)
                if value > 3.0:
                    changed.append((value, (x + left, y + top, min(16, w - left), min(16, h - top))))
        if mean > 1.5 or changed:
            blocks = tuple(block for _, block in sorted(changed, reverse=True)[:5])
            return FrameGuardReport(False, "区域平均变化" if mean > 1.5 else "局部块变化",
                                    (x, y, w, h), mean, max_block, len(changed), blocks)
    return FrameGuardReport(True, "画面未变化")


def save_guard_evidence(recorder, previous, current, report, label):
    """Optional bounded debug evidence; errors must not change control flow."""
    if recorder is None or not recorder.enabled or report.unchanged:
        return
    try:
        tag = f"{label}_{time.monotonic_ns()}"
        marked = current.copy()
        for x, y, w, h in report.blocks:
            cv2.rectangle(marked, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 2)
        for suffix, image in (("before", previous), ("after", current), ("changed", marked)):
            path = recorder.save(image, category="debug", tag=f"{tag}_{suffix}")
            if path is not None:
                logger.info("复用校验截图：%s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("复用校验证据保存失败: %s", exc)
